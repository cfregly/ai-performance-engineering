"""Reproduce original arithmetic/metadata defects without mocking GPU execution.

Run with PYTHONPATH=code /opt/miniconda3/bin/python this_file.py OUTPUT.json.
Output uses exclusive creation. Original source is read from the reviewed commit.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Optional

import torch
from core.benchmark.metrics import BLACKWELL_B200, compute_copy_bandwidth_metrics, compute_roofline_metrics
from core.benchmark.performance_targets import _get_peak_values
from core.diagnostics.microbench import _prepare_fp8_matmul
from core.harness.arch_config import ArchitectureConfig

ROOT = Path(__file__).resolve().parents[5]
BASE = "b57e4c6a9e261c09ac09208705d040c81b03d35e"


def original(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT, text=True)


def module_from_original(path, name):
    source = original(path)
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / path)
    sys.modules[name] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def assignment_expression(path, function_name, assigned_name):
    tree = ast.parse(original(path))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == function_name)
    assignment = next(node for node in ast.walk(function) if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == assigned_name for target in node.targets))
    return compile(ast.Expression(assignment.value), f"original_{function_name}", "eval")


def main():
    old_metrics = module_from_original("code/core/benchmark/metrics.py", "audit_original_metrics")
    old_targets = module_from_original("code/core/benchmark/performance_targets.py", "audit_original_targets")
    old_hbm_expr = assignment_expression("code/core/benchmark/benchmark_peak.py", "measure_hbm_bandwidth", "bandwidth_gbs")
    old_peer_expr = assignment_expression("code/core/benchmark/benchmark_peak.py", "measure_nvlink_bandwidth", "bandwidth_gbs")
    tree = ast.parse(original("code/core/harness/arch_config.py"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ArchitectureConfig")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_sanitize_arch_value")
    namespace = {"Optional": Optional}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "original_arch_method", "exec"), namespace)
    with tempfile.TemporaryDirectory(prefix="aisp-peak-original-") as tmp:
        path = Path(tmp) / "benchmark_peak_results_original_fixture.json"
        path.write_text(json.dumps({"hbm": {"peak_bandwidth_tbs": 3.5}}))
        before_bytes = path.read_bytes()
        old_peaks = old_targets._get_peak_values(Path(tmp))
        new_peaks = _get_peak_values(Path(tmp))
        preserved = path.read_bytes() == before_bytes
    try:
        torch.randn((16, 16), dtype=torch.float8_e4m3fn, device="cpu")
        old_fp8 = {"succeeded": True}
    except Exception as exc:
        old_fp8 = {"succeeded": False, "error_type": type(exc).__name__, "error": str(exc)}
    a, b, _ = _prepare_fp8_matmul(16, torch.device("cpu"))
    output = {
        "reviewed_commit": BASE,
        "environment": {"python": sys.version, "torch": torch.__version__, "cuda_available": torch.cuda.is_available()},
        "scope": "CPU arithmetic, real artifact loader, real CPU tensor preparation, and source-extracted metadata method; no GPU measurements",
        "hbm_bytes": {
            "original_expression_gbs": eval(old_hbm_expr, {"size_bytes": 2**30, "iterations": 5, "elapsed_s": 5}),
            "corrected_gbs": compute_copy_bandwidth_metrics(2**30, 5, 5000)["peak_bandwidth_gbs"],
        },
        "peer_bytes": {
            "original_expression_gbs": eval(old_peer_expr, {"size_mb": 1024, "iterations": 20, "elapsed_ms": 20000}),
            "corrected_gbs": compute_copy_bandwidth_metrics(2**30, 20, 20000, traffic="one_way_payload")["peak_bandwidth_gbs"],
        },
        "b200_fp8_dense_tflops": {"original": old_metrics.BLACKWELL_B200.fp8_tflops, "corrected": BLACKWELL_B200.fp8_tflops},
        "ridge_classification_ai400": {
            "original": old_metrics.compute_roofline_metrics(4e12, 1e10, None, "fp8", old_metrics.BLACKWELL_B200),
            "corrected": compute_roofline_metrics(4e12, 1e10, None, "fp8", BLACKWELL_B200),
        },
        "memory_ceiling": {
            "original_1ms": old_metrics.compute_roofline_metrics(4e9, 1e9, 1, "fp8", old_metrics.BLACKWELL_B200)["roofline.memory_ceiling_tflops"],
            "original_2ms": old_metrics.compute_roofline_metrics(4e9, 1e9, 2, "fp8", old_metrics.BLACKWELL_B200)["roofline.memory_ceiling_tflops"],
            "corrected_1ms": compute_roofline_metrics(4e9, 1e9, 1, "fp8", BLACKWELL_B200)["roofline.memory_ceiling_tflops"],
            "corrected_2ms": compute_roofline_metrics(4e9, 1e9, 2, "fp8", BLACKWELL_B200)["roofline.memory_ceiling_tflops"],
        },
        "legacy_hbm_artifact": {"original_peaks": old_peaks[0], "corrected_peaks": new_peaks[0], "corrected_warnings": new_peaks[1], "preserved": preserved},
        "target_rewrite": {"original": namespace["_sanitize_arch_value"](None, "sm_121a"), "corrected": ArchitectureConfig._sanitize_arch_value(None, "sm_121a")},
        "float8_operand_preparation": {"original_randn": old_fp8, "corrected_a_dtype": str(a.dtype), "corrected_b_dtype": str(b.dtype), "a_stride": a.stride(), "b_stride": b.stride()},
    }
    with Path(sys.argv[1]).open("x") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
