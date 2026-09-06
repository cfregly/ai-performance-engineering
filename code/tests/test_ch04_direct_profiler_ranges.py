from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.profiling.nvtx_helper import canonicalize_nvtx_name
from core.profiling.profiler_config import build_profiler_config_from_benchmark

PROFILE_CASES = (
    (
        "ch04.baseline_pipeline_parallel",
        "BaselinePipelineParallelBenchmark",
        "compute_kernel:pipeline_parallel_gpipe",
    ),
    (
        "ch04.optimized_pipeline_parallel_1f1b",
        "OptimizedPipelineParallelBenchmark",
        "compute_kernel:pipeline_parallel_1f1b",
    ),
    (
        "ch04.baseline_pipeline_parallel_multigpu",
        "BaselinePipelineParallelBenchmark",
        "compute_kernel:pipeline_parallel_multigpu_gpipe",
    ),
    (
        "ch04.optimized_pipeline_parallel_multigpu_1f1b",
        "OptimizedPipelineParallelBenchmark",
        "compute_kernel:pipeline_parallel_multigpu_1f1b",
    ),
    (
        "ch04.baseline_tensor_parallel",
        "BaselineTensorParallelBenchmark",
        "compute_kernel:tensor_parallel_sync_allgather",
    ),
    (
        "ch04.optimized_tensor_parallel_async",
        "OptimizedTensorParallelBenchmark",
        "compute_kernel:tensor_parallel_async_allgather",
    ),
    (
        "ch04.baseline_tensor_parallel_allgather_multigpu",
        "BaselineTensorParallelAllGatherBenchmark",
        "compute_kernel:tensor_parallel_allgather_reference",
    ),
    (
        "ch04.optimized_tensor_parallel_allgather_multigpu",
        "OptimizedTensorParallelAllGatherBenchmark",
        "compute_kernel:tensor_parallel_allgather_inplace",
    ),
    (
        "ch04.baseline_tensor_parallel_multigpu",
        "BaselineTensorParallelBenchmark",
        "compute_kernel:tensor_parallel_allgather_with_barriers",
    ),
    (
        "ch04.optimized_tensor_parallel_multigpu",
        "OptimizedTensorParallelBenchmark",
        "compute_kernel:tensor_parallel_allgather_reduced_sync",
    ),
    (
        "ch04.baseline_torchcomms",
        "BaselineTorchcommsBenchmark",
        "compute_kernel:torchcomms_legacy",
    ),
    (
        "ch04.optimized_torchcomms",
        "OptimizedTorchcommsBenchmark",
        "compute_kernel:torchcomms_functional",
    ),
    (
        "ch04.baseline_torchcomms_multigpu",
        "BaselineTorchcommsBenchmark",
        "compute_kernel:torchcomms_blocking_allreduce",
    ),
    (
        "ch04.optimized_torchcomms_multigpu",
        "OptimizedTorchcommsBenchmark",
        "compute_kernel:torchcomms_async_allreduce",
    ),
)


@pytest.mark.parametrize(("module_name", "benchmark_class", "expected_range"), PROFILE_CASES)
def test_direct_profile_config_selects_the_target_owned_range(
    module_name: str,
    benchmark_class: str,
    expected_range: str,
) -> None:
    module = importlib.import_module(module_name)
    benchmark_type = getattr(module, benchmark_class)
    benchmark = benchmark_type.__new__(benchmark_type)

    config = benchmark.get_config()
    assert expected_range == module.PROFILE_NVTX_RANGE
    assert canonicalize_nvtx_name(expected_range) == expected_range
    assert config.profiling.nsys_nvtx_include == (expected_range,)

    profiler = build_profiler_config_from_benchmark(config)
    command = profiler.get_ncu_command_for_target(
        "/tmp/ch04-direct-profile",
        ["python", str(Path(module.__file__).resolve())],
    )
    assert command.count("--nvtx-include") == 1
    include_index = command.index("--nvtx-include")
    assert command[include_index + 1] == expected_range
    assert command[command.index("--replay-mode") + 1] == "app-range"


@pytest.mark.parametrize(("module_name", "_benchmark_class", "_expected_range"), PROFILE_CASES)
def test_explicit_worker_adapter_invokes_the_real_main_cli(
    module_name: str, _benchmark_class: str, _expected_range: str,
) -> None:
    code_root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-m", "core.harness.benchmark_worker", "--module", module_name,
         "--callable", "main", "--", "--iters", "1", "--help"],
        cwd=code_root,
        env={**os.environ, "PYTHONPATH": str(code_root), "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True, text=True, timeout=30, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # These flags belong to the requested main, not the adapter's own parser.
    assert "--iters" in result.stdout and "--warmup" in result.stdout


def _call_name(node: ast.AST) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    return ast.unparse(node.func)


@pytest.mark.parametrize(("module_name", "_benchmark_class", "_expected_range"), PROFILE_CASES)
def test_target_range_encloses_only_the_post_warmup_timed_loop(
    module_name: str,
    _benchmark_class: str,
    _expected_range: str,
) -> None:
    module = importlib.import_module(module_name)
    source = Path(module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    worker = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_run_worker"
    )
    profile_ranges = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.With)
        and any(_call_name(item.context_expr) == "nvtx_range" for item in node.items)
    ]
    assert len(profile_ranges) == 1
    profile_range = profile_ranges[0]
    range_call = profile_range.items[0].context_expr
    assert isinstance(range_call, ast.Call)
    assert isinstance(range_call.args[0], ast.Name)
    assert range_call.args[0].id == "PROFILE_NVTX_RANGE"
    assert any(
        keyword.arg == "enable"
        and isinstance(keyword.value, ast.Constant)
        and keyword.value.value is True
        for keyword in range_call.keywords
    )

    warmup_loops = [
        node
        for node in ast.walk(worker)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "range(max(warmup, 0))"
    ]
    timed_loops = [
        node
        for node in ast.walk(profile_range)
        if isinstance(node, ast.For) and ast.unparse(node.iter) == "range(max(iters, 1))"
    ]
    assert len(warmup_loops) == 1
    assert len(timed_loops) == 1
    assert warmup_loops[0].end_lineno < profile_range.lineno < timed_loops[0].lineno

    range_calls = {
        _call_name(node)
        for node in ast.walk(profile_range)
        if isinstance(node, ast.Call)
    }
    assert "time.perf_counter" in range_calls
    assert "torch.cuda.synchronize" in range_calls
    assert any(
        isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "elapsed" for target in node.targets)
        for node in profile_range.body
    )
