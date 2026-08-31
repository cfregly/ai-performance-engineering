#!/usr/bin/env python3
"""Bound the actual two-GPU/Triton accuracy and memory-sanitizer gate."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import torch

from validation_process import run_command


REQUIRED_GPU_CASES = {
    "test_actual_cuda_sorted_pytorch_reference_covers_full_output": 1,
    "test_active_triton_activation_matches_full_reference_and_updates_tail": 27,
    "test_cuda_trainable_activation_uses_pytorch_and_preserves_backward": 1,
    "test_triton_inference_honors_noncurrent_device_and_restores_caller": 1,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    if not 0 < args.timeout <= 900:
        parser.error("timeout must be positive and at most 900 seconds")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir.resolve()
    report = {"status": "HOLD", "torch": torch.__version__, "cuda": torch.version.cuda,
              "required_gpu_cases": REQUIRED_GPU_CASES, "runs": [],
              "scope": "Active activation and explicit PyTorch reference only; retired FFN kernels remain unavailable"}
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            report["reason"] = "Actual two CUDA devices required; no GPU or Triton execution attempted"
            return 3
        try:
            import triton
        except ImportError as exc:
            report["reason"] = f"Actual Triton dependency unavailable: {exc}"
            return 3
        sanitizer = shutil.which("compute-sanitizer")
        if sanitizer is None:
            report["reason"] = "compute-sanitizer required for the memory gate"
            return 3
        report["triton"] = triton.__version__
        report["devices"] = []
        for index in (0, 1):
            with torch.cuda.device(index):
                report["devices"].append({"index": index, "name": torch.cuda.get_device_name(index),
                                          "capability": list(torch.cuda.get_device_capability(index))})
                if not torch.cuda.is_bf16_supported():
                    report["reason"] = f"GPU {index} does not support the required BF16 cases"
                    return 3
        code = Path(__file__).resolve().parents[2]
        os.chdir(code)
        os.environ["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
        for mode in ("plain", "memcheck"):
            xml = output / f"{mode}.xml"
            command = [sys.executable, "-m", "pytest", "-q", "-s", "-rs", "-p", "no:cacheprovider",
                       "tests/test_audit_wave1_moe_triton.py", f"--junitxml={xml}"]
            if mode == "memcheck":
                command = [sanitizer, "--tool", "memcheck", "--target-processes", "all",
                           "--error-exitcode", "9", *command]
            result = run_command(command, timeout=args.timeout)
            (output / f"{mode}.stdout.txt").write_text(result.stdout)
            (output / f"{mode}.stderr.txt").write_text(result.stderr)
            run = {"mode": mode, "command": command, "exit_code": result.returncode}
            report["runs"].append(run)
            if result.returncode != 0 or not xml.exists():
                report.update(status="FAIL", reason=f"{mode} failed, timed out, or omitted results")
                return 1
            cases = list(ET.parse(xml).iter("testcase"))
            counts = {name: sum(case.attrib.get("name", "").split("[", 1)[0] == name for case in cases)
                      for name in REQUIRED_GPU_CASES}
            run["observed_gpu_case_counts"] = counts
            if counts != REQUIRED_GPU_CASES or any(child.tag in {"skipped", "failure", "error"} for case in cases for child in case):
                report.update(status="FAIL", reason=f"{mode} had a missing, skipped, failed, or error case")
                return 1
        report["status"] = "PASS"
        return 0
    except Exception as exc:
        report.update(status="FAIL", reason=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        (output / "receipt.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
