"""Compile and run a real CUDA TMA sample; never treat missing tools as success."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path


def tma_sample_source() -> Path:
    # This sample takes the descriptor TMA path or exits unsupported; unlike the
    # pipeline demo, it does not silently switch to a manual-copy baseline.
    return Path(__file__).resolve().parents[2] / "ch07/async_prefetch_2d_demo.cu"


def run_cuda_tma_sample(*, nvcc: str = "nvcc") -> str:
    """Return pass/fail/skip for compilation and sample execution only.

    The caller must check that the active device is SM121 before invoking this
    operation. A sample exit code is not full-output or TMA-instruction evidence.
    """
    compiler = shutil.which(nvcc)
    if compiler is None:
        print("SKIPPED: nvcc is unavailable; no CUDA compilation or execution occurred")
        return "skip"
    source = tma_sample_source()
    if not source.is_file():
        print(f"ERROR: CUDA TMA sample is missing: {source}")
        return "fail"

    with tempfile.TemporaryDirectory(prefix="aisp-tma-sm121-") as directory:
        binary = Path(directory) / "tma_sample_sm121"
        command = [compiler, "-O3", "-std=c++17", "--expt-relaxed-constexpr",
                   "-arch=sm_121", str(source), "-o", str(binary), "-lcuda"]
        print(f"CUDA sample source: {source}")
        print(f"CUDA compile command: {command!r}")
        try:
            compilation = subprocess.run(command, capture_output=True, text=True, check=False, timeout=120)
            print(compilation.stdout, end="")
            print(compilation.stderr, end="")
            if compilation.returncode != 0 or not binary.is_file():
                print(f"ERROR: CUDA sample compilation failed (exit {compilation.returncode})")
                return "fail"
            execution = subprocess.run([str(binary)], capture_output=True, text=True, check=False, timeout=30)
            print(execution.stdout, end="")
            print(execution.stderr, end="")
            if execution.returncode == 3:
                print("SKIPPED: the CUDA sample reported unsupported TMA hardware/runtime")
                return "skip"
            if execution.returncode != 0:
                print(f"ERROR: CUDA sample execution failed (exit {execution.returncode})")
                return "fail"
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"ERROR: CUDA sample compile/run failed: {exc}")
            return "fail"
    print("CUDA sample compilation and execution: PASSED")
    print("Scope: sample execution only; full-output and instruction-level TMA validation remain separate.")
    return "pass"
