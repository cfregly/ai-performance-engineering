"""Environment probe for nvfp4_dual_gemm tuning on B200.

Captures:
- driver/CUDA/GPU clock metadata from nvidia-smi
- BF16 GEMM throughput under locked SM clocks (default 1500 and max app clock)
- NVML current/app clock snapshots around the GEMM run
"""

from __future__ import annotations

import argparse
import json
import subprocess
from typing import Any

import torch

from core.harness.benchmark_harness import lock_gpu_clocks


def _run_cmd(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)


def _nvidia_smi_query(fields: list[str]) -> dict[str, str]:
    q = ",".join(fields)
    out = _run_cmd(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"]).strip()
    vals = [x.strip() for x in out.split(",")]
    return dict(zip(fields, vals))


def _nvml_clocks(device: int = 0) -> dict[str, int]:
    import pynvml

    pynvml.nvmlInit()
    try:
        h = pynvml.nvmlDeviceGetHandleByIndex(device)
        app_sm = int(pynvml.nvmlDeviceGetApplicationsClock(h, pynvml.NVML_CLOCK_SM))
        app_mem = int(pynvml.nvmlDeviceGetApplicationsClock(h, pynvml.NVML_CLOCK_MEM))
        cur_sm = int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
        cur_mem = int(pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
        return {
            "app_sm_mhz": app_sm,
            "app_mem_mhz": app_mem,
            "cur_sm_mhz": cur_sm,
            "cur_mem_mhz": cur_mem,
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _gemm_tflops(size: int, iters: int, warmup: int, dtype: torch.dtype) -> dict[str, Any]:
    a = torch.randn((size, size), device="cuda", dtype=dtype)
    b = torch.randn((size, size), device="cuda", dtype=dtype)

    for _ in range(warmup):
        _ = a @ b
    torch.cuda.synchronize()

    # Snapshot under load after warmup
    clocks_before = _nvml_clocks(0)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    start.record(current_stream)
    for _ in range(iters):
        _ = a @ b
    end.record(current_stream)
    end.synchronize()

    clocks_after = _nvml_clocks(0)

    total_ms = float(start.elapsed_time(end))
    total_s = total_ms / 1000.0
    flops = 2.0 * (size**3) * iters
    tflops = flops / total_s / 1e12

    return {
        "size": size,
        "iters": iters,
        "warmup": warmup,
        "dtype": str(dtype),
        "total_ms": total_ms,
        "tflops": tflops,
        "clocks_before": clocks_before,
        "clocks_after": clocks_after,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=8192)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--sm-clock-low", type=int, default=1500)
    parser.add_argument("--sm-clock-high", type=int, default=None)
    parser.add_argument("--mem-clock-mhz", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    base_info = _nvidia_smi_query(
        [
            "name",
            "driver_version",
            "clocks.max.sm",
            "clocks.max.memory",
            "clocks.applications.graphics",
            "clocks.applications.memory",
            "temperature.gpu",
            "power.draw",
            "pstate",
        ]
    )

    max_sm = int(base_info["clocks.max.sm"])
    high_sm = int(args.sm_clock_high) if args.sm_clock_high is not None else max_sm

    runs = []
    for sm in [int(args.sm_clock_low), int(high_sm)]:
        with lock_gpu_clocks(device=0, sm_clock_mhz=sm, mem_clock_mhz=args.mem_clock_mhz):
            torch.cuda.synchronize()
            bench = _gemm_tflops(
                size=int(args.size),
                iters=int(args.iters),
                warmup=int(args.warmup),
                dtype=torch.bfloat16,
            )
            bench["requested_sm_clock_mhz"] = sm
            runs.append(bench)

    payload = {
        "gpu": base_info,
        "runs": runs,
        "notes": [
            "If cur_sm_mhz in clocks_before/clocks_after is far below requested_sm_clock_mhz under load, clock lock or power/thermal policy may be limiting.",
            "Compare TFLOPS between low/high SM clock runs for sanity on clock scaling.",
        ],
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
