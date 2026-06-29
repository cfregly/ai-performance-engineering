"""Exploratory sweep for tcgen05 CUTLASS 1-SM vs 2-SM vs 4-SM cluster shapes."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Callable

import torch

from core.benchmark.tcgen05_requirements import ensure_tcgen05_supported
from core.common.tcgen05 import (
    load_tcgen05_warp_specialized_cutlass_module,
    load_tcgen05_warpgroup_specialized_module,
)
from core.harness.benchmark_harness import lock_gpu_clocks
from labs.tcgen05_cluster_shapes import cutlass_sm100_supports_4sm


def _benchmark(fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], a: torch.Tensor, b: torch.Tensor, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    start.record(current_stream)
    for _ in range(iterations):
        fn(a, b)
    end.record(current_stream)
    end.synchronize()
    return start.elapsed_time(end) / iterations


def _run_variant(name: str, fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], a: torch.Tensor, b: torch.Tensor, warmup: int, iterations: int) -> dict:
    mean_ms = _benchmark(fn, a, b, warmup=warmup, iterations=iterations)
    return {
        "variant": name,
        "status": "ok",
        "mean_ms": mean_ms,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--no-lock-gpu-clocks",
        dest="lock_gpu_clocks",
        action="store_false",
        help="Skip harness clock locking for local experimentation",
    )
    parser.add_argument("--sm-clock-mhz", type=int, default=None, help="Optional SM application clock")
    parser.add_argument("--mem-clock-mhz", type=int, default=None, help="Optional memory application clock")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.set_defaults(lock_gpu_clocks=True)
    args = parser.parse_args(argv)

    ensure_tcgen05_supported(
        loader=load_tcgen05_warp_specialized_cutlass_module,
        module_name="labs tcgen05 cluster-shape sweep (1sm)",
    )
    ensure_tcgen05_supported(
        loader=load_tcgen05_warpgroup_specialized_module,
        module_name="labs tcgen05 cluster-shape sweep (2sm)",
    )

    lock_ctx = (
        lock_gpu_clocks(device=0, sm_clock_mhz=args.sm_clock_mhz, mem_clock_mhz=args.mem_clock_mhz)
        if args.lock_gpu_clocks
        else nullcontext()
    )
    with lock_ctx:
        torch.manual_seed(0)
        a = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16).contiguous()
        b = torch.randn((args.n, args.k), device="cuda", dtype=torch.float16).contiguous()

        module_1sm = load_tcgen05_warp_specialized_cutlass_module()
        module_2sm = load_tcgen05_warpgroup_specialized_module()

        results = [
            _run_variant("1sm", module_1sm.matmul_tcgen05_warp_specialized_cutlass, a, b, args.warmup, args.iterations),
            _run_variant("2sm", module_2sm.matmul_tcgen05_warpgroup_specialized, a, b, args.warmup, args.iterations),
        ]

        if cutlass_sm100_supports_4sm():
            results.append(
                {
                    "variant": "4sm",
                    "status": "not_implemented",
                    "reason": "CUTLASS exposes a 4-SM schedule symbol, but this lab does not yet ship a 4-SM kernel wrapper.",
                }
            )
        else:
            results.append(
                {
                    "variant": "4sm",
                    "status": "unsupported_by_cutlass",
                    "reason": "Current CUTLASS SM100 dense-gemm headers expose 1-SM and 2-SM schedules only.",
                }
            )

        baseline_ms = next((row["mean_ms"] for row in results if row.get("variant") == "1sm" and row.get("status") == "ok"), None)
        for row in results:
            if baseline_ms and row.get("status") == "ok":
                row["relative_to_1sm"] = baseline_ms / row["mean_ms"]

    payload = {
        "problem": {"m": args.m, "n": args.n, "k": args.k},
        "warmup": args.warmup,
        "iterations": args.iterations,
        "lock_gpu_clocks": bool(args.lock_gpu_clocks),
        "sm_clock_mhz": args.sm_clock_mhz,
        "mem_clock_mhz": args.mem_clock_mhz,
        "cutlass_supports_4sm": cutlass_sm100_supports_4sm(),
        "results": results,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
