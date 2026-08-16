#!/usr/bin/env python3
"""
Control-plane collective benchmark.

Compares synchronization/completion-signaling patterns used by distributed apps:
1) all_gather_object (Python object collective)
2) all_gather on a 1-element tensor
3) all_reduce on a 1-element tensor

This mirrors the high-impact ml-engineering microbenchmarks that show how
`all_gather_object` can become a major latency bottleneck in tight loops.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os

from core.common.device_utils import resolve_local_rank
import socket
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev

import torch
import torch.distributed as dist

if __package__ in {None, ""}:
    _repo_root = Path(__file__).resolve().parents[2]
    _env = os.environ.copy()
    _pythonpath = _env.get("PYTHONPATH")
    _env["PYTHONPATH"] = str(_repo_root) if not _pythonpath else os.pathsep.join([str(_repo_root), _pythonpath])
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "cluster.scripts.allgather_control_plane_bench", *sys.argv[1:]],
        _env,
    )

try:
    from core.harness.benchmark_harness import _resolve_physical_device_index  # type: ignore
except Exception:
    _resolve_physical_device_index = None  # type: ignore[assignment]


def _percentile(samples: list[float], pct: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return samples[0]
    ordered = sorted(samples)
    position = (len(ordered) - 1) * (pct / 100.0)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _nccl_version_str() -> str:
    try:
        version = torch.cuda.nccl.version()
    except Exception:
        return "unknown"
    if isinstance(version, (tuple, list)):
        return ".".join(str(x) for x in version)
    return str(version)


def _clock_snapshot(logical_device: int, hostname: str, rank: int) -> dict:
    physical_device = logical_device
    if _resolve_physical_device_index is not None:
        try:
            physical_device = int(_resolve_physical_device_index(logical_device))
        except Exception:
            physical_device = logical_device

    try:
        import pynvml
    except Exception as exc:
        return {
            "rank": rank,
            "hostname": hostname,
            "logical_device": logical_device,
            "physical_device": physical_device,
            "error": f"pynvml unavailable: {exc}",
        }

    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_device)
        gpu_name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(gpu_name, bytes):
            gpu_name = gpu_name.decode("utf-8")
        return {
            "rank": rank,
            "hostname": hostname,
            "logical_device": logical_device,
            "physical_device": physical_device,
            "gpu_name": gpu_name,
            "app_sm_mhz": int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM)),
            "app_mem_mhz": int(pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM)),
            "cur_sm_mhz": int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)),
            "cur_mem_mhz": int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)),
        }
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _run_method(
    name: str,
    fn,
    iters: int,
    warmup: int,
    local_rank: int,
    is_rank0: bool,
) -> dict:
    # Warm up comm/runtime path first.
    for _ in range(warmup):
        dist.barrier()
        fn()
        torch.cuda.synchronize(local_rank)
        dist.barrier()

    per_iter_ms: list[float] = []
    for _ in range(iters):
        dist.barrier()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize(local_rank)
        t1 = time.perf_counter()

        local_ms = (t1 - t0) * 1000.0
        slowest_rank_ms = torch.tensor([local_ms], dtype=torch.float64, device=f"cuda:{local_rank}")
        dist.reduce(slowest_rank_ms, dst=0, op=dist.ReduceOp.MAX)

        if is_rank0:
            per_iter_ms.append(float(slowest_rank_ms.item()))

    if not is_rank0:
        return {}

    ms_mean = mean(per_iter_ms)
    ms_std = stdev(per_iter_ms) if len(per_iter_ms) > 1 else 0.0
    return {
        "method": name,
        "iters": iters,
        "warmup": warmup,
        "latency_ms": {
            "mean": round(ms_mean, 6),
            "median": round(median(per_iter_ms), 6),
            "min": round(min(per_iter_ms), 6),
            "max": round(max(per_iter_ms), 6),
            "p95": round(_percentile(per_iter_ms, 95.0), 6),
            "p99": round(_percentile(per_iter_ms, 99.0), 6),
            "std": round(ms_std, 6),
            "cv_pct": round((ms_std / ms_mean * 100.0) if ms_mean > 0 else 0.0, 6),
            "samples": [round(x, 6) for x in per_iter_ms],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Control-plane collective benchmark")
    parser.add_argument("--iters", type=int, default=2000, help="Measured iterations per method (default: 2000)")
    parser.add_argument("--warmup", type=int, default=200, help="Warmup iterations per method (default: 200)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    if os.environ.get("AISP_CLOCK_LOCKED") != "1":
        raise SystemExit(
            "ERROR: GPU clock lock is required for control-plane collective benchmark.\n"
            "Run via scripts/run_with_gpu_clocks.sh (or wrapper scripts that call it)."
        )
    if args.iters <= 0:
        raise SystemExit(f"ERROR: --iters must be > 0 (got {args.iters})")
    if args.warmup < 0:
        raise SystemExit(f"ERROR: --warmup must be >= 0 (got {args.warmup})")

    dist.init_process_group(backend="nccl")
    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_rank0 = rank == 0
    hostname = socket.gethostname()

    local_clock = _clock_snapshot(local_rank, hostname, rank)
    all_clocks = [None for _ in range(world_size)]
    dist.all_gather_object(all_clocks, local_clock)

    done = torch.zeros(1, dtype=torch.int32, device=f"cuda:{local_rank}")
    gathered_tensors = [torch.zeros_like(done) for _ in range(world_size)]
    gathered_objects = [False for _ in range(world_size)]

    def _all_gather_object_fn() -> None:
        nonlocal gathered_objects
        dist.all_gather_object(gathered_objects, True)

    def _all_gather_tensor_fn() -> None:
        nonlocal done, gathered_tensors
        done.fill_(rank)
        dist.all_gather(gathered_tensors, done)

    def _all_reduce_tensor_fn() -> None:
        nonlocal done
        done.fill_(1)
        dist.all_reduce(done, op=dist.ReduceOp.SUM)

    methods = [
        ("all_gather_object", _all_gather_object_fn),
        ("all_gather_tensor", _all_gather_tensor_fn),
        ("all_reduce_tensor", _all_reduce_tensor_fn),
    ]

    results: dict[str, dict] = {}
    for method_name, method_fn in methods:
        result = _run_method(
            name=method_name,
            fn=method_fn,
            iters=args.iters,
            warmup=args.warmup,
            local_rank=local_rank,
            is_rank0=is_rank0,
        )
        if is_rank0:
            results[method_name] = result

    del done, gathered_tensors, gathered_objects
    gc.collect()

    if is_rank0:
        object_mean = results["all_gather_object"]["latency_ms"]["mean"]
        gather_tensor_mean = results["all_gather_tensor"]["latency_ms"]["mean"]
        reduce_tensor_mean = results["all_reduce_tensor"]["latency_ms"]["mean"]

        object_vs_gather_tensor = (object_mean / gather_tensor_mean) if gather_tensor_mean > 0 else None
        object_vs_reduce_tensor = (object_mean / reduce_tensor_mean) if reduce_tensor_mean > 0 else None

        fastest_method = min(
            results.items(),
            key=lambda kv: float(kv[1]["latency_ms"]["mean"]),
        )[0]

        output = {
            "test": "allgather_control_plane",
            "world_size": world_size,
            "iters": args.iters,
            "warmup": args.warmup,
            "hostname": hostname,
            "app_clocks": all_clocks,
            "methods": results,
            "comparison": {
                "all_gather_object_over_all_gather_tensor_speedup": (
                    round(object_vs_gather_tensor, 6) if object_vs_gather_tensor is not None else None
                ),
                "all_gather_object_over_all_reduce_tensor_speedup": (
                    round(object_vs_reduce_tensor, 6) if object_vs_reduce_tensor is not None else None
                ),
                "fastest_method": fastest_method,
                "fastest_latency_ms": round(float(results[fastest_method]["latency_ms"]["mean"]), 6),
            },
            "software": {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "nccl_version": _nccl_version_str(),
                "clock_lock_required": True,
            },
        }

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

        print("")
        print("=" * 72)
        print("ALL-GATHER CONTROL-PLANE COMPARISON")
        print(
            "  all_gather_object: "
            f"mean={results['all_gather_object']['latency_ms']['mean']:.6f} ms "
            f"p99={results['all_gather_object']['latency_ms']['p99']:.6f} ms"
        )
        print(
            "  all_gather_tensor: "
            f"mean={results['all_gather_tensor']['latency_ms']['mean']:.6f} ms "
            f"p99={results['all_gather_tensor']['latency_ms']['p99']:.6f} ms"
        )
        print(
            "  all_reduce_tensor: "
            f"mean={results['all_reduce_tensor']['latency_ms']['mean']:.6f} ms "
            f"p99={results['all_reduce_tensor']['latency_ms']['p99']:.6f} ms"
        )
        if object_vs_gather_tensor is not None:
            print(f"  all_gather_object vs all_gather_tensor: {object_vs_gather_tensor:.3f}x")
        if object_vs_reduce_tensor is not None:
            print(f"  all_gather_object vs all_reduce_tensor: {object_vs_reduce_tensor:.3f}x")
        print(f"  Fastest method: {fastest_method}")
        print(f"  Output: {out_path}")
        print("=" * 72)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
