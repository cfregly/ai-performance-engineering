#!/usr/bin/env python3
"""
All-reduce latency comparison benchmark.

Compares:
1) one large all-reduce payload
2) many small all-reduces that sum to the same total payload

This mirrors the intent of ml-engineering's all_reduce_latency_comp.py and is
useful for exposing latency-driven inefficiency from over-fragmented collectives
(e.g., too many small gradient buckets).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import socket
import sys
from pathlib import Path
from statistics import mean, median, stdev

import torch
import torch.distributed as dist

from core.common.device_utils import resolve_local_rank

if __package__ in {None, ""}:
    _repo_root = Path(__file__).resolve().parents[2]
    _env = os.environ.copy()
    _pythonpath = _env.get("PYTHONPATH")
    _env["PYTHONPATH"] = str(_repo_root) if not _pythonpath else os.pathsep.join([str(_repo_root), _pythonpath])
    os.execvpe(
        sys.executable,
        [sys.executable, "-m", "cluster.scripts.allreduce_latency_comp", *sys.argv[1:]],
        _env,
    )

try:
    from core.harness.benchmark_harness import _resolve_physical_device_index  # type: ignore
except Exception:
    _resolve_physical_device_index = None  # type: ignore[assignment]


def _busbw_factor(world_size: int) -> float:
    # Same all-reduce correction used by nccl-tests.
    return 2.0 * (world_size - 1) / world_size


def _to_gbps(bytes_per_sec: float) -> float:
    return bytes_per_sec / 1e9


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


def _run_case(
    ops: list[tuple[torch.Tensor, int]],
    payload_bytes_total: int,
    iters: int,
    warmup: int,
    busbw_coeff: float,
    local_rank: int,
    world_size: int,
    is_rank0: bool,
) -> dict:
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        dist.barrier()
        for tensor, repeat in ops:
            for _ in range(repeat):
                dist.all_reduce(tensor)
        torch.cuda.synchronize()

    durations_ms = []
    busbw_gbps = []
    algbw_gbps = []
    for _ in range(iters):
        dist.barrier()
        start_event.record()
        for tensor, repeat in ops:
            for _ in range(repeat):
                dist.all_reduce(tensor)
        end_event.record()
        torch.cuda.synchronize()

        duration_ms = start_event.elapsed_time(end_event)
        duration_s = duration_ms / 1000.0
        local_algbw = payload_bytes_total / duration_s
        mean_algbw_tensor = torch.tensor([local_algbw], dtype=torch.float64, device=f"cuda:{local_rank}")
        dist.reduce(mean_algbw_tensor, dst=0, op=dist.ReduceOp.SUM)

        if is_rank0:
            mean_algbw = mean_algbw_tensor.item() / world_size
            mean_busbw = mean_algbw * busbw_coeff
            durations_ms.append(duration_ms)
            algbw_gbps.append(_to_gbps(mean_algbw))
            busbw_gbps.append(_to_gbps(mean_busbw))

    if not is_rank0:
        return {}

    d_mean = mean(durations_ms)
    d_std = stdev(durations_ms) if len(durations_ms) > 1 else 0.0
    b_mean = mean(busbw_gbps)
    b_std = stdev(busbw_gbps) if len(busbw_gbps) > 1 else 0.0

    return {
        "payload_bytes_total": payload_bytes_total,
        "iters": iters,
        "warmup": warmup,
        "duration_ms": {
            "mean": round(d_mean, 4),
            "median": round(median(durations_ms), 4),
            "min": round(min(durations_ms), 4),
            "max": round(max(durations_ms), 4),
            "std": round(d_std, 4),
            "cv_pct": round((d_std / d_mean * 100.0) if d_mean > 0 else 0.0, 4),
            "samples": [round(x, 4) for x in durations_ms],
        },
        "busbw_gbps": {
            "mean": round(b_mean, 4),
            "median": round(median(busbw_gbps), 4),
            "min": round(min(busbw_gbps), 4),
            "max": round(max(busbw_gbps), 4),
            "std": round(b_std, 4),
            "cv_pct": round((b_std / b_mean * 100.0) if b_mean > 0 else 0.0, 4),
            "samples": [round(x, 4) for x in busbw_gbps],
        },
        "algbw_gbps": {
            "mean": round(mean(algbw_gbps), 4),
            "median": round(median(algbw_gbps), 4),
            "min": round(min(algbw_gbps), 4),
            "max": round(max(algbw_gbps), 4),
            "samples": [round(x, 4) for x in algbw_gbps],
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="All-reduce latency comparison benchmark")
    parser.add_argument("--payload-gib", type=float, default=4.0, help="Total payload per trial in GiB (default: 4.0)")
    parser.add_argument("--chunks", type=int, default=1000, help="Number of chunks for small-payload case (default: 1000)")
    parser.add_argument("--iters", type=int, default=5, help="Measured iterations (default: 5)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations (default: 1)")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    if os.environ.get("AISP_CLOCK_LOCKED") != "1":
        raise SystemExit(
            "ERROR: GPU clock lock is required for all-reduce latency comparison.\n"
            "Run via scripts/run_with_gpu_clocks.sh (or wrapper scripts that call it)."
        )
    if args.payload_gib <= 0:
        raise SystemExit(f"ERROR: --payload-gib must be > 0 (got {args.payload_gib})")
    if args.chunks <= 0:
        raise SystemExit(f"ERROR: --chunks must be > 0 (got {args.chunks})")
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

    payload_bytes = int(args.payload_gib * (2**30))
    dtype = torch.float32
    elem_bytes = torch.finfo(dtype).bits // 8
    large_elems = max(1, payload_bytes // elem_bytes)
    # Keep total payload for small case as close as possible to large case.
    effective_chunks = min(args.chunks, large_elems)
    small_elems = max(1, large_elems // effective_chunks)
    remainder_elems = large_elems - (small_elems * effective_chunks)
    small_payload_bytes_total = (small_elems * effective_chunks + remainder_elems) * elem_bytes
    large_payload_bytes_total = large_elems * elem_bytes

    if is_rank0:
        print("All-reduce latency comparison")
        print(f"  ranks={world_size}")
        print(f"  target_payload={args.payload_gib:.4f} GiB ({payload_bytes:,} bytes)")
        print(f"  large_case: 1 x {large_payload_bytes_total / (2**30):.4f} GiB")
        print(f"  small_case: {effective_chunks} x {(small_elems * elem_bytes) / (2**20):.4f} MiB")
        if remainder_elems > 0:
            print(f"              + 1 x {(remainder_elems * elem_bytes) / (2**20):.4f} MiB remainder")
        print(f"  iters={args.iters} warmup={args.warmup}")

    large_tensor = torch.rand(large_elems, dtype=dtype, device=f"cuda:{local_rank}")
    small_tensor = torch.rand(small_elems, dtype=dtype, device=f"cuda:{local_rank}")
    remainder_tensor = None
    if remainder_elems > 0:
        remainder_tensor = torch.rand(remainder_elems, dtype=dtype, device=f"cuda:{local_rank}")
    busbw_coeff = _busbw_factor(world_size)

    large_case = _run_case(
        ops=[(large_tensor, 1)],
        payload_bytes_total=large_payload_bytes_total,
        iters=args.iters,
        warmup=args.warmup,
        busbw_coeff=busbw_coeff,
        local_rank=local_rank,
        world_size=world_size,
        is_rank0=is_rank0,
    )

    small_ops: list[tuple[torch.Tensor, int]] = [(small_tensor, effective_chunks)]
    if remainder_tensor is not None:
        small_ops.append((remainder_tensor, 1))

    small_case = _run_case(
        ops=small_ops,
        payload_bytes_total=small_payload_bytes_total,
        iters=args.iters,
        warmup=args.warmup,
        busbw_coeff=busbw_coeff,
        local_rank=local_rank,
        world_size=world_size,
        is_rank0=is_rank0,
    )

    del large_tensor, small_tensor, remainder_tensor
    gc.collect()

    if is_rank0:
        large_d = large_case["duration_ms"]["mean"]
        small_d = small_case["duration_ms"]["mean"]
        large_bw = large_case["busbw_gbps"]["mean"]
        small_bw = small_case["busbw_gbps"]["mean"]

        duration_ratio = (small_d / large_d) if large_d > 0 else None
        bandwidth_ratio = (large_bw / small_bw) if small_bw > 0 else None

        result = {
            "test": "allreduce_latency_comp",
            "world_size": world_size,
            "payload_target_gib": args.payload_gib,
            "payload_target_bytes": payload_bytes,
            "chunks": args.chunks,
            "effective_chunks": effective_chunks,
            "remainder_elements": remainder_elems,
            "hostname": hostname,
            "app_clocks": all_clocks,
            "cases": {
                "one_large": large_case,
                "many_small": small_case,
            },
            "comparison": {
                "duration_ratio_small_over_large": round(duration_ratio, 6) if duration_ratio is not None else None,
                "bandwidth_ratio_large_over_small": round(bandwidth_ratio, 6) if bandwidth_ratio is not None else None,
                "one_large_busbw_gbps_mean": round(large_bw, 4),
                "many_small_busbw_gbps_mean": round(small_bw, 4),
                "one_large_duration_ms_mean": round(large_d, 4),
                "many_small_duration_ms_mean": round(small_d, 4),
            },
            "software": {
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "nccl_version": ".".join(str(x) for x in torch.cuda.nccl.version()),
                "clock_lock_required": True,
            },
        }

        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print("")
        print("=" * 72)
        print("ALL-REDUCE LATENCY COMPARISON")
        print(f"  1x large: busbw={large_bw:.2f} GBps, duration={large_d:.2f} ms")
        print(f"  {args.chunks}x small: busbw={small_bw:.2f} GBps, duration={small_d:.2f} ms")
        if bandwidth_ratio is not None:
            print(f"  Large-vs-small bandwidth ratio: {bandwidth_ratio:.3f}x")
        if duration_ratio is not None:
            print(f"  Small-vs-large duration ratio: {duration_ratio:.3f}x")
        print(f"  Output: {out_path}")
        print("=" * 72)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
