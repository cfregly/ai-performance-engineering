#!/usr/bin/env python3
from __future__ import annotations

"""
NVSHMEM vs NCCL Benchmark (Conceptual)
=====================================

Micro-benchmark comparing NCCL broadcast against PyTorch 2.10 symmetric
memory (NVSHMEM-backed) direct copies on multi-GPU Blackwell B200 systems.

Measurements:
- Latency (µs) for small message sizes (1 KB - 1 MB)
- Bandwidth (GB/s) for larger message sizes (16 MB - 512 MB)

The script now fails fast with `SKIPPED:` when NVSHMEM/symmetric memory is
missing instead of publishing NCCL-only results under the same benchmark name.

Usage:
    torchrun --nproc_per_node=<num_gpus> nvshmem_vs_nccl_benchmark.py \
        --min-bytes 1024 --max-bytes 67108864 --steps 6 --mode nccl
"""

import argparse
import datetime
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

from ch04.distributed_helper import run_main_with_skip_status, setup_single_gpu_env
from core.common.device_utils import resolve_local_rank
from core.optimization.symmetric_memory_patch import (
    maybe_create_symmetric_memory_handle,
    symmetric_memory_available,
)

FLOAT16_BYTES = torch.finfo(torch.float16).bits // 8


# ============================================================================
# Utilities
# ============================================================================


def init_distributed() -> int:
    setup_single_gpu_env("nvshmem_vs_nccl_benchmark", min_world_size=2)
    if not symmetric_memory_available():
        raise RuntimeError("SKIPPED: nvshmem_vs_nccl_benchmark requires NVSHMEM or SymmetricMemory support")
    
    if not dist.is_initialized():
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", torch.cuda.device_count()))
        local_rank = resolve_local_rank()

        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60),
            device_id=local_rank,
        )
    else:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    
    return rank


@dataclass
class BenchmarkResult:
    bytes: int
    latency_us: float
    bandwidth_gbps: float


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(num_bytes)
    unit = 0
    while value >= 1024 and unit < len(units) - 1:
        value /= 1024.0
        unit += 1
    return f"{value:.1f} {units[unit]}"


def _measure_nccl_broadcast(bytes_per_rank: int, iterations: int) -> BenchmarkResult:
    device = torch.cuda.current_device()
    dtype = torch.float16
    numel = bytes_per_rank // FLOAT16_BYTES
    numel = max(1, numel)

    tensor = torch.randn(numel, device=device, dtype=dtype)
    compute = torch.randn_like(tensor)
    overlap_compute = os.environ.get("AISP_BROADCAST_OVERLAP", "").lower() in {"1", "true", "yes"}
    compute_passes = max(1, int(os.environ.get("AISP_BROADCAST_COMPUTE_PASSES", "1")))
    comm_stream = torch.cuda.Stream(device=device) if overlap_compute else None
    done = torch.cuda.Event() if overlap_compute and comm_stream is not None else None

    # Warmup
    for _ in range(5):
        dist.broadcast(tensor, src=0)
        for _ in range(compute_passes):
            compute.mul_(1.0001)

    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()

    start.record(current_stream)
    if comm_stream is not None:
        comm_stream.wait_stream(current_stream)
    for _ in range(iterations):
        if overlap_compute and comm_stream is not None:
            with torch.cuda.stream(comm_stream):
                work = dist.broadcast(tensor, src=0, async_op=True)
            for _ in range(compute_passes):
                compute.mul_(1.0001)
            work.wait()
            current_stream.wait_stream(comm_stream)
        else:
            dist.broadcast(tensor, src=0)
            for _ in range(compute_passes):
                compute.mul_(1.0001)
    end.record(current_stream)
    torch.cuda.synchronize(device)

    elapsed_ms = start.elapsed_time(end)
    latency_us = (elapsed_ms * 1000.0) / iterations
    total_bytes = bytes_per_rank * max(dist.get_world_size() - 1, 1)
    bandwidth_gbps = (total_bytes / (elapsed_ms / iterations / 1000.0)) / 1e9
    return BenchmarkResult(bytes=bytes_per_rank, latency_us=latency_us, bandwidth_gbps=bandwidth_gbps)


def _measure_symmetric_broadcast(bytes_per_rank: int, iterations: int) -> Optional[BenchmarkResult]:
    if not symmetric_memory_available():
        return None

    device = torch.cuda.current_device()
    dtype = torch.float16
    numel = bytes_per_rank // FLOAT16_BYTES
    numel = max(1, numel)

    local = torch.randn(numel, device=device, dtype=dtype)
    compute = torch.randn_like(local)
    sym_handle = maybe_create_symmetric_memory_handle(local)
    if sym_handle is None:
        return None
    buffer = sym_handle.buffer

    rank = dist.get_rank()
    remote_buffers = []
    if rank == 0:
        remote_buffers = [sym_handle.get_buffer(peer) for peer in range(1, dist.get_world_size())]
    overlap_compute = os.environ.get("AISP_BROADCAST_OVERLAP", "").lower() in {"1", "true", "yes"}
    compute_passes = max(1, int(os.environ.get("AISP_BROADCAST_COMPUTE_PASSES", "1")))
    comm_stream = torch.cuda.Stream(device=device) if overlap_compute else None
    done = torch.cuda.Event() if overlap_compute and comm_stream is not None else None

    for _ in range(5):
        if rank == 0:
            for remote in remote_buffers:
                remote.copy_(buffer, non_blocking=True)
        torch.cuda.synchronize(device)
        for _ in range(compute_passes):
            compute.mul_(1.0001)
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize(device)
    current_stream = torch.cuda.current_stream()

    start.record(current_stream)
    if comm_stream is not None:
        comm_stream.wait_stream(current_stream)
    for _ in range(iterations):
        if overlap_compute and comm_stream is not None and done is not None:
            if rank == 0:
                with torch.cuda.stream(comm_stream):
                    for remote in remote_buffers:
                        remote.copy_(buffer, non_blocking=True)
                    done.record(comm_stream)
            else:
                done.record(current_stream)
            for _ in range(compute_passes):
                compute.mul_(1.0001)
            current_stream.wait_event(done)
        else:
            if rank == 0:
                for remote in remote_buffers:
                    remote.copy_(buffer, non_blocking=True)
            torch.cuda.synchronize(device)
            for _ in range(compute_passes):
                compute.mul_(1.0001)
    end.record(current_stream)
    torch.cuda.synchronize(device)
    dist.barrier()

    elapsed_ms = start.elapsed_time(end)
    latency_us = (elapsed_ms * 1000.0) / iterations
    total_bytes = bytes_per_rank * max(dist.get_world_size() - 1, 1)
    bandwidth_gbps = (total_bytes / (elapsed_ms / iterations / 1000.0)) / 1e9
    return BenchmarkResult(bytes=bytes_per_rank, latency_us=latency_us, bandwidth_gbps=bandwidth_gbps)


def sweep_sizes(min_bytes: int, max_bytes: int, steps: int) -> List[int]:
    if steps <= 1:
        return [min_bytes]
    start_exp = math.log10(min_bytes)
    end_exp = math.log10(max_bytes)
    if steps == 2:
        return [min_bytes, max_bytes]
    return [
        int(round(10 ** (start_exp + (end_exp - start_exp) * idx / (steps - 1))))
        for idx in range(steps)
    ]


def benchmark(args: argparse.Namespace) -> Dict[str, List[BenchmarkResult]]:
    results: Dict[str, List[BenchmarkResult]] = {"nccl": [], "nvshmem": []}
    sizes = sweep_sizes(args.min_bytes, args.max_bytes, args.steps)

    for message_size in sizes:
        dist.barrier()
        if args.mode in ("nccl", "both"):
            nccl = _measure_nccl_broadcast(message_size, args.iterations)
            results["nccl"].append(nccl)
        dist.barrier()
        if args.mode in ("nvshmem", "both"):
            nvshmem_res = _measure_symmetric_broadcast(message_size, args.iterations)
            if nvshmem_res is not None:
                results["nvshmem"].append(nvshmem_res)
        dist.barrier()

    return results


def main(destroy_process_group: bool = True) -> None:
    parser = argparse.ArgumentParser(description="Compare NVSHMEM vs NCCL broadcast latency/bandwidth")
    parser.add_argument("--min-bytes", type=int, default=1024, help="Smallest message size per rank")
    parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024, help="Largest message size per rank")
    parser.add_argument("--steps", type=int, default=6, help="Number of message sizes to sample")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations per measurement")
    parser.add_argument(
        "--mode",
        choices=("nccl", "nvshmem", "both"),
        default="both",
        help="Which transport(s) to benchmark",
    )
    args = parser.parse_args()

    rank = init_distributed()
    results = benchmark(args)

    if rank == 0:
        print("\nNVSHMEM vs NCCL Broadcast Benchmark (conceptual)")
        print("------------------------------------------------------")
        print(f"Symmetric memory available: {symmetric_memory_available()}")
        print("Message Size | NCCL Latency (us) | NCCL BW (GB/s) | NVSHMEM Latency (us) | NVSHMEM BW (GB/s)")
        print("-------------------------------------------------------------------------------------------")

        nvshmem_dict = {res.bytes: res for res in results["nvshmem"]}
        for res in results["nccl"]:
            nv = nvshmem_dict.get(res.bytes)
            nv_lat = f"{nv.latency_us:8.2f}" if nv else "   n/a "
            nv_bw = f"{nv.bandwidth_gbps:8.2f}" if nv else "   n/a "
            print(
                f"{_format_bytes(res.bytes):>12} | {res.latency_us:16.2f} | {res.bandwidth_gbps:13.2f} | "
                f"{nv_lat:>18} | {nv_bw:>15}"
            )

    dist.barrier()
    if destroy_process_group:
        dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
