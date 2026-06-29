from __future__ import annotations

"""Comprehensive NCCL benchmark for PyTorch 2.10 + CUDA 13.0."""


import argparse
import os
import time

import torch
import torch.distributed as dist

from core.common.device_utils import resolve_local_rank


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="NCCL Benchmark Suite")
    parser.add_argument("--operation", nargs="+",
                        choices=["allreduce", "allgather", "reducescatter", "broadcast", "reduce"],
                        help="Collective operations to benchmark.")
    parser.add_argument("--dtype", nargs="+",
                        choices=["float32", "float16", "bfloat16"],
                        help="Data types to test.")
    parser.add_argument("--max-size", type=int, default=256,
                        help="Maximum data size in MB (default: 256).")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Number of warmup iterations (default: 5).")
    parser.add_argument("--trials", type=int, default=10,
                        help="Number of benchmark trials (default: 10).")
    return parser.parse_args()


def init_runtime() -> tuple[int, int, torch.device, bool]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for NCCL benchmark.")

    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size_env == 1 and "RANK" not in os.environ:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device.index or 0)
        return 0, 1, device, False

    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group("nccl", init_method="env://", device_id=local_rank)

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device, True


def benchmark_collective(rank: int, world_size: int, tensor: torch.Tensor,
                         op_type: str, warmup: int, trials: int) -> tuple[float, float, float]:
    device = tensor.device
    allgather_outputs = [torch.empty_like(tensor) for _ in range(world_size)] if op_type == "allgather" else None
    shard_size = (tensor.numel() + world_size - 1) // world_size
    reducescatter_output = (
        torch.empty(shard_size, device=device, dtype=tensor.dtype)
        if op_type == "reducescatter"
        else None
    )
    reducescatter_inputs = None
    if op_type == "reducescatter":
        padded_numel = shard_size * world_size
        reducescatter_source = (
            tensor
            if padded_numel == tensor.numel()
            else torch.empty(padded_numel, device=device, dtype=tensor.dtype)
        )
        reducescatter_inputs = reducescatter_source.view(world_size, shard_size)

    def _run_collective() -> None:
        if op_type == "allreduce":
            dist.all_reduce(tensor)
        elif op_type == "allgather":
            if allgather_outputs is None:
                raise RuntimeError("allgather outputs were not initialized")
            dist.all_gather(allgather_outputs, tensor)
        elif op_type == "reducescatter":
            if reducescatter_output is None or reducescatter_inputs is None:
                raise RuntimeError("reduce-scatter buffers were not initialized")
            dist.reduce_scatter_tensor(reducescatter_output, reducescatter_inputs)
        elif op_type == "broadcast":
            dist.broadcast(tensor, src=0)
        elif op_type == "reduce":
            dist.reduce(tensor, dst=0)
        else:
            raise ValueError(f"Unsupported op: {op_type}")

    for _ in range(warmup):
        _run_collective()
        torch.cuda.synchronize(device)

    total_time = 0.0
    min_time = float("inf")
    max_time = float("-inf")
    trial_count = 0
    for _ in range(trials):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        _run_collective()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        total_time += elapsed
        min_time = min(min_time, elapsed)
        max_time = max(max_time, elapsed)
        trial_count += 1

    avg_time = total_time / trial_count
    return avg_time, min_time, max_time


def format_bandwidth(tensor: torch.Tensor, op_type: str, avg_time: float, world_size: int) -> float:
    bytes_transferred = tensor.numel() * tensor.element_size()
    if op_type == "allreduce":
        effective = bytes_transferred * 2 * (world_size - 1) / world_size
    elif op_type == "allgather":
        effective = bytes_transferred * world_size
    elif op_type == "reducescatter":
        effective = bytes_transferred * (world_size - 1) / world_size
    else:
        effective = bytes_transferred
    return effective / avg_time / 1e9


def run_single_gpu(args: argparse.Namespace) -> None:
    device = torch.device("cuda:0")
    torch.cuda.set_device(device.index or 0)

    ops = args.operation or ["allreduce"]
    dtypes = args.dtype or ["float32"]
    sizes_mb = sorted({4, 64, args.max_size})

    for op in ops:
        for dtype_name in dtypes:
            dtype = getattr(torch, dtype_name)
            bytes_per_elem = torch.finfo(dtype).bits // 8
            for size_mb in sizes_mb:
                total_bytes = size_mb * 1024 * 1024
                numel = max(1, total_bytes // bytes_per_elem)
                tensor = torch.randn(numel, device=device, dtype=dtype)
                torch.cuda.synchronize(device)
                start = time.perf_counter()
                _ = tensor * 2
                torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - start
                bandwidth = (tensor.numel() * tensor.element_size()) / elapsed / 1e9
                print(f"SINGLE_GPU {op.upper()} {dtype_name} {size_mb}MB: {bandwidth:.2f} GB/s")


def run_distributed(rank: int, world_size: int, device: torch.device,
                    args: argparse.Namespace) -> None:
    if rank == 0:
        print(f"NCCL Benchmark - World Size: {world_size}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA: {torch.version.cuda}")
        print("=" * 60)

    ops = args.operation or ["allreduce", "allgather", "reducescatter", "broadcast"]
    dtypes = args.dtype or ["float32", "float16", "bfloat16"]
    sizes = [1024, 1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024]

    for op in ops:
        for dtype_name in dtypes:
            dtype = getattr(torch, dtype_name)
            for elements in sizes:
                if args.max_size and elements * torch.finfo(dtype).bits / 8 > args.max_size * 1024 * 1024:
                    continue
                tensor = torch.randn(elements, device=device, dtype=dtype)
                avg_time, min_time, max_time = benchmark_collective(
                    rank, world_size, tensor, op, args.warmup, args.trials
                )
                bandwidth = format_bandwidth(tensor, op, avg_time, world_size)
                if rank == 0:
                    print(f"{op.upper()} {dtype_name} {elements} elements:")
                    print(f"  Avg: {avg_time*1000:.2f} ms, Min: {min_time*1000:.2f} ms, Max: {max_time*1000:.2f} ms")
                    print(f"  Bandwidth: {bandwidth:.2f} GB/s")
                    print("-" * 40)

    if rank == 0 and world_size == 8:
        print("=" * 60)
        print("Multi-GPU NVLink expectations:")
        print("  AllReduce 1GB: 700-800 GB/s")
        print("  P2P: ~850 GB/s per GPU pair")


def main() -> None:
    args = parse_args()

    try:
        rank, world_size, device, distributed = init_runtime()
    except RuntimeError as exc:
        print(f"Skipping NCCL benchmark: {exc}")
        return

    if not distributed:
        run_single_gpu(args)
        return

    run_distributed(rank, world_size, device, args)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
