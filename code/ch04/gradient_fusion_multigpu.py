#!/usr/bin/env python3
"""Gradient fusion benchmark for multi-GPU all-reduce."""

from __future__ import annotations

import argparse
import datetime
import os
from typing import Tuple

import torch
import torch.distributed as dist

from ch04.distributed_helper import run_main_with_skip_status, setup_single_gpu_env
from core.benchmark.gpu_requirements import require_min_gpus
from core.common.device_utils import resolve_local_rank

FLOAT16_BYTES = torch.finfo(torch.float16).bits // 8


def init_distributed() -> Tuple[int, int, torch.device]:
    setup_single_gpu_env("gradient_fusion_multigpu", min_world_size=2)
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
    return dist.get_rank(), dist.get_world_size(), torch.device(f"cuda:{torch.cuda.current_device()}")


def run_benchmark(
    *,
    mode: str,
    num_tensors: int,
    tensor_kb: int,
    iterations: int,
) -> None:
    require_min_gpus(2)
    rank, world_size, device = init_distributed()
    if world_size < 2:
        raise RuntimeError("gradient_fusion_multigpu requires >=2 GPUs")

    dtype = torch.float16
    numel = max(1, (tensor_kb * 1024) // FLOAT16_BYTES)

    tensors = [
        torch.randn(numel, device=device, dtype=dtype) for _ in range(num_tensors)
    ]
    fused = None
    if mode == "optimized":
        fused = torch.empty(
            num_tensors * numel,
            device=device,
            dtype=dtype,
        )
        offset = 0
        for tensor in tensors:
            next_offset = offset + numel
            fused[offset:next_offset].copy_(tensor.view(-1))
            offset = next_offset

    for _ in range(5):
        if mode == "baseline":
            for t in tensors:
                dist.all_reduce(t)
        else:
            dist.all_reduce(fused)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        if mode == "baseline":
            for t in tensors:
                dist.all_reduce(t)
        else:
            dist.all_reduce(fused)
    end.record()
    torch.cuda.synchronize(device)

    elapsed_ms = start.elapsed_time(end)
    total_bytes = num_tensors * numel * FLOAT16_BYTES
    bw_gbps = (total_bytes / (elapsed_ms / iterations / 1000.0)) / 1e9

    if rank == 0:
        print(
            f"[gradient_fusion:{mode}] tensors={num_tensors} size={tensor_kb}KB "
            f"time={elapsed_ms / iterations:.4f} ms/iter bw={bw_gbps:.2f} GB/s"
        )

    dist.barrier()
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient fusion benchmark")
    parser.add_argument("--mode", choices=("baseline", "optimized"), default="baseline")
    parser.add_argument("--num-tensors", type=int, default=128, help="Number of small tensors.")
    parser.add_argument("--tensor-kb", type=int, default=64, help="Size per tensor (KB).")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations to time.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_benchmark(
        mode=args.mode,
        num_tensors=args.num_tensors,
        tensor_kb=args.tensor_kb,
        iterations=args.iterations,
    )


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
