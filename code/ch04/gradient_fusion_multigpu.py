#!/usr/bin/env python3
"""Gradient fusion benchmark for stable multi-GPU gradient averaging.

Both variants average the same FP16 gradients for every warmup and measured
iteration. The baseline issues one collective per gradient; the optimized path
packs the same bytes into one collective. Repeated SUM is intentionally avoided
because it changes the workload state exponentially and overflows FP16.
"""

from __future__ import annotations

import argparse
import datetime
import math
import os
from typing import Any

import torch
import torch.distributed as dist

from ch04.distributed_helper import run_main_with_skip_status, setup_single_gpu_env
from ch04.gradient_fusion_result import write_gradient_fusion_child_result
from core.benchmark.gpu_requirements import require_min_gpus
from core.common.device_utils import resolve_local_rank
from core.profiling.nvtx_helper import nvtx_range

FLOAT16_BYTES = torch.finfo(torch.float16).bits // 8
BASELINE_PROFILE_NVTX_RANGE = "compute_kernel:gradient_fusion_many_allreduces"
OPTIMIZED_PROFILE_NVTX_RANGE = "compute_kernel:gradient_fusion_fused_allreduce"
PAIR_NUM_TENSORS = 2048
PAIR_TENSOR_KB = 4
PAIR_TIMED_ITERATIONS = 50


def init_distributed() -> tuple[int, int, torch.device]:
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


def _run_collectives(
    mode: str,
    tensors: list[torch.Tensor],
    fused: torch.Tensor | None,
    *,
    iterations: int,
    reduce_op: Any,
) -> None:
    if mode not in {"baseline", "optimized"}:
        raise ValueError(f"Unsupported gradient-fusion mode: {mode!r}")
    if iterations <= 0:
        raise ValueError("Gradient-fusion iterations must be positive")
    for _ in range(iterations):
        if mode == "baseline":
            for tensor in tensors:
                dist.all_reduce(tensor, op=reduce_op)
        else:
            if fused is None:
                raise RuntimeError("optimized gradient fusion requires a fused tensor")
            dist.all_reduce(fused, op=reduce_op)


def _stable_average_reduce_op(world_size: int) -> Any:
    """Return the backend's explicit stable gradient-average primitive."""
    if not dist.is_initialized():
        raise RuntimeError("Gradient-fusion process group must be initialized")
    if world_size < 2:
        raise RuntimeError("Gradient-fusion average requires world_size >= 2")

    backend = str(dist.get_backend()).lower()
    if backend == "nccl":
        make_premul_sum = getattr(dist, "_make_nccl_premul_sum", None)
        if not callable(make_premul_sum):
            raise RuntimeError(
                "SKIPPED: this PyTorch build lacks NCCL premultiplied SUM support "
                "required for stable gradient averaging"
            )
        return make_premul_sum(1.0 / world_size)
    if backend == "gloo":
        average = getattr(dist.ReduceOp, "AVG", None)
        if average is None:
            raise RuntimeError(
                "SKIPPED: this Gloo build lacks ReduceOp.AVG for the CPU-only "
                "gradient-fusion control"
            )
        return average
    raise RuntimeError(
        f"SKIPPED: unsupported gradient-fusion average backend {backend!r}"
    )


def run_benchmark(
    *,
    mode: str,
    num_tensors: int,
    tensor_kb: int,
    iterations: int,
    profile_rank: int | None = None,
) -> None:
    if mode not in {"baseline", "optimized"}:
        raise ValueError(f"Unsupported gradient-fusion mode: {mode!r}")
    if num_tensors <= 0 or tensor_kb <= 0 or iterations <= 0:
        raise ValueError("Gradient-fusion workload dimensions must be positive")
    require_min_gpus(2)
    rank, world_size, device = init_distributed()
    if world_size < 2:
        raise RuntimeError("gradient_fusion_multigpu requires >=2 GPUs")
    if profile_rank is not None and not 0 <= profile_rank < world_size:
        raise ValueError("profile_rank must identify a participating rank")

    dtype = torch.float16
    numel = max(1, (tensor_kb * 1024) // FLOAT16_BYTES)
    generator = torch.Generator(device=device)
    generator.manual_seed((torch.initial_seed() + rank) % (2**63 - 1))

    initial_tensors = [
        torch.randn(numel, device=device, dtype=dtype, generator=generator)
        for _ in range(num_tensors)
    ]
    initial_gradients = torch.cat(initial_tensors)
    reference_average_fp32 = initial_gradients.float()
    dist.all_reduce(reference_average_fp32, op=dist.ReduceOp.SUM)
    reference_average_fp32.div_(world_size)
    reference_average = reference_average_fp32.to(dtype=dtype)

    tensors = [tensor.clone() for tensor in initial_tensors]
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

    average_op = _stable_average_reduce_op(world_size)

    _run_collectives(mode, tensors, fused, iterations=5, reduce_op=average_op)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    profile_range = (
        BASELINE_PROFILE_NVTX_RANGE
        if mode == "baseline"
        else OPTIMIZED_PROFILE_NVTX_RANGE
    )
    # Select one rank for NCU while every rank still executes every collective.
    # Profiling concurrent NCCL ranges from both processes can serialize peers.
    with nvtx_range(profile_range, enable=profile_rank is None or rank == profile_rank):
        _run_collectives(
            mode,
            tensors,
            fused,
            iterations=iterations,
            reduce_op=average_op,
        )
        end.record()
        torch.cuda.synchronize(device)

    elapsed_ms = start.elapsed_time(end)
    time_per_iter_ms = elapsed_ms / iterations
    if not math.isfinite(time_per_iter_ms) or time_per_iter_ms <= 0:
        raise RuntimeError(
            "Gradient-fusion CUDA-event time per iteration must be finite and positive"
        )
    total_bytes = num_tensors * numel * FLOAT16_BYTES
    bw_gbps = (total_bytes / (time_per_iter_ms / 1000.0)) / 1e9
    verify_output = torch.cat(tensors) if mode == "baseline" else fused
    if verify_output is None:
        raise RuntimeError("Gradient-fusion timed output is missing")
    write_gradient_fusion_child_result(
        variant=mode,
        rank=rank,
        world_size=world_size,
        num_tensors=num_tensors,
        tensor_kb=tensor_kb,
        iterations=iterations,
        initial_gradients=initial_gradients,
        reference_average=reference_average,
        verify_output=verify_output,
    )

    if rank == 0:
        print(f"rank0 time_per_iter_ms: {time_per_iter_ms:.9f}", flush=True)
        print(
            f"[gradient_fusion:{mode}] tensors={num_tensors} size={tensor_kb}KB "
            f"time={time_per_iter_ms:.4f} ms/iter bw={bw_gbps:.2f} GB/s",
            flush=True,
        )

    dist.barrier()
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gradient fusion benchmark")
    parser.add_argument("--mode", choices=("baseline", "optimized"), default="baseline")
    parser.add_argument("--num-tensors", type=int, default=128, help="Number of small tensors.")
    parser.add_argument("--tensor-kb", type=int, default=64, help="Size per tensor (KB).")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations to time.")
    parser.add_argument(
        "--profile-rank", type=int, default=None,
        help="Emit the measured NVTX range only on this rank; all ranks execute.",
    )
    args = parser.parse_args()
    if args.num_tensors <= 0:
        parser.error("--num-tensors must be positive")
    if args.tensor_kb <= 0:
        parser.error("--tensor-kb must be positive")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    return args


def main() -> None:
    args = parse_args()
    run_benchmark(
        mode=args.mode,
        num_tensors=args.num_tensors,
        tensor_kb=args.tensor_kb,
        iterations=args.iterations,
        profile_rank=args.profile_rank,
    )


if __name__ == "__main__":
    raise SystemExit(run_main_with_skip_status(main))
