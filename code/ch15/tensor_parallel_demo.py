"""tensor_parallel_demo.py - Chapter 15 tensor-parallel (TP) demo (tool).

Run with torchrun, e.g.:
  torchrun --nproc_per_node <num_gpus> ch15/tensor_parallel_demo.py
"""

from __future__ import annotations

import argparse
import os

from core.benchmark.utils import scalar_tensor_to_float
from core.common.device_utils import resolve_local_rank

import torch
import torch.distributed as dist


def _init_dist() -> tuple[int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for tensor_parallel_demo.py")
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError("Run with torchrun (missing RANK/WORLD_SIZE env vars).")
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = resolve_local_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    return rank, world_size, device


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 15 tensor parallel demo")
    parser.add_argument("--batch", type=int, default=64, help="Batch size.")
    parser.add_argument("--in-features", type=int, default=4096, help="Input feature dimension.")
    parser.add_argument("--out-features", type=int, default=4096, help="Output feature dimension.")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16", help="Compute dtype.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tensor_parallel_demo.py")

    rank, world_size, device = _init_dist()
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    in_features = int(args.in_features)
    out_features = int(args.out_features)
    if in_features % world_size != 0:
        raise ValueError(f"in_features={in_features} must be divisible by world_size={world_size}")

    shard_in = in_features // world_size
    shard_start = rank * shard_in
    shard_end = shard_start + shard_in

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    x = torch.randn(int(args.batch), in_features, device=device, dtype=dtype)
    w_full = torch.randn(out_features, in_features, device=device, dtype=dtype)
    w_shard = w_full[:, shard_start:shard_end].contiguous()
    x_shard = x[:, shard_start:shard_end].contiguous()

    def tp_forward() -> torch.Tensor:
        partial = x_shard @ w_shard.t()
        dist.all_reduce(partial, op=dist.ReduceOp.SUM)
        return partial

    with torch.inference_mode():
        for _ in range(int(args.warmup)):
            _ = tp_forward()
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream(device)
        start.record(current_stream)
        for _ in range(int(args.iters)):
            y_tp = tp_forward()
        end.record(current_stream)
        torch.cuda.synchronize(device)
        per_iter_ms = start.elapsed_time(end) / max(int(args.iters), 1)

        y_ref = x @ w_full.t()
        max_diff = scalar_tensor_to_float((y_tp.float() - y_ref.float()).abs().max())

    worst = torch.tensor(per_iter_ms, device=device, dtype=torch.float32)
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    worst_ms = scalar_tensor_to_float(worst)

    if rank == 0:
        print(f"tensor_parallel_demo: world={world_size} -> {worst_ms:.3f} ms/iter, max_abs_diff={max_diff:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
