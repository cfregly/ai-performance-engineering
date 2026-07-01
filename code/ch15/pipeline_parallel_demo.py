"""pipeline_parallel_demo.py - Chapter 15 pipeline-parallel (PP) demo (tool).

Minimal multi-stage pipeline example using torch.distributed send/recv.

Run with torchrun, e.g.:
  torchrun --nproc_per_node <num_gpus> ch15/pipeline_parallel_demo.py
"""

from __future__ import annotations

import argparse
import os

from core.benchmark.utils import scalar_tensor_to_float
from core.common.device_utils import resolve_local_rank
import time

import torch
import torch.distributed as dist


def _init_dist() -> tuple[int, int, torch.device]:
    if not dist.is_available():
        raise RuntimeError("torch.distributed is required for pipeline_parallel_demo.py")
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


def _weight_for_stage(stage_idx: int, hidden: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(1234 + stage_idx)
    weight = torch.randn(hidden, hidden, generator=gen, dtype=torch.float32)
    return weight.to(device=device, dtype=dtype)


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 15 pipeline parallel demo (multi-stage)")
    parser.add_argument("--batch", type=int, default=64, help="Batch size.")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden dimension.")
    parser.add_argument("--iters", type=int, default=50, help="Timed iterations.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="bf16", help="Compute dtype.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for pipeline_parallel_demo.py")

    rank, world_size, device = _init_dist()
    if world_size < 2:
        raise RuntimeError("pipeline_parallel_demo.py requires world_size >= 2")

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    hidden = int(args.hidden)
    batch = int(args.batch)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Stage-specific weight (deterministic across ranks).
    w_stage = _weight_for_stage(rank, hidden, device, dtype)
    w_stage_t = w_stage.t()

    # Buffers for send/recv.
    act = torch.empty(batch, hidden, device=device, dtype=dtype)
    out = torch.empty(batch, hidden, device=device, dtype=dtype)

    x0 = None
    if rank == 0:
        x0 = torch.randn(batch, hidden, device=device, dtype=dtype)

    def stage_forward_into(x: torch.Tensor, out_buf: torch.Tensor, *, is_last: bool) -> torch.Tensor:
        torch.mm(x, w_stage_t, out=out_buf)
        return out_buf if is_last else torch.relu_(out_buf)

    # Warmup.
    with torch.inference_mode():
        for _ in range(int(args.warmup)):
            if rank == 0:
                assert x0 is not None
                stage_forward_into(x0, act, is_last=False)
                dist.send(act, dst=1)
                dist.recv(out, src=world_size - 1)
            elif rank == world_size - 1:
                dist.recv(act, src=rank - 1)
                stage_forward_into(act, out, is_last=True)
                dist.send(out, dst=0)
            else:
                dist.recv(act, src=rank - 1)
                stage_forward_into(act, out, is_last=False)
                dist.send(out, dst=rank + 1)
        torch.cuda.synchronize(device)
        dist.barrier()

    # Timed loop (measure wall-clock on rank 0, take max over ranks).
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(int(args.iters)):
            if rank == 0:
                assert x0 is not None
                stage_forward_into(x0, act, is_last=False)
                dist.send(act, dst=1)
                dist.recv(out, src=world_size - 1)
            elif rank == world_size - 1:
                dist.recv(act, src=rank - 1)
                stage_forward_into(act, out, is_last=True)
                dist.send(out, dst=0)
            else:
                dist.recv(act, src=rank - 1)
                stage_forward_into(act, out, is_last=False)
                dist.send(out, dst=rank + 1)
        torch.cuda.synchronize(device)
        dist.barrier()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0 / max(int(args.iters), 1)

    worst = torch.tensor(elapsed_ms, device=device, dtype=torch.float32)
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    worst_ms = scalar_tensor_to_float(worst)

    max_diff = 0.0
    if rank == 0:
        assert x0 is not None
        ref = x0
        for stage_idx in range(world_size):
            w_ref = _weight_for_stage(stage_idx, hidden, device, dtype)
            ref = ref @ w_ref.t()
            if stage_idx < world_size - 1:
                ref = torch.relu_(ref)
        max_diff = scalar_tensor_to_float((out.float() - ref.float()).abs().max())
        print(
            f"pipeline_parallel_demo: world={world_size} -> {worst_ms:.3f} ms/iter, "
            f"max_abs_diff={max_diff:.3e}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
