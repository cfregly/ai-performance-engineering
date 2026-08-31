"""Optimized ZeRO-2: overlap reduce-scatter gradients with sharded optimizer states."""

from __future__ import annotations

import argparse
from time import perf_counter
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks as comm_hooks
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

from labs.train_distributed.training_utils.memory import print_memory_stats
from labs.train_distributed.training_utils.utils import get
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--hidden-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on the DDP module.")
    return parser.parse_args()


def _build_model(hidden_size: int, device):
    layers = []
    for _ in range(6):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
    layers.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*layers).to(device)


def _optimizer_cfg(params, lr):
    try:
        return dict(
            optimizer_class=torch.optim.AdamW,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.05,
            fused=True,
        )
    except TypeError:
        return dict(
            optimizer_class=torch.optim.AdamW,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.05,
        )


def main():
    args = parse_args()
    local_rank = get("lrank")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=local_rank)
    if get("ws") < 2:
        print("Warning: Optimized ZeRO-2 running with world_size < 2; no sharding benefit will be observed.")
    rank = get("rank")
    device = torch.device(f"cuda:{local_rank}")


    model = _build_model(args.hidden_size, device)
    ddp_model = DDP(
        model,
        device_ids=[device],
        static_graph=True,
        gradient_as_bucket_view=True,
        bucket_cap_mb=25,
    )
    hook = getattr(comm_hooks, "batched_reduce_scatter_hook", None)
    if hook is None:
        raise RuntimeError("SKIPPED: PyTorch build missing batched_reduce_scatter_hook required for optimized ZeRO-2.")
    ddp_model.register_comm_hook(state=dist.group.WORLD, hook=hook)

    if args.compile:
        ddp_model = torch.compile(ddp_model, mode="reduce-overhead")

    optimizer = ZeroRedundancyOptimizer(
        ddp_model.parameters(),
        overlap_with_ddp=True,
        gradient_as_bucket_view=True,
        **_optimizer_cfg(ddp_model.parameters(), args.learning_rate),
    )

    total_tokens = 0
    start = perf_counter()
    grad_clip = 1.0
    loss_value_buffer = torch.empty(1, dtype=torch.float64, device=device)
    x = torch.empty(args.batch_size, args.hidden_size, device=device)
    y = torch.empty_like(x)

    # Warmup
    x.normal_()
    y.normal_()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        warm_loss = nn.functional.mse_loss(ddp_model(x), y)
    warm_loss.backward()
    optimizer.step()

    if rank == 0:
        print_memory_stats("optimized-zero2 warmup", ddp_model, optimizer, rank, device)
    dist.barrier()

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            x.normal_()
            y.normal_()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = nn.functional.mse_loss(ddp_model(x), y) / args.grad_accum
            loss.backward()
            total_tokens += x.numel()

        torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), grad_clip)
        optimizer.step()

        if rank == 0 and step % 10 == 0:
            elapsed = perf_counter() - start
            toks_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
            loss_value_buffer[0].copy_(loss.detach())
            loss_value = float(loss_value_buffer.detach().cpu()[0])
            print(
                f"[optimized-zero2] step {step}/{args.steps} "
                f"loss={loss_value:.4f} "
                f"tokens/s per rank={toks_per_sec:,.0f}"
            )

    torch.cuda.synchronize(device)
    total_time = perf_counter() - start
    if rank == 0:
        toks_per_sec = total_tokens / total_time
        print(f"[optimized-zero2] finished {args.steps} steps | {toks_per_sec:,.0f} toks/s per rank")

    dist.destroy_process_group()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "zero2.py",
        base_args=[
            "--mode",
            "optimized",
            "--variant",
            "single",
            "--batch-size",
            "16",
            "--hidden-size",
            "10000",
            "--grad-accum",
            "1",
        ],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:zero2",
        default_nproc_per_node=1,
        multi_gpu_required=False,
        name="optimized_zero2",
    )
