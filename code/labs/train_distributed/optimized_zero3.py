"""Optimized ZeRO-3 via PyTorch FSDP on a synthetic MLP."""

from __future__ import annotations

import argparse
from time import perf_counter
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
)

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
    parser.add_argument("--compile", action="store_true", help="Use torch.compile on the FSDP module.")
    return parser.parse_args()


def _maybe_fused_adamw(params, lr):
    try:
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.05,
            fused=True,
        )
    except TypeError:
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.05)


def _build_model(hidden_size: int):
    layers = []
    for _ in range(6):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
    layers.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*layers)


def main():
    args = parse_args()
    local_rank = get("lrank")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=local_rank)
    if get("ws") < 2:
        print("Warning: Optimized ZeRO-3 running with world_size < 2; no sharding benefit will be observed.")
    rank = get("rank")
    device = torch.device(f"cuda:{local_rank}")


    base_model = _build_model(args.hidden_size)

    fsdp_model = FSDP(
        base_model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        forward_prefetch=True,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        use_orig_params=True,
        cpu_offload=CPUOffload(offload_params=False),
    )

    if args.compile:
        fsdp_model = torch.compile(fsdp_model, mode="reduce-overhead")

    optimizer = _maybe_fused_adamw(fsdp_model.parameters(), args.learning_rate)
    x = torch.empty(args.batch_size, args.hidden_size, device=device)
    y = torch.empty_like(x)

    # Warmup to stabilize allocator and buckets.
    x.normal_()
    y.normal_()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        warm_loss = nn.functional.mse_loss(fsdp_model(x), y)
    warm_loss.backward()
    optimizer.step()
    if rank == 0:
        print_memory_stats("optimized-zero3 warmup", fsdp_model, optimizer, rank, device)
    dist.barrier()

    total_tokens = 0
    start = perf_counter()
    grad_clip = 1.0
    loss_value_buffer = torch.empty(1, dtype=torch.float64, device=device)

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(args.grad_accum):
            x.normal_()
            y.normal_()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                loss = nn.functional.mse_loss(fsdp_model(x), y) / args.grad_accum
            loss.backward()
            total_tokens += x.numel()

        torch.nn.utils.clip_grad_norm_(fsdp_model.parameters(), grad_clip)
        optimizer.step()

        if rank == 0 and step % 10 == 0:
            elapsed = perf_counter() - start
            toks_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
            loss_value_buffer[0].copy_(loss.detach())
            loss_value = float(loss_value_buffer.detach().cpu()[0])
            print(
                f"[optimized-zero3] step {step}/{args.steps} "
                f"loss={loss_value:.4f} tokens/s per rank={toks_per_sec:,.0f}"
            )

    torch.cuda.synchronize(device)
    total_time = perf_counter() - start
    if rank == 0:
        toks_per_sec = total_tokens / total_time
        print(f"[optimized-zero3] finished {args.steps} steps | {toks_per_sec:,.0f} toks/s per rank")

    dist.destroy_process_group()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "zero3.py",
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
        target_label="labs/train_distributed:zero3",
        default_nproc_per_node=1,
        multi_gpu_required=False,
        name="optimized_zero3",
    )
