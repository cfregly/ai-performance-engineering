"""Optimized ZeRO-1 using PyTorch's ZeroRedundancyOptimizer + DDP overlap."""

from __future__ import annotations

import argparse
from time import perf_counter
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP

from core.benchmark.gpu_requirements import require_min_gpus
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


def _maybe_fused_adamw(params, lr):
    try:
        return dict(
            optimizer_class=torch.optim.AdamW,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )
    except TypeError:
        return dict(
            optimizer_class=torch.optim.AdamW,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )


def _build_model(hidden_size: int, device):
    layers = []
    for _ in range(6):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU(inplace=True)])
    layers.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*layers).to(device)


def main():
    require_min_gpus(2, script_name="optimized_zero1_multigpu.py")
    args = parse_args()
    local_rank = get("lrank")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=local_rank)
    if get("ws") < 2:
        print("Warning: Optimized ZeRO-1 running with world_size < 2; no sharding benefit will be observed.")
    rank = get("rank")
    device = torch.device(f"cuda:{local_rank}")

    model = _build_model(args.hidden_size, device)
    ddp_model = DDP(
        model,
        device_ids=[device],
        static_graph=True,
        gradient_as_bucket_view=True,
    )

    if args.compile:
        ddp_model = torch.compile(ddp_model, mode="reduce-overhead")

    optimizer_cfg = _maybe_fused_adamw(ddp_model.parameters(), args.learning_rate)
    optimizer = ZeroRedundancyOptimizer(
        ddp_model.parameters(),
        overlap_with_ddp=True,
        gradient_as_bucket_view=True,
        **optimizer_cfg,
    )

    grad_clip = 1.0
    total_tokens = 0
    start = perf_counter()
    loss_value_buffer = torch.empty(1, dtype=torch.float64, device=device)
    x = torch.empty(args.batch_size, args.hidden_size, device=device)
    y = torch.empty_like(x)

    # Warmup
    x.normal_()
    y.normal_()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        warmup_loss = ddp_model(x)
        warmup_loss = nn.functional.mse_loss(warmup_loss, y)
    warmup_loss.backward()
    optimizer.step()

    if rank == 0:
        print_memory_stats("optimized-zero1 warmup", ddp_model, optimizer, rank, device)
    dist.barrier()

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        for micro in range(args.grad_accum):
            x.normal_()
            y.normal_()
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = ddp_model(x)
                loss = nn.functional.mse_loss(out, y) / args.grad_accum
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
                f"[optimized-zero1] step {step}/{args.steps} "
                f"loss={loss_value:.4f} "
                f"tokens/s per rank={toks_per_sec:,.0f}"
            )

    torch.cuda.synchronize(device)
    total_time = perf_counter() - start
    if rank == 0:
        toks_per_sec = total_tokens / total_time
        print(
            f"[optimized-zero1] finished {args.steps} steps | "
            f"{toks_per_sec:,.0f} toks/s per rank"
        )

    dist.destroy_process_group()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "zero1.py",
        base_args=[
            "--mode",
            "optimized",
            "--variant",
            "multigpu",
            "--batch-size",
            "16",
            "--hidden-size",
            "10000",
            "--grad-accum",
            "1",
        ],
        config_arg_map={"iterations": "--steps"},
        multi_gpu_required=True,
        target_label="labs/train_distributed:zero1_multigpu",
        default_nproc_per_node=None,
        name="optimized_zero1_multigpu",
    )
