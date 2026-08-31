"""Baseline ZeRO-2 comparison: standard DDP all-reduce without sharding."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

from core.benchmark.gpu_requirements import require_min_gpus
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark
from labs.train_distributed.training_utils.utils import get


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--extra-grad-mb",
        type=int,
        default=0,
        help="Extra gradient payload size (MB) to amplify communication.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    return parser.parse_args()


def _build_model(hidden_size: int, device: torch.device) -> nn.Sequential:
    layers = []
    for _ in range(6):
        layers.extend([nn.Linear(hidden_size, hidden_size), nn.GELU()])
    layers.append(nn.Linear(hidden_size, hidden_size))
    return nn.Sequential(*layers).to(device)


def main():
    require_min_gpus(2, script_name="baseline_zero2_multigpu.py")
    args = parse_args()
    local_rank = get("lrank")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=local_rank)

    rank = get("rank")
    device = torch.device(f"cuda:{local_rank}")

    model = _build_model(args.hidden_size, device)
    extra_param = None
    if args.extra_grad_mb > 0:
        elem_bytes = torch.finfo(torch.bfloat16).bits // 8
        numel = (args.extra_grad_mb * 1024 * 1024) // elem_bytes
        extra_param = torch.nn.Parameter(torch.zeros(numel, device=device, dtype=torch.bfloat16))
        model.register_parameter("extra_grad_payload", extra_param)
    ddp_model = DDP(
        model,
        device_ids=[device],
        bucket_cap_mb=1,
        gradient_as_bucket_view=False,
    )

    optimizer = torch.optim.AdamW(
        ddp_model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )

    x = torch.empty(args.batch_size, args.hidden_size, device=device)
    y = torch.empty_like(x)
    x.normal_()
    y.normal_()
    optimizer.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        warm_loss = nn.functional.mse_loss(ddp_model(x), y)
    if extra_param is not None:
        warm_loss = warm_loss + extra_param.sum() * 0.0
    warm_loss.backward()
    optimizer.step()
    dist.barrier()
    torch.cuda.synchronize(device)

    total_tokens = 0
    loss_value_buffer = torch.empty(1, dtype=torch.float64, device=device)
    start = perf_counter()

    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        x.normal_()
        y.normal_()
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            loss = nn.functional.mse_loss(ddp_model(x), y)
        if extra_param is not None:
            loss = loss + extra_param.sum() * 0.0
        loss.backward()
        optimizer.step()
        total_tokens += x.numel()

        if rank == 0 and step % 10 == 0:
            loss_value_buffer[0].copy_(loss.detach())
            loss_value = float(loss_value_buffer.detach().cpu()[0])
            print(
                f"[baseline-zero2] step {step}/{args.steps} loss={loss_value:.4f} "
                f"tokens/step={x.numel():,}"
            )

    torch.cuda.synchronize(device)
    total_time = perf_counter() - start
    if rank == 0:
        toks_per_sec = total_tokens / total_time if total_time > 0 else 0.0
        print(f"[baseline-zero2] finished {args.steps} steps | {toks_per_sec:,.0f} toks/s per rank")

    dist.destroy_process_group()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "zero2.py",
        base_args=[
            "--mode",
            "baseline",
            "--variant",
            "multigpu",
            "--batch-size",
            "16",
            "--hidden-size",
            "10000",
            "--extra-grad-mb",
            "12288",
        ],
        config_arg_map={"iterations": "--steps"},
        multi_gpu_required=True,
        target_label="labs/train_distributed:zero2_multigpu",
        default_nproc_per_node=None,
        name="baseline_zero2_multigpu",
    )
