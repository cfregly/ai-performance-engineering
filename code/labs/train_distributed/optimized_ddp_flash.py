"""Optimized DDP training loop using flash attention (requires compatible kernels)."""

from __future__ import annotations

import argparse
import os

from core.common.device_utils import resolve_local_rank
from time import perf_counter
from pathlib import Path
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import fp16_compress_hook

from labs.train_distributed.training_utils.utils import (
    build_dataloader,
    build_text_model_flash,
    build_tokenizer,
    get_dataset,
)
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200, help="Number of optimization steps.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-rank microbatch size.")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--max-length", type=int, default=1024, help="Pad sequences to a fixed length.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on the model.")
    return parser.parse_args()


def _maybe_fused_adamw(params, lr):
    try:
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
            fused=True,
        )
    except TypeError:
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )


def main():
    args = parse_args()
    local_rank = resolve_local_rank()
    if not torch.cuda.is_available():
        raise RuntimeError("DDP optimized run requires CUDA GPUs.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if not dist.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="nccl", device_id=local_rank)
        else:
            raise RuntimeError("DDP optimized run requires torch.distributed process group to be initialized.")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = rank == 0
    use_ddp = world_size > 1

    tokenizer = build_tokenizer()
    dataset = get_dataset()["train"]

    dataloader = build_dataloader(
        dataset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        distributed=use_ddp,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True,
        max_length=args.max_length,
    )

    model = build_text_model_flash()
    model.to(device)
    model.train()

    ddp_model = model
    if use_ddp:
        ddp_model = DDP(
            model,
            device_ids=[local_rank],
            static_graph=True,
            bucket_cap_mb=50,
            gradient_as_bucket_view=True,
        )
        ddp_model.register_comm_hook(state=torch.distributed.group.WORLD, hook=fp16_compress_hook)

    if args.compile:
        ddp_model = torch.compile(ddp_model, mode="reduce-overhead")

    optimizer = _maybe_fused_adamw(ddp_model.parameters(), args.learning_rate)

    num_steps = min(args.steps, len(dataloader))
    total_tokens = 0
    start_time = perf_counter()
    loss_value_buffer = torch.empty(1, dtype=torch.float64, device=device)

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        micro_step = step % args.grad_accum
        sync_ctx = (
            ddp_model.no_sync()
            if use_ddp and args.grad_accum > 1 and micro_step != args.grad_accum - 1
            else nullcontext()
        )
        with torch.cuda.amp.autocast(dtype=torch.bfloat16), sync_ctx:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch["labels"] = batch["input_ids"].clone()
            outputs = ddp_model(**batch)
            loss = outputs.loss / args.grad_accum
        loss.backward()

        if micro_step == args.grad_accum - 1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_tokens += batch["input_ids"].numel()

        if step % 10 == 0 and is_main:
            loss_value_buffer[0].copy_(loss.detach())
            loss_value = float(loss_value_buffer.detach().cpu()[0])
            print(
                f"[optimized-ddp-flash] step {step}/{num_steps} "
                f"loss={loss_value:.4f} "
                f"tokens/step={batch['input_ids'].numel():,}"
            )

    torch.cuda.synchronize(device)
    total_time = perf_counter() - start_time
    if is_main:
        toks_sec = total_tokens / total_time if total_time > 0 else 0.0
        effective_bs = args.batch_size * args.grad_accum * world_size
        print(
            f"[optimized-ddp-flash] {num_steps} steps | total tokens {total_tokens:,} | "
            f"global batch {effective_bs} | {toks_sec:,.0f} toks/s per rank"
        )

    if dist.is_initialized():
        dist.destroy_process_group()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "ddp.py",
        base_args=[
            "--mode",
            "optimized_flash",
            "--variant",
            "single",
            "--batch-size",
            "16",
            "--grad-accum",
            "1",
            "--max-length",
            "1024",
        ],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:ddp_flash",
        default_nproc_per_node=1,
        multi_gpu_required=False,
        name="optimized_ddp_flash",
    )
