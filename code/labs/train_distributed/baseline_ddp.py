"""Baseline DDP training loop kept intentionally simple for comparison."""

from __future__ import annotations

import argparse
import os

from core.common.device_utils import resolve_local_rank
from time import perf_counter
from pathlib import Path

from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50, help="Number of optimization steps.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-rank batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    return parser.parse_args()


def _maybe_fused_adamw(params, lr):
    import torch

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
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    from labs.train_distributed.training_utils.utils import (
    build_dataloader,
    build_text_model,
    build_tokenizer,
    configure_training_matmul_policy,
    get_dataset,
    set_seed,
)

    args = parse_args()
    local_rank = resolve_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    if not dist.is_initialized() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl", device_id=local_rank)

    rank = dist.get_rank() if dist.is_initialized() else 0
    is_main = rank == 0
    set_seed(42)
    configure_training_matmul_policy()

    tokenizer = build_tokenizer()
    dataset = get_dataset()["train"]
    dataloader = build_dataloader(
        dataset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        distributed=dist.is_initialized(),
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True,
    )

    model_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = build_text_model(dtype=model_dtype)
    model.to(device)
    model.train()

    ddp_model = model
    if dist.is_initialized():
        ddp_model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            gradient_as_bucket_view=True,
            find_unused_parameters=False,
        )

    optimizer = _maybe_fused_adamw(ddp_model.parameters(), args.learning_rate)

    num_steps = min(args.steps, len(dataloader))
    start = perf_counter()
    total_tokens = 0
    loss_value_buffer = torch.empty(1, dtype=torch.float64, device=device)

    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        optimizer.zero_grad(set_to_none=True)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        batch["labels"] = batch["input_ids"].clone()
        outputs = ddp_model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_tokens += batch["input_ids"].numel()

        if is_main and step % 10 == 0:
            loss_value_buffer[0].copy_(loss.detach())
            loss_value = float(loss_value_buffer.detach().cpu()[0])
            print(f"[baseline-ddp] step {step}/{num_steps} | loss={loss_value:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    if is_main:
        toks_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
        print(f"[baseline-ddp] finished {num_steps} steps in {elapsed:.1f}s "
              f"({toks_per_sec:,.0f} toks/s per rank)")

    if dist.is_initialized():
        dist.destroy_process_group()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "ddp.py",
        base_args=["--mode", "baseline", "--variant", "single", "--batch-size", "16"],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:ddp",
        default_nproc_per_node=1,
        multi_gpu_required=False,
        name="baseline_ddp",
    )
