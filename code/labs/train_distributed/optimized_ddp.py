"""Optimized DDP training loop showcasing perf levers from the book/labs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DistributedModel

from core.common.device_utils import resolve_local_rank
from labs.train_distributed.training_utils.child_result import child_result_requested
from labs.train_distributed.training_utils.ddp_child_result import (
    bind_distributed_sampler_seed,
    initialize_ddp_seed,
    make_ddp_adamw,
    make_ddp_child_result_contract,
    publish_ddp_child_result,
)
from labs.train_distributed.training_utils.gradient_accumulation import (
    build_gradient_accumulation_plan,
    gradient_sync_context,
    validate_gradient_accumulation,
)
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark
from labs.train_distributed.training_utils.utils import (
    build_dataloader,
    build_text_model,
    build_tokenizer,
    configure_training_matmul_policy,
    get_dataset,
    make_causal_lm_labels,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50, help="Number of optimization steps.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-rank microbatch size.")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    return parser.parse_args()


def main():
    args = parse_args()
    validate_gradient_accumulation(args.steps, args.grad_accum)
    local_rank = resolve_local_rank()
    if not torch.cuda.is_available():
        raise RuntimeError("DDP optimized run requires CUDA GPUs.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if not dist.is_initialized():
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            dist.init_process_group(backend="nccl", device_id=local_rank)
        else:
            raise RuntimeError(
                "DDP optimized run requires torch.distributed process group to be initialized."
            )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main = rank == 0
    active_seed = initialize_ddp_seed()
    configure_training_matmul_policy()
    use_ddp = dist.is_initialized()
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
    )
    bind_distributed_sampler_seed(dataloader, active_seed)

    model = build_text_model(dtype=torch.bfloat16)
    model.to(device)
    model.train()

    ddp_model = model
    if use_ddp:
        ddp_model = DistributedModel(
            model,
            device_ids=[local_rank],
            static_graph=True,
            bucket_cap_mb=50,
            gradient_as_bucket_view=True,
        )

    optimizer = make_ddp_adamw(ddp_model.parameters(), args.learning_rate, prefer_fused=True)
    num_steps = min(args.steps, len(dataloader))
    accumulation_plan = build_gradient_accumulation_plan(num_steps, args.grad_accum)
    total_tokens = 0
    start_time = perf_counter()
    loss_value_buffer = torch.empty(1, dtype=torch.float64, device=device)

    completed_steps = 0
    final_batch = None
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        accumulation = accumulation_plan[step]
        sync_ctx = gradient_sync_context(
            ddp_model,
            accumulation,
            distributed=use_ddp,
        )
        with sync_ctx:
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            batch["labels"] = make_causal_lm_labels(batch["input_ids"], batch["attention_mask"])
            outputs = ddp_model(**batch)
            loss = outputs.loss / accumulation.group_size
            loss.backward()

        if accumulation.should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        completed_steps += 1
        final_batch = batch

        total_tokens += batch["input_ids"].numel()

        if step % 10 == 0 and is_main:
            loss_value_buffer[0].copy_(loss.detach())
            loss_value = float(loss_value_buffer.detach().cpu()[0])
            print(
                f"[optimized-ddp] step {step}/{num_steps} "
                f"loss={loss_value:.4f} "
                f"tokens/step={batch['input_ids'].numel():,}"
            )

    torch.cuda.synchronize(device)
    total_time = perf_counter() - start_time
    if completed_steps <= 0:
        raise RuntimeError("DDP training completed no optimization steps")
    if is_main:
        toks_sec = total_tokens / total_time if total_time > 0 else 0.0
        effective_bs = args.batch_size * args.grad_accum * world_size
        print(
            f"[optimized-ddp] {completed_steps} steps | total tokens {total_tokens:,} | "
            f"global batch {effective_bs} | {toks_sec:,.0f} toks/s per rank"
        )
        if child_result_requested():
            print(
                f"rank0 time_per_iter_ms: {total_time * 1000.0 / completed_steps:.9f}",
                flush=True,
            )

    publish_ddp_child_result(
        candidate_model=ddp_model,
        reference_model=model,
        final_batch=final_batch,
        completed_iterations=completed_steps,
    )

    if dist.is_initialized():
        dist.destroy_process_group()


def get_benchmark():
    """Expose torchrun-wrapped benchmark for the harness."""
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "ddp.py",
        base_args=[
            "--mode",
            "optimized",
            "--variant",
            "single",
            "--batch-size",
            "16",
            "--grad-accum",
            "1",
        ],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:ddp",
        default_nproc_per_node=1,
        multi_gpu_required=False,
        name="optimized_ddp",
        child_result_contract=make_ddp_child_result_contract(multigpu=False),
    )
