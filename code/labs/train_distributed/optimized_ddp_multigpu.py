"""Optimized DDP training loop showcasing perf levers from the book/labs."""

from __future__ import annotations

import argparse
import os
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DistributedModel

from core.benchmark.gpu_requirements import require_min_gpus
from core.common.device_utils import resolve_local_rank
from core.utils.compile_utils import get_optimal_compile_mode

try:
    from arch_config import prefer_sdpa_backends  # type: ignore
except Exception:  # pragma: no cover - defensive
    prefer_sdpa_backends = None  # type: ignore

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
from labs.train_distributed.training_utils.deferred_metrics import DeferredTrainingProgress
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
    parser.add_argument("--steps", type=int, default=200, help="Maximum number of training microbatches.")
    parser.add_argument("--batch-size", type=int, default=16, help="Per-rank microbatch size.")
    parser.add_argument("--grad-accum", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile on the model.")
    return parser.parse_args()


def main():
    require_min_gpus(2, script_name="optimized_ddp_multigpu.py")
    args = parse_args()
    validate_gradient_accumulation(args.steps, args.grad_accum)
    local_rank = resolve_local_rank()
    if not torch.cuda.is_available():
        raise RuntimeError("DDP optimized run requires CUDA GPUs.")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    configure_training_matmul_policy()

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
    tokenizer = build_tokenizer()
    dataset = get_dataset(tokenizer=tokenizer)["train"]

    dataloader = build_dataloader(
        dataset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        distributed=True,
        num_workers=4,
        prefetch_factor=4,
        pin_memory=True,
    )
    bind_distributed_sampler_seed(dataloader, active_seed)

    model = build_text_model()
    model.to(device)
    model.train()

    ddp_model = DistributedModel(
        model,
        device_ids=[local_rank],
        bucket_cap_mb=50,
        gradient_as_bucket_view=True,
    )

    if args.compile:
        # get_optimal_compile_mode keeps max-autotune on the pinned toolchain but
        # falls back to "default" on sm_103 + Triton >= 3.6 (unloadable tcgen05).
        ddp_model = torch.compile(
            ddp_model,
            mode=get_optimal_compile_mode("max-autotune"),
            fullgraph=True,
            dynamic=False,
        )

    optimizer = make_ddp_adamw(ddp_model.parameters(), args.learning_rate, prefer_fused=True)

    num_steps = min(args.steps, len(dataloader))
    accumulation_plan = build_gradient_accumulation_plan(num_steps, args.grad_accum)
    total_tokens = 0
    progress = (
        DeferredTrainingProgress(num_steps=num_steps, interval=10, device=device)
        if is_main and num_steps > 0
        else None
    )
    start_time = perf_counter()

    completed_steps = 0
    final_batch = None
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        accumulation = accumulation_plan[step]
        sdpa_ctx = prefer_sdpa_backends() if prefer_sdpa_backends is not None else nullcontext()
        sync_ctx = gradient_sync_context(
            ddp_model,
            accumulation,
            distributed=True,
        )
        with sync_ctx:
            with sdpa_ctx:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                batch["labels"] = make_causal_lm_labels(
                    batch["input_ids"], batch["attention_mask"]
                )
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
            if progress is None:
                raise RuntimeError("Training progress buffer was not initialized")
            progress.record(step=step, loss=outputs.loss.detach(), tokens=batch["input_ids"].numel())

    torch.cuda.synchronize(device)
    total_time = perf_counter() - start_time
    if completed_steps <= 0:
        raise RuntimeError("DDP training completed no optimization steps")
    if is_main:
        if progress is None:
            raise RuntimeError("Training progress buffer was not initialized")
        for sample in progress.read():
            print(
                f"[optimized-ddp] step {sample.step}/{num_steps} "
                f"loss={sample.loss:.4f} tokens/step={sample.tokens:,}"
            )
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
            "multigpu",
            "--batch-size",
            "32",
            "--grad-accum",
            "1",
        ],
        config_arg_map={"iterations": "--steps"},
        multi_gpu_required=True,
        target_label="labs/train_distributed:ddp_multigpu",
        default_nproc_per_node=None,
        name="optimized_ddp_multigpu",
        child_result_contract=make_ddp_child_result_contract(multigpu=True),
    )
