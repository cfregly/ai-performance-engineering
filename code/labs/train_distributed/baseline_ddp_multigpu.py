"""Baseline DDP training loop kept intentionally simple for comparison."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import perf_counter

from core.benchmark.gpu_requirements import require_min_gpus
from core.common.device_utils import resolve_local_rank
from labs.train_distributed.training_utils.child_result import child_result_requested
from labs.train_distributed.training_utils.ddp_child_result import (
    bind_distributed_sampler_seed,
    initialize_ddp_seed,
    make_ddp_adamw,
    make_ddp_child_result_contract,
    publish_ddp_child_result,
)
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50, help="Number of optimization steps.")
    parser.add_argument("--batch-size", type=int, default=8, help="Per-rank batch size.")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="AdamW learning rate.")
    return parser.parse_args()


def main():
    require_min_gpus(2, script_name="baseline_ddp_multigpu.py")
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DistributedModel

    from labs.train_distributed.training_utils.utils import (
        build_dataloader,
        build_text_model,
        build_tokenizer,
        configure_training_matmul_policy,
        get_dataset,
        make_causal_lm_labels,
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
    active_seed = initialize_ddp_seed()
    configure_training_matmul_policy()

    tokenizer = build_tokenizer()
    dataset = get_dataset()["train"]
    dataloader = build_dataloader(
        dataset,
        tokenizer,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        distributed=dist.is_initialized() and dist.get_world_size() > 1,
        num_workers=2,
        prefetch_factor=2,
    )
    bind_distributed_sampler_seed(dataloader, active_seed)

    model = build_text_model()
    model.to(device)
    model.train()

    ddp_model = model
    if dist.is_initialized() and dist.get_world_size() > 1:
        ddp_model = DistributedModel(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            gradient_as_bucket_view=False,
            find_unused_parameters=False,
            bucket_cap_mb=1,
        )

    # BF16 fused and unfused AdamW round updates differently. Keep optimizer
    # arithmetic shared so the pair isolates DDP communication and input loading.
    optimizer = make_ddp_adamw(ddp_model.parameters(), args.learning_rate, prefer_fused=True)

    num_steps = min(args.steps, len(dataloader))
    start = perf_counter()
    total_tokens = 0
    loss_value_buffer = torch.empty(1, dtype=torch.float64, device=device)

    completed_steps = 0
    final_batch = None
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break

        optimizer.zero_grad(set_to_none=True)

        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

        batch["labels"] = make_causal_lm_labels(batch["input_ids"], batch["attention_mask"])
        outputs = ddp_model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        completed_steps += 1
        final_batch = batch

        total_tokens += batch["input_ids"].numel()

        if is_main and step % 10 == 0:
            loss_value_buffer[0].copy_(loss.detach())
            loss_value = float(loss_value_buffer.detach().cpu()[0])
            print(f"[baseline-ddp] step {step}/{num_steps} | loss={loss_value:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = perf_counter() - start
    if completed_steps <= 0:
        raise RuntimeError("DDP training completed no optimization steps")
    if is_main:
        toks_per_sec = total_tokens / elapsed if elapsed > 0 else 0.0
        print(
            f"[baseline-ddp] finished {completed_steps} steps in {elapsed:.1f}s "
            f"({toks_per_sec:,.0f} toks/s per rank)"
        )
        if child_result_requested():
            print(
                f"rank0 time_per_iter_ms: {elapsed * 1000.0 / completed_steps:.9f}",
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
        base_args=["--mode", "baseline", "--variant", "multigpu", "--batch-size", "32"],
        config_arg_map={"iterations": "--steps"},
        multi_gpu_required=True,
        target_label="labs/train_distributed:ddp_multigpu",
        default_nproc_per_node=None,
        name="baseline_ddp_multigpu",
        child_result_contract=make_ddp_child_result_contract(multigpu=True),
    )
