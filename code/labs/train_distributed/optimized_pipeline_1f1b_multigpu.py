"""Optimized 1F1B demo with tuned microbatching and async transfers."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

from core.benchmark.gpu_requirements import require_min_gpus
from labs.train_distributed.pipeline import (
    PipelineConfig,
    PipelineExperiment,
    PipelineTelemetry,
    add_pipeline_args,
    format_telemetry,
    resolve_n_stages,
    parse_device_ids,
)
from labs.train_distributed.training_utils.torchrun_harness import TorchrunScriptBenchmark

def parse_args():
    parser = argparse.ArgumentParser(description="Optimized 1F1B pipeline.")
    add_pipeline_args(parser)
    parser.set_defaults(batch_size=256, hidden_dim=2048, depth=12)
    parser.add_argument(
        "--micro-batch-target",
        type=int,
        default=4,
        help="Number of microbatches to target when not explicitly configured.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device_ids = parse_device_ids(args.device_ids)
    require_min_gpus(2, script_name="optimized_pipeline_1f1b_multigpu.py")
    stage_count = resolve_n_stages(args.n_stages)
    micro = args.micro_batch_size
    if micro is None:
        target_microbatches = max(1, stage_count * args.micro_batch_target)
        micro = max(1, args.batch_size // target_microbatches)
        if args.batch_size % micro != 0:
            micro = max(1, args.batch_size // max(1, stage_count * max(1, args.micro_batch_target // 2)))
        if args.batch_size % micro != 0:
            micro = args.batch_size

    config = PipelineConfig(
        schedule="1f1b",
        n_stages=stage_count,
        batch_size=args.batch_size,
        micro_batch_size=micro,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        learning_rate=args.learning_rate * 2,
        non_blocking=True,
        dtype=torch.float32,
        device_ids=device_ids,
        seed=args.seed,
    )

    experiment = PipelineExperiment(config)
    cumulative = PipelineTelemetry(config.n_stages, schedule=config.schedule)
    total_loss = 0.0
    inputs = torch.empty(config.batch_size, config.input_dim, dtype=config.dtype)
    targets = torch.empty_like(inputs)
    start = perf_counter()

    for step in range(args.steps):
        inputs.normal_()
        targets.normal_()
        loss, telemetry = experiment.run_batch(inputs, targets)
        cumulative.merge(telemetry)
        total_loss += loss

        if step % args.log_every == 0:
            print(
                f"[optimized-1f1b] step {step + 1}/{args.steps} loss={loss:.4f} "
                f"micro_batch={config.micro_batch_size}"
            )

    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    elems = args.steps * config.batch_size * config.input_dim
    elems_per_sec = elems / elapsed if elapsed > 0 else 0.0
    avg_loss = total_loss / max(1, args.steps)

    print(
        f"[optimized-1f1b] done in {elapsed:.2f}s | avg_loss={avg_loss:.4f} | "
        f"elements/s={elems_per_sec:,.0f}"
    )
    print(format_telemetry("optimized-1f1b", cumulative))


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "pipeline_1f1b_multigpu.py",
        base_args=[
            "--mode",
            "optimized",
            "--batch-size",
            "1024",
            "--micro-batch-size",
            "256",
            "--hidden-dim",
            "2048",
            "--depth",
            "12",
        ],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:1f1b_multigpu_2stages",
        default_nproc_per_node=None,
        default_iterations=10,
        multi_gpu_required=True,
        name="optimized_pipeline_1f1b_multigpu_2stages",
    )
