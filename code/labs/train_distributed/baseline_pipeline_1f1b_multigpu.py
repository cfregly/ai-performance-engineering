"""Baseline 1F1B demo with over-fragmented micro-batching."""

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
    parser = argparse.ArgumentParser(description="Baseline 1F1B pipeline.")
    add_pipeline_args(parser)
    parser.set_defaults(batch_size=256, hidden_dim=2048, depth=12)
    return parser.parse_args()


def main():
    args = parse_args()
    device_ids = parse_device_ids(args.device_ids)
    require_min_gpus(2, script_name="baseline_pipeline_1f1b_multigpu.py")
    stage_count = resolve_n_stages(args.n_stages)
    default_micro = args.micro_batch_size
    if default_micro is None:
        default_micro = max(1, args.batch_size // max(1, stage_count))
        if args.batch_size % default_micro != 0:
            default_micro = max(1, args.batch_size // max(1, stage_count * 2))
        if args.batch_size % default_micro != 0:
            default_micro = args.batch_size

    config = PipelineConfig(
        schedule="1f1b",
        n_stages=stage_count,
        batch_size=args.batch_size,
        micro_batch_size=default_micro,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        learning_rate=args.learning_rate,
        non_blocking=False,
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
                f"[baseline-1f1b] step {step + 1}/{args.steps} loss={loss:.4f} "
                f"micro_batch={config.micro_batch_size}"
            )

    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    elems = args.steps * config.batch_size * config.input_dim
    elems_per_sec = elems / elapsed if elapsed > 0 else 0.0
    avg_loss = total_loss / max(1, args.steps)

    print(
        f"[baseline-1f1b] done in {elapsed:.2f}s | avg_loss={avg_loss:.4f} | "
        f"elements/s={elems_per_sec:,.0f}"
    )
    print(format_telemetry("baseline-1f1b", cumulative))


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "pipeline_1f1b_multigpu.py",
        base_args=[
            "--mode",
            "baseline",
            "--batch-size",
            "1024",
            "--micro-batch-size",
            "32",
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
        name="baseline_pipeline_1f1b_multigpu_2stages",
    )
