"""Optimized GPipe pipeline demo with tuned micro-batching."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import torch

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
    parser = argparse.ArgumentParser(description="Optimized GPipe pipeline.")
    add_pipeline_args(parser)
    parser.add_argument(
        "--micro-batch-overlap",
        type=int,
        default=None,
        help="Default microbatch fan-out used when --micro-batch-size is not provided.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device_ids = parse_device_ids(args.device_ids)
    stage_count = resolve_n_stages(args.n_stages)
    overlap = stage_count if args.micro_batch_overlap is None else args.micro_batch_overlap
    target_micro = args.micro_batch_size
    if target_micro is None:
        target_micro = max(4, args.batch_size // max(1, overlap))

    config = PipelineConfig(
        schedule="gpipe",
        n_stages=stage_count,
        batch_size=args.batch_size,
        micro_batch_size=target_micro,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        learning_rate=args.learning_rate * 2,
        non_blocking=True,
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
                f"[optimized-gpipe] step {step + 1}/{args.steps} loss={loss:.4f} "
                f"micro_batch={config.micro_batch_size}"
            )

    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    elems = args.steps * config.batch_size * config.input_dim
    elems_per_sec = elems / elapsed if elapsed > 0 else 0.0
    avg_loss = total_loss / max(1, args.steps)

    print(
        f"[optimized-gpipe] done in {elapsed:.2f}s | avg_loss={avg_loss:.4f} | "
        f"elements/s={elems_per_sec:,.0f}"
    )
    print(format_telemetry("optimized-gpipe", cumulative))


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "pipeline_gpipe.py",
        base_args=[
            "--mode",
            "optimized",
            "--n-stages",
            "2",
            "--device-ids",
            "0,0",
            "--batch-size",
            "256",
            "--micro-batch-size",
            "64",
            "--hidden-dim",
            "2048",
            "--depth",
            "12",
        ],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:pipeline_gpipe_2stages",
        default_nproc_per_node=1,
        multi_gpu_required=False,
        name="optimized_pipeline_gpipe_2stages",
    )
