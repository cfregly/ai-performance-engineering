"""Optimized DualPipe demo with deeper windows and async handoffs."""

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
    parser = argparse.ArgumentParser(description="Optimized DualPipe schedule.")
    add_pipeline_args(parser)
    parser.add_argument(
        "--dual-window-multiplier",
        type=int,
        default=2,
        help="Scale factor applied to n_stages when picking a default DualPipe window.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device_ids = parse_device_ids(args.device_ids)
    stage_count = resolve_n_stages(args.n_stages)
    default_window = stage_count * max(1, args.dual_window_multiplier)
    dual_window = args.dual_window or default_window
    micro = args.micro_batch_size or (stage_count * 2)

    config = PipelineConfig(
        schedule="dualpipe",
        n_stages=stage_count,
        batch_size=args.batch_size,
        micro_batch_size=micro,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        learning_rate=args.learning_rate * 2,
        non_blocking=True,
        dual_window=dual_window,
        device_ids=device_ids,
        seed=args.seed,
    )

    experiment = PipelineExperiment(config)
    telemetry_total = PipelineTelemetry(config.n_stages, schedule=config.schedule)
    total_loss = 0.0
    inputs = torch.empty(config.batch_size, config.input_dim, dtype=config.dtype)
    targets = torch.empty_like(inputs)
    start = perf_counter()

    for step in range(args.steps):
        inputs.normal_()
        targets.normal_()
        loss, telemetry = experiment.run_batch(inputs, targets)
        telemetry_total.merge(telemetry)
        total_loss += loss

        if step % args.log_every == 0:
            print(
                f"[optimized-dualpipe] step {step + 1}/{args.steps} loss={loss:.4f} "
                f"dual_window={config.dual_window} micro_batch={config.micro_batch_size}"
            )

    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    avg_loss = total_loss / max(1, args.steps)
    elems = args.steps * config.batch_size * config.input_dim
    elems_per_sec = elems / elapsed if elapsed > 0 else 0.0

    print(
        f"[optimized-dualpipe] done in {elapsed:.2f}s | avg_loss={avg_loss:.4f} | "
        f"elements/s={elems_per_sec:,.0f}"
    )
    print(format_telemetry("optimized-dualpipe", telemetry_total))


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "pipeline_dualpipe.py",
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
            "--dual-window",
            "4",
            "--hidden-dim",
            "2048",
            "--depth",
            "12",
        ],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:dualpipe_2stages",
        default_nproc_per_node=1,
        multi_gpu_required=False,
        name="optimized_pipeline_dualpipe_2stages",
    )
