"""Baseline DualPipeV demo approximating the V-shape (cut-in-half) schedule."""

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
    parser = argparse.ArgumentParser(description="Baseline DualPipeV schedule.")
    add_pipeline_args(parser)
    parser.add_argument(
        "--dual-window-default",
        type=int,
        default=2,
        help="Maximum inflight microbatches on stage0 when --dual-window is not provided.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device_ids = parse_device_ids(args.device_ids)
    stage_count = resolve_n_stages(args.n_stages)
    dual_window = args.dual_window or max(stage_count, args.dual_window_default)

    config = PipelineConfig(
        schedule="dualpipev",
        n_stages=stage_count,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size or stage_count,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        learning_rate=args.learning_rate,
        non_blocking=False,
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
                f"[baseline-dualpipev] step {step + 1}/{args.steps} loss={loss:.4f} "
                f"dual_window={config.dual_window} micro_batch={config.micro_batch_size}"
            )

    torch.cuda.synchronize()
    elapsed = perf_counter() - start
    avg_loss = total_loss / max(1, args.steps)
    elems = args.steps * config.batch_size * config.input_dim
    elems_per_sec = elems / elapsed if elapsed > 0 else 0.0

    print(
        f"[baseline-dualpipev] done in {elapsed:.2f}s | avg_loss={avg_loss:.4f} | "
        f"elements/s={elems_per_sec:,.0f}"
    )
    print(format_telemetry("baseline-dualpipev", telemetry_total))


def get_benchmark():
    return TorchrunScriptBenchmark(
        script_path=Path(__file__).parent / "pipeline_dualpipev.py",
        base_args=[
            "--mode",
            "baseline",
            "--n-stages",
            "2",
            "--device-ids",
            "0,0",
            "--batch-size",
            "256",
            "--micro-batch-size",
            "16",
            "--dual-window",
            "2",
            "--hidden-dim",
            "2048",
            "--depth",
            "12",
        ],
        config_arg_map={"iterations": "--steps"},
        target_label="labs/train_distributed:dualpipev_2stages",
        default_nproc_per_node=1,
        multi_gpu_required=False,
        name="baseline_pipeline_dualpipev_2stages",
    )
