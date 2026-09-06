#!/usr/bin/env python3
"""Torchrun worker for the symmetric-memory performance benchmark pair."""

from __future__ import annotations

import argparse
import importlib
import math
import statistics
from collections.abc import Sequence

import torch
import torch.distributed as dist

from ch04.distributed_helper import run_main_with_skip_status
from ch04.symmetric_memory_perf_common import (
    SYMMETRIC_MEMORY_PERF_BASELINE_NVTX_RANGE,
    SYMMETRIC_MEMORY_PERF_OPTIMIZED_NVTX_RANGE,
    write_symmetric_memory_perf_child_result,
)
from core.profiling.nvtx_helper import nvtx_range

_VARIANTS = {
    "baseline": (
        "ch04.baseline_symmetric_memory_perf_multigpu",
        SYMMETRIC_MEMORY_PERF_BASELINE_NVTX_RANGE,
    ),
    "optimized": (
        "ch04.optimized_symmetric_memory_perf_multigpu",
        SYMMETRIC_MEMORY_PERF_OPTIMIZED_NVTX_RANGE,
    ),
}


def _mean_measured_iteration_ms(samples: Sequence[float]) -> float:
    """Return the mean of real, post-warmup CUDA-event samples."""
    if not samples:
        raise RuntimeError("No measured symmetric-memory perf timing samples")
    normalized = [float(sample) for sample in samples]
    if any(not math.isfinite(sample) or sample <= 0 for sample in normalized):
        raise RuntimeError(
            "Symmetric-memory perf timing samples must be finite and positive"
        )
    return float(statistics.fmean(normalized))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one symmetric-memory performance benchmark variant."
    )
    parser.add_argument("--variant", choices=tuple(_VARIANTS), required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    return args


def main() -> None:
    args = _parse_args()
    module_name, profile_range = _VARIANTS[args.variant]
    module = importlib.import_module(module_name)
    benchmark = module.get_benchmark()

    try:
        benchmark.setup()
        for _ in range(args.warmup):
            benchmark.benchmark_fn()
            benchmark.finalize_iteration_metrics()

        torch.cuda.synchronize()
        measured_times_ms: list[float] = []
        with nvtx_range(profile_range, enable=True):
            for _ in range(args.iterations):
                benchmark.benchmark_fn()
                benchmark.finalize_iteration_metrics()
                measured_times_ms.append(float(benchmark._last_avg_ms))
            torch.cuda.synchronize()
        mean_iteration_ms = _mean_measured_iteration_ms(measured_times_ms)

        validation_error = benchmark.validate_result()
        if validation_error:
            raise RuntimeError(
                f"Symmetric-memory perf {args.variant} validation failed: "
                f"{validation_error}"
            )
        benchmark.capture_verification_payload()
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        write_symmetric_memory_perf_child_result(
            benchmark,
            variant=args.variant,
            rank=rank,
            world_size=world_size,
        )
        dist.barrier()
        if rank == 0:
            print(f"rank0 time_per_iter_ms: {mean_iteration_ms:.9f}", flush=True)
            print(
                f"symmetric_memory_perf variant={args.variant} "
                f"warmup={args.warmup} iterations={args.iterations} validated",
                flush=True,
            )
    finally:
        try:
            benchmark.teardown()
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()


if __name__ == "__main__":
    exit_code = run_main_with_skip_status(main)
    if exit_code:
        raise SystemExit(exit_code)
