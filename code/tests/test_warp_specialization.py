#!/usr/bin/env python3
"""Compare the baseline and optimized warp specialization benchmark pairs.

Run from code/ so the chapter and core packages import:
    python -m tests.test_warp_specialization [--iterations N] [--warmup N]

core/scripts/run_warp_specialization_ci.sh runs this module with CI loop counts.
"""

import argparse
import importlib
import os
import sys
import time
from typing import NamedTuple

import torch

IS_CI = os.environ.get("CI", "").lower() in {"1", "true", "yes"}
DEV_ITER_CAP = int(os.environ.get("WARP_SPEC_DEV_ITER_CAP", "20"))
DEV_WARMUP_CAP = int(os.environ.get("WARP_SPEC_DEV_WARMUP_CAP", "3"))

# Global CLI arguments (set by main())
_cli_iterations = None
_cli_warmup = None


class ComparisonPair(NamedTuple):
    """A baseline/optimized pair. Each target is written as "module:ClassName"."""

    label: str
    baseline: str
    optimized: str
    default_iterations: int
    default_warmup: int


# run_benchmark() times benchmark_fn() in this process with host timers. CUDA
# binary benchmarks launch an executable on every call, so they do not belong here.
COMPARISON_PAIRS = (
    ComparisonPair(
        label="ch13",
        baseline="ch13.baseline_warp_specialization_training:BaselineWarpSpecializationTrainingBenchmark",
        optimized="ch13.optimized_warp_specialization_training:OptimizedWarpSpecializationTrainingBenchmark",
        default_iterations=10,
        default_warmup=5,
    ),
)


def _load_benchmark_class(target: str):
    module_name, class_name = target.split(":")
    return getattr(importlib.import_module(module_name), class_name)


def resolve_comparison_pairs():
    """Import every pair before timing starts so a missing target fails the run."""
    return [
        (pair, _load_benchmark_class(pair.baseline), _load_benchmark_class(pair.optimized))
        for pair in COMPARISON_PAIRS
    ]


def _resolve_loop_counts(benchmark, default_iterations: int, default_warmup: int):
    """Resolve iteration/warmup counts based on CLI flags, BenchmarkConfig, and defaults."""
    default_iterations = max(1, int(default_iterations))
    default_warmup = max(0, int(default_warmup))
    config = None

    if hasattr(benchmark, "get_config"):
        config = benchmark.get_config()

    # CLI flags take precedence
    if _cli_iterations is not None:
        iterations = _cli_iterations
    elif IS_CI:
        # CI mode: use config or default (no cap)
        iterations = config.iterations if config else default_iterations
    else:
        # Dev mode: use config or default with cap
        iterations = config.iterations if config else default_iterations
        iterations = min(iterations, DEV_ITER_CAP)

    if _cli_warmup is not None:
        warmup = _cli_warmup
    elif IS_CI:
        # CI mode: use config or default (no cap)
        warmup = config.warmup if config else default_warmup
    else:
        # Dev mode: use config or default with cap
        warmup = config.warmup if config else default_warmup
        warmup = min(warmup, DEV_WARMUP_CAP)

    return max(1, iterations), max(0, warmup)


def run_benchmark(benchmark_class, iterations=50, warmup=5):
    """Run a benchmark and return average time in ms."""
    benchmark = benchmark_class()
    iterations, warmup = _resolve_loop_counts(benchmark, iterations, warmup)

    # Setup
    benchmark.setup()
    torch.cuda.synchronize()

    # Warmup
    for _ in range(warmup):
        benchmark.benchmark_fn()
    torch.cuda.synchronize()

    # Benchmark
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        benchmark.benchmark_fn()
        torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms

    # Teardown
    benchmark.teardown()

    avg_time = sum(times) / len(times)
    median_time = sorted(times)[len(times) // 2]
    return avg_time, median_time, times

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare warp specialization benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Number of benchmark iterations (overrides defaults and config)"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=None,
        help="Number of warmup iterations (overrides defaults and config)"
    )

    args = parser.parse_args(argv)

    # Set global CLI arguments for _resolve_loop_counts
    global _cli_iterations, _cli_warmup
    _cli_iterations = args.iterations
    _cli_warmup = args.warmup

    pairs = resolve_comparison_pairs()
    if not torch.cuda.is_available():
        labels = ", ".join(pair.label for pair, _, _ in pairs)
        raise SystemExit(
            f"SKIPPED: warp specialization comparison ({labels}) requires a CUDA GPU"
        )

    print("=" * 80)
    print("Warp Specialization Benchmark Comparison")
    print("=" * 80)
    if _cli_iterations is not None or _cli_warmup is not None:
        print(f"CLI overrides: iterations={_cli_iterations}, warmup={_cli_warmup}")
    print()

    results = {}

    for pair, baseline_class, optimized_class in pairs:
        print(f"\n--- {pair.label} ---")
        print("Running baseline...")
        baseline_avg, baseline_median, _ = run_benchmark(
            baseline_class, iterations=pair.default_iterations, warmup=pair.default_warmup
        )
        print(f"  Baseline: {baseline_avg:.3f} ms (median: {baseline_median:.3f} ms)")

        print("Running optimized...")
        optimized_avg, optimized_median, _ = run_benchmark(
            optimized_class, iterations=pair.default_iterations, warmup=pair.default_warmup
        )
        print(f"  Optimized: {optimized_avg:.3f} ms (median: {optimized_median:.3f} ms)")

        speedup = baseline_avg / optimized_avg
        print(f"  Speedup: {speedup:.2f}x")
        results[pair.label] = (baseline_avg, optimized_avg, speedup)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for chapter, (baseline, optimized, speedup) in results.items():
        better = "✓ Optimized better" if speedup > 1.0 else "✗ Optimized worse" if speedup < 1.0 else "≈ Same"
        print(f"{chapter.upper()}: Baseline={baseline:.3f}ms, Optimized={optimized:.3f}ms, Speedup={speedup:.2f}x {better}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
