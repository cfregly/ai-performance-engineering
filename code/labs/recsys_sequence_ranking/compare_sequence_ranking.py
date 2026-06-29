"""Direct baseline-vs-optimized runner for the RecSys sequence-ranking lab."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch

from core.benchmark.utils import scalar_tensor_to_float
from core.harness.benchmark_harness import lock_gpu_clocks
from labs.recsys_sequence_ranking.baseline_sequence_ranking import BaselineSequenceRankingBenchmark
from labs.recsys_sequence_ranking.optimized_sequence_ranking import (
    OptimizedSequenceRankingBenchmark,
)
from labs.recsys_sequence_ranking.recsys_sequence_ranking_common import (
    apply_cli_overrides,
    default_workload,
)


def _measure(bench, *, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        bench.benchmark_fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream()
        start.record(current_stream)
        for _ in range(iterations):
            bench.benchmark_fn()
        end.record(current_stream)
        end.synchronize()
        return float(start.elapsed_time(end) / iterations)

    total_ms = 0.0
    for _ in range(iterations):
        t0 = time.perf_counter()
        bench.benchmark_fn()
        total_ms += (time.perf_counter() - t0) * 1000.0
    return float(total_ms / iterations)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per path")
    parser.add_argument("--iterations", type=int, default=20, help="Timed iterations per path")
    parser.add_argument(
        "--no-lock-gpu-clocks",
        dest="lock_gpu_clocks",
        action="store_false",
        help="Skip harness clock locking for local experimentation",
    )
    parser.add_argument("--sm-clock-mhz", type=int, default=None, help="Optional SM application clock")
    parser.add_argument("--mem-clock-mhz", type=int, default=None, help="Optional memory application clock")
    parser.add_argument("--json", action="store_true", help="Print JSON payload")
    parser.add_argument("--json-out", type=Path, default=None, help="Write JSON payload to file")
    parser.set_defaults(lock_gpu_clocks=True)
    args, workload_argv = parser.parse_known_args()

    workload = apply_cli_overrides(default_workload(), workload_argv)
    lock_ctx = (
        lock_gpu_clocks(device=0, sm_clock_mhz=args.sm_clock_mhz, mem_clock_mhz=args.mem_clock_mhz)
        if args.lock_gpu_clocks
        else nullcontext()
    )
    with lock_ctx:
        baseline = BaselineSequenceRankingBenchmark()
        optimized = OptimizedSequenceRankingBenchmark()
        baseline.workload = workload
        optimized.workload = workload
        baseline._refresh_workload_metadata()
        optimized._refresh_workload_metadata()

        baseline.setup()
        optimized.setup()
        try:
            baseline_ms = _measure(baseline, warmup=args.warmup, iterations=args.iterations)
            optimized_ms = _measure(optimized, warmup=args.warmup, iterations=args.iterations)
            max_abs_diff = scalar_tensor_to_float(
                (baseline.output - optimized.output).abs().max()
            )
            payload = {
                "batch_size": workload.batch_size,
                "seq_len": workload.seq_len,
                "num_tables": workload.num_tables,
                "embedding_dim": workload.embedding_dim,
                "hidden_dim": workload.hidden_dim,
                "num_candidates": workload.num_candidates,
                "score_backend": optimized.score_backend,
                "compile_enabled": optimized.compile_enabled,
                "baseline_latency_ms": baseline_ms,
                "optimized_latency_ms": optimized_ms,
                "speedup": baseline_ms / optimized_ms if optimized_ms > 0 else float("inf"),
                "max_abs_diff": max_abs_diff,
                "lock_gpu_clocks": args.lock_gpu_clocks,
                "sm_clock_mhz": args.sm_clock_mhz,
                "mem_clock_mhz": args.mem_clock_mhz,
            }
        finally:
            baseline.teardown()
            optimized.teardown()

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("RecSys sequence ranking direct compare")
        print(
            f"shape=batch{workload.batch_size} seq{workload.seq_len} "
            f"tables{workload.num_tables} candidates{workload.num_candidates}"
        )
        print(f"baseline:  {baseline_ms:.6f} ms")
        print(f"optimized: {optimized_ms:.6f} ms")
        print(f"speedup:   {payload['speedup']:.3f}x")
        print(f"max_abs_diff: {max_abs_diff:.8f}")
        print(f"score_backend: {optimized.score_backend} compile_enabled={optimized.compile_enabled}")
        if args.json_out is not None:
            print(f"wrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
