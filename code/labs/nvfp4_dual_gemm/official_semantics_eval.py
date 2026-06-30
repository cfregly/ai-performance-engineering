"""Leaderboard-style evaluator mirroring reference-kernels eval_better_bench semantics.

This script intentionally follows `problems/nvidia/eval_better_bench.py` behavior for
benchmark mode used by nvfp4 dual gemm leaderboard runs:
- pre-generate NUM_ITERATIONS_PER_BENCHMARK inputs (seed += 42 each sample)
- one mandatory correctness pass before timing
- optional recheck every timed iteration (enabled for leaderboard-style mode)
- per-iteration L2 cache clear before timing
- adaptive stopping based on relative error / max_time / wallclock
"""

from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from core.harness.benchmark_harness import lock_gpu_clocks
from labs.nvfp4_dual_gemm.gpu_isolation import ensure_gpu_isolation
from labs.nvfp4_dual_gemm.local_eval_loader import (
    load_reference_module,
    load_submission_module,
    load_utils_module,
)
from labs.nvfp4_dual_gemm.utils import clear_l2_cache_large

BENCHMARKS = (
    {"name": "case0", "m": 256, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
    {"name": "case1", "m": 512, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
    {"name": "case2", "m": 256, "n": 3072, "k": 4096, "l": 1, "seed": 1111},
    {"name": "case3", "m": 512, "n": 3072, "k": 7168, "l": 1, "seed": 1111},
)

TOP_SCORE_SECONDS_598 = 1.2913403524642259e-05
TOP_SCORE_US_598 = TOP_SCORE_SECONDS_598 * 1e6


@dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def _clone_tree(x: Any) -> Any:
    if isinstance(x, tuple):
        return tuple(_clone_tree(v) for v in x)
    if isinstance(x, list):
        return [_clone_tree(v) for v in x]
    if isinstance(x, dict):
        return {k: _clone_tree(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.clone()
    return x


@contextmanager
def _null_ctx():
    yield None


def _calculate_stats(durations_ns: list[float]) -> Stats:
    runs = 0
    mean = 0.0
    m2 = 0.0
    best = float("inf")
    worst = float("-inf")
    for value in durations_ns:
        runs += 1
        delta = value - mean
        mean += delta / runs
        m2 += delta * (value - mean)
        if value < best:
            best = value
        if value > worst:
            worst = value
    if runs == 0:
        raise ValueError("durations_ns must not be empty")
    if runs > 1:
        variance = m2 / (runs - 1)
        std = math.sqrt(variance)
        err = std / math.sqrt(runs)
    else:
        std = 0.0
        err = 0.0
    return Stats(
        runs=runs,
        mean=mean,
        std=std,
        err=err,
        best=best,
        worst=worst,
    )


def _build_data_list(reference_mod: Any, case: dict[str, int | str], num_iterations: int) -> list[Any]:
    args = {
        "m": int(case["m"]),
        "n": int(case["n"]),
        "k": int(case["k"]),
        "l": int(case["l"]),
        "seed": int(case["seed"]),
    }
    data_list = []
    for _ in range(num_iterations):
        args["seed"] += 42
        data_list.append(reference_mod.generate_input(**args))
    return data_list


def _run_single_benchmark_case(
    submission_mod: Any,
    reference_mod: Any,
    case: dict[str, int | str],
    *,
    num_iterations: int,
    max_repeats: int,
    max_time_ns: float,
    recheck: bool,
    clear_l2: bool,
) -> dict[str, Any]:
    data_list = _build_data_list(reference_mod, case, num_iterations=num_iterations)

    # Mandatory correctness check before timing.
    outputs = []
    for data in data_list:
        outputs.append(submission_mod.custom_kernel(_clone_tree(data)))
    torch.cuda.synchronize()

    for reference_input, custom_output in zip(data_list, outputs):
        ok, msg = reference_mod.check_implementation(reference_input, custom_output)
        if not ok:
            raise RuntimeError(f"{case['name']} verify failed: {msg}")

    durations_ns: list[float] = []
    benchmark_start_ns = time.perf_counter_ns()

    for i in range(int(max_repeats)):
        torch.cuda.synchronize()

        if clear_l2:
            clear_l2_cache_large()

        outputs_iter = []
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream()

        start.record(current_stream)
        for data in data_list:
            outputs_iter.append(submission_mod.custom_kernel(data))
        end.record(current_stream)
        end.synchronize()

        duration_ns = (start.elapsed_time(end) / float(num_iterations)) * 1e6

        if recheck:
            for reference_input, custom_output in zip(data_list, outputs_iter):
                ok, msg = reference_mod.check_implementation(reference_input, custom_output)
                if not ok:
                    raise RuntimeError(f"{case['name']} verify failed during timed loop: {msg}")

        durations_ns.append(float(duration_ns))

        total_bench_ns = time.perf_counter_ns() - benchmark_start_ns
        if i > 1 and total_bench_ns > 1e8:
            stats = _calculate_stats(durations_ns)
            if (
                (stats.err / stats.mean) < 0.001
                or (stats.mean * stats.runs) > max_time_ns
                or total_bench_ns > 120e9
            ):
                break

    stats = _calculate_stats(durations_ns)

    return {
        "name": case["name"],
        "m": int(case["m"]),
        "n": int(case["n"]),
        "k": int(case["k"]),
        "l": int(case["l"]),
        "seed": int(case["seed"]),
        "runs": stats.runs,
        "mean_ns": stats.mean,
        "std_ns": stats.std,
        "err_ns": stats.err,
        "best_ns": stats.best,
        "worst_ns": stats.worst,
        "mean_us": stats.mean / 1000.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-file",
        type=Path,
        default=Path("labs/nvfp4_dual_gemm/optimized_submission.py"),
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=Path("labs/nvfp4_dual_gemm/reference_submission.py"),
    )
    parser.add_argument("--num-iterations-per-benchmark", type=int, default=50)

    # Mirrors leaderboard mode in eval_better_bench.py
    parser.add_argument("--warmup-max-repeats", type=int, default=1000)
    parser.add_argument("--warmup-max-time-ns", type=float, default=5e8)
    parser.add_argument("--bench-max-repeats", type=int, default=1000)
    parser.add_argument("--bench-max-time-ns", type=float, default=30e9)

    parser.add_argument("--clear-l2", action="store_true", default=True)
    parser.add_argument("--no-clear-l2", dest="clear_l2", action="store_false")

    parser.add_argument("--lock-gpu-clocks", action="store_true", default=True)
    parser.add_argument("--no-lock-gpu-clocks", dest="lock_gpu_clocks", action="store_false")
    parser.add_argument("--sm-clock-mhz", type=int, default=1500)
    parser.add_argument("--mem-clock-mhz", type=int, default=None)
    parser.add_argument("--require-idle-gpu", action="store_true", default=False)
    parser.add_argument("--kill-foreign-gpu-jobs", action="store_true", default=False)
    parser.add_argument("--isolation-owner-pid", type=int, default=None)
    parser.add_argument("--isolation-settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--isolation-allow-cmd-substring",
        action="append",
        default=["python -m mcp.mcp_server --serve"],
    )

    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    if args.num_iterations_per_benchmark <= 0:
        raise ValueError("--num-iterations-per-benchmark must be > 0")

    utils_mod = load_utils_module(
        args.submission_file.parent / "utils.py",
        "nvfp4_dual_utils_for_official_semantics",
    )
    reference_mod = load_reference_module(
        args.reference_file,
        module_name="nvfp4_dual_reference_for_official_semantics",
        utils_module=utils_mod,
    )
    submission_mod = load_submission_module(
        args.submission_file,
        module_name="nvfp4_dual_submission_for_official_semantics",
        reference_module=reference_mod,
        utils_module=utils_mod,
    )

    isolation_preflight = None
    if args.require_idle_gpu or args.kill_foreign_gpu_jobs:
        isolation_preflight = ensure_gpu_isolation(
            owner_pid=args.isolation_owner_pid,
            kill_foreign=args.kill_foreign_gpu_jobs,
            require_idle=args.require_idle_gpu,
            settle_seconds=args.isolation_settle_seconds,
            context="official_semantics_preflight",
            allow_cmd_substrings=args.isolation_allow_cmd_substring,
        )

    lock_ctx = (
        lock_gpu_clocks(device=0, sm_clock_mhz=args.sm_clock_mhz, mem_clock_mhz=args.mem_clock_mhz)
        if args.lock_gpu_clocks
        else _null_ctx()
    )

    with lock_ctx:
        # Warmup all benchmark cases first (leaderboard behavior).
        for case in BENCHMARKS:
            _run_single_benchmark_case(
                submission_mod,
                reference_mod,
                case,
                num_iterations=args.num_iterations_per_benchmark,
                max_repeats=args.warmup_max_repeats,
                max_time_ns=args.warmup_max_time_ns,
                recheck=False,
                clear_l2=args.clear_l2,
            )

        # Timed leaderboard-style pass with recheck enabled.
        cases = [
            _run_single_benchmark_case(
                submission_mod,
                reference_mod,
                case,
                num_iterations=args.num_iterations_per_benchmark,
                max_repeats=args.bench_max_repeats,
                max_time_ns=args.bench_max_time_ns,
                recheck=True,
                clear_l2=args.clear_l2,
            )
            for case in BENCHMARKS
        ]

    score_us = float(math.exp(sum(math.log(case["mean_us"]) for case in cases) / len(cases)))
    score_seconds = score_us / 1e6

    payload = {
        "submission_file": str(args.submission_file),
        "reference_file": str(args.reference_file),
        "score_seconds": score_seconds,
        "score_us": score_us,
        "top_score_seconds_598": TOP_SCORE_SECONDS_598,
        "top_score_us_598": TOP_SCORE_US_598,
        "delta_vs_top_us": score_us - TOP_SCORE_US_598,
        "beats_top_598": bool(score_us < TOP_SCORE_US_598),
        "settings": {
            "num_iterations_per_benchmark": args.num_iterations_per_benchmark,
            "warmup_max_repeats": args.warmup_max_repeats,
            "warmup_max_time_ns": args.warmup_max_time_ns,
            "bench_max_repeats": args.bench_max_repeats,
            "bench_max_time_ns": args.bench_max_time_ns,
            "clear_l2": args.clear_l2,
            "lock_gpu_clocks": args.lock_gpu_clocks,
            "sm_clock_mhz": args.sm_clock_mhz,
            "mem_clock_mhz": args.mem_clock_mhz,
            "require_idle_gpu": args.require_idle_gpu,
            "kill_foreign_gpu_jobs": args.kill_foreign_gpu_jobs,
            "isolation_owner_pid": args.isolation_owner_pid,
            "isolation_settle_seconds": args.isolation_settle_seconds,
            "isolation_allow_cmd_substring": args.isolation_allow_cmd_substring,
        },
        "cases": cases,
    }
    if isolation_preflight is not None:
        payload["isolation_preflight"] = isolation_preflight

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"submission_file={payload['submission_file']}")
        print(f"score_us={payload['score_us']:.6f} (seconds={payload['score_seconds']:.12f})")
        print(
            "top_598_us="
            f"{payload['top_score_us_598']:.6f} delta_us={payload['delta_vs_top_us']:+.6f} "
            f"beats_top={payload['beats_top_598']}"
        )
        for case in cases:
            print(
                f"{case['name']}: m={case['m']} n={case['n']} k={case['k']} "
                f"mean={case['mean_us']:.6f}us runs={case['runs']} "
                f"err={case['err_ns'] / 1000.0:.6f}us"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
