"""Official-semantics local evaluator for GPUMODE leaderboard 597 (nvfp4_gemm).

This mirrors `problems/nvidia/eval_better_bench.py` leaderboard-mode behavior:
- NUM_ITERATIONS_PER_BENCHMARK = 50
- warm up all benchmark shapes first (max_repeats=1000, max_time_ns=5e8)
- timed run with recheck=True (max_repeats=1000, max_time_ns=30e9)
- dynamic stop rule based on relative error / total measured kernel time / wallclock
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from core.harness.benchmark_harness import lock_gpu_clocks
from labs.nvfp4_gemm.local_eval_loader import (
    load_reference_module,
    load_submission_module,
    load_utils_module,
)

NUM_ITERATIONS_PER_BENCHMARK = 50
DEFAULT_WARMUP_MAX_REPEATS = 1000
DEFAULT_WARMUP_MAX_TIME_NS = 5e8
DEFAULT_LEADERBOARD_MAX_REPEATS = 1000
DEFAULT_LEADERBOARD_MAX_TIME_NS = 30e9

# Queried from https://www.gpumode.com/api/leaderboard/597 on 2026-02-28.
TOP_SCORE_SECONDS_597 = 9.981888843481874e-06

BENCHMARKS = (
    {"name": "case0", "m": 128, "n": 7168, "k": 16384, "l": 1, "seed": 1111},
    {"name": "case1", "m": 128, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
    {"name": "case2", "m": 128, "n": 7168, "k": 2048, "l": 1, "seed": 1111},
)

ALL_CASE_NAMES = tuple(case["name"] for case in BENCHMARKS)


@contextmanager
def _null_ctx():
    yield None


@dataclasses.dataclass
class Stats:
    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


def _calculate_stats(durations_ns: list[float]) -> Stats:
    runs = len(durations_ns)
    if runs <= 0:
        raise ValueError("durations_ns must be non-empty")

    total = float(sum(durations_ns))
    best = float(min(durations_ns))
    worst = float(max(durations_ns))
    avg = total / runs
    if runs == 1:
        std = 0.0
    else:
        variance = sum((x - avg) ** 2 for x in durations_ns)
        std = math.sqrt(variance / (runs - 1))
    err = std / math.sqrt(runs)
    return Stats(runs=runs, mean=avg, std=std, err=err, best=best, worst=worst)


def _clone_data(x: Any) -> Any:
    if isinstance(x, tuple):
        return tuple(_clone_data(v) for v in x)
    if isinstance(x, list):
        return [_clone_data(v) for v in x]
    if isinstance(x, dict):
        return {k: _clone_data(v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.clone()
    return x


_WORKER_SUBMISSION: Any | None = None
_WORKER_REFERENCE: Any | None = None
_WORKER_UTILS: Any | None = None


def _init_worker(submission_file: str, reference_file: str, utils_file: str) -> None:
    # Match official harness behavior for stable benchmarking.
    os.environ["CUTE_DSL_DISABLE_FILE_CACHING"] = "1"

    submission_path = Path(submission_file).resolve()
    reference_path = Path(reference_file).resolve()
    utils_path = Path(utils_file).resolve()

    global _WORKER_SUBMISSION
    global _WORKER_REFERENCE
    global _WORKER_UTILS
    _WORKER_UTILS = load_utils_module(utils_path, f"utils_worker_{os.getpid()}")
    _WORKER_REFERENCE = load_reference_module(
        reference_path,
        module_name=f"reference_worker_{os.getpid()}",
        utils_module=_WORKER_UTILS,
    )
    _WORKER_SUBMISSION = load_submission_module(
        submission_path,
        module_name=f"submission_worker_{os.getpid()}",
        reference_module=_WORKER_REFERENCE,
        utils_module=_WORKER_UTILS,
    )

    set_seed = getattr(_WORKER_UTILS, "set_seed", None)
    if callable(set_seed):
        set_seed(42)


def _worker_run_single_benchmark(
    test_args: dict[str, int | str],
    recheck: bool,
    max_repeats: int,
    max_time_ns: float,
) -> dict[str, Any]:
    if _WORKER_SUBMISSION is None or _WORKER_REFERENCE is None or _WORKER_UTILS is None:
        raise RuntimeError("Worker modules are not initialized")

    custom_kernel = getattr(_WORKER_SUBMISSION, "custom_kernel")
    generate_input = getattr(_WORKER_REFERENCE, "generate_input")
    check_implementation = getattr(_WORKER_REFERENCE, "check_implementation")
    clear_l2_cache_large = getattr(_WORKER_UTILS, "clear_l2_cache_large")

    durations: list[float] = []
    args = {
        "m": int(test_args["m"]),
        "n": int(test_args["n"]),
        "k": int(test_args["k"]),
        "l": int(test_args["l"]),
        "seed": int(test_args["seed"]),
    }
    data_list = []

    # Match official data generation behavior: advance seed each sample.
    for _ in range(NUM_ITERATIONS_PER_BENCHMARK):
        if "seed" in args:
            args["seed"] = int(args["seed"]) + 42
        data_list.append(generate_input(**args))

    check_copy = _clone_data(data_list)

    # One obligatory correctness check over the generated input set.
    outputs = []
    try:
        for data in data_list:
            outputs.append(custom_kernel(_clone_data(data)))
    except Exception as exc:
        return {"ok": False, "error": f"custom_kernel failed: {exc}"}
    for reference_output, custom_output in zip(check_copy, outputs):
        good, message = check_implementation(reference_output, custom_output)
        if not good:
            return {"ok": False, "error": f"correctness check failed: {message}"}

    bm_start_time = time.perf_counter_ns()
    stop_reason = "max_repeats"
    current_stream = torch.cuda.current_stream()
    for i in range(max_repeats):
        torch.cuda.synchronize()
        outputs = []
        clear_l2_cache_large()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record(current_stream)
        for data in data_list:
            outputs.append(custom_kernel(data))
        end_event.record(current_stream)
        end_event.synchronize()

        duration_ns = (start_event.elapsed_time(end_event) / NUM_ITERATIONS_PER_BENCHMARK) * 1e6
        durations.append(float(duration_ns))

        if recheck:
            for reference_output, custom_output in zip(check_copy, outputs):
                good, message = check_implementation(reference_output, custom_output)
                if not good:
                    return {"ok": False, "error": f"recheck failed: {message}"}

        total_bm_duration = time.perf_counter_ns() - bm_start_time
        if i > 1 and total_bm_duration > 1e8:
            stats = _calculate_stats(durations)
            if stats.err / stats.mean < 0.001:
                stop_reason = "relative_error"
                break
            if stats.mean * stats.runs > max_time_ns:
                stop_reason = "max_kernel_time_budget"
                break
            if total_bm_duration > 120e9:
                stop_reason = "wallclock_budget"
                break

    stats = _calculate_stats(durations)
    return {
        "ok": True,
        "stats": dataclasses.asdict(stats),
        "stop_reason": stop_reason,
    }


def _score_submission(
    *,
    submission_file: Path,
    reference_file: Path,
    utils_file: Path,
    benchmarks: list[dict[str, int | str]],
    warmup_max_repeats: int,
    warmup_max_time_ns: float,
    leaderboard_max_repeats: int,
    leaderboard_max_time_ns: float,
    lock_gpu_clocks_enabled: bool,
    sm_clock_mhz: int,
    mem_clock_mhz: int | None,
) -> dict[str, Any]:
    mp_ctx = multiprocessing.get_context("spawn")

    lock_ctx = (
        lock_gpu_clocks(device=0, sm_clock_mhz=sm_clock_mhz, mem_clock_mhz=mem_clock_mhz)
        if lock_gpu_clocks_enabled
        else _null_ctx()
    )

    cases: list[dict[str, Any]] = []
    with lock_ctx:
        with mp_ctx.Pool(
            1,
            initializer=_init_worker,
            initargs=(str(submission_file), str(reference_file), str(utils_file)),
        ) as pool:
            for test in benchmarks:
                warmup_result = pool.apply(
                    _worker_run_single_benchmark,
                    (dict(test), False, warmup_max_repeats, warmup_max_time_ns),
                )
                if not warmup_result.get("ok", False):
                    raise RuntimeError(f"Warmup failed for {test['name']}: {warmup_result.get('error')}")

            for test in benchmarks:
                result = pool.apply(
                    _worker_run_single_benchmark,
                    (dict(test), True, leaderboard_max_repeats, leaderboard_max_time_ns),
                )
                if not result.get("ok", False):
                    raise RuntimeError(f"Benchmark failed for {test['name']}: {result.get('error')}")

                stats = result["stats"]
                # Official stats.mean is in ns.
                mean_ns = float(stats["mean"])
                cases.append(
                    {
                        "name": test["name"],
                        "m": int(test["m"]),
                        "n": int(test["n"]),
                        "k": int(test["k"]),
                        "l": int(test["l"]),
                        "seed": int(test["seed"]),
                        "runs": int(stats["runs"]),
                        "mean_ns": mean_ns,
                        "mean_us": mean_ns / 1e3,
                        "std_ns": float(stats["std"]),
                        "err_ns": float(stats["err"]),
                        "best_ns": float(stats["best"]),
                        "worst_ns": float(stats["worst"]),
                        "stop_reason": result["stop_reason"],
                    }
                )

    score_ns = float(math.exp(sum(math.log(case["mean_ns"]) for case in cases) / len(cases)))
    score_us = score_ns / 1e3
    score_seconds = score_ns / 1e9

    selected_case_names = tuple(case["name"] for case in benchmarks)
    full_official_case_set = set(selected_case_names) == set(ALL_CASE_NAMES)

    return {
        "submission_file": str(submission_file),
        "reference_file": str(reference_file),
        "utils_file": str(utils_file),
        "selected_cases": list(selected_case_names),
        "score_ns": score_ns,
        "score_us": score_us,
        "score_seconds": score_seconds,
        "top_score_seconds_597": TOP_SCORE_SECONDS_597,
        "top_score_us_597": TOP_SCORE_SECONDS_597 * 1e6,
        "delta_vs_top_us": (score_us - (TOP_SCORE_SECONDS_597 * 1e6)) if full_official_case_set else None,
        "beats_top_597": bool(score_seconds < TOP_SCORE_SECONDS_597) if full_official_case_set else None,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-file",
        type=Path,
        default=Path("labs/nvfp4_gemm/optimized_submission.py"),
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=Path("labs/nvfp4_gemm/reference_submission.py"),
    )
    parser.add_argument(
        "--utils-file",
        type=Path,
        default=Path("labs/nvfp4_gemm/utils.py"),
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="all",
        help="Comma-separated case names to run (e.g. 'case0' or 'case0,case1'). Default: all",
    )
    parser.add_argument("--warmup-max-repeats", type=int, default=DEFAULT_WARMUP_MAX_REPEATS)
    parser.add_argument("--warmup-max-time-ns", type=float, default=DEFAULT_WARMUP_MAX_TIME_NS)
    parser.add_argument("--leaderboard-max-repeats", type=int, default=DEFAULT_LEADERBOARD_MAX_REPEATS)
    parser.add_argument("--leaderboard-max-time-ns", type=float, default=DEFAULT_LEADERBOARD_MAX_TIME_NS)
    parser.add_argument("--lock-gpu-clocks", action="store_true", default=True)
    parser.add_argument("--no-lock-gpu-clocks", dest="lock_gpu_clocks", action="store_false")
    parser.add_argument("--sm-clock-mhz", type=int, default=1500)
    parser.add_argument("--mem-clock-mhz", type=int, default=None)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for leaderboard evaluation")
    if not args.submission_file.exists():
        raise FileNotFoundError(f"Submission file not found: {args.submission_file}")
    if not args.reference_file.exists():
        raise FileNotFoundError(f"Reference file not found: {args.reference_file}")
    if not args.utils_file.exists():
        raise FileNotFoundError(f"Utils file not found: {args.utils_file}")

    if args.cases.strip().lower() == "all":
        selected_benchmarks = [dict(case) for case in BENCHMARKS]
    else:
        requested = [token.strip() for token in args.cases.split(",") if token.strip()]
        requested_set = set(requested)
        unknown = sorted(requested_set.difference(ALL_CASE_NAMES))
        if unknown:
            raise ValueError(f"Unknown case names in --cases: {unknown}; valid cases: {list(ALL_CASE_NAMES)}")
        selected_benchmarks = [dict(case) for case in BENCHMARKS if case["name"] in requested_set]
        if not selected_benchmarks:
            raise ValueError("--cases selected no benchmarks")

    payload = _score_submission(
        submission_file=args.submission_file.resolve(),
        reference_file=args.reference_file.resolve(),
        utils_file=args.utils_file.resolve(),
        benchmarks=selected_benchmarks,
        warmup_max_repeats=args.warmup_max_repeats,
        warmup_max_time_ns=args.warmup_max_time_ns,
        leaderboard_max_repeats=args.leaderboard_max_repeats,
        leaderboard_max_time_ns=args.leaderboard_max_time_ns,
        lock_gpu_clocks_enabled=args.lock_gpu_clocks,
        sm_clock_mhz=args.sm_clock_mhz,
        mem_clock_mhz=args.mem_clock_mhz,
    )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("NVFP4 GEMM official-semantics eval (leaderboard 597)")
        print(f"submission: {payload['submission_file']}")
        print(f"cases: {','.join(payload['selected_cases'])}")
        print(f"score: {payload['score_us']:.6f} us ({payload['score_seconds']:.12e} s)")
        if payload["delta_vs_top_us"] is not None:
            print(
                f"top: {payload['top_score_us_597']:.6f} us "
                f"({payload['top_score_seconds_597']:.12e} s)"
            )
            print(f"delta vs top: {payload['delta_vs_top_us']:+.6f} us")
            print(f"beats top: {payload['beats_top_597']}")
        else:
            print("top comparison: skipped (subset run)")
        for case in payload["cases"]:
            print(
                f"  {case['name']} m={case['m']} n={case['n']} k={case['k']} "
                f"mean={case['mean_us']:.6f} us runs={case['runs']} stop={case['stop_reason']}"
            )
        if args.json_out is not None:
            print(f"wrote: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
