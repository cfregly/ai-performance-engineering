"""Local leaderboard-style evaluator for GPUMODE nvfp4_gemm (leaderboard 597)."""

from __future__ import annotations

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from core.harness.benchmark_harness import lock_gpu_clocks
from labs.nvfp4_gemm import utils as nvfp4_utils
from labs.nvfp4_gemm.local_eval_loader import (
    load_reference_module,
    load_submission_module,
    load_utils_module,
)

# Queried from https://www.gpumode.com/api/leaderboard/597 on 2026-02-28.
TOP_SCORE_SECONDS_597 = 9.981888843481874e-06
TOP_SCORE_US_597 = TOP_SCORE_SECONDS_597 * 1e6

BENCHMARKS = (
    {"name": "case0", "m": 128, "n": 7168, "k": 16384, "l": 1, "seed": 1111},
    {"name": "case1", "m": 128, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
    {"name": "case2", "m": 128, "n": 7168, "k": 2048, "l": 1, "seed": 1111},
)


@contextmanager
def _null_ctx():
    yield None


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


def _build_data_batch(reference_mod: Any, case: dict[str, int | str], *, count: int) -> list[Any]:
    seed = int(case["seed"])
    out = []
    for _ in range(count):
        out.append(
            reference_mod.generate_input(
                m=int(case["m"]),
                n=int(case["n"]),
                k=int(case["k"]),
                l=int(case["l"]),
                seed=seed,
            )
        )
        seed += 42
    return out


def _run_case(
    submission_mod: Any,
    reference_mod: Any,
    case: dict[str, int | str],
    *,
    warmup: int,
    repeats: int,
    inputs_per_repeat: int,
    verify: bool,
    verify_count: int,
    flush_l2: bool,
) -> dict[str, Any]:
    data_batch = _build_data_batch(reference_mod, case, count=inputs_per_repeat)

    if verify:
        count = min(max(1, verify_count), len(data_batch))
        for idx in range(count):
            data = _clone_tree(data_batch[idx])
            got = submission_mod.custom_kernel(data)
            ok, msg = reference_mod.check_implementation(data, got)
            if not ok:
                raise RuntimeError(f"{case['name']} verify failed at sample {idx}: {msg}")
        torch.cuda.synchronize()

    for _ in range(max(0, warmup)):
        for data in data_batch:
            submission_mod.custom_kernel(data)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    samples_us: list[float] = []

    for _ in range(repeats):
        if flush_l2:
            nvfp4_utils.clear_l2_cache_large()
        start.record(current_stream)
        for data in data_batch:
            submission_mod.custom_kernel(data)
        end.record(current_stream)
        end.synchronize()
        repeat_us = float(start.elapsed_time(end) * 1000.0)
        samples_us.append(repeat_us / float(inputs_per_repeat))

    sample_count = len(samples_us)
    total_us = 0.0
    total_squares_us = 0.0
    for sample_us in samples_us:
        total_us += sample_us
        total_squares_us += sample_us * sample_us

    samples_us.sort()
    mean_us = float(total_us / sample_count)
    variance_us = max(0.0, (total_squares_us / sample_count) - mean_us * mean_us)
    stdev_us = float(math.sqrt(variance_us)) if sample_count > 1 else 0.0
    p50_us = float(samples_us[len(samples_us) // 2])
    p99_us = float(samples_us[max(0, int(len(samples_us) * 0.99) - 1)])

    return {
        "name": case["name"],
        "m": int(case["m"]),
        "n": int(case["n"]),
        "k": int(case["k"]),
        "l": int(case["l"]),
        "seed": int(case["seed"]),
        "mean_us": mean_us,
        "stdev_us": stdev_us,
        "p50_us": p50_us,
        "p99_us": p99_us,
        "min_us": float(samples_us[0]),
        "max_us": float(samples_us[-1]),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "inputs_per_repeat": int(inputs_per_repeat),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-file",
        type=Path,
        default=Path("labs/nvfp4_gemm/optimized_submission.py"),
        help="Path to candidate submission file.",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=Path("labs/nvfp4_gemm/reference_submission.py"),
        help="Path to reference implementation file.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--inputs-per-repeat", type=int, default=50)
    parser.add_argument("--verify", action="store_true", default=True)
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument(
        "--verify-count",
        type=int,
        default=6,
        help="Number of inputs to correctness-check per case when --verify is enabled.",
    )
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", dest="flush_l2", action="store_false")
    parser.add_argument("--lock-gpu-clocks", action="store_true", default=True)
    parser.add_argument("--no-lock-gpu-clocks", dest="lock_gpu_clocks", action="store_false")
    parser.add_argument("--sm-clock-mhz", type=int, default=1500)
    parser.add_argument("--mem-clock-mhz", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0 or args.inputs_per_repeat <= 0:
        raise ValueError("--warmup >= 0, --repeats > 0, and --inputs-per-repeat > 0 are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nvfp4_gemm local eval")
    if not args.submission_file.exists():
        raise FileNotFoundError(f"Submission file not found: {args.submission_file}")
    if not args.reference_file.exists():
        raise FileNotFoundError(f"Reference file not found: {args.reference_file}")

    utils_path = args.reference_file.parent.resolve() / "utils.py"
    utils_mod = load_utils_module(utils_path, "nvfp4_gemm_utils")
    reference_mod = load_reference_module(
        args.reference_file,
        module_name="nvfp4_gemm_reference",
        utils_module=utils_mod,
    )
    submission_mod = load_submission_module(
        args.submission_file,
        module_name="nvfp4_gemm_submission",
        reference_module=reference_mod,
        utils_module=utils_mod,
    )
    if not callable(getattr(submission_mod, "custom_kernel", None)):
        raise RuntimeError(f"{args.submission_file} must define callable custom_kernel(data)")
    if not callable(getattr(reference_mod, "generate_input", None)):
        raise RuntimeError(f"{args.reference_file} must define callable generate_input(m,n,k,l,seed)")
    if not callable(getattr(reference_mod, "check_implementation", None)):
        raise RuntimeError(f"{args.reference_file} must define callable check_implementation(data, output)")

    lock_ctx = (
        lock_gpu_clocks(device=0, sm_clock_mhz=args.sm_clock_mhz, mem_clock_mhz=args.mem_clock_mhz)
        if args.lock_gpu_clocks
        else _null_ctx()
    )

    with lock_ctx:
        cases = [
            _run_case(
                submission_mod,
                reference_mod,
                case,
                warmup=args.warmup,
                repeats=args.repeats,
                inputs_per_repeat=args.inputs_per_repeat,
                verify=args.verify,
                verify_count=args.verify_count,
                flush_l2=args.flush_l2,
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
        "top_score_seconds_597": TOP_SCORE_SECONDS_597,
        "top_score_us_597": TOP_SCORE_US_597,
        "delta_vs_top_us": score_us - TOP_SCORE_US_597,
        "beats_top_597": bool(score_us < TOP_SCORE_US_597),
        "settings": {
            "warmup": args.warmup,
            "repeats": args.repeats,
            "inputs_per_repeat": args.inputs_per_repeat,
            "verify": args.verify,
            "verify_count": args.verify_count,
            "flush_l2": args.flush_l2,
            "lock_gpu_clocks": args.lock_gpu_clocks,
            "sm_clock_mhz": args.sm_clock_mhz,
            "mem_clock_mhz": args.mem_clock_mhz,
        },
        "cases": cases,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("NVFP4 GEMM leaderboard-style local eval (597)")
        print(f"score: {score_us:.6f} us ({score_seconds:.12e} s)")
        print(
            "top: "
            f"{TOP_SCORE_US_597:.6f} us ({TOP_SCORE_SECONDS_597:.12e} s); "
            f"delta: {payload['delta_vs_top_us']:+.6f} us"
        )
        print(f"beats top: {payload['beats_top_597']}")
        for case in cases:
            print(
                f"  {case['name']} m={case['m']} n={case['n']} k={case['k']} "
                f"mean={case['mean_us']:.6f} us stdev={case['stdev_us']:.6f}"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
