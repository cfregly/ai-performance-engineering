"""Local leaderboard-style evaluator for NVFP4 dual GEMM (leaderboard 598).

Semantics intentionally mirror the official benchmark flow:
- correctness check on a batch of distinct inputs per case
- timing measures one repeat as N distinct `custom_kernel(data_i)` calls
- reported per-call latency is repeat_time / N
"""

from __future__ import annotations

if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    import os
    import sys

    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "labs.nvfp4_dual_gemm.local_eval",
            *sys.argv[1:],
        ],
    )

import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from core.harness.benchmark_harness import lock_gpu_clocks
from core.harness.l2_cache_utils import create_l2_flush_buffer, flush_l2_cache
from labs.nvfp4_dual_gemm.gpu_isolation import ensure_gpu_isolation
from labs.nvfp4_dual_gemm.local_eval_loader import (
    load_reference_module,
    load_submission_module,
    load_utils_module,
)


BENCHMARKS = (
    {"name": "case0", "m": 256, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
    {"name": "case1", "m": 512, "n": 4096, "k": 7168, "l": 1, "seed": 1111},
    {"name": "case2", "m": 256, "n": 3072, "k": 4096, "l": 1, "seed": 1111},
    {"name": "case3", "m": 512, "n": 3072, "k": 7168, "l": 1, "seed": 1111},
)

# Queried from GPUMODE/kernelbot-data on 2026-02-28.
TOP_SCORE_SECONDS_598 = 1.2913403524642259e-05
TOP_SCORE_US_598 = TOP_SCORE_SECONDS_598 * 1e6


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


def _case_args(case: dict[str, int | str], *, seed: int) -> dict[str, int]:
    return {
        "m": int(case["m"]),
        "n": int(case["n"]),
        "k": int(case["k"]),
        "l": int(case["l"]),
        "seed": int(seed),
    }


def _build_data_batch(reference_mod: Any, case: dict[str, int | str], *, count: int) -> list[Any]:
    base_seed = int(case["seed"])
    return [reference_mod.generate_input(**_case_args(case, seed=base_seed + 42 * i)) for i in range(int(count))]


def _run_case(
    submission_mod: Any,
    reference_mod: Any,
    case: dict[str, int | str],
    *,
    warmup: int,
    repeats: int,
    inputs_per_repeat: int,
    flush_l2: bool,
    flush_buffer: torch.Tensor | None,
    verify: bool,
) -> dict[str, Any]:
    data_batch = _build_data_batch(reference_mod, case, count=inputs_per_repeat)

    if verify:
        for data in data_batch:
            got = submission_mod.custom_kernel(_clone_tree(data))
            ok, msg = reference_mod.check_implementation(data, got)
            if not ok:
                raise RuntimeError(f"{case['name']} verify failed: {msg}")
        torch.cuda.synchronize()

    for _ in range(max(0, int(warmup))):
        for data in data_batch:
            submission_mod.custom_kernel(data)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    samples_us: list[float] = []
    for _ in range(int(repeats)):
        if flush_l2 and flush_buffer is not None:
            flush_l2_cache(buffer=flush_buffer)

        start.record(current_stream)
        for data in data_batch:
            submission_mod.custom_kernel(data)
        end.record(current_stream)
        end.synchronize()

        per_call_us = (start.elapsed_time(end) * 1000.0) / float(inputs_per_repeat)
        samples_us.append(float(per_call_us))

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
        "repeats": int(repeats),
        "warmup": int(warmup),
        "inputs_per_repeat": int(inputs_per_repeat),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-file",
        type=Path,
        default=Path("labs/nvfp4_dual_gemm/optimized_submission.py"),
        help="Path to candidate submission file.",
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=Path("labs/nvfp4_dual_gemm/reference_submission.py"),
        help="Path to reference implementation file.",
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=16)
    parser.add_argument("--inputs-per-repeat", type=int, default=50)
    parser.add_argument("--verify", action="store_true", default=True)
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--flush-l2", action="store_true", default=True)
    parser.add_argument("--no-flush-l2", dest="flush_l2", action="store_false")
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
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    if args.warmup < 0 or args.repeats <= 0 or args.inputs_per_repeat <= 0:
        raise ValueError("--warmup >= 0, --repeats > 0, --inputs-per-repeat > 0 required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nvfp4_dual_gemm local eval")
    if not args.submission_file.exists():
        raise FileNotFoundError(f"Submission file not found: {args.submission_file}")
    if not args.reference_file.exists():
        raise FileNotFoundError(f"Reference file not found: {args.reference_file}")

    isolation_preflight = None
    if args.require_idle_gpu or args.kill_foreign_gpu_jobs:
        isolation_preflight = ensure_gpu_isolation(
            owner_pid=args.isolation_owner_pid,
            kill_foreign=args.kill_foreign_gpu_jobs,
            require_idle=args.require_idle_gpu,
            settle_seconds=args.isolation_settle_seconds,
            context="local_eval_preflight",
            allow_cmd_substrings=args.isolation_allow_cmd_substring,
        )

    utils_mod = load_utils_module(args.submission_file.parent / "utils.py", "nvfp4_dual_utils_for_local_eval")
    reference_mod = load_reference_module(
        args.reference_file,
        module_name="nvfp4_dual_reference_for_local_eval",
        utils_module=utils_mod,
    )
    submission_mod = load_submission_module(
        args.submission_file,
        module_name="nvfp4_dual_submission_for_local_eval",
        reference_module=reference_mod,
        utils_module=utils_mod,
    )

    if not callable(getattr(submission_mod, "custom_kernel", None)):
        raise RuntimeError(f"{args.submission_file} does not define callable custom_kernel(data)")
    if not callable(getattr(reference_mod, "generate_input", None)):
        raise RuntimeError(f"{args.reference_file} does not define callable generate_input(...)")
    if not callable(getattr(reference_mod, "check_implementation", None)):
        raise RuntimeError(f"{args.reference_file} does not define callable check_implementation(data, output)")

    flush_buffer = create_l2_flush_buffer() if args.flush_l2 else None
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
                flush_l2=args.flush_l2,
                flush_buffer=flush_buffer,
                verify=args.verify,
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
            "warmup": args.warmup,
            "repeats": args.repeats,
            "inputs_per_repeat": args.inputs_per_repeat,
            "verify": args.verify,
            "flush_l2": args.flush_l2,
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
                f"{case['name']}: m={case['m']} n={case['n']} k={case['k']} l={case['l']} "
                f"mean={case['mean_us']:.6f}us p50={case['p50_us']:.6f}us p99={case['p99_us']:.6f}us "
                f"stdev={case['stdev_us']:.6f}us"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
