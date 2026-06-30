#!/usr/bin/env python3
"""Strict isolated A/B validation with promotion report output.

This runner enforces GPU isolation before every A/B sample and emits a single
JSON report suitable for promotion decisions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any

from gpu_isolation import ensure_gpu_isolation


TOP_SCORE_US_598 = 12.91340352464226


def _extract_json_blob(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("No JSON object found in local_eval output")
    return json.loads(text[start : end + 1])


def _summarize_deltas(deltas: list[float]) -> tuple[float, float, float]:
    count = len(deltas)
    if count == 0:
        raise ValueError("Cannot summarize an empty delta set")

    total = 0.0
    total_squares = 0.0
    for delta in deltas:
        total += delta
        total_squares += delta * delta

    mean = total / count
    deltas.sort()
    midpoint = count // 2
    if count % 2:
        median = deltas[midpoint]
    else:
        median = (deltas[midpoint - 1] + deltas[midpoint]) / 2.0

    if count > 1:
        variance = max(0.0, (total_squares - (total * total / count)) / (count - 1))
        stdev = math.sqrt(variance)
    else:
        stdev = 0.0
    return float(mean), float(median), float(stdev)


def _summarize_pair_rows(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    deltas: list[float] = []
    baseline_log_total = 0.0
    candidate_log_total = 0.0
    wins_candidate = 0
    wins_baseline = 0

    for row in pair_rows:
        baseline_score = float(row["baseline_score_us"])
        candidate_score = float(row["candidate_score_us"])
        delta = float(row["delta_candidate_minus_baseline_us"])
        deltas.append(delta)
        baseline_log_total += math.log(baseline_score)
        candidate_log_total += math.log(candidate_score)
        if delta < 0.0:
            wins_candidate += 1
        elif delta > 0.0:
            wins_baseline += 1

    mean_delta, median_delta, stdev_delta = _summarize_deltas(deltas)
    pair_count = len(deltas)
    ties = pair_count - wins_candidate - wins_baseline
    baseline_geo = float(math.exp(baseline_log_total / pair_count))
    candidate_geo = float(math.exp(candidate_log_total / pair_count))
    promote_candidate = bool(
        mean_delta < 0.0 and wins_candidate > wins_baseline and candidate_geo < baseline_geo
    )

    return {
        "pairs": pair_count,
        "baseline_geomean_us": baseline_geo,
        "candidate_geomean_us": candidate_geo,
        "delta_geomean_candidate_minus_baseline_us": candidate_geo - baseline_geo,
        "delta_mean_candidate_minus_baseline_us": mean_delta,
        "delta_median_candidate_minus_baseline_us": median_delta,
        "delta_stdev_candidate_minus_baseline_us": stdev_delta,
        "wins_candidate": wins_candidate,
        "wins_baseline": wins_baseline,
        "ties": ties,
        "promote_candidate": promote_candidate,
        "delta_candidate_vs_top598_us": candidate_geo - TOP_SCORE_US_598,
        "delta_baseline_vs_top598_us": baseline_geo - TOP_SCORE_US_598,
    }


def _run_local_eval(
    *,
    repo_root: Path,
    local_eval_script: Path,
    submission_file: Path,
    reference_file: Path,
    warmup: int,
    repeats: int,
    inputs_per_repeat: int,
    owner_pid: int,
    require_idle_gpu: bool,
    kill_foreign_gpu_jobs: bool,
    isolation_settle_seconds: float,
    isolation_allow_cmd_substring: list[str],
    torch_extensions_dir: str | None,
    timeout_seconds: int,
) -> dict[str, Any]:
    cmd = [
        "python",
        "-u",
        str(local_eval_script),
        "--submission-file",
        str(submission_file),
        "--reference-file",
        str(reference_file),
        "--warmup",
        str(int(warmup)),
        "--repeats",
        str(int(repeats)),
        "--inputs-per-repeat",
        str(int(inputs_per_repeat)),
        "--json",
        "--isolation-owner-pid",
        str(int(owner_pid)),
        "--isolation-settle-seconds",
        str(float(isolation_settle_seconds)),
    ]
    if require_idle_gpu:
        cmd.append("--require-idle-gpu")
    if kill_foreign_gpu_jobs:
        cmd.append("--kill-foreign-gpu-jobs")
    for allow in isolation_allow_cmd_substring:
        cmd += ["--isolation-allow-cmd-substring", str(allow)]

    t0 = time.perf_counter()
    env = os.environ.copy()
    if torch_extensions_dir:
        env["TORCH_EXTENSIONS_DIR"] = str(torch_extensions_dir)
    proc = subprocess.run(
        cmd,
        cwd=repo_root,
        capture_output=True,
        text=True,
        env=env,
        timeout=int(timeout_seconds),
        check=False,
    )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise RuntimeError(
            f"local_eval failed rc={proc.returncode} for {submission_file}\n"
            f"stdout_tail:\n{proc.stdout[-2000:]}\n"
            f"stderr_tail:\n{proc.stderr[-2000:]}"
        )
    payload = _extract_json_blob(proc.stdout)
    payload["wall_seconds"] = elapsed
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-file",
        type=Path,
        default=Path("labs/nvfp4_dual_gemm/cand_stage64_7_stage128_5_cacheA_first.py"),
    )
    parser.add_argument(
        "--candidate-file",
        type=Path,
        default=Path("labs/nvfp4_dual_gemm/optimized_submission.py"),
    )
    parser.add_argument(
        "--reference-file",
        type=Path,
        default=Path("labs/nvfp4_dual_gemm/reference_submission.py"),
    )
    parser.add_argument(
        "--local-eval-script",
        type=Path,
        default=Path("labs/nvfp4_dual_gemm/local_eval.py"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("labs/nvfp4_dual_gemm/promotion_report_strict_ab.json"),
    )
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--inputs-per-repeat", type=int, default=20)
    parser.add_argument("--timeout-seconds-per-run", type=int, default=240)
    parser.add_argument("--run-window-seconds", type=int, default=1800)
    parser.add_argument("--owner-pid", type=int, default=None)
    parser.add_argument("--require-idle-gpu", action="store_true", default=True)
    parser.add_argument("--allow-foreign-gpu-jobs", dest="require_idle_gpu", action="store_false")
    parser.add_argument("--kill-foreign-gpu-jobs", action="store_true", default=True)
    parser.add_argument("--no-kill-foreign-gpu-jobs", dest="kill_foreign_gpu_jobs", action="store_false")
    parser.add_argument("--isolation-settle-seconds", type=float, default=1.0)
    parser.add_argument(
        "--isolation-allow-cmd-substring",
        action="append",
        default=["python -m mcp.mcp_server --serve"],
    )
    parser.add_argument("--torch-extensions-dir", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    owner_pid = int(args.owner_pid) if args.owner_pid is not None else int(os.getpid())

    report: dict[str, Any] = {
        "status": "running",
        "started_at_unix": time.time(),
        "owner_pid": owner_pid,
        "paths": {
            "repo_root": str(repo_root),
            "baseline_file": str(args.baseline_file),
            "candidate_file": str(args.candidate_file),
            "reference_file": str(args.reference_file),
            "local_eval_script": str(args.local_eval_script),
            "output": str(args.output),
        },
        "settings": {
            "pairs": int(args.pairs),
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
            "inputs_per_repeat": int(args.inputs_per_repeat),
            "timeout_seconds_per_run": int(args.timeout_seconds_per_run),
            "run_window_seconds": int(args.run_window_seconds),
            "require_idle_gpu": bool(args.require_idle_gpu),
            "kill_foreign_gpu_jobs": bool(args.kill_foreign_gpu_jobs),
            "isolation_settle_seconds": float(args.isolation_settle_seconds),
            "isolation_allow_cmd_substring": list(args.isolation_allow_cmd_substring),
            "torch_extensions_dir": args.torch_extensions_dir,
        },
        "pair_rows": [],
        "top_score_us_598": TOP_SCORE_US_598,
    }

    start = time.perf_counter()
    try:
        # Global preflight before any timed run.
        report["initial_isolation_preflight"] = ensure_gpu_isolation(
            owner_pid=owner_pid,
            kill_foreign=bool(args.kill_foreign_gpu_jobs),
            require_idle=bool(args.require_idle_gpu),
            settle_seconds=float(args.isolation_settle_seconds),
            context="strict_ab_initial",
            allow_cmd_substrings=list(args.isolation_allow_cmd_substring),
        )

        for i in range(1, int(args.pairs) + 1):
            elapsed_window = time.perf_counter() - start
            if elapsed_window > float(args.run_window_seconds):
                raise RuntimeError(
                    f"run window exceeded before pair {i}: "
                    f"{elapsed_window:.2f}s > {args.run_window_seconds}s"
                )

            pre_a = ensure_gpu_isolation(
                owner_pid=owner_pid,
                kill_foreign=bool(args.kill_foreign_gpu_jobs),
                require_idle=bool(args.require_idle_gpu),
                settle_seconds=float(args.isolation_settle_seconds),
                context=f"pair{i}_pre_A",
                allow_cmd_substrings=list(args.isolation_allow_cmd_substring),
            )
            run_a = _run_local_eval(
                repo_root=repo_root,
                local_eval_script=args.local_eval_script,
                submission_file=args.baseline_file,
                reference_file=args.reference_file,
                warmup=args.warmup,
                repeats=args.repeats,
                inputs_per_repeat=args.inputs_per_repeat,
                owner_pid=owner_pid,
                require_idle_gpu=bool(args.require_idle_gpu),
                kill_foreign_gpu_jobs=bool(args.kill_foreign_gpu_jobs),
                isolation_settle_seconds=float(args.isolation_settle_seconds),
                isolation_allow_cmd_substring=list(args.isolation_allow_cmd_substring),
                torch_extensions_dir=args.torch_extensions_dir,
                timeout_seconds=int(args.timeout_seconds_per_run),
            )
            post_a = ensure_gpu_isolation(
                owner_pid=owner_pid,
                kill_foreign=bool(args.kill_foreign_gpu_jobs),
                require_idle=bool(args.require_idle_gpu),
                settle_seconds=float(args.isolation_settle_seconds),
                context=f"pair{i}_post_A",
                allow_cmd_substrings=list(args.isolation_allow_cmd_substring),
            )

            pre_b = ensure_gpu_isolation(
                owner_pid=owner_pid,
                kill_foreign=bool(args.kill_foreign_gpu_jobs),
                require_idle=bool(args.require_idle_gpu),
                settle_seconds=float(args.isolation_settle_seconds),
                context=f"pair{i}_pre_B",
                allow_cmd_substrings=list(args.isolation_allow_cmd_substring),
            )
            run_b = _run_local_eval(
                repo_root=repo_root,
                local_eval_script=args.local_eval_script,
                submission_file=args.candidate_file,
                reference_file=args.reference_file,
                warmup=args.warmup,
                repeats=args.repeats,
                inputs_per_repeat=args.inputs_per_repeat,
                owner_pid=owner_pid,
                require_idle_gpu=bool(args.require_idle_gpu),
                kill_foreign_gpu_jobs=bool(args.kill_foreign_gpu_jobs),
                isolation_settle_seconds=float(args.isolation_settle_seconds),
                isolation_allow_cmd_substring=list(args.isolation_allow_cmd_substring),
                torch_extensions_dir=args.torch_extensions_dir,
                timeout_seconds=int(args.timeout_seconds_per_run),
            )
            post_b = ensure_gpu_isolation(
                owner_pid=owner_pid,
                kill_foreign=bool(args.kill_foreign_gpu_jobs),
                require_idle=bool(args.require_idle_gpu),
                settle_seconds=float(args.isolation_settle_seconds),
                context=f"pair{i}_post_B",
                allow_cmd_substrings=list(args.isolation_allow_cmd_substring),
            )

            delta = float(run_b["score_us"]) - float(run_a["score_us"])
            report["pair_rows"].append(
                {
                    "pair": i,
                    "baseline_score_us": float(run_a["score_us"]),
                    "candidate_score_us": float(run_b["score_us"]),
                    "delta_candidate_minus_baseline_us": delta,
                    "baseline_wall_seconds": float(run_a["wall_seconds"]),
                    "candidate_wall_seconds": float(run_b["wall_seconds"]),
                    "pre_a": pre_a,
                    "post_a": post_a,
                    "pre_b": pre_b,
                    "post_b": post_b,
                }
            )

        report["summary"] = _summarize_pair_rows(report["pair_rows"])
        report["status"] = "ok"
    except Exception as exc:  # noqa: BLE001
        report["status"] = "error"
        report["error"] = str(exc)
    finally:
        report["finished_at_unix"] = time.time()
        report["total_wall_seconds"] = float(report["finished_at_unix"] - report["started_at_unix"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
        print(f"wrote {args.output}")

    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
