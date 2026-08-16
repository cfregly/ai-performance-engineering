"""Run challenge-595 (`nvfp4_gemv`) with exact official leaderboard harness semantics.

This wrapper stages your local submission and executes upstream
`problems/nvidia/eval.py` in `leaderboard` mode, then parses the emitted
`benchmark.*` records into a compact score report.
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
            "labs.nvfp4_gemv.local_eval",
            *sys.argv[1:],
        ],
    )

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

from core.harness.benchmark_harness import lock_gpu_clocks


REPO_ROOT = Path(__file__).resolve().parents[2]


BENCHMARKS = (
    {"name": "case0", "m": 7168, "k": 16384, "l": 1, "seed": 1111},
    {"name": "case1", "m": 4096, "k": 7168, "l": 8, "seed": 1111},
    {"name": "case2", "m": 7168, "k": 2048, "l": 4, "seed": 1111},
)


@contextmanager
def _null_ctx():
    yield None


def _format_case_line(case: dict[str, int | str]) -> str:
    return (
        f"m: {int(case['m'])}; "
        f"k: {int(case['k'])}; "
        f"l: {int(case['l'])}; "
        f"seed: {int(case['seed'])}"
    )


def _stage_official_files(
    *,
    work_dir: Path,
    submission_file: Path,
    official_root: Path,
) -> Path:
    run_dir = work_dir / "labs" / "nvfp4_gemv"
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_src = official_root / "eval.py"
    utils_src = official_root / "utils.py"
    task_src = official_root / "nvfp4_gemv" / "task.py"
    ref_src = official_root / "nvfp4_gemv" / "reference.py"

    for p in (eval_src, utils_src, task_src, ref_src, submission_file):
        if not p.exists():
            raise FileNotFoundError(f"Required file missing: {p}")

    shutil.copy2(eval_src, run_dir / "eval.py")
    shutil.copy2(utils_src, run_dir / "utils.py")
    shutil.copy2(task_src, run_dir / "task.py")
    shutil.copy2(ref_src, run_dir / "reference.py")
    shutil.copy2(submission_file, run_dir / "submission.py")

    # Preserve parent-relative lab imports used by candidate submissions
    # (e.g. Path(__file__).parents[1] / "nvfp4_gemm" / ...).
    src_dep_dir = REPO_ROOT / "labs" / "nvfp4_gemm"
    dst_dep_dir = work_dir / "labs" / "nvfp4_gemm"
    if src_dep_dir.exists():
        dst_dep_dir.mkdir(parents=True, exist_ok=True)
        for src in src_dep_dir.glob("*.py"):
            shutil.copy2(src, dst_dep_dir / src.name)

    cases_file = run_dir / "benchmarks.txt"
    cases_file.write_text(
        "\n".join(_format_case_line(case) for case in BENCHMARKS) + "\n",
        encoding="utf-8",
    )
    return run_dir


def _parse_log(log_text: str) -> dict[str, Any]:
    kv_re = re.compile(r"^([^:]+):\s*(.*)$")
    parsed: dict[str, Any] = {}
    benchmarks: dict[int, dict[str, Any]] = {}

    for raw in log_text.splitlines():
        m = kv_re.match(raw.strip())
        if not m:
            continue
        key, value = m.group(1), m.group(2)
        parsed[key] = value
        b = re.match(r"^benchmark\.(\d+)\.(.+)$", key)
        if b:
            idx = int(b.group(1))
            field = b.group(2)
            benchmarks.setdefault(idx, {})[field] = value

    rows: list[dict[str, Any]] = []
    for idx in sorted(benchmarks.keys()):
        row = benchmarks[idx]
        mean_ns = float(row.get("mean", "nan"))
        runs = int(float(row.get("runs", "0"))) if "runs" in row else None
        rows.append(
            {
                "index": idx,
                "name": BENCHMARKS[idx]["name"] if idx < len(BENCHMARKS) else f"case{idx}",
                "spec": row.get("spec"),
                "runs": runs,
                "mean_ns": mean_ns,
                "mean_us": mean_ns / 1e3,
                "std_ns": float(row.get("std", "nan")) if "std" in row else None,
                "err_ns": float(row.get("err", "nan")) if "err" in row else None,
                "best_ns": float(row.get("best", "nan")) if "best" in row else None,
                "worst_ns": float(row.get("worst", "nan")) if "worst" in row else None,
            }
        )

    means_ns = [x["mean_ns"] for x in rows]
    geom_ns = float("nan")
    if means_ns and all(math.isfinite(v) and v > 0 for v in means_ns):
        geom_ns = float(math.exp(sum(math.log(v) for v in means_ns) / len(means_ns)))

    return {
        "raw": parsed,
        "rows": rows,
        "geom_ns": geom_ns,
        "geom_us": geom_ns / 1e3 if math.isfinite(geom_ns) else float("nan"),
        "geom_seconds": geom_ns / 1e9 if math.isfinite(geom_ns) else float("nan"),
        "check": parsed.get("check"),
    }


def _run_official_eval(
    *,
    work_dir: Path,
    seed: int | None,
) -> tuple[int, str]:
    read_fd, write_fd = os.pipe()
    env = os.environ.copy()
    env["POPCORN_FD"] = str(write_fd)
    if seed is not None:
        env["POPCORN_SEED"] = str(int(seed))

    cmd = [sys.executable, "eval.py", "leaderboard", "benchmarks.txt"]
    proc = subprocess.Popen(
        cmd,
        cwd=str(work_dir),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        pass_fds=(write_fd,),
    )
    os.close(write_fd)
    with os.fdopen(read_fd, "r", encoding="utf-8", errors="replace") as f:
        popcorn_log = f.read()
    stdout, stderr = proc.communicate()

    if stdout:
        popcorn_log = popcorn_log + "\n[eval.stdout]\n" + stdout
    if stderr:
        popcorn_log = popcorn_log + "\n[eval.stderr]\n" + stderr
    return proc.returncode, popcorn_log


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-file",
        type=Path,
        default=Path("labs/nvfp4_gemv/optimized_submission.py"),
        help="Submission file to benchmark (staged as submission.py).",
    )
    parser.add_argument(
        "--official-root",
        type=Path,
        default=Path("/tmp/reference-kernels-gpumode/problems/nvidia"),
        help="Path to upstream reference-kernels problems/nvidia root.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional POPCORN_SEED override.")
    parser.add_argument("--lock-gpu-clocks", action="store_true", default=True)
    parser.add_argument("--no-lock-gpu-clocks", dest="lock_gpu_clocks", action="store_false")
    parser.add_argument("--sm-clock-mhz", type=int, default=1500)
    parser.add_argument("--mem-clock-mhz", type=int, default=None)
    parser.add_argument("--keep-workdir", action="store_true", help="Keep staged temp dir for debugging.")
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for nvfp4_gemv eval")
    if not args.submission_file.exists():
        raise FileNotFoundError(f"Submission file not found: {args.submission_file}")

    if args.keep_workdir:
        work_dir = Path(tempfile.mkdtemp(prefix="nvfp4_gemv_official_eval_"))
        cleanup = False
    else:
        tmp = tempfile.TemporaryDirectory(prefix="nvfp4_gemv_official_eval_")
        work_dir = Path(tmp.name)
        cleanup = True

    try:
        run_dir = _stage_official_files(
            work_dir=work_dir,
            submission_file=args.submission_file.resolve(),
            official_root=args.official_root.resolve(),
        )

        lock_ctx = (
            lock_gpu_clocks(device=0, sm_clock_mhz=args.sm_clock_mhz, mem_clock_mhz=args.mem_clock_mhz)
            if args.lock_gpu_clocks
            else _null_ctx()
        )
        with lock_ctx:
            returncode, log_text = _run_official_eval(work_dir=run_dir, seed=args.seed)

        parsed = _parse_log(log_text)
        payload = {
            "submission_file": str(args.submission_file),
            "official_root": str(args.official_root),
            "work_dir": str(work_dir),
            "returncode": int(returncode),
            "check": parsed["check"],
            "score_seconds": parsed["geom_seconds"],
            "score_us": parsed["geom_us"],
            "settings": {
                "seed": args.seed,
                "lock_gpu_clocks": args.lock_gpu_clocks,
                "sm_clock_mhz": args.sm_clock_mhz,
                "mem_clock_mhz": args.mem_clock_mhz,
            },
            "cases": parsed["rows"],
        }

        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(f"submission_file={payload['submission_file']}")
            print(f"returncode={payload['returncode']} check={payload['check']}")
            print(f"score_us={payload['score_us']:.6f} (seconds={payload['score_seconds']:.12f})")
            for row in payload["cases"]:
                best_us = row["best_ns"] / 1e3 if row["best_ns"] is not None else float("nan")
                worst_us = row["worst_ns"] / 1e3 if row["worst_ns"] is not None else float("nan")
                err_us = row["err_ns"] / 1e3 if row["err_ns"] is not None else float("nan")
                print(
                    f"{row['name']}: runs={row['runs']} mean={row['mean_us']:.6f}us "
                    f"best={best_us:.6f}us worst={worst_us:.6f}us "
                    f"err={err_us:.6f}us"
                )

            # Keep full evaluator logs visible for debugging and parity checks.
            print("\n--- official-eval-log ---")
            print(log_text.rstrip())

        if returncode != 0 or payload["check"] != "pass":
            return 112
        return 0
    finally:
        if cleanup:
            tmp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
