"""Case0-only official-semantics sweep for nvfp4_gemm submissions.

Runs `local_eval_official597.py --cases case0` with different case0 kernel variants
and reports mean-us ranking for fast structural iteration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


LAB_DIR = Path(__file__).resolve().parent
EVAL = LAB_DIR / "local_eval_official597.py"
DEFAULT_SUBMISSION = LAB_DIR / "optimized_submission.py"

CASE0_VARIANTS = (
    "v4_default",
    "v4_n64_s1",
    "v4_n64_split2",
    "v4_n128_s1",
    "v4_m64_n128_split2_s8",
    "v4_m64_n128_split1_s8",
    "v4_m64_n64_split2_s8",
    "v4_m64_n64_split1_s8",
    "v4_m64_n128_bk512_s3",
    "v3b",
)


def _run_variant(
    *,
    submission_file: Path,
    variant: str,
    sm_clock_mhz: int,
    mem_clock_mhz: int | None,
) -> dict:
    env = os.environ.copy()
    env["AISP_NVFP4_CASE0_VARIANT"] = variant

    cmd = [
        sys.executable,
        str(EVAL),
        "--submission-file",
        str(submission_file),
        "--cases",
        "case0",
        "--sm-clock-mhz",
        str(sm_clock_mhz),
    ]
    json_out = Path(f"/tmp/nvfp4_case0_eval_{variant}_{int(time.time()*1e6)}.json")
    cmd.extend(["--json-out", str(json_out)])
    if mem_clock_mhz is not None:
        cmd.extend(["--mem-clock-mhz", str(mem_clock_mhz)])

    proc = subprocess.run(
        cmd,
        cwd=LAB_DIR.parent.parent,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "variant": variant,
            "ok": False,
            "error": proc.stderr.strip() or proc.stdout.strip() or f"rc={proc.returncode}",
        }

    if not json_out.exists():
        return {
            "variant": variant,
            "ok": False,
            "error": f"missing json output: {json_out}",
        }
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    case0 = payload["cases"][0]
    return {
        "variant": variant,
        "ok": True,
        "mean_us": float(case0["mean_us"]),
        "runs": int(case0["runs"]),
        "stop_reason": case0["stop_reason"],
        "file": payload["submission_file"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--submission-file",
        type=Path,
        default=DEFAULT_SUBMISSION,
    )
    parser.add_argument(
        "--variants",
        type=str,
        default=",".join(CASE0_VARIANTS),
        help="Comma-separated case0 variants",
    )
    parser.add_argument("--sm-clock-mhz", type=int, default=1500)
    parser.add_argument("--mem-clock-mhz", type=int, default=None)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not args.submission_file.exists():
        raise FileNotFoundError(f"Submission file not found: {args.submission_file}")

    variants = [token.strip() for token in args.variants.split(",") if token.strip()]
    if not variants:
        raise ValueError("No variants provided")

    rows = []
    ok_rows = []
    failed = []
    for variant in variants:
        row = _run_variant(
            submission_file=args.submission_file.resolve(),
            variant=variant,
            sm_clock_mhz=args.sm_clock_mhz,
            mem_clock_mhz=args.mem_clock_mhz,
        )
        rows.append(row)
        if row.get("ok"):
            ok_rows.append(row)
        else:
            failed.append(row)

    ok_rows.sort(key=lambda r: r["mean_us"])

    out = {
        "timestamp": int(time.time()),
        "submission_file": str(args.submission_file.resolve()),
        "variants": rows,
        "ranked_ok": ok_rows,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("Case0 official-semantics sweep")
    for i, row in enumerate(ok_rows, start=1):
        print(
            f"{i}. {row['variant']}: {row['mean_us']:.6f} us "
            f"(runs={row['runs']} stop={row['stop_reason']})"
        )
    if failed:
        print("failed variants:")
        for row in failed:
            print(f"- {row['variant']}: {row['error']}")
    if args.json_out is not None:
        print(f"wrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
