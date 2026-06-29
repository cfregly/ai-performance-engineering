#!/usr/bin/env python3
"""Focused verify-green sweep for optimized_submission.py kernel parameters.

Sweeps:
- cache_A / cache_B eviction policy
- NUM_STAGES split for BLOCK_N==64 and BLOCK_N==128

Writes a JSON artifact with only verify-green (`status=ok`) candidates sorted by score.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gpu_isolation import ensure_gpu_isolation

ROOT = Path(__file__).resolve().parent
BASE_SUBMISSION = ROOT / "optimized_submission.py"
REFERENCE_SUBMISSION = ROOT / "reference_submission.py"
EVAL_SCRIPT = ROOT / "local_eval.py"
OUT_JSON = ROOT / "focused_kernel_sweep_verify_green_v2.json"


@dataclass(frozen=True)
class Candidate:
    cache_a: str
    cache_b: str
    stage64: int
    stage128: int

    @property
    def tag(self) -> str:
        return (
            f"cacheA_{self.cache_a.lower()}__cacheB_{self.cache_b.lower()}__"
            f"s64_{self.stage64}__s128_{self.stage128}"
        )


def _extract_json_blob(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("No JSON object found in eval output")
    return json.loads(text[start : end + 1])


def _cleanup_orphans() -> None:
    # Some candidate runs can leave detached multiprocessing workers alive.
    # Keep the device clean between candidates to avoid cross-run contamination.
    subprocess.run(
        ["bash", "-lc", "pkill -9 -f 'multiprocessing.spawn' || true"],
        check=False,
        capture_output=True,
        text=True,
    )


def _patch_source(src: str, cand: Candidate) -> str:
    out = re.sub(
        r"constexpr uint64_t cache_A = [^;]+;",
        f"constexpr uint64_t cache_A = {cand.cache_a};",
        src,
        count=1,
    )
    out = re.sub(
        r"constexpr uint64_t cache_B = [^;]+;",
        f"constexpr uint64_t cache_B = {cand.cache_b};",
        out,
        count=1,
    )
    out = re.sub(
        r"constexpr int NUM_STAGES = \(BLOCK_N == 64 \? \d+ : \d+\);",
        f"constexpr int NUM_STAGES = (BLOCK_N == 64 ? {cand.stage64} : {cand.stage128});",
        out,
        count=1,
    )
    return out


def _run_candidate(
    cand: Candidate,
    src: str,
    *,
    owner_pid: int,
    require_idle_gpu: bool,
    kill_foreign_gpu_jobs: bool,
    isolation_settle_seconds: float,
    isolation_allow_cmd_substring: list[str],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="nvfp4_dual_focused_") as td:
        sub_path = Path(td) / f"{cand.tag}.py"
        sub_path.write_text(_patch_source(src, cand))
        isolation_preflight = ensure_gpu_isolation(
            owner_pid=owner_pid,
            kill_foreign=kill_foreign_gpu_jobs,
            require_idle=require_idle_gpu,
            settle_seconds=isolation_settle_seconds,
            context=f"sweep_preflight:{cand.tag}",
            allow_cmd_substrings=isolation_allow_cmd_substring,
        )
        row: dict[str, Any] = {
            "tag": cand.tag,
            "cache_a": cand.cache_a,
            "cache_b": cand.cache_b,
            "stage64": cand.stage64,
            "stage128": cand.stage128,
            "isolation_preflight": isolation_preflight,
        }
        cmd = [
            "python",
            "-u",
            str(EVAL_SCRIPT),
            "--submission-file",
            str(sub_path),
            "--reference-file",
            str(REFERENCE_SUBMISSION),
            "--warmup",
            "1",
            "--repeats",
            "3",
            "--inputs-per-repeat",
            "20",
            "--json",
        ]
        if require_idle_gpu:
            cmd.append("--require-idle-gpu")
        if kill_foreign_gpu_jobs:
            cmd.append("--kill-foreign-gpu-jobs")
        cmd += [
            "--isolation-owner-pid",
            str(owner_pid),
            "--isolation-settle-seconds",
            str(float(isolation_settle_seconds)),
        ]
        for allow in isolation_allow_cmd_substring:
            cmd += ["--isolation-allow-cmd-substring", str(allow)]
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=240,
            )
        except subprocess.TimeoutExpired as exc:
            _cleanup_orphans()
            row["status"] = "timeout"
            row["seconds"] = time.perf_counter() - t0
            row["stdout_tail"] = (exc.stdout or "")[-4000:]
            row["stderr_tail"] = (exc.stderr or "")[-4000:]
            return row
        dt = time.perf_counter() - t0
        row["seconds"] = dt
        if proc.returncode != 0:
            _cleanup_orphans()
            row["status"] = "error"
            row["returncode"] = proc.returncode
            row["stdout_tail"] = proc.stdout[-4000:]
            row["stderr_tail"] = proc.stderr[-4000:]
            return row
        try:
            payload = _extract_json_blob(proc.stdout)
        except Exception as exc:  # noqa: BLE001
            _cleanup_orphans()
            row["status"] = "bad_json"
            row["error"] = str(exc)
            row["stdout_tail"] = proc.stdout[-4000:]
            row["stderr_tail"] = proc.stderr[-4000:]
            return row
        row["status"] = "ok"
        row["score_us"] = payload["score_us"]
        row["delta_vs_top_us"] = payload["delta_vs_top_us"]
        row["cases"] = payload["cases"]
        _cleanup_orphans()
        return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()

    owner_pid = int(args.owner_pid) if args.owner_pid is not None else int(os.getpid())
    src = BASE_SUBMISSION.read_text()
    cache_opts = ["EVICT_FIRST", "EVICT_NORMAL"]
    stage64_opts = [6, 7, 8]
    stage128_opts = [4, 5, 6]
    candidates = [
        Candidate(cache_a=a, cache_b=b, stage64=s64, stage128=s128)
        for a in cache_opts
        for b in cache_opts
        for s64 in stage64_opts
        for s128 in stage128_opts
    ]

    results = []
    verify_green = []
    for i, cand in enumerate(candidates, start=1):
        print(f"[{i}/{len(candidates)}] {cand.tag}", flush=True)
        result = _run_candidate(
            cand,
            src,
            owner_pid=owner_pid,
            require_idle_gpu=bool(args.require_idle_gpu),
            kill_foreign_gpu_jobs=bool(args.kill_foreign_gpu_jobs),
            isolation_settle_seconds=float(args.isolation_settle_seconds),
            isolation_allow_cmd_substring=list(args.isolation_allow_cmd_substring),
        )
        results.append(result)
        if result.get("status") == "ok":
            verify_green.append(result)

    verify_green.sort(key=lambda r: r["score_us"])

    payload = {
        "base_submission": str(BASE_SUBMISSION),
        "reference_submission": str(REFERENCE_SUBMISSION),
        "grid": {
            "cache_opts": cache_opts,
            "stage64_opts": stage64_opts,
            "stage128_opts": stage128_opts,
            "count": len(candidates),
        },
        "isolation": {
            "owner_pid": owner_pid,
            "require_idle_gpu": bool(args.require_idle_gpu),
            "kill_foreign_gpu_jobs": bool(args.kill_foreign_gpu_jobs),
            "isolation_settle_seconds": float(args.isolation_settle_seconds),
            "isolation_allow_cmd_substring": list(args.isolation_allow_cmd_substring),
        },
        "verify_green_count": len(verify_green),
        "verify_green_results": verify_green,
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print(f"Wrote {OUT_JSON}", flush=True)
    if verify_green:
        top = verify_green[0]
        print(
            f"best={top['tag']} score_us={top['score_us']:.6f} "
            f"delta_vs_top_us={top['delta_vs_top_us']:+.6f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
