"""Interleaved A/B rerank runner for labs/nvfp4_gemm on B200.

This script builds and times the baseline/optimized CUDA binaries from this lab and
reports geomean deltas over interleaved A/B pairs.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from core.harness.benchmark_harness import lock_gpu_clocks

LAB_DIR = Path(__file__).resolve().parent

TIME_MS_RE = re.compile(r"TIME_MS:\s*([0-9.eE+-]+)")
GEOMEAN_RE = re.compile(r"GEOMEAN_MS:\s*([0-9.eE+-]+)")
SHAPE_RE = re.compile(
    r"shape=\((\d+),(\d+),(\d+)\)[^\n]*TIME_MS:\s*([0-9.eE+-]+)",
    re.IGNORECASE,
)
VERIFY_RE = re.compile(r"VERIFY_CHECKSUM:\s*([0-9.eE+-]+)")

ARCH_SUFFIX = {
    "sm_100": "_sm100",
    "sm_103": "_sm103",
    "sm_121": "_sm121",
}


@contextmanager
def _null_ctx():
    yield None


def _run(cmd: list[str], *, cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )


def _build(binary_name: str, *, arch: str, verify: bool, timeout_seconds: int) -> Path:
    suffix = ARCH_SUFFIX[arch]
    target = f"{binary_name}{'_verify' if verify else ''}{suffix}"
    completed = _run(
        ["make", f"ARCH={arch}", target, *( ["VERIFY=1"] if verify else [] )],
        cwd=LAB_DIR,
        timeout_seconds=timeout_seconds,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Build failed for {target}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    out_path = LAB_DIR / target
    if not out_path.exists():
        raise FileNotFoundError(f"Expected built binary not found: {out_path}")
    return out_path


def _parse_output(stdout: str) -> dict[str, Any]:
    times = [float(x) for x in TIME_MS_RE.findall(stdout)]
    if not times:
        raise RuntimeError(f"Could not parse TIME_MS from output:\n{stdout}")

    geomean_match = GEOMEAN_RE.findall(stdout)
    geomean_ms = float(geomean_match[-1]) if geomean_match else float(times[-1])

    shape_times = {}
    for m_str, n_str, k_str, t_str in SHAPE_RE.findall(stdout):
        shape_times[f"{m_str}x{n_str}x{k_str}"] = float(t_str)

    verify_match = VERIFY_RE.search(stdout)
    verify_checksum = float(verify_match.group(1)) if verify_match else None

    return {
        "time_ms_tokens": times,
        "geomean_ms": geomean_ms,
        "shape_times_ms": shape_times,
        "verify_checksum": verify_checksum,
    }


def _run_binary(binary_path: Path, *, timeout_seconds: int) -> dict[str, Any]:
    completed = _run([str(binary_path)], cwd=LAB_DIR, timeout_seconds=timeout_seconds)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Binary failed: {binary_path.name} rc={completed.returncode}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    parsed = _parse_output(completed.stdout)
    parsed["stdout"] = completed.stdout
    parsed["stderr"] = completed.stderr
    return parsed


def _summarize(values: list[float]) -> dict[str, float]:
    count = len(values)
    if count == 0:
        raise ValueError("Cannot summarize an empty sample set")

    total = 0.0
    total_squares = 0.0
    for value in values:
        total += value
        total_squares += value * value

    mean = total / count
    variance = max(0.0, (total_squares / count) - mean * mean)
    values.sort()
    midpoint = count // 2
    if count % 2:
        median = values[midpoint]
    else:
        median = (values[midpoint - 1] + values[midpoint]) / 2.0

    return {
        "mean": float(mean),
        "median": float(median),
        "stdev": float(math.sqrt(variance)) if count > 1 else 0.0,
        "min": float(values[0]),
        "max": float(values[-1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=8, help="Number of interleaved AB pairs")
    parser.add_argument("--arch", choices=sorted(ARCH_SUFFIX), default="sm_100")
    parser.add_argument("--timeout-seconds", type=int, default=600)
    parser.add_argument("--verify", action="store_true", default=True)
    parser.add_argument("--no-verify", dest="verify", action="store_false")
    parser.add_argument("--lock-gpu-clocks", action="store_true", default=True)
    parser.add_argument("--no-lock-gpu-clocks", dest="lock_gpu_clocks", action="store_false")
    parser.add_argument("--sm-clock-mhz", type=int, default=1500)
    parser.add_argument("--mem-clock-mhz", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Print JSON payload")
    parser.add_argument("--json-out", type=Path, default=None, help="Write JSON payload to file")
    args = parser.parse_args()

    if args.pairs <= 0:
        raise ValueError("--pairs must be > 0")

    baseline_bin = _build("baseline_nvfp4_gemm", arch=args.arch, verify=False, timeout_seconds=args.timeout_seconds)
    optimized_bin = _build("optimized_nvfp4_gemm", arch=args.arch, verify=False, timeout_seconds=args.timeout_seconds)

    verify_payload: dict[str, Any] = {}
    if args.verify:
        baseline_verify = _build("baseline_nvfp4_gemm", arch=args.arch, verify=True, timeout_seconds=args.timeout_seconds)
        optimized_verify = _build("optimized_nvfp4_gemm", arch=args.arch, verify=True, timeout_seconds=args.timeout_seconds)

        baseline_v = _run_binary(baseline_verify, timeout_seconds=args.timeout_seconds)
        optimized_v = _run_binary(optimized_verify, timeout_seconds=args.timeout_seconds)
        b_checksum = baseline_v.get("verify_checksum")
        o_checksum = optimized_v.get("verify_checksum")
        if b_checksum is None or o_checksum is None:
            raise RuntimeError("Verification enabled but VERIFY_CHECKSUM was not found in verify binary output")
        checksum_delta = abs(float(b_checksum) - float(o_checksum))
        verify_payload = {
            "baseline_checksum": float(b_checksum),
            "optimized_checksum": float(o_checksum),
            "abs_delta": float(checksum_delta),
        }

    pair_rows = []
    baseline_values = []
    optimized_values = []
    deltas = []

    lock_ctx = (
        lock_gpu_clocks(device=0, sm_clock_mhz=args.sm_clock_mhz, mem_clock_mhz=args.mem_clock_mhz)
        if args.lock_gpu_clocks
        else _null_ctx()
    )

    with lock_ctx:
        for pair_idx in range(args.pairs):
            b = _run_binary(baseline_bin, timeout_seconds=args.timeout_seconds)
            o = _run_binary(optimized_bin, timeout_seconds=args.timeout_seconds)
            b_ms = float(b["geomean_ms"])
            o_ms = float(o["geomean_ms"])
            delta_ms = o_ms - b_ms
            baseline_values.append(b_ms)
            optimized_values.append(o_ms)
            deltas.append(delta_ms)
            pair_rows.append(
                {
                    "pair": pair_idx + 1,
                    "baseline_geomean_ms": b_ms,
                    "optimized_geomean_ms": o_ms,
                    "delta_ms_optimized_minus_baseline": delta_ms,
                }
            )

    baseline_stats = _summarize(baseline_values)
    optimized_stats = _summarize(optimized_values)
    delta_stats = _summarize(deltas)

    payload = {
        "timestamp": int(time.time()),
        "arch": args.arch,
        "pairs": args.pairs,
        "lock_gpu_clocks": bool(args.lock_gpu_clocks),
        "sm_clock_mhz": args.sm_clock_mhz,
        "mem_clock_mhz": args.mem_clock_mhz,
        "verify": bool(args.verify),
        "verify_result": verify_payload,
        "baseline_binary": str(baseline_bin),
        "optimized_binary": str(optimized_bin),
        "baseline_geomean_ms": baseline_stats,
        "optimized_geomean_ms": optimized_stats,
        "delta_ms_optimized_minus_baseline": delta_stats,
        "optimized_beats_baseline_mean": bool(optimized_stats["mean"] < baseline_stats["mean"]),
        "pair_results": pair_rows,
    }

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("NVFP4 GEMM A/B rerank")
        print(f"pairs={args.pairs} arch={args.arch} verify={args.verify} lock_gpu_clocks={args.lock_gpu_clocks}")
        if verify_payload:
            print(
                "verify checksums: "
                f"baseline={verify_payload['baseline_checksum']:.6f} "
                f"optimized={verify_payload['optimized_checksum']:.6f} "
                f"abs_delta={verify_payload['abs_delta']:.6f}"
            )
        print(
            "baseline mean geomean: "
            f"{baseline_stats['mean']:.9f} ms "
            f"(stdev {baseline_stats['stdev']:.9f})"
        )
        print(
            "optimized mean geomean: "
            f"{optimized_stats['mean']:.9f} ms "
            f"(stdev {optimized_stats['stdev']:.9f})"
        )
        print(
            "delta (optimized-baseline): "
            f"mean {delta_stats['mean']:+.9f} ms, "
            f"median {delta_stats['median']:+.9f} ms"
        )
        print(f"optimized beats baseline mean: {payload['optimized_beats_baseline_mean']}")
        if args.json_out is not None:
            print(f"wrote: {args.json_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
