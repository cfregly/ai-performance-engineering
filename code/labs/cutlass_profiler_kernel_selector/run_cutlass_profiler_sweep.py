"""Run CUTLASS profiler across transformer-ish GEMM shapes and capture the best kernels."""

from __future__ import annotations

import argparse
import csv
import math
import json
import os
import subprocess
import sys
from itertools import chain
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .shapes import GemmShape, transformer_gemm_shapes


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO_ROOT / "artifacts" / "cutlass_profiler"


def profiler_binary(explicit: Optional[str] = None) -> Path:
    """Resolve the cutlass_profiler binary location."""

    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Explicit cutlass_profiler not found: {candidate}")

    env_candidate = os.environ.get("CUTLASS_PROFILER_BIN")
    if env_candidate:
        candidate = Path(env_candidate)
        if candidate.is_file():
            return candidate

    default = REPO_ROOT / "third_party" / "cutlass" / "build_profiler" / "tools" / "profiler" / "cutlass_profiler"
    if default.is_file():
        return default

    raise FileNotFoundError(
        "cutlass_profiler binary not found. Run ./setup.sh to build it or pass --profiler-bin explicitly."
    )


def _to_float(value: str) -> float:
    raw = value.strip()
    for token in (",",):
        raw = raw.replace(token, "")
    try:
        return float(raw)
    except ValueError:
        return float("nan")


def _select_field(fields: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    lowered = {f.lower(): f for f in fields}
    for want in candidates:
        for key, original in lowered.items():
            if want in key:
                return original
    return None


def parse_best_result(csv_path: Path) -> Dict[str, object]:
    """Parse the CSV emitted by cutlass_profiler and return the best row by FLOP/s."""

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"No rows found in profiler output {csv_path}")

        gflops_field = _select_field(reader.fieldnames or [], ("gflop", "tflop", "flops"))
        runtime_field = _select_field(reader.fieldnames or [], ("runtime", "time"))
        kernel_field = _select_field(reader.fieldnames or [], ("kernel", "name"))

        if gflops_field is None or runtime_field is None:
            raise ValueError(f"Missing gflops/time fields in {csv_path} (fields: {reader.fieldnames})")

        best_row = None
        best_score = -float("inf")
        for row in chain((first_row,), reader):
            gflops_raw = _to_float(row[gflops_field])
            if "tflop" in gflops_field.lower():
                tflops = gflops_raw
                gflops = gflops_raw * 1000.0
            else:
                gflops = gflops_raw
                tflops = gflops_raw / 1000.0

            if gflops > best_score:
                best_score = gflops
                runtime_ms = _to_float(row[runtime_field])
                # If runtime was missing/NaN, derive it from FLOP/s if possible.
                if math.isnan(runtime_ms) or runtime_ms <= 0:
                    try:
                        flops = _to_float(row.get("Flops", "")) * 1.0  # reported as raw FLOPs
                        runtime_ms = (flops / (gflops * 1e9)) * 1e3 if gflops > 0 else float("nan")
                    except Exception:
                        runtime_ms = float("nan")

                best_row = {
                    "kernel": row.get(kernel_field, row.get("op", "")) if kernel_field else "",
                    "gflops": gflops,
                    "tflops": tflops,
                    "runtime_ms": runtime_ms,
                }

    assert best_row is not None
    return best_row


def run_profiler_for_shape(binary: Path, shape: GemmShape, output_dir: Path, extra_flags: Optional[List[str]] = None) -> Dict:
    """Invoke cutlass_profiler for a specific GEMM shape and parse the best kernel."""

    output_dir.mkdir(parents=True, exist_ok=True)
    # CUTLASS appends the operation kind as a suffix (e.g., ".gemm.csv"), ignoring
    # any extension we pass. Precompute the expected stem and later probe for files
    # matching that stem.
    csv_path = output_dir / f"{shape.name}"
    log_path = output_dir / f"{shape.name}.log"

    cmd = [
        str(binary),
        "--operation=Gemm",
        f"--m={shape.m}",
        f"--n={shape.n}",
        f"--k={shape.k}",
        f"--A={shape.dtype}:row",
        f"--B={shape.dtype}:row",
        f"--C={shape.dtype}",
        "--accumulator-type=f32",
        "--op_class=tensorop",
        "--providers=cutlass",
        "--enable-kernel-performance-search",
        "--enable-best-kernel-for-fixed-shape",
        "--sort-results-flops-per-sec",
        f"--output={csv_path}",
    ]

    if extra_flags:
        cmd.extend(extra_flags)

    with log_path.open("w") as log_file:
        process = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)

    if process.returncode != 0:
        raise RuntimeError(f"{binary.name} failed for {shape.name}; see {log_path}")

    csv_file = max(output_dir.glob(f"{shape.name}*.csv"), key=lambda p: p.stat().st_mtime, default=None)
    if csv_file is None:
        raise FileNotFoundError(
            f"Profiler did not emit CSV for {shape.name}; expected something like {csv_path}*.csv; log: {log_path}"
        )

    best_row = parse_best_result(csv_file)
    best_row.update(shape.as_dict())
    best_row["csv"] = str(csv_file)
    best_row["log"] = str(log_path)
    return best_row


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Sweep CUTLASS profiler across transformer-ish GEMM shapes.")
    parser.add_argument(
        "--profiler-bin",
        type=str,
        default=None,
        help="Path to cutlass_profiler binary (defaults to env CUTLASS_PROFILER_BIN or third_party/cutlass/build_profiler/... ).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help="Where to write CSVs/logs/results JSON.",
    )
    parser.add_argument(
        "--shapes",
        type=str,
        nargs="*",
        default=None,
        help="Subset of shape names to run (defaults to all transformer shapes).",
    )
    parser.add_argument(
        "--extra-profiler-flags",
        type=str,
        nargs="*",
        default=None,
        help="Additional flags forwarded verbatim to cutlass_profiler.",
    )
    args = parser.parse_args(argv)

    binary = profiler_binary(args.profiler_bin)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    shapes = transformer_gemm_shapes()
    if args.shapes:
        name_set = set(args.shapes)
        shapes = [s for s in shapes if s.name in name_set]
        if not shapes:
            print(f"No matching shapes for {args.shapes}", file=sys.stderr)
            return 1

    print(f"Using cutlass_profiler at: {binary}")
    print(f"Writing artifacts to: {output_dir}")

    results: List[Dict[str, object]] = []
    failed: List[str] = []
    for shape in shapes:
        try:
            print(f"→ Profiling {shape.name} (m={shape.m}, n={shape.n}, k={shape.k})...")
            best = run_profiler_for_shape(binary, shape, output_dir, args.extra_profiler_flags)
            print(
                f"   best kernel={best['kernel']} runtime={best['runtime_ms']:.3f} ms "
                f"throughput={best['tflops']:.2f} TFLOP/s"
            )
            results.append(best)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"   FAILED: {exc}", file=sys.stderr)
            failed.append(shape.name)

    summary = {
        "provider": "cutlass_profiler",
        "binary": str(binary),
        "results": results,
        "failed": failed,
    }

    summary_path = output_dir / "cutlass_profiler_results.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSummary written to {summary_path}")
    if failed:
        print(f"{len(failed)} shape(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
