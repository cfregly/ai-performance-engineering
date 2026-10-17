"""Run CUTLASS profiler across transformer-ish GEMM shapes and capture the best kernels."""

from __future__ import annotations

import argparse
import csv
import math
import json
import os
import subprocess
import sys
from collections import Counter
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from labs.cutlass_profiler_kernel_selector.shapes import (  # noqa: E402
    GemmShape,
    transformer_gemm_shapes,
)


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
        candidate = Path(env_candidate).expanduser()
        if candidate.is_file():
            return candidate
        raise FileNotFoundError(f"CUTLASS_PROFILER_BIN does not name a file: {candidate}")

    default = REPO_ROOT / "third_party" / "cutlass" / "build_profiler" / "tools" / "profiler" / "cutlass_profiler"
    if default.is_file():
        return default

    raise FileNotFoundError(
        f"cutlass_profiler binary not found at {default}. setup.sh clones CUTLASS into "
        "third_party/cutlass but does not build the profiler. Build the cutlass_profiler "
        "CMake target in third_party/cutlass/build_profiler, or point CUTLASS_PROFILER_BIN "
        "or --profiler-bin at an existing binary."
    )


# Exact column names from CUTLASS tools/profiler/src/performance_report.cpp.
# The same header also has "Flops", the total FLOP count of one launch. That
# value is identical for every kernel of a problem and is not a throughput.
PROVIDER_COLUMN = "Provider"
# performance_report.cpp writes to_string(result.provider, true), and
# tools/library/src/util.cu maps Provider::kCUTLASS to "CUTLASS". The GEMM
# profiler gives every row this provider. It uses cuBLAS only as a reference
# check, so a row from any other provider is not a CUTLASS kernel and is skipped.
CUTLASS_PROVIDER = "CUTLASS"
KERNEL_COLUMN = "Operation"
STATUS_COLUMN = "Status"
DISPOSITION_COLUMN = "Disposition"
RUNTIME_MS_COLUMN = "Runtime"
GFLOPS_COLUMN = "GFLOPs"
# With --enable-best-kernel-for-fixed-shape, gemm_operation_profiler.cu times
# each kernel across raster orders, swizzle sizes and, for kernels built with a
# dynamic cluster shape, cluster shapes. It reports only the fastest
# combination, so the kernel name alone cannot reproduce the reported runtime.
RUNTIME_PARAM_COLUMNS = (
    "raster_order",
    "swizzle_size",
    "cluster_m",
    "cluster_n",
    "cluster_k",
    "cluster_m_fallback",
    "cluster_n_fallback",
    "cluster_k_fallback",
)
REQUIRED_COLUMNS = (
    PROVIDER_COLUMN,
    KERNEL_COLUMN,
    STATUS_COLUMN,
    DISPOSITION_COLUMN,
    RUNTIME_MS_COLUMN,
    GFLOPS_COLUMN,
    *RUNTIME_PARAM_COLUMNS,
)
# cutlass_profiler still times a kernel after its reference check reports
# "incorrect" or "failed", so only the dispositions below may win the selection.
SELECTABLE_DISPOSITIONS = ("passed", "not_verified")
# cutlass_profiler checks each kernel against the reference only once, before
# the best-kernel search. That check uses the problem's runtime parameters,
# which default to heuristic raster order, swizzle size 1 and the kernel's
# default cluster shape unless --extra-profiler-flags sets them.
# operation_profiler.cu calls verify_cutlass before profile, and in
# gemm_operation_profiler.cu every timed combination starts as a copy of that
# verified result. The recorded "disposition" therefore describes the default
# configuration, not the "runtime_params" recorded next to it.


def _positive_measurement(row: Dict[str, str], column: str, csv_path: Path) -> float:
    text = row[column].strip()
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(
            f"{csv_path}: {column}={text!r} for kernel {row[KERNEL_COLUMN]!r} is not a number"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(
            f"{csv_path}: {column}={text!r} for kernel {row[KERNEL_COLUMN]!r} "
            "is not a positive finite value"
        )
    return value


def _runtime_params(row: Dict[str, str], csv_path: Path) -> Dict[str, str]:
    params = {column: row[column].strip() for column in RUNTIME_PARAM_COLUMNS}
    blank = [column for column, value in params.items() if not value]
    if blank:
        raise ValueError(
            f"{csv_path}: kernel {row[KERNEL_COLUMN]!r} has blank runtime parameter columns {blank}"
        )
    return params


def parse_best_result(csv_path: Path) -> Dict[str, object]:
    """Return the timed, non-failing CUTLASS kernel with the highest GFLOP/s in a cutlass_profiler CSV."""

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        first_row = next(reader, None)
        if first_row is None:
            raise ValueError(f"No rows found in profiler output {csv_path}")

        fieldnames = reader.fieldnames or []
        missing = [column for column in REQUIRED_COLUMNS if column not in fieldnames]
        if missing:
            raise ValueError(
                f"{csv_path} is not a cutlass_profiler CSV report: missing columns {missing} "
                f"(fields: {fieldnames})"
            )

        best_row: Optional[Dict[str, object]] = None
        rejected: Counter = Counter()
        for row in chain((first_row,), reader):
            if any(row[column] is None for column in REQUIRED_COLUMNS):
                raise ValueError(f"{csv_path}:{reader.line_num}: row is missing required columns")
            provider = row[PROVIDER_COLUMN].strip()
            status = row[STATUS_COLUMN].strip()
            disposition = row[DISPOSITION_COLUMN].strip()
            # cutlass_profiler leaves GFLOPs blank for a kernel it did not time.
            timed = bool(row[GFLOPS_COLUMN].strip())
            if (
                provider != CUTLASS_PROVIDER
                or status != "success"
                or disposition not in SELECTABLE_DISPOSITIONS
                or not timed
            ):
                rejected[
                    f"provider={provider} status={status} disposition={disposition} timed={timed}"
                ] += 1
                continue

            gflops = _positive_measurement(row, GFLOPS_COLUMN, csv_path)
            runtime_ms = _positive_measurement(row, RUNTIME_MS_COLUMN, csv_path)
            if best_row is None or gflops > best_row["gflops"]:
                best_row = {
                    "kernel": row[KERNEL_COLUMN].strip(),
                    "disposition": disposition,
                    "gflops": gflops,
                    "tflops": gflops / 1000.0,
                    "runtime_ms": runtime_ms,
                    "runtime_params": _runtime_params(row, csv_path),
                }

    if best_row is None:
        raise ValueError(
            f"No selectable kernel in {csv_path}: no row has Provider={CUTLASS_PROVIDER}, "
            f"Status=success, a Disposition in {SELECTABLE_DISPOSITIONS}, and a GFLOPs value. "
            f"Rejected rows: {dict(rejected)}"
        )
    return best_row


def run_profiler_for_shape(binary: Path, shape: GemmShape, output_dir: Path, extra_flags: Optional[List[str]] = None) -> Dict:
    """Invoke cutlass_profiler for a specific GEMM shape and parse the best kernel."""

    output_dir.mkdir(parents=True, exist_ok=True)
    # CUTLASS appends the operation kind as a suffix (e.g., ".gemm.csv"), ignoring
    # any extension we pass. Precompute the expected stem and later probe for files
    # matching that stem.
    csv_path = output_dir / f"{shape.name}"
    log_path = output_dir / f"{shape.name}.log"
    prior_csv_state = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in output_dir.glob(f"{shape.name}*.csv")
    }

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
        raise RuntimeError(
            f"{binary.name} exited with code {process.returncode} for {shape.name}. See {log_path}"
        )

    def was_emitted_by_this_run(path: Path) -> bool:
        current = (path.stat().st_mtime_ns, path.stat().st_size)
        return path not in prior_csv_state or current != prior_csv_state[path]

    csv_file = max(
        (path for path in output_dir.glob(f"{shape.name}*.csv") if was_emitted_by_this_run(path)),
        key=lambda path: path.stat().st_mtime_ns,
        default=None,
    )
    if csv_file is None:
        raise FileNotFoundError(
            f"Profiler did not emit a fresh CSV for {shape.name}. "
            f"Expected a new or changed file matching {csv_path}*.csv. Log: {log_path}"
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
                f"   best kernel={best['kernel']} ({best['disposition']}) runtime={best['runtime_ms']:.3f} ms "
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
