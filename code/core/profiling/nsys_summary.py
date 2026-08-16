#!/usr/bin/env python3
"""
Utility to summarise Nsight Systems reports across the Blackwell codebase.

Example:
    python -m core.profiling.nsys_summary --glob "artifacts/runs/**/*.nsys-rep" \
        --kernel-regex "attn|mma" --top-k 8 --output artifacts/runs/analysis/nsys_summary.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

from ch17.blackwell_profiling_guide import NsightSystemsProfiler  # noqa: E402


def _collect_reports(explicit: Iterable[str], pattern: str | None) -> list[Path]:
    reports: list[Path] = []

    for item in explicit:
        path = Path(item).expanduser()
        if path.is_dir():
            reports.extend(sorted(path.glob("*.nsys-rep")))
            reports.extend(sorted(path.glob("*.qdrep")))
        elif path.is_file():
            reports.append(path)

    if pattern:
        reports.extend(sorted(Path().glob(pattern)))

    # Deduplicate while preserving order
    unique: list[Path] = []
    seen = set()
    for report in reports:
        resolved = report.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _format_summary(report: Path, data: dict) -> str:
    lines = [
        f"=== Nsight Systems Summary ({report}) ===",
    ]
    kernels = data["kernels"]
    if not kernels:
        lines.append("No CUDA GPU kernels recorded (check trace configuration).")
        return "\n".join(lines)

    lines.append("Top CUDA Kernels:")
    for idx, row in enumerate(kernels, 1):
        name = row.get("Name", "Unknown")
        time_pct = row.get("Time (%)") or row.get("Time (%) [sum]", "0")
        total_ns = (
            row.get("Total Time (ns)")
            or row.get("Time (ns)")
            or row.get("Total Time (ns) [sum]")
            or row.get("Time (ns) [sum]")
            or "0"
        )
        try:
            time_ms = float(total_ns) / 1e6
        except (TypeError, ValueError):
            time_ms = 0.0
        try:
            pct = float(str(time_pct).replace('"', ""))
        except (TypeError, ValueError):
            pct = 0.0
        lines.append(f"  {idx:>2}. {name}")
        lines.append(f"       Time: {time_ms:.3f} ms   Share: {pct:.2f}%")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarise Nsight Systems reports (cuda_gpu_kern_sum)."
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="Explicit .nsys-rep/.qdrep file or directory containing reports. "
        "May be supplied multiple times.",
    )
    parser.add_argument(
        "--glob",
        default=None,
        help="Glob pattern (relative to CWD) to locate reports "
        "(e.g. 'artifacts/runs/**/*.nsys-rep').",
    )
    parser.add_argument(
        "--kernel-regex",
        default=None,
        help="Optional regex filter applied to kernel names (e.g. 'attn|mma').",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of kernels to include per report (default: 5).",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional file to write the summary. Printed to stdout otherwise.",
    )

    args = parser.parse_args()

    reports = _collect_reports(args.report, args.glob)
    if not reports:
        print("No Nsight Systems reports found.", file=sys.stderr)
        return 0

    summaries: list[str] = []
    for report in reports:
        try:
            data = NsightSystemsProfiler.summarize_report(
                str(report),
                kernel_regex=args.kernel_regex,
                top_k=args.top_k,
                print_summary=False,
            )
        except Exception as exc:
            summaries.append(f"=== {report} ===\nFailed to summarise: {exc}")
            continue
        summaries.append(_format_summary(report, data))

    output_text = "\n\n".join(summaries)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output_text + "\n")
        print(f"Wrote Nsight Systems summary to {output_path}")
    else:
        print(output_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
