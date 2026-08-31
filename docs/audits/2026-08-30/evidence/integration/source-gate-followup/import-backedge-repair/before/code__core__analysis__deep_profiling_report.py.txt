#!/usr/bin/env python3
"""
Deep profiling workflow for performance-engineering chapters.

This utility stitches together three pillars that the book's TODO list calls
out as missing:

1. Nsight Systems kernel ranking (timeline level context)
2. Nsight Compute kernel metrics + roofline classification
3. Automated optimisation advice based on key bottleneck signals

Typical usage:

    python -m core.analysis.deep_profiling_report \\
        --ncu-csv artifacts/runs/analysis/double_buffered_pipeline_512.csv \\
        --nsys-report artifacts/runs/<run_id>/profiles/bench/ch10/pipeline_async_verified.nsys-rep \\
        --output-json artifacts/runs/analysis/double_buffered_pipeline_analysis.json

The script understands the CSV format produced by either:
* `ncu --set roofline --csv ...`
* `python core/profiling/extract_ncu_metrics.py --example <name>`

It does not require GPU access to run; it analyses offline artifacts.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import os
import re
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ch08.roofline import RooflineAnalyzer
from ch17.blackwell_profiling_guide import NsightSystemsProfiler


@dataclass
class RawMetric:
    """Single Nsight Compute metric entry."""

    name: str
    value: float
    unit: Optional[str] = None
    section: Optional[str] = None


@dataclass
class KernelMetrics:
    """Aggregated metrics for a kernel."""

    name: str
    metrics: Dict[str, RawMetric] = field(default_factory=dict)

    def get(self, *candidates: str) -> Optional[RawMetric]:
        for candidate in candidates:
            if candidate in self.metrics:
                return self.metrics[candidate]
        return None

    def get_value(self, *candidates: str) -> Optional[float]:
        entry = self.get(*candidates)
        return entry.value if entry else None


@dataclass
class RooflineSummary:
    achieved_tflops: float
    achieved_bandwidth_gbs: float
    arithmetic_intensity: float
    compute_utilization_pct: float
    memory_utilization_pct: float
    tmem_utilization_pct: Optional[float]
    l2_utilization_pct: Optional[float]
    binding: str
    is_memory_bound: bool
    is_compute_bound: bool
    is_tmem_bound: bool
    ridge_point: float
    memory_bound_limit_tflops: float
    peak_tflops: float
    peak_bandwidth_gbs: float


@dataclass
class Advisory:
    kernel: str
    precision: str
    duration_ms: Optional[float]
    flops: Optional[float]
    bytes_transferred: Optional[float]
    roofline: Optional[RooflineSummary]
    sm_util_pct: Optional[float]
    dram_util_pct: Optional[float]
    tmem_util_pct: Optional[float]
    occupancy_pct: Optional[float]
    tensor_util_pct: Optional[float]
    warp_exec_pct: Optional[float]
    l2_hit_pct: Optional[float]
    recommendations: List[str]


NCU_KERNEL_KEYS = [
    "Kernel Name",
    "Kernel Name/Id",
    "Kernel Name/ID",
    "Kernel",
    "ID",
]

NCU_METRIC_KEYS = [
    "Metric Name",
    "Metric Name/Description",
    "Name",
]

NCU_VALUE_KEYS = [
    "Metric Value",
    "Metric Value [Latest]",
    "Value",
    "Metric Value (%)",
]

NCU_UNIT_KEYS = [
    "Metric Unit",
    "Unit",
]

NCU_SECTION_KEYS = [
    "Section",
    "Metric Section",
]


def parse_float(text: str) -> Optional[float]:
    """Best-effort float parser that tolerates Nsight formatting."""
    if text is None:
        return None
    stripped = text.strip()
    if not stripped or stripped in {"N/A", "nan"}:
        return None
    stripped = stripped.replace(",", "")
    if stripped.endswith("%"):
        stripped = stripped[:-1]
    # Remove parenthetical units e.g. "123.4 (bytes)"
    stripped = re.sub(r"\([^)]*\)", "", stripped).strip()
    match = re.match(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", stripped)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def first_key(row: Dict[str, str], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in row and row[key].strip():
            return row[key].strip()
    return None


def parse_ncu_csv(path: Path) -> Dict[str, KernelMetrics]:
    """Parse Nsight Compute CSV export into per-kernel metrics."""
    kernels: Dict[str, KernelMetrics] = {}
    with path.open(newline="") as fh:
        # Skip preamble until header row begins (first cell "ID")
        header_line: Optional[str] = None
        preamble: List[str] = []
        for line in fh:
            if line.lstrip().startswith('"ID"'):
                header_line = line
                break
            preamble.append(line)
        if header_line is None:
            return {}
        reader = csv.DictReader(itertools.chain([header_line], fh))
        for row in reader:
            kernel_name = first_key(row, NCU_KERNEL_KEYS) or "kernel"
            metric_name_raw = first_key(row, NCU_METRIC_KEYS)
            metric_value_raw = first_key(row, NCU_VALUE_KEYS)
            if not metric_name_raw or metric_value_raw is None:
                continue
            value = parse_float(metric_value_raw)
            if value is None:
                continue
            metric_name = metric_name_raw.strip()
            unit = first_key(row, NCU_UNIT_KEYS)
            section = first_key(row, NCU_SECTION_KEYS)
            kernel = kernels.setdefault(kernel_name, KernelMetrics(kernel_name))
            kernel.metrics[metric_name] = RawMetric(metric_name, value, unit, section)
    return kernels


def merge_kernel_metrics(dicts: Iterable[Dict[str, KernelMetrics]]) -> Dict[str, KernelMetrics]:
    result: Dict[str, KernelMetrics] = {}
    for kernel_map in dicts:
        for name, metrics in kernel_map.items():
            combined = result.setdefault(name, KernelMetrics(name))
            combined.metrics.update(metrics.metrics)
    return result


def metric_in_unit(entry: Optional[RawMetric]) -> Optional[float]:
    if entry is None:
        return None
    value = entry.value
    unit = (entry.unit or "").lower()
    if unit.endswith("ns"):
        return value / 1e6
    if unit.endswith("us"):
        return value / 1e3
    if unit.endswith("ms"):
        return value
    if unit.endswith("s"):
        return value * 1e3
    return value  # Fallback assume already in ms


def pick_precision(metrics: KernelMetrics) -> str:
    fp32 = metrics.get_value("flop_count_sp", "flop_count_sp_sum")
    fp16 = metrics.get_value("flop_count_hp", "flop_count_hp_sum", "flop_count_half_precision")
    fp8 = metrics.get_value("flop_count_fp8")
    if fp8 and fp8 > 0:
        return "fp8"
    if fp16 and fp16 > 0 and (fp32 is None or fp16 >= fp32):
        return "fp16"
    return "fp32"


def compute_flops(metrics: KernelMetrics) -> Optional[float]:
    total = 0.0
    present = False

    # Legacy Nsight Compute metric names
    legacy_flop_metrics = [
        ("flop_count_sp", 1.0),
        ("flop_count_hp", 1.0),
        ("flop_count_fp8", 1.0),
        ("flop_count_dp", 1.0),
        ("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum", 2.0),
        ("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum", 2.0),
        ("smsp__inst_executed_pipe_tensor.sum", 2.0),
    ]
    for key, weight in legacy_flop_metrics:
        value = metrics.get_value(key)
        if value is not None:
            total += value * weight
            present = True

    # Blackwell SM 12.x metric names (thread-level instruction counters)
    ffma = metrics.get_value("sm__sass_thread_inst_executed_op_ffma_pred_on.sum")
    if ffma is not None:
        total += ffma * 2.0
        present = True

    for key in [
        "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
        "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
    ]:
        value = metrics.get_value(key)
        if value is not None:
            total += value
            present = True

    return total if present else None


def compute_bytes(metrics: KernelMetrics) -> Optional[float]:
    value = 0.0
    present = False
    byte_metrics = [
        ("dram__bytes_read.sum", None),
        ("dram__bytes_write.sum", None),
        ("dram__bytes.sum", None),
        ("gpu__dram_sectors_read.sum", "sector"),
        ("gpu__dram_sectors_write.sum", "sector"),
        ("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum", "byte"),
        ("l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum", "byte"),
        ("l1tex__t_bytes_pipe_lsu_mem_global_op_tma_ld.sum", "byte"),
        ("l1tex__t_bytes_pipe_lsu_mem_global_op_tma_st.sum", "byte"),
    ]
    for key, unit_hint in byte_metrics:
        metric = metrics.get(key)
        if metric is None:
            continue
        val = metric.value
        if val is None:
            continue
        unit = (metric.unit or unit_hint or "").lower()
        if "sector" in unit:
            val *= 32.0
        value += val
        present = True
    return value if present else None


def safe_pct(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return value if value > 1.0 else value * 100.0


def determine_binding_roof(
    compute_pct: Optional[float],
    tmem_pct: Optional[float],
    dram_pct: Optional[float],
    l2_pct: Optional[float],
    threshold: float = 5.0,
) -> str:
    """Roughly classify whether compute, TMEM, or DRAM is the binding roof."""
    best = "compute"
    best_value = compute_pct or 0.0
    for label, value in (
        ("tmem", tmem_pct),
        ("l2", l2_pct),
        ("dram", dram_pct),
    ):
        if value is None:
            continue
        if value > best_value + threshold:
            best = label
            best_value = value
    return best


def derive_roofline(metrics: KernelMetrics) -> Tuple[Optional[RooflineSummary], Optional[float], Optional[float], Optional[float], str]:
    analyzer = RooflineAnalyzer()
    precision = pick_precision(metrics)
    duration_metric = metrics.get(
        "gpu__time_duration.sum",
        "gpu__time_duration.avg",
        "Duration",
        "Kernel Duration"
    )
    duration_ms = metric_in_unit(duration_metric)
    flops = compute_flops(metrics)
    bytes_transferred = compute_bytes(metrics)
    compute_util_pct = safe_pct(metrics.get_value(
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "Compute (SM) Throughput",
        "Compute Pipe Throughput",
        "SM Throughput",
    ))
    memory_util_pct = safe_pct(metrics.get_value(
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "Memory Throughput",
        "Memory (Device) Throughput",
        "Device Memory Throughput",
    ))
    tmem_util_pct = safe_pct(metrics.get_value(
        "tmem__throughput.avg.pct_of_peak_sustained_elapsed",
        "Tensor Memory Throughput",
        "Tensor Memory (TMEM) Throughput",
    ))
    l2_util_pct = safe_pct(metrics.get_value(
        "l2__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "L2 Throughput",
    ))
    binding = determine_binding_roof(compute_util_pct, tmem_util_pct, memory_util_pct, l2_util_pct)
    is_tmem_bound = binding == "tmem"
    roofline_summary: Optional[RooflineSummary] = None
    if duration_ms and flops and bytes_transferred and duration_ms > 0 and bytes_transferred > 0:
        results = analyzer.analyze_kernel(duration_ms, flops, bytes_transferred, precision)
        roofline_summary = RooflineSummary(
            achieved_tflops=results["achieved_tflops"],
            achieved_bandwidth_gbs=results["achieved_bandwidth_gbs"],
            arithmetic_intensity=results["arithmetic_intensity"],
            compute_utilization_pct=results["compute_utilization_pct"],
            memory_utilization_pct=results["memory_utilization_pct"],
            tmem_utilization_pct=tmem_util_pct,
            l2_utilization_pct=l2_util_pct,
            binding=binding,
            is_memory_bound=results["is_memory_bound"],
            is_compute_bound=results["is_compute_bound"],
            is_tmem_bound=is_tmem_bound,
            ridge_point=results["ridge_point"],
            memory_bound_limit_tflops=results["memory_bound_tflops"],
            peak_tflops=results["peak_tflops"],
            peak_bandwidth_gbs=results["peak_bandwidth_gbs"],
        )
    elif duration_ms and compute_util_pct is not None:
        peak_by_precision = {
            "fp32": analyzer.specs.peak_fp32_tflops,
            "fp16": analyzer.specs.peak_fp16_tflops,
            "fp8": analyzer.specs.peak_fp8_tflops,
            "tf32": analyzer.specs.peak_tf32_tflops,
        }
        peak_tflops = peak_by_precision.get(precision, analyzer.specs.peak_fp32_tflops)
        achieved_tflops = peak_tflops * (compute_util_pct / 100.0)
        achieved_bandwidth_gbs = None
        if memory_util_pct is not None:
            achieved_bandwidth_gbs = analyzer.specs.memory_bandwidth_gbs * (memory_util_pct / 100.0)
        arithmetic_intensity = None
        if achieved_bandwidth_gbs and achieved_bandwidth_gbs > 0:
            arithmetic_intensity = 1000.0 * (achieved_tflops / achieved_bandwidth_gbs)
        ridge_point = (peak_tflops * 1000.0) / analyzer.specs.memory_bandwidth_gbs if analyzer.specs.memory_bandwidth_gbs else 0.0
        memory_bound_limit_tflops = None
        is_memory_bound = False
        is_compute_bound = False
        if arithmetic_intensity is not None and analyzer.specs.memory_bandwidth_gbs > 0:
            memory_bound_limit_tflops = min(
                peak_tflops,
                (arithmetic_intensity * analyzer.specs.memory_bandwidth_gbs) / 1000.0,
            )
            if arithmetic_intensity < ridge_point * 0.95:
                is_memory_bound = True
            elif arithmetic_intensity >= ridge_point:
                is_compute_bound = True
        if not is_memory_bound and not is_compute_bound:
            if compute_util_pct >= (memory_util_pct or 0):
                is_compute_bound = True
            else:
                is_memory_bound = True
        if flops is None:
            flops = achieved_tflops * (duration_ms / 1000.0) * 1e12
        if bytes_transferred is None and achieved_bandwidth_gbs:
            bytes_transferred = achieved_bandwidth_gbs * (duration_ms / 1000.0) * 1e9
        roofline_summary = RooflineSummary(
            achieved_tflops=achieved_tflops,
            achieved_bandwidth_gbs=achieved_bandwidth_gbs or 0.0,
            arithmetic_intensity=arithmetic_intensity or 0.0,
            compute_utilization_pct=compute_util_pct,
            memory_utilization_pct=memory_util_pct or 0.0,
            tmem_utilization_pct=tmem_util_pct,
            l2_utilization_pct=l2_util_pct,
            binding=binding,
            is_memory_bound=is_memory_bound,
            is_compute_bound=is_compute_bound,
            is_tmem_bound=is_tmem_bound,
            ridge_point=ridge_point,
            memory_bound_limit_tflops=memory_bound_limit_tflops or 0.0,
            peak_tflops=peak_tflops,
            peak_bandwidth_gbs=analyzer.specs.memory_bandwidth_gbs,
        )
    return roofline_summary, duration_ms, flops, bytes_transferred, precision


def build_advisory(metrics: KernelMetrics) -> Advisory:
    roofline_summary, duration_ms, flops, bytes_transferred, precision = derive_roofline(metrics)
    sm_util = safe_pct(metrics.get_value(
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "Compute (SM) Throughput",
        "Compute Pipe Throughput",
        "SM Throughput",
    ))
    dram_util = safe_pct(metrics.get_value(
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "Memory Throughput",
        "Memory (Device) Throughput",
        "Device Memory Throughput",
    ))
    tmem_util = safe_pct(metrics.get_value(
        "tmem__throughput.avg.pct_of_peak_sustained_elapsed",
        "Tensor Memory Throughput",
        "Tensor Memory (TMEM) Throughput",
    ))
    occupancy = safe_pct(metrics.get_value("sm__warps_active.avg.pct_of_peak_sustained_active"))
    tensor_util = safe_pct(
        metrics.get_value(
            "smpp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        )
    )
    warp_exec = safe_pct(
        metrics.get_value(
            "smsp__warp_execution_efficiency.avg.pct",
            "warp_execution_efficiency",
            "Warp Execution Efficiency",
        )
    )
    l2_hit = safe_pct(
        metrics.get_value(
            "lts__t_sectors_hit_rate.pct",
            "l2_tex_hit_rate",
            "L2 Hit Rate",
        )
    )

    recommendations: List[str] = []

    if roofline_summary:
        if roofline_summary.is_memory_bound:
            recommendations.append(
                "Kernel is memory-bound; focus on increasing arithmetic intensity (fuse ops, reuse data, leverage shared memory/TMA)."
            )
            if dram_util and dram_util > 80:
                recommendations.append(
                    "HBM bandwidth already saturated (>80%); investigate cache blocking or compression to reduce traffic."
                )
        if roofline_summary.is_tmem_bound:
            recommendations.append(
                "TMEM throughput is binding; fix tensor-map alignment, reduce multicast fan-out, or streamline TMA descriptors."
            )
            recommendations.append(
                "Capture Nsight metrics + labs/blackwell_matmul metadata and run core/analysis/dual_roofline_plot.py to visualise the dual ceilings."
            )
        elif roofline_summary.is_compute_bound:
            recommendations.append(
                "Kernel is compute-bound; pursue higher SM utilisation (occupancy tuning, instruction-level parallelism)."
            )
            if tensor_util and tensor_util < 40:
                recommendations.append(
                    "Tensor cores underutilised; migrate hot loops to tensor core MMA (tcgen05) or lower precision paths."
                )
        if roofline_summary.arithmetic_intensity < roofline_summary.ridge_point * 0.8:
            recommendations.append(
                "Arithmetic intensity below ridge point; restructure data movement to perform more FLOPs per byte."
            )
        if roofline_summary.compute_utilization_pct < 40:
            recommendations.append(
                "Compute utilisation <40%; experiment with launch bounds, register capping, or persistent CTAs."
            )

    if occupancy and occupancy < 50:
        recommendations.append(
            "Achieved occupancy below 50%; reduce register/shared-memory pressure or increase block size."
        )
    if warp_exec and warp_exec < 70:
        recommendations.append(
            "Warp execution efficiency under 70%; eliminate divergence and ensure coalesced memory patterns."
        )
    if l2_hit and l2_hit < 60:
        recommendations.append(
            "L2 hit rate below 60%; add blocking, use cp.async/TMA, or stage data in shared memory."
        )
    if not recommendations:
        recommendations.append("No major bottlenecks detected; kernel appears well optimised.")

    return Advisory(
        kernel=metrics.name,
        precision=precision,
        duration_ms=duration_ms,
        flops=flops,
        bytes_transferred=bytes_transferred,
        roofline=roofline_summary,
        sm_util_pct=sm_util,
        dram_util_pct=dram_util,
        tmem_util_pct=tmem_util,
        occupancy_pct=occupancy,
        tensor_util_pct=tensor_util,
        warp_exec_pct=warp_exec,
        l2_hit_pct=l2_hit,
        recommendations=recommendations,
    )


def summarise_nsys(report_path: Optional[Path], top_k: int) -> Optional[Dict[str, object]]:
    if report_path is None or NsightSystemsProfiler is None:
        return None
    try:
        summary = NsightSystemsProfiler.summarize_report(
            str(report_path),
            top_k=top_k,
            print_summary=False,
        )
    except Exception as exc:  # pragma: no cover - nsys may be unavailable
        return {"error": f"Failed to parse Nsight Systems report: {exc}"}
    return {
        "report": summary["report"],
        "kernels": summary["kernels"],
    }


def render_table(advisories: Sequence[Advisory], top_k: int) -> str:
    lines = []
    header = [
        "Kernel",
        "Dur (ms)",
        "TFLOPS",
        "GB/s",
        "AI",
        "SM %",
        "TMEM %",
        "HBM %",
        "Occ %",
        "Advice",
    ]
    widths = [len(h) for h in header]
    table_rows = []
    for advisory in advisories[:top_k]:
        roof = advisory.roofline
        ai = roof.arithmetic_intensity if roof else None
        row = [
            advisory.kernel,
            f"{advisory.duration_ms:.3f}" if advisory.duration_ms is not None else "n/a",
            f"{roof.achieved_tflops:.2f}" if roof else "n/a",
            f"{roof.achieved_bandwidth_gbs:.1f}" if roof else "n/a",
            f"{ai:.2f}" if ai is not None else "n/a",
            f"{advisory.sm_util_pct:.1f}" if advisory.sm_util_pct is not None else "n/a",
            f"{advisory.tmem_util_pct:.1f}" if advisory.tmem_util_pct is not None else "n/a",
            f"{advisory.dram_util_pct:.1f}" if advisory.dram_util_pct is not None else "n/a",
            f"{advisory.occupancy_pct:.1f}" if advisory.occupancy_pct is not None else "n/a",
            advisory.recommendations[0] if advisory.recommendations else "",
        ]
        widths = [max(w, len(cell)) for w, cell in zip(widths, row)]
        table_rows.append(row)

    def fmt(row: Sequence[str]) -> str:
        return " | ".join(cell.ljust(width) for cell, width in zip(row, widths))

    lines.append(fmt(header))
    lines.append("-+-".join("-" * width for width in widths))
    for row in table_rows:
        lines.append(fmt(row))
    return "\n".join(lines)


def aggregate_stats(advisories: Sequence[Advisory]) -> Dict[str, object]:
    ai_values = [adv.roofline.arithmetic_intensity for adv in advisories if adv.roofline]
    compute_util = [adv.roofline.compute_utilization_pct for adv in advisories if adv.roofline]
    memory_util = [adv.roofline.memory_utilization_pct for adv in advisories if adv.roofline]
    tmem_util = [adv.roofline.tmem_utilization_pct for adv in advisories if adv.roofline and adv.roofline.tmem_utilization_pct is not None]
    return {
        "kernel_count": len(advisories),
        "mean_arithmetic_intensity": statistics.mean(ai_values) if ai_values else None,
        "median_arithmetic_intensity": statistics.median(ai_values) if ai_values else None,
        "mean_compute_util_pct": statistics.mean(compute_util) if compute_util else None,
        "mean_memory_util_pct": statistics.mean(memory_util) if memory_util else None,
        "mean_tmem_util_pct": statistics.mean(tmem_util) if tmem_util else None,
        "memory_bound_kernels": [
            adv.kernel for adv in advisories if adv.roofline and adv.roofline.is_memory_bound
        ],
        "compute_bound_kernels": [
            adv.kernel for adv in advisories if adv.roofline and adv.roofline.is_compute_bound
        ],
        "tmem_bound_kernels": [
            adv.kernel for adv in advisories if adv.roofline and adv.roofline.is_tmem_bound
        ],
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deep profiling report generator.")
    parser.add_argument(
        "--ncu-csv",
        action="append",
        type=Path,
        help="Nsight Compute CSV export (can be provided multiple times).",
    )
    parser.add_argument(
        "--nsys-report",
        type=Path,
        help="Optional Nsight Systems .nsys-rep to identify top kernels.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write machine-readable summary.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Limit output to top-k kernels (default: 5).",
    )
    parser.add_argument(
        "--print-markdown",
        action="store_true",
        help="Render a Markdown table instead of plain text.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.ncu_csv:
        print("Provide at least one --ncu-csv export from Nsight Compute.", file=sys.stderr)
        return 1

    kernel_maps = []
    for csv_path in args.ncu_csv:
        if not csv_path.exists():
            print(f"[warn] Missing Nsight Compute CSV: {csv_path}", file=sys.stderr)
            continue
        kernel_maps.append(parse_ncu_csv(csv_path))
    if not kernel_maps:
        print("No Nsight Compute CSV files could be parsed.", file=sys.stderr)
        return 2

    merged = merge_kernel_metrics(kernel_maps)
    advisories = sorted(
        (build_advisory(metrics) for metrics in merged.values()),
        key=lambda adv: (
            -(adv.duration_ms if adv.duration_ms is not None else -math.inf),
            adv.kernel,
        ),
    )

    if args.print_markdown:
        print("```markdown")
    print(render_table(advisories, args.top_k))
    if args.print_markdown:
        print("```")
    print()

    if args.nsys_report:
        systems_summary = summarise_nsys(args.nsys_report, args.top_k)
        if systems_summary and "error" not in systems_summary:
            print("Nsight Systems top kernels:")
            for idx, kernel in enumerate(systems_summary["kernels"][: args.top_k], 1):
                name = kernel.get("Name", "kernel")
                pct = kernel.get("Time (%)") or kernel.get("Time (%) [sum]", "0")
                print(f"  {idx}. {name} ({pct}%)")
        elif systems_summary:
            print(systems_summary["error"])
        print()

    if args.output_json:
        summary = {
            "advisories": [
                {
                    "kernel": adv.kernel,
                    "precision": adv.precision,
                    "duration_ms": adv.duration_ms,
                    "flops": adv.flops,
                    "bytes_transferred": adv.bytes_transferred,
                    "roofline": None
                    if adv.roofline is None
                    else {
                        "achieved_tflops": adv.roofline.achieved_tflops,
                        "achieved_bandwidth_gbs": adv.roofline.achieved_bandwidth_gbs,
                        "arithmetic_intensity": adv.roofline.arithmetic_intensity,
                        "compute_utilization_pct": adv.roofline.compute_utilization_pct,
                        "memory_utilization_pct": adv.roofline.memory_utilization_pct,
                        "tmem_utilization_pct": adv.roofline.tmem_utilization_pct,
                        "l2_utilization_pct": adv.roofline.l2_utilization_pct,
                        "binding": adv.roofline.binding,
                        "is_memory_bound": adv.roofline.is_memory_bound,
                        "is_compute_bound": adv.roofline.is_compute_bound,
                        "is_tmem_bound": adv.roofline.is_tmem_bound,
                        "ridge_point": adv.roofline.ridge_point,
                        "memory_bound_limit_tflops": adv.roofline.memory_bound_limit_tflops,
                        "peak_tflops": adv.roofline.peak_tflops,
                        "peak_bandwidth_gbs": adv.roofline.peak_bandwidth_gbs,
                    },
                    "sm_util_pct": adv.sm_util_pct,
                    "dram_util_pct": adv.dram_util_pct,
                    "tmem_util_pct": adv.tmem_util_pct,
                    "occupancy_pct": adv.occupancy_pct,
                    "tensor_util_pct": adv.tensor_util_pct,
                    "warp_execution_pct": adv.warp_exec_pct,
                    "l2_hit_pct": adv.l2_hit_pct,
                    "recommendations": adv.recommendations,
                }
                for adv in advisories
            ],
            "stats": aggregate_stats(advisories),
        }
        if args.nsys_report:
            summary["nsight_systems"] = summarise_nsys(args.nsys_report, args.top_k)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2))
        print(f"Wrote JSON summary to {args.output_json}")

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
