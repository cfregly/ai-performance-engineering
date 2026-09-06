"""
Shared profiling insights and heuristics used across dashboard, CLI, and MCP.

These helpers stay free of HTTP/handler state and operate on plain data
structures so they can be reused by any interface.
"""

from __future__ import annotations

import csv
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.profiling.ncu_summary import (
    _is_ncu_range_record,
    _ncu_nvtx_range_metadata,
    _range_metrics_from_model,
)

_BASELINE_ROLE_TOKENS = {"baseline", "base", "before", "reference", "ref"}
_OPTIMIZED_ROLE_TOKENS = {"optimized", "opt", "after", "tuned", "candidate"}
_ROLE_TOKENS = _BASELINE_ROLE_TOKENS | _OPTIMIZED_ROLE_TOKENS
_PAIR_MANIFEST_FILENAME = "pair_manifest.json"
_NCU_COMPARISON_METRICS = (
    "gpu__time_duration.avg",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
)
_DEFAULT_KERNEL_ALIAS_GROUPS: Dict[str, set[str]] = {
    # Alias families for structurally similar kernels that use different codegen
    # stacks (for example, CUTLASS vs vendor kernels).
    "sm100_fp4_gemm_family": {
        "cutlass",
        "cutlass3x",
        "nvjet",
        "gemm",
        "bstensorop",
        "ue4m3",
        "vec16ue4m3",
        "tnt",
        "tnn",
    }
}


def _materialize_profile_if_needed(path: Path, *, root: Optional[Path] = None) -> Path:
    """Copy symlinked profile artifacts to real files for stable downstream tooling."""
    try:
        if not path.is_symlink():
            return path
        resolved = path.resolve(strict=True)
    except Exception:
        return path

    materialized_root = (root or path.parent) / ".materialized_profiles"
    try:
        materialized_root.mkdir(parents=True, exist_ok=True)
    except Exception:
        return path

    suffix = "".join(path.suffixes)
    stem = path.name[: -len(suffix)] if suffix else path.name
    digest_input = f"{path.absolute()}::{resolved}".encode("utf-8", errors="ignore")
    digest = hashlib.sha1(digest_input).hexdigest()[:12]
    dst_name = f"{stem}__{digest}{suffix}"
    dst = materialized_root / dst_name
    try:
        if not dst.exists() or resolved.stat().st_mtime > dst.stat().st_mtime:
            shutil.copy2(resolved, dst)
        return dst
    except Exception:
        return path


def _stage_profile_pair(
    baseline_path: Path,
    optimized_path: Path,
    *,
    root: Path,
    label: str,
) -> tuple[Path, Path]:
    """Copy a selected pair into a concrete baseline/optimized staging directory."""
    # Prefer concrete files for comparison tools; symlink-heavy layouts can make
    # auto-pairing and downstream report generation brittle.
    baseline_src = _materialize_profile_if_needed(baseline_path, root=root)
    optimized_src = _materialize_profile_if_needed(optimized_path, root=root)

    baseline_suffix = "".join(baseline_path.suffixes) or baseline_path.suffix
    optimized_suffix = "".join(optimized_path.suffixes) or optimized_path.suffix
    digest_src = (
        f"{baseline_path.absolute()}::{optimized_path.absolute()}::"
        f"{baseline_src.absolute()}::{optimized_src.absolute()}::{label}"
    )
    digest = hashlib.sha1(digest_src.encode("utf-8", errors="ignore")).hexdigest()[:12]
    stage_dir = root / ".staged_profile_pairs" / f"pair_{digest}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    baseline_dst = stage_dir / f"baseline{baseline_suffix}"
    optimized_dst = stage_dir / f"optimized{optimized_suffix}"

    try:
        if not baseline_dst.exists() or baseline_src.stat().st_mtime > baseline_dst.stat().st_mtime:
            shutil.copy2(baseline_src, baseline_dst)
    except Exception:
        baseline_dst = baseline_src

    try:
        if not optimized_dst.exists() or optimized_src.stat().st_mtime > optimized_dst.stat().st_mtime:
            shutil.copy2(optimized_src, optimized_dst)
    except Exception:
        optimized_dst = optimized_src

    manifest_payload = {
        "schema_version": "1.0",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "label": label,
        "pair_status": (
            "complete"
            if baseline_dst.exists() and optimized_dst.exists()
            else "partial"
        ),
        "staged_pair_dir": str(stage_dir),
        "artifacts": {
            "baseline": {
                "source_path": str(baseline_path),
                "materialized_source_path": str(baseline_src),
                "staged_path": str(baseline_dst),
                "exists": baseline_dst.exists(),
            },
            "optimized": {
                "source_path": str(optimized_path),
                "materialized_source_path": str(optimized_src),
                "staged_path": str(optimized_dst),
                "exists": optimized_dst.exists(),
            },
        },
    }
    _write_pair_manifest(stage_dir / _PAIR_MANIFEST_FILENAME, manifest_payload)

    return baseline_dst, optimized_dst


def _write_pair_manifest(path: Path, payload: Dict[str, Any]) -> None:
    """Best-effort write of a stable pair manifest JSON."""
    try:
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    except Exception:
        return


def _load_kernel_alias_groups() -> Dict[str, set[str]]:
    """Return kernel alias groups with optional env override."""
    env_payload = os.environ.get("AISP_KERNEL_ALIAS_GROUPS_JSON", "").strip()
    if not env_payload:
        return _DEFAULT_KERNEL_ALIAS_GROUPS
    try:
        parsed = json.loads(env_payload)
        if not isinstance(parsed, dict):
            return _DEFAULT_KERNEL_ALIAS_GROUPS
        normalized: Dict[str, set[str]] = {}
        for group_name, values in parsed.items():
            if not isinstance(group_name, str):
                continue
            if not isinstance(values, list):
                continue
            tokens = {
                str(value).strip().lower()
                for value in values
                if str(value).strip()
            }
            if tokens:
                normalized[group_name.strip().lower()] = tokens
        return normalized or _DEFAULT_KERNEL_ALIAS_GROUPS
    except Exception:
        return _DEFAULT_KERNEL_ALIAS_GROUPS


def _extract_ncu_sources(ncu_path: Path, limit: int = 12) -> List[Dict[str, str]]:
    """
    Best-effort extraction of source/line info from an ncu-rep.
    Parses the Source page CSV and keeps the first few entries with a Source column.
    """
    try:
        result = subprocess.run(
            ["ncu", "--import", str(ncu_path), "--csv", "--page", "source"],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if result.returncode != 0:
            return []
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        kernel_name: Optional[str] = None
        header: Optional[List[str]] = None
        samples: List[Dict[str, str]] = []
        for row in csv.reader(lines):
            if not row:
                continue
            if row[0] == "Kernel Name":
                kernel_name = row[1] if len(row) > 1 else kernel_name
                header = None
                continue
            if "Source" in row and "Address" in row:
                header = row
                continue
            if header and len(row) >= len(header):
                entry = dict(zip(header, row))
                source = entry.get("Source", "").strip()
                if not source:
                    continue
                samples.append(
                    {
                        "kernel": kernel_name or "",
                        "source": source,
                        "address": entry.get("Address", "").strip(),
                        "stall": entry.get("Warp Stall Sampling (All Samples)", "").strip(),
                    }
                )
                if len(samples) >= limit:
                    break
        return samples
    except Exception:
        return []


def _extract_ncu_disassembly(ncu_path: Path, limit: int = 40) -> List[str]:
    """
    Best-effort disassembly from an ncu-rep via the Source page (SASS view).
    Returns a list of text lines (truncated) for quick inspection.
    """
    try:
        result = subprocess.run(
            ["ncu", "--import", str(ncu_path), "--csv", "--page", "source", "--print-source", "sass"],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if result.returncode != 0:
            return []
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        payload: List[str] = []
        header: Optional[List[str]] = None
        for row in csv.reader(lines):
            if not row:
                continue
            if row[0] == "Kernel Name":
                header = None
                continue
            if "Address" in row and "Source" in row:
                header = row
                continue
            if header and len(row) >= 2:
                address = row[0].strip()
                source = row[1].strip()
                if not source:
                    continue
                payload.append(f"{address} {source}".strip())
                if len(payload) >= limit:
                    break
        return payload
    except Exception:
        return []


def _extract_nsys_sources(nsys_path: Path, limit: int = 12) -> List[Dict[str, Any]]:
    """
    Best-effort extraction of source/line info from an nsys-rep by exporting trace JSON.
    Broadened to capture CPU/GPU/NVTX/driver events with any file/line/symbol hints.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="nsys_src_") as tmp:
            out_prefix = Path(tmp) / "trace"
            export_cmd = [
                "nsys",
                "export",
                "--type",
                "timeline",  # richer than stats; still JSON
                "--format",
                "json",
                "--force",
                "true",
                "--output",
                str(out_prefix),
                str(nsys_path),
            ]
            subprocess.run(export_cmd, check=True, capture_output=True, text=True, timeout=90)
            json_path = out_prefix.with_suffix(".json")
            if not json_path.exists():
                return []
            data = json.loads(json_path.read_text())
            events = data if isinstance(data, list) else data.get("traceEvents", [])
            samples: List[Dict[str, Any]] = []
            seen = set()
            for ev in events:
                if not isinstance(ev, dict):
                    continue
                cat = (ev.get("cat") or "").lower()
                # Broader cat coverage: cuda, gpu, cpu, nvtx, driver/cu/rt, cupti
                if not any(tok in cat for tok in ("cuda", "gpu", "cpu", "nvtx", "driver", "cupti")):
                    continue
                args = ev.get("args") or {}
                file = ev.get("file") or args.get("file") or args.get("Source File")
                line = ev.get("line") or args.get("line") or args.get("Source Line")
                symbol = args.get("External Symbol") or args.get("Symbol") or ev.get("name")
                key = (file, line, symbol, ev.get("pid"), ev.get("tid"))
                if key in seen:
                    continue
                seen.add(key)
                samples.append(
                    {
                        "name": ev.get("name", ""),
                        "cat": ev.get("cat", ""),
                        "file": file,
                        "line": line,
                        "symbol": symbol,
                        "ts": ev.get("ts"),
                        "dur": ev.get("dur"),
                        "pid": ev.get("pid"),
                        "tid": ev.get("tid"),
                    }
                )
                if len(samples) >= limit:
                    break
            return samples
    except Exception:
        return []


def detect_bottlenecks(
    flame_data: Optional[Dict[str, Any]],
    kernel_data: Optional[Dict[str, Any]],
    hw_caps: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Analyze profile data to detect performance bottlenecks."""
    flame_data = flame_data or {}
    kernel_data = kernel_data or {}
    hw_caps = hw_caps or {}

    bottlenecks: List[Dict[str, Any]] = []
    total_time = float(flame_data.get("value") or 0)
    if total_time <= 0:
        return {"bottlenecks": [], "message": "No profile data available"}

    features = hw_caps.get("features", []) or []
    feature_names = [f.get("name", "") for f in features if isinstance(f, dict)]

    # Analyze by category
    for category in flame_data.get("children", []):
        cat_name = category.get("name", "")
        cat_time = float(category.get("value") or 0)
        cat_pct = (cat_time / total_time * 100) if total_time else 0.0

        # Memory transfer bottleneck
        if cat_name in ["gpu_memcpy", "cuda_runtime"] and cat_pct > 10:
            bottlenecks.append(
                {
                    "type": "memory_transfer",
                    "severity": "high" if cat_pct > 20 else "medium",
                    "category": cat_name,
                    "time_us": cat_time,
                    "percentage": round(cat_pct, 1),
                    "description": f"Memory transfers taking {cat_pct:.1f}% of execution time",
                    "suggestions": [
                        "Use pinned memory for host-to-device transfers",
                        "Overlap transfers with compute using CUDA streams",
                        "Consider using TMA for async memory operations"
                        if any(name.startswith("TMA") for name in feature_names)
                        else "Batch small transfers into larger ones",
                        "Use cudaMemcpyAsync instead of synchronous copies",
                    ],
                    "potential_speedup": f"{cat_pct * 0.5:.0f}% reduction possible",
                }
            )

        # CPU overhead bottleneck
        if cat_name == "python_function" and cat_pct > 15:
            bottlenecks.append(
                {
                    "type": "cpu_overhead",
                    "severity": "high" if cat_pct > 30 else "medium",
                    "category": cat_name,
                    "time_us": cat_time,
                    "percentage": round(cat_pct, 1),
                    "description": f"Python/CPU overhead taking {cat_pct:.1f}% of execution time",
                    "suggestions": [
                        "Use torch.compile() to fuse operations",
                        "Enable CUDA graphs for repeated kernel launches",
                        "Reduce Python-level loops - vectorize operations",
                        "Use torch.jit.script for hot paths",
                    ],
                    "potential_speedup": f"{cat_pct * 0.7:.0f}% reduction possible",
                }
            )

        # Profiler overhead (informational)
        if cat_name == "overhead" and cat_pct > 5:
            bottlenecks.append(
                {
                    "type": "profiler_overhead",
                    "severity": "info",
                    "category": cat_name,
                    "time_us": cat_time,
                    "percentage": round(cat_pct, 1),
                    "description": f"Profiler overhead is {cat_pct:.1f}% - actual performance is better",
                    "suggestions": [
                        "This overhead is from profiling only",
                        f"Real execution will be faster by ~{cat_pct:.0f}%",
                    ],
                    "potential_speedup": "N/A - profiling artifact",
                }
            )

    # Analyze individual kernels for optimization opportunities
    kernels = kernel_data.get("kernels", []) or []
    total_kernel_time = float(kernel_data.get("summary", {}).get("total_time_us", 1) or 1)

    for kernel in kernels[:5]:  # Top 5 kernels
        k_name = kernel.get("name", "")
        k_time = float(kernel.get("time_us", 0) or 0)
        k_pct = (k_time / total_kernel_time * 100) if total_kernel_time else 0.0

        if k_pct > 15:  # Significant kernel
            suggestions: List[str] = []
            lower_name = k_name.lower()

            # Pattern-based suggestions
            if any(term in lower_name for term in ["gemm", "matmul", "mm_"]):
                suggestions.extend(
                    [
                        "Ensure matrix dimensions are multiples of 16",
                        "Consider using cuBLAS or cuBLASLt for GEMM",
                        "Try FP8 for inference if accuracy permits"
                        if (hw_caps.get("architecture") or "").lower().startswith("blackwell")
                        else "Use TF32 for faster FP32 operations",
                    ]
                )
            elif "conv" in lower_name:
                suggestions.extend(
                    [
                        "Use cuDNN with autotuning enabled",
                        "Consider channels-last memory format (NHWC)",
                        "Try depthwise-separable convolutions if applicable",
                    ]
                )
            elif any(term in lower_name for term in ["softmax", "attention", "sdpa"]):
                suggestions.extend(
                    [
                        "Use Flash Attention for memory efficiency",
                        "Consider fused attention kernels",
                        "Use scaled_dot_product_attention with efficient backend",
                    ]
                )
            elif any(term in lower_name for term in ["copy", "memcpy", "_to_"]):
                suggestions.extend(
                    [
                        "Minimize data movement between CPU and GPU",
                        "Use in-place operations where possible",
                        "Consider memory layout optimization",
                    ]
                )
            else:
                suggestions.extend(
                    [
                        "Profile with NCU for detailed metrics",
                        "Check occupancy and register usage",
                        "Consider kernel fusion opportunities",
                    ]
                )

            bottlenecks.append(
                {
                    "type": "hot_kernel",
                    "severity": "high" if k_pct > 25 else "medium",
                    "category": "kernel",
                    "kernel_name": k_name,
                    "time_us": k_time,
                    "percentage": round(k_pct, 1),
                    "description": f"Kernel '{k_name[:50]}' taking {k_pct:.1f}% of GPU time",
                    "suggestions": suggestions,
                    "potential_speedup": "Varies by optimization",
                }
            )

    severity_order = {"high": 0, "medium": 1, "info": 2}
    bottlenecks.sort(key=lambda x: severity_order.get(x.get("severity", "info"), 3))

    return {
        "bottlenecks": bottlenecks,
        "total_time_us": total_time,
        "hardware": {
            "name": hw_caps.get("gpu", {}).get("name", "Unknown"),
            "architecture": hw_caps.get("architecture", "unknown"),
            "features_available": len(features),
        },
    }


def calculate_optimization_score(
    hw_caps: Dict[str, Any], bottlenecks: Dict[str, Any], kernel_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Calculate an optimization opportunity score based on profile analysis."""
    hw_caps = hw_caps or {}
    bottlenecks = bottlenecks or {}
    kernel_data = kernel_data or {}

    score = 100  # Start with perfect score
    deductions: List[str] = []
    quick_wins: List[Dict[str, Any]] = []
    advanced_opts: List[Dict[str, Any]] = []
    already_optimized: List[str] = []

    # Check for bottlenecks and deduct points
    for bottleneck in bottlenecks.get("bottlenecks", []) or []:
        severity = bottleneck.get("severity")
        if severity == "high":
            score -= 15
            deductions.append(f"{bottleneck.get('type')}: -15 pts")
        elif severity == "medium":
            score -= 8
            deductions.append(f"{bottleneck.get('type')}: -8 pts")

    # Check for unused hardware features
    for feature in hw_caps.get("features", []) or []:
        if not isinstance(feature, dict) or not feature.get("supported"):
            continue
        name = feature.get("name", "")
        opt_text = feature.get("optimization", "")
        if "TMA" in name:
            quick_wins.append(
                {
                    "feature": name,
                    "potential": "+15% speedup",
                    "effort": "medium",
                    "description": opt_text,
                }
            )
        elif "FP8" in name:
            quick_wins.append(
                {
                    "feature": name,
                    "potential": "+25% throughput",
                    "effort": "low",
                    "description": opt_text,
                }
            )
        elif "Cluster" in name:
            advanced_opts.append(
                {
                    "feature": name,
                    "potential": "+10% speedup",
                    "effort": "high",
                    "description": opt_text,
                }
            )
        elif "DSMEM" in name:
            advanced_opts.append(
                {
                    "feature": name,
                    "potential": "+5-15% speedup",
                    "effort": "high",
                    "description": opt_text,
                }
            )

    # Check what's already optimized (look for patterns in kernel names)
    kernels = kernel_data.get("kernels", []) or []
    kernel_names = [k.get("name", "").lower() for k in kernels]

    if any("flash" in k for k in kernel_names):
        already_optimized.append("Flash Attention enabled")
        score += 5
    if any("fused" in k for k in kernel_names):
        already_optimized.append("Kernel fusion detected")
        score += 3
    if any("cudnn" in k for k in kernel_names):
        already_optimized.append("cuDNN optimized kernels")
        score += 3

    score = max(0, min(100, score))  # Clamp to 0-100

    return {
        "score": score,
        "grade": "A"
        if score >= 90
        else "B"
        if score >= 75
        else "C"
        if score >= 60
        else "D"
        if score >= 40
        else "F",
        "deductions": deductions,
        "quick_wins": quick_wins,
        "advanced_optimizations": advanced_opts,
        "already_optimized": already_optimized,
        "summary": {
            "quick_win_count": len(quick_wins),
            "advanced_opt_count": len(advanced_opts),
            "already_optimized_count": len(already_optimized),
        },
    }


def compare_nsys_files(
    profiles_dir: Path,
    pair_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Extract and compare nsys metrics between baseline and optimized."""
    pair_health = assess_profile_pair_health(profiles_dir, pair_key=pair_key)
    baseline_nsys, optimized_nsys = _collect_profile_role_files(profiles_dir, ".nsys-rep")

    if not baseline_nsys or not optimized_nsys:
        pair_dir, error = _select_pair_dir(profiles_dir, pair_key, label="nsys")
        if error:
            return {**error, "pair_health": pair_health}
        if not pair_dir:
            return None
        baseline_nsys, optimized_nsys = _collect_profile_role_files(pair_dir, ".nsys-rep")
        if not baseline_nsys or not optimized_nsys:
            return None

    pair, selected_pair_key, error = _select_profile_pair(
        baseline_nsys,
        optimized_nsys,
        pair_key=pair_key,
        label="nsys",
    )
    if error:
        return {**error, "pair_health": pair_health}
    if not pair:
        return None

    baseline_path, optimized_path = pair
    baseline_path, optimized_path = _stage_profile_pair(
        baseline_path,
        optimized_path,
        root=profiles_dir,
        label="nsys",
    )

    try:
        from core.profiling.extract_nsys_summary import harvest

        baseline_metrics = harvest(baseline_path)
        optimized_metrics = harvest(optimized_path)

        comparison = {
            "baseline_file": baseline_path.name,
            "optimized_file": optimized_path.name,
            "baseline_path": str(baseline_path),
            "optimized_path": str(optimized_path),
            "staged_pair_dir": str(baseline_path.parent),
            "pair_key": pair_key or selected_pair_key,
            "pair_health": pair_health,
            "metrics": [],
            "baseline_sources": _extract_nsys_sources(baseline_path),
            "optimized_sources": _extract_nsys_sources(optimized_path),
        }

        opt_lookup = {m.get("metric", ""): m.get("value", "") for m in optimized_metrics}

        for base_metric in baseline_metrics:
            metric = base_metric.get("metric", "")
            baseline_val = base_metric.get("value", "")
            optimized_val = opt_lookup.get(metric, "")

            delta = None
            delta_pct = None
            try:
                b_num = float(str(baseline_val).replace(",", ""))
                o_num = float(str(optimized_val).replace(",", ""))
                delta = o_num - b_num
                delta_pct = ((o_num - b_num) / b_num * 100) if b_num != 0 else 0
            except (ValueError, TypeError):
                pass

            comparison["metrics"].append(
                {
                    "name": metric,
                    "baseline": baseline_val,
                    "optimized": optimized_val,
                    "delta": delta,
                    "delta_pct": delta_pct,
                }
            )

        return comparison
    except Exception as exc:  # pragma: no cover - best-effort helper
        return {"error": str(exc)}


def _ncu_time_to_ms(value: float, unit: str) -> float:
    normalized = (unit or "").strip().lower()
    if normalized.endswith("ns"):
        return value / 1e6
    if normalized.endswith("us"):
        return value / 1e3
    if normalized.endswith("ms"):
        return value
    if normalized.endswith("s"):
        return value * 1e3
    return value


def _read_ncu_csv_measurement(path: Path) -> Dict[str, Any]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))

    range_rows = [row for row in rows if _is_ncu_range_record(row)]
    numeric_rows = [row for row in rows if str(row.get("ID") or "").strip().isdigit()]
    if range_rows:
        non_range_rows = [row for row in numeric_rows if not _is_ncu_range_record(row)]
        if non_range_rows:
            raise ValueError("NCU CSV mixes range and kernel result rows")
        result_ids = {str(row.get("ID") or "").strip() for row in range_rows}
        ranges = {_ncu_nvtx_range_metadata(row)[0] for row in range_rows}
        ranges.discard(None)
        if len(result_ids) != 1 or len(ranges) != 1:
            raise ValueError("NCU range CSV must describe exactly one selected NVTX range")

        metrics: Dict[str, float] = {}
        metric_units: Dict[str, str] = {}
        has_tall_metrics = any((row.get("Metric Name") or "").strip() for row in range_rows)
        if has_tall_metrics:
            for row in range_rows:
                name = (row.get("Metric Name") or "").strip()
                if name not in _NCU_COMPARISON_METRICS:
                    continue
                value = _safe_float((row.get("Metric Value") or "").strip().rstrip("%"))
                if value is None:
                    continue
                unit = (row.get("Metric Unit") or "").strip()
                if name.startswith("gpu__time_duration"):
                    value = _ncu_time_to_ms(value, unit)
                if name in metrics and metrics[name] != value:
                    raise ValueError(f"NCU range CSV disagrees on metric {name!r}")
                metrics[name] = value
                metric_units[name] = unit
            source_format = "ncu_details_csv"
        else:
            units_row = next(
                (
                    row
                    for row in rows
                    if not str(row.get("ID") or "").strip()
                    and not str(row.get("Kernel Name") or "").strip()
                ),
                {},
            )
            row = range_rows[0]
            for name in _NCU_COMPARISON_METRICS:
                value = _safe_float((row.get(name) or "").strip().rstrip("%"))
                if value is None:
                    continue
                unit = (units_row.get(name) or "").strip()
                if name.startswith("gpu__time_duration"):
                    value = _ncu_time_to_ms(value, unit)
                metrics[name] = value
                metric_units[name] = unit
            source_format = "ncu_raw_csv"

        if not metrics:
            raise ValueError("NCU range CSV contains no supported numeric metrics")
        range_row = range_rows[0]
        nvtx_range, nvtx_range_raw = _ncu_nvtx_range_metadata(range_row)
        provenance = {
            "schema_version": "1.0",
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "replay_mode": None,
            "result_scope": "range",
            "result_id": int(next(iter(result_ids))),
            "result_count": 1,
            "nvtx_range": nvtx_range,
            "nvtx_range_raw": nvtx_range_raw,
            "coverage_policy": "range_row_only_unverified",
            "requested_metrics": None,
            "observed_metrics": sorted(metrics),
            "metric_units": metric_units,
            "session_capture": None,
            "source_format": source_format,
        }
        return {
            "result_scope": "range",
            "metrics": metrics,
            "range_time_ms": metrics.get("gpu__time_duration.avg"),
            "provenance": provenance,
        }

    metrics: Dict[str, Any] = {}
    for row in rows:
        name = row.get("Metric Name", row.get("Name", ""))
        value = row.get("Metric Value", row.get("Avg", row.get("Value", "")))
        if name:
            metrics[name] = value
    return {"result_scope": "kernel", "metrics": metrics}


def _ncu_range_comparison(
    baseline: Dict[str, Any],
    optimized: Dict[str, Any],
) -> Dict[str, Any]:
    baseline_provenance = baseline["provenance"]
    optimized_provenance = optimized["provenance"]
    baseline_range = baseline_provenance.get("nvtx_range")
    optimized_range = optimized_provenance.get("nvtx_range")
    if baseline_range != optimized_range:
        raise ValueError(
            "NCU range pair selects different NVTX ranges: "
            f"{baseline_range!r} != {optimized_range!r}"
        )
    baseline_coverage = baseline_provenance.get("coverage_policy")
    optimized_coverage = optimized_provenance.get("coverage_policy")
    if baseline_coverage != optimized_coverage:
        raise ValueError(
            "NCU range pair uses different coverage policies: "
            f"{baseline_coverage!r} != {optimized_coverage!r}"
        )

    metric_rows: List[Dict[str, Any]] = []
    all_metrics = set(baseline["metrics"]) | set(optimized["metrics"])
    for name in sorted(all_metrics):
        baseline_value = baseline["metrics"].get(name)
        optimized_value = optimized["metrics"].get(name)
        is_range_time = name == "gpu__time_duration.avg"
        delta = None
        ratio = None
        if (
            not is_range_time
            and isinstance(baseline_value, (int, float))
            and isinstance(optimized_value, (int, float))
        ):
            delta = optimized_value - baseline_value
            ratio = optimized_value / baseline_value if baseline_value != 0 else None
        metric_rows.append(
            {
                "name": name,
                "baseline": baseline_value,
                "optimized": optimized_value,
                "delta": delta,
                "ratio": ratio,
                "comparison_scope": "range",
                "speedup_eligible": False if is_range_time else None,
            }
        )

    return {
        "nvtx_range": baseline_range,
        "coverage_policy": baseline_coverage,
        "baseline_range_time_ms": baseline.get("range_time_ms"),
        "optimized_range_time_ms": optimized.get("range_time_ms"),
        "canonical_speedup_eligible": False,
        "timing_semantics": "profiler selected-range duration; descriptive only",
        "metrics": metric_rows,
    }


def _inspect_ncu_range_measurement(path: Path, rows: List[Dict[str, str]]) -> Dict[str, Any]:
    numeric_rows = [row for row in rows if str(row.get("ID") or "").strip().isdigit()]
    range_rows = [row for row in numeric_rows if _is_ncu_range_record(row)]
    non_range_rows = [row for row in numeric_rows if not _is_ncu_range_record(row)]
    if non_range_rows:
        raise ValueError("NCU report mixes range and kernel result rows")
    result_ids = {str(row.get("ID") or "").strip() for row in range_rows}
    ranges = {_ncu_nvtx_range_metadata(row)[0] for row in range_rows}
    ranges.discard(None)
    if len(result_ids) != 1 or len(ranges) != 1:
        raise ValueError("NCU range report must describe exactly one selected NVTX range")

    from core.profiling.metrics_extractor import inspect_ncu_app_range_report

    expected_range = next(iter(ranges))
    metrics, provenance = inspect_ncu_app_range_report(
        path,
        expected_nvtx_range=expected_range,
        requested_metrics=_NCU_COMPARISON_METRICS,
        timeout=45,
    )
    if (
        provenance.get("result_scope") != "range"
        or getattr(metrics, "kernel_time_ms", None) is not None
        or getattr(metrics, "range_time_ms", None) is None
    ):
        raise ValueError("NCU range inspector returned inconsistent scope or timing semantics")
    return {
        "result_scope": "range",
        "metrics": _range_metrics_from_model(metrics),
        "range_time_ms": float(metrics.range_time_ms),
        "provenance": provenance,
    }


def compare_ncu_files(
    profiles_dir: Path,
    pair_key: Optional[str] = None,
    include_ncu_details: bool = False,
) -> Optional[Dict[str, Any]]:
    """Extract and compare ncu metrics between baseline and optimized."""
    pair_health = assess_profile_pair_health(profiles_dir, pair_key=pair_key)
    baseline_ncu, optimized_ncu = _collect_profile_role_files(profiles_dir, ".ncu-rep")
    baseline_csv, optimized_csv = _collect_profile_role_files(
        profiles_dir,
        ".csv",
        name_predicate=lambda p: "ncu" in p.name.lower(),
    )

    if (not baseline_ncu or not optimized_ncu) and (not baseline_csv or not optimized_csv):
        pair_dir, error = _select_pair_dir(profiles_dir, pair_key, label="ncu")
        if error:
            return {**error, "pair_health": pair_health}
        if pair_dir:
            baseline_ncu, optimized_ncu = _collect_profile_role_files(pair_dir, ".ncu-rep")
            baseline_csv, optimized_csv = _collect_profile_role_files(
                pair_dir,
                ".csv",
                name_predicate=lambda p: "ncu" in p.name.lower(),
            )

    if baseline_csv and optimized_csv:
        pair, selected_key, error = _select_profile_pair(
            baseline_csv,
            optimized_csv,
            pair_key=pair_key,
            label="ncu csv",
        )
        if error:
            return {**error, "pair_health": pair_health}
        if not pair:
            return None

        baseline_csv_path, optimized_csv_path = pair
        baseline_csv_path, optimized_csv_path = _stage_profile_pair(
            baseline_csv_path,
            optimized_csv_path,
            root=profiles_dir,
            label="ncu_csv",
        )
        selected_pair_key = pair_key or selected_key

        try:
            baseline_measurement = _read_ncu_csv_measurement(baseline_csv_path)
            optimized_measurement = _read_ncu_csv_measurement(optimized_csv_path)
            baseline_scope = baseline_measurement["result_scope"]
            optimized_scope = optimized_measurement["result_scope"]
            if baseline_scope != optimized_scope:
                return {
                    "error": (
                        "NCU baseline/optimized result scopes differ: "
                        f"{baseline_scope!r} != {optimized_scope!r}"
                    ),
                    "baseline_file": baseline_csv_path.name,
                    "optimized_file": optimized_csv_path.name,
                    "pair_health": pair_health,
                }

            if baseline_scope == "range":
                range_comparison = _ncu_range_comparison(
                    baseline_measurement, optimized_measurement
                )
                return {
                    "success": True,
                    "result_scope": "range",
                    "baseline_file": baseline_csv_path.name,
                    "optimized_file": optimized_csv_path.name,
                    "baseline_path": str(baseline_csv_path),
                    "optimized_path": str(optimized_csv_path),
                    "staged_pair_dir": str(baseline_csv_path.parent),
                    "pair_key": pair_key or selected_pair_key,
                    "pair_health": pair_health,
                    "baseline_provenance": baseline_measurement["provenance"],
                    "optimized_provenance": optimized_measurement["provenance"],
                    "range_comparison": range_comparison,
                }

            baseline_metrics = baseline_measurement["metrics"]
            optimized_metrics = optimized_measurement["metrics"]

            comparison = {
                "result_scope": "kernel",
                "baseline_file": baseline_csv_path.name,
                "optimized_file": optimized_csv_path.name,
                "baseline_path": str(baseline_csv_path),
                "optimized_path": str(optimized_csv_path),
                "staged_pair_dir": str(baseline_csv_path.parent),
                "pair_key": pair_key or selected_pair_key,
                "pair_health": pair_health,
                "metrics": [],
            }
            if include_ncu_details and baseline_ncu and optimized_ncu:
                rep_pair, _, rep_error = _select_profile_pair(
                    baseline_ncu,
                    optimized_ncu,
                    pair_key=selected_pair_key,
                    label="ncu",
                )
                if rep_error:
                    return {**rep_error, "pair_health": pair_health}
                if rep_pair:
                    baseline_rep, optimized_rep = rep_pair
                    baseline_rep, optimized_rep = _stage_profile_pair(
                        baseline_rep,
                        optimized_rep,
                        root=profiles_dir,
                        label="ncu_rep_details",
                    )
                    comparison["baseline_sources"] = _extract_ncu_sources(baseline_rep)
                    comparison["optimized_sources"] = _extract_ncu_sources(optimized_rep)
                    comparison["baseline_disassembly"] = _extract_ncu_disassembly(baseline_rep)
                    comparison["optimized_disassembly"] = _extract_ncu_disassembly(optimized_rep)

            all_keys = set(baseline_metrics.keys()) | set(optimized_metrics.keys())
            for key in sorted(all_keys):
                b_val = baseline_metrics.get(key, "")
                o_val = optimized_metrics.get(key, "")

                delta = None
                try:
                    b_num = float(str(b_val).replace(",", ""))
                    o_num = float(str(o_val).replace(",", ""))
                    delta = o_num - b_num
                except (ValueError, TypeError):
                    pass

                comparison["metrics"].append(
                    {"name": key, "baseline": b_val, "optimized": o_val, "delta": delta}
                )

            return comparison
        except Exception as exc:
            return {"error": str(exc)}

    if not baseline_ncu or not optimized_ncu:
        return None

    def _parse_float(text: str) -> Optional[float]:
        stripped = (text or "").strip().replace(",", "")
        if not stripped:
            return None
        if stripped.endswith("%"):
            stripped = stripped[:-1]
        match = re.match(r"^[+-]?\d*\.?\d+(?:[eE][+-]?\d+)?", stripped)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None

    def _time_to_ms(value: float, unit: str) -> float:
        u = (unit or "").strip().lower()
        if u.endswith("ns"):
            return value / 1e6
        if u.endswith("us"):
            return value / 1e3
        if u.endswith("ms"):
            return value
        if u.endswith("s"):
            return value * 1e3
        return value

    def _load_metrics(ncu_path: Path) -> Dict[str, Any]:
        metrics = list(_NCU_COMPARISON_METRICS)
        metric_aliases = {
            # Newer/older NCU "details" metric naming variants.
            "Duration": "gpu__time_duration.avg",
            "Compute (SM) Throughput": "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "DRAM Throughput": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "Memory Throughput": "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            "L2 Cache Throughput": "lts__throughput.avg.pct_of_peak_sustained_elapsed",
            "Achieved Occupancy": "sm__warps_active.avg.pct_of_peak_sustained_active",
        }
        accepted_metrics = set(metrics) | set(metric_aliases.keys())
        per_kernel: Dict[str, Dict[str, float]] = {}

        details = subprocess.run(
            ["ncu", "--import", str(ncu_path), "--csv", "--page", "details"],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if details.returncode == 0 and details.stdout.strip():
            rows = list(csv.DictReader(io.StringIO(details.stdout)))
            if any(_is_ncu_range_record(row) for row in rows):
                return _inspect_ncu_range_measurement(ncu_path, rows)
            for row in rows:
                kernel = (row.get("Kernel Name") or "").strip() or "kernel"
                metric = (row.get("Metric Name") or "").strip()
                value_raw = (row.get("Metric Value") or "").strip()
                if not metric or not value_raw:
                    continue
                normalized_metric = metric_aliases.get(metric, metric)
                if normalized_metric not in accepted_metrics:
                    continue
                value = _parse_float(value_raw)
                if value is None:
                    continue
                unit = (row.get("Metric Unit") or "").strip()
                if normalized_metric.startswith("gpu__time_duration"):
                    value = _time_to_ms(value, unit)
                per_kernel.setdefault(kernel, {})[normalized_metric] = value
            if per_kernel:
                return {"result_scope": "kernel", "per_kernel": per_kernel}

        # Some NCU captures do not populate the "details" page. Fallback to "raw".
        raw = subprocess.run(
            ["ncu", "--import", str(ncu_path), "--csv", "--page", "raw"],
            capture_output=True,
            text=True,
            timeout=45,
        )
        if raw.returncode != 0 or not raw.stdout.strip():
            return {"result_scope": "kernel", "per_kernel": {}}
        rows = list(csv.DictReader(io.StringIO(raw.stdout)))
        if any(_is_ncu_range_record(row) for row in rows):
            return _inspect_ncu_range_measurement(ncu_path, rows)
        for row in rows:
            kernel = (row.get("Kernel Name") or "").strip()
            if not kernel:
                continue
            for metric in metrics:
                value_raw = (row.get(metric) or "").strip()
                if not value_raw:
                    continue
                value = _parse_float(value_raw)
                if value is None:
                    continue
                per_kernel.setdefault(kernel, {})[metric] = value
        return {"result_scope": "kernel", "per_kernel": per_kernel}

    pair, selected_pair_key, error = _select_profile_pair(
        baseline_ncu,
        optimized_ncu,
        pair_key=pair_key,
        label="ncu",
    )
    if error:
        return {**error, "pair_health": pair_health}
    if not pair:
        return None

    baseline_path, optimized_path = pair
    baseline_path, optimized_path = _stage_profile_pair(
        baseline_path,
        optimized_path,
        root=profiles_dir,
        label="ncu",
    )

    try:
        baseline_measurement = _load_metrics(baseline_path)
        optimized_measurement = _load_metrics(optimized_path)
    except Exception as exc:  # pragma: no cover - tool availability varies
        return {"error": f"NCU extraction failed: {exc}"}

    baseline_scope = baseline_measurement["result_scope"]
    optimized_scope = optimized_measurement["result_scope"]
    if baseline_scope != optimized_scope:
        return {
            "error": (
                "NCU baseline/optimized result scopes differ: "
                f"{baseline_scope!r} != {optimized_scope!r}"
            ),
            "baseline_file": baseline_path.name,
            "optimized_file": optimized_path.name,
            "pair_health": pair_health,
        }
    if baseline_scope == "range":
        try:
            range_comparison = _ncu_range_comparison(
                baseline_measurement, optimized_measurement
            )
        except ValueError as exc:
            return {
                "error": str(exc),
                "baseline_file": baseline_path.name,
                "optimized_file": optimized_path.name,
                "pair_health": pair_health,
            }
        return {
            "success": True,
            "result_scope": "range",
            "baseline_file": baseline_path.name,
            "optimized_file": optimized_path.name,
            "baseline_path": str(baseline_path),
            "optimized_path": str(optimized_path),
            "staged_pair_dir": str(baseline_path.parent),
            "pair_key": pair_key or selected_pair_key,
            "pair_health": pair_health,
            "baseline_provenance": baseline_measurement["provenance"],
            "optimized_provenance": optimized_measurement["provenance"],
            "range_comparison": range_comparison,
        }

    baseline_per_kernel = baseline_measurement["per_kernel"]
    optimized_per_kernel = optimized_measurement["per_kernel"]

    if not baseline_per_kernel and not optimized_per_kernel:
        return {
            "error": (
                "No extractable NCU metrics found in either report "
                "(the capture may not include the requested metric pages)."
            ),
            "baseline_file": baseline_path.name,
            "optimized_file": optimized_path.name,
        }

    paired_kernels = _pair_kernel_metrics(baseline_per_kernel, optimized_per_kernel, limit=12)

    def _kernel_payload(pair: Dict[str, Any]) -> Dict[str, Any]:
        baseline_name = pair.get("baseline_kernel")
        optimized_name = pair.get("optimized_kernel")
        b = baseline_per_kernel.get(baseline_name, {}) if baseline_name else {}
        o = optimized_per_kernel.get(optimized_name, {}) if optimized_name else {}
        keys = set(b.keys()) | set(o.keys())
        diffs: Dict[str, Any] = {}
        for key in sorted(keys):
            bv = b.get(key)
            ov = o.get(key)
            delta = None
            ratio = None
            if isinstance(bv, (int, float)) and isinstance(ov, (int, float)):
                delta = ov - bv
                ratio = (ov / bv) if bv != 0 else None
            diffs[key] = {"baseline": bv, "optimized": ov, "delta": delta, "ratio": ratio}

        display_kernel = baseline_name or optimized_name or "kernel"
        if baseline_name and optimized_name and baseline_name != optimized_name:
            display_kernel = f"{baseline_name} => {optimized_name}"
        return {
            "kernel": display_kernel,
            "baseline_kernel": baseline_name,
            "optimized_kernel": optimized_name,
            "match_type": pair.get("match_type"),
            "match_score": pair.get("match_score"),
            "metrics": diffs,
        }

    exact_count = sum(1 for pair in paired_kernels if pair.get("match_type") == "exact")
    fuzzy_count = sum(1 for pair in paired_kernels if pair.get("match_type") == "fuzzy")
    rank_count = sum(1 for pair in paired_kernels if pair.get("match_type") == "rank")
    unmatched_count = sum(1 for pair in paired_kernels if pair.get("match_type") == "unmatched")

    comparison = {
        "result_scope": "kernel",
        "baseline_file": baseline_path.name,
        "optimized_file": optimized_path.name,
        "baseline_path": str(baseline_path),
        "optimized_path": str(optimized_path),
        "staged_pair_dir": str(baseline_path.parent),
        "pair_key": pair_key or selected_pair_key,
        "pair_health": pair_health,
        "kernel_pairing": {
            "paired_count": len(paired_kernels),
            "exact_count": exact_count,
            "fuzzy_count": fuzzy_count,
            "rank_count": rank_count,
            "unmatched_count": unmatched_count,
        },
        "kernel_comparison": [_kernel_payload(pair) for pair in paired_kernels],
    }
    alignment_valid = (exact_count + fuzzy_count) > 0
    if exact_count > 0:
        alignment_quality = "exact"
    elif fuzzy_count > 0:
        alignment_quality = "fuzzy"
    elif rank_count > 0:
        alignment_quality = "rank_only"
    elif unmatched_count > 0:
        alignment_quality = "unmatched_only"
    else:
        alignment_quality = "none"
    comparison["alignment_quality"] = alignment_quality
    comparison["alignment_valid_for_tuning"] = alignment_valid
    if exact_count == 0 and fuzzy_count == 0 and paired_kernels:
        comparison["pairing_warning"] = (
            "Kernel symbols did not align (rank-only/unmatched mapping). "
            "Treat per-kernel deltas as low-confidence; capture with tighter NVTX/kernel filters for decisive tuning."
        )
        comparison["low_confidence_for_tuning"] = True
        comparison["success"] = True
        comparison["advisory_warning"] = (
            "NCU kernel comparison alignment is low-confidence (rank-only/unmatched symbols). "
            "Per-kernel deltas should be treated as advisory."
        )
        comparison["hint"] = (
            "Use profile_ncu with kernel_filter + nvtx_include and compare a concrete baseline/optimized pair directory."
        )
    else:
        comparison["low_confidence_for_tuning"] = False
        comparison["success"] = True

    baseline_agg = _aggregate_ncu_metrics(baseline_per_kernel)
    optimized_agg = _aggregate_ncu_metrics(optimized_per_kernel)
    metric_keys = set(baseline_agg.get("weighted_metrics", {}).keys()) | set(
        optimized_agg.get("weighted_metrics", {}).keys()
    )
    aggregate_delta: Dict[str, Any] = {}
    for key in sorted(metric_keys):
        b_val = baseline_agg.get("weighted_metrics", {}).get(key)
        o_val = optimized_agg.get("weighted_metrics", {}).get(key)
        delta = None
        ratio = None
        if isinstance(b_val, (int, float)) and isinstance(o_val, (int, float)):
            delta = o_val - b_val
            ratio = (o_val / b_val) if b_val != 0 else None
        aggregate_delta[key] = {"baseline": b_val, "optimized": o_val, "delta": delta, "ratio": ratio}

    base_total = baseline_agg.get("total_gpu_time_ms")
    opt_total = optimized_agg.get("total_gpu_time_ms")
    time_speedup = None
    time_delta_ms = None
    if isinstance(base_total, (int, float)) and isinstance(opt_total, (int, float)):
        time_delta_ms = opt_total - base_total
        if opt_total > 0:
            time_speedup = base_total / opt_total

    comparison["aggregate"] = {
        "baseline": baseline_agg,
        "optimized": optimized_agg,
        "delta": aggregate_delta,
        "total_gpu_time_ms": {
            "baseline": base_total,
            "optimized": opt_total,
            "delta": time_delta_ms,
            "speedup": time_speedup,
        },
    }
    if include_ncu_details:
        comparison["baseline_sources"] = _extract_ncu_sources(baseline_path)
        comparison["optimized_sources"] = _extract_ncu_sources(optimized_path)
        comparison["baseline_disassembly"] = _extract_ncu_disassembly(baseline_path)
        comparison["optimized_disassembly"] = _extract_ncu_disassembly(optimized_path)
    return comparison


def _kernel_time_ms(per_kernel: Dict[str, Dict[str, float]], kernel_name: str) -> float:
    try:
        return float(per_kernel.get(kernel_name, {}).get("gpu__time_duration.avg", 0.0) or 0.0)
    except Exception:
        return 0.0


def _normalize_kernel_symbol(name: str) -> str:
    text = str(name or "")
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"\([^)]*\)", " ", text)
    text = text.replace("::", "_")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).lower()
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def _kernel_tokens(name: str) -> set[str]:
    stopwords = {
        "kernel",
        "void",
        "const",
        "unsigned",
        "int",
        "float",
        "double",
        "half",
        "device",
        "global",
    }
    tokens = {
        token
        for token in _normalize_kernel_symbol(name).split("_")
        if token and token not in stopwords
    }
    alias_groups = _load_kernel_alias_groups()
    for group_name, group_tokens in alias_groups.items():
        if tokens & group_tokens:
            tokens.add(group_name)
    return tokens


def _kernel_similarity(left: str, right: str) -> float:
    left_norm = _normalize_kernel_symbol(left)
    right_norm = _normalize_kernel_symbol(right)
    if not left_norm or not right_norm:
        return 0.0
    if left_norm == right_norm:
        return 2.0

    left_tokens = _kernel_tokens(left)
    right_tokens = _kernel_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0

    intersection = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    jaccard = (intersection / union) if union else 0.0
    alias_group_keys = set(_load_kernel_alias_groups().keys())
    if left_tokens & right_tokens & alias_group_keys:
        # Alias families indicate strong semantic relatedness even when
        # generated symbol names differ substantially (e.g., CUTLASS vs NVJET).
        jaccard += 0.4
    if left_norm in right_norm or right_norm in left_norm:
        jaccard += 0.15
    return jaccard


def _pair_kernel_metrics(
    baseline_per_kernel: Dict[str, Dict[str, float]],
    optimized_per_kernel: Dict[str, Dict[str, float]],
    limit: int = 12,
) -> List[Dict[str, Any]]:
    baseline_ranked = sorted(
        baseline_per_kernel.keys(),
        key=lambda name: _kernel_time_ms(baseline_per_kernel, name),
        reverse=True,
    )
    optimized_ranked = sorted(
        optimized_per_kernel.keys(),
        key=lambda name: _kernel_time_ms(optimized_per_kernel, name),
        reverse=True,
    )

    if not baseline_ranked and not optimized_ranked:
        return []

    pairs: List[Dict[str, Any]] = []
    used_optimized: set[str] = set()

    # Step 1: exact symbol matches.
    for baseline_name in baseline_ranked:
        if baseline_name in optimized_per_kernel and baseline_name not in used_optimized:
            used_optimized.add(baseline_name)
            pairs.append(
                {
                    "baseline_kernel": baseline_name,
                    "optimized_kernel": baseline_name,
                    "match_type": "exact",
                    "match_score": 2.0,
                }
            )

    # Step 2: fuzzy symbol matches.
    for baseline_name in baseline_ranked:
        if any(pair["baseline_kernel"] == baseline_name for pair in pairs):
            continue
        best_name: Optional[str] = None
        best_score = 0.0
        for optimized_name in optimized_ranked:
            if optimized_name in used_optimized:
                continue
            score = _kernel_similarity(baseline_name, optimized_name)
            if score > best_score:
                best_score = score
                best_name = optimized_name
        if best_name is not None and best_score >= 0.35:
            used_optimized.add(best_name)
            pairs.append(
                {
                    "baseline_kernel": baseline_name,
                    "optimized_kernel": best_name,
                    "match_type": "fuzzy",
                    "match_score": round(best_score, 4),
                }
            )

    # Step 3: rank fallback if we still have no useful pairs (symbol mismatch).
    if not pairs and baseline_ranked and optimized_ranked:
        pair_count = min(len(baseline_ranked), len(optimized_ranked))
        for idx in range(pair_count):
            optimized_name = optimized_ranked[idx]
            used_optimized.add(optimized_name)
            pairs.append(
                {
                    "baseline_kernel": baseline_ranked[idx],
                    "optimized_kernel": optimized_name,
                    "match_type": "rank",
                    "match_score": 0.0,
                }
            )

    # Step 4: include remaining baseline kernels for visibility.
    for baseline_name in baseline_ranked:
        if any(pair["baseline_kernel"] == baseline_name for pair in pairs):
            continue
        pairs.append(
            {
                "baseline_kernel": baseline_name,
                "optimized_kernel": None,
                "match_type": "unmatched",
                "match_score": 0.0,
            }
        )

    # Step 5: include remaining optimized kernels for visibility.
    for optimized_name in optimized_ranked:
        if optimized_name in used_optimized:
            continue
        pairs.append(
            {
                "baseline_kernel": None,
                "optimized_kernel": optimized_name,
                "match_type": "unmatched",
                "match_score": 0.0,
            }
        )

    pairs.sort(
        key=lambda pair: _kernel_time_ms(baseline_per_kernel, pair.get("baseline_kernel") or ""),
        reverse=True,
    )
    return pairs[:limit]


def _aggregate_ncu_metrics(per_kernel: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """Summarize per-kernel NCU metrics into weighted aggregate guidance."""
    kernel_rows = []
    for name, metrics in per_kernel.items():
        time_ms = _kernel_time_ms(per_kernel, name)
        kernel_rows.append((name, time_ms, metrics))

    kernel_rows.sort(key=lambda row: row[1], reverse=True)
    total_time_ms = sum(row[1] for row in kernel_rows)

    weighted_metrics: Dict[str, float] = {}
    metric_keys = {
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    }
    if total_time_ms > 0:
        for key in metric_keys:
            weighted_sum = 0.0
            covered_time = 0.0
            for _, time_ms, metrics in kernel_rows:
                value = metrics.get(key)
                if isinstance(value, (int, float)):
                    weighted_sum += float(value) * time_ms
                    covered_time += time_ms
            if covered_time > 0:
                weighted_metrics[key] = weighted_sum / covered_time

    dominant = kernel_rows[0] if kernel_rows else None
    dominant_payload = None
    if dominant is not None:
        dominant_payload = {
            "kernel": dominant[0],
            "gpu__time_duration.avg_ms": dominant[1],
            "metrics": {
                key: value
                for key, value in dominant[2].items()
                if key in metric_keys or key.startswith("gpu__time_duration")
            },
        }

    return {
        "kernel_count": len(kernel_rows),
        "total_gpu_time_ms": total_time_ms,
        "dominant_kernel": dominant_payload,
        "weighted_metrics": weighted_metrics,
    }


def _extract_nsys_cuda_api_stats(nsys_path: Path) -> Dict[str, Any]:
    """Extract CUDA API summary stats from nsys report."""
    try:
        with tempfile.TemporaryDirectory(prefix="nsys_api_") as tmp_dir:
            output_prefix = Path(tmp_dir) / "report"
            result = subprocess.run(
                ["nsys", "stats", "--report", "cuda_api_sum", "--format", "csv",
                 "--force-export", "true", "-o", str(output_prefix), str(nsys_path)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                return {}
            
            # Find the generated CSV file
            csv_files = sorted(Path(tmp_dir).glob("report*cuda_api_sum.csv"))
            if not csv_files:
                return {}
            
            api_stats = {}
            with open(csv_files[0], newline='') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    name = row.get('Name', '')
                    if name:
                        try:
                            api_stats[name] = {
                                'time_pct': float(row.get('Time (%)', 0) or 0),
                                'total_time_ns': int(float(row.get('Total Time (ns)', 0) or 0)),
                                'num_calls': int(row.get('Num Calls', 0) or 0),
                                'avg_ns': float(row.get('Avg (ns)', 0) or 0),
                            }
                        except (ValueError, TypeError):
                            continue
            return api_stats
    except Exception:
        return {}


def _extract_nsys_kernel_stats(nsys_path: Path) -> Dict[str, Any]:
    """Extract GPU kernel summary stats from nsys report."""
    try:
        with tempfile.TemporaryDirectory(prefix="nsys_kern_") as tmp_dir:
            output_prefix = Path(tmp_dir) / "report"
            result = subprocess.run(
                ["nsys", "stats", "--report", "cuda_gpu_kern_sum", "--format", "csv",
                 "--force-export", "true", "-o", str(output_prefix), str(nsys_path)],
                capture_output=True, text=True, timeout=60
            )
            if result.returncode != 0:
                return {}
            
            # Find the generated CSV file
            csv_files = sorted(Path(tmp_dir).glob("report*cuda_gpu_kern_sum.csv"))
            if not csv_files:
                return {}
            
            kernel_stats = {}
            with open(csv_files[0], newline='') as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    name = row.get('Name', '')
                    if name:
                        try:
                            # Shorten kernel names for display while retaining full name.
                            short_name = name.split('<')[0].split('(')[0][-40:]
                            kernel_stats[short_name] = {
                                'full_name': name,
                                'short_name': short_name,
                                'time_pct': float(row.get('Time (%)', 0) or 0),
                                'total_time_ns': int(float(row.get('Total Time (ns)', 0) or 0)),
                                'instances': int(row.get('Instances', 0) or 0),
                                'avg_ns': float(row.get('Avg (ns)', 0) or 0),
                            }
                        except (ValueError, TypeError):
                            continue
            return kernel_stats
    except Exception:
        return {}


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", ""))
    except (ValueError, TypeError):
        return None


def _select_profile_pair_paths(
    profiles_dir: Path,
    pair_key: Optional[str],
    suffix: str,
    label: str,
) -> tuple[Optional[Path], Optional[Path], Optional[str], Optional[Dict[str, Any]]]:
    baseline_files, optimized_files = _collect_profile_role_files(profiles_dir, suffix)

    if not baseline_files or not optimized_files:
        pair_dir, error = _select_pair_dir(profiles_dir, pair_key, label=label)
        if error:
            return None, None, None, error
        if pair_dir:
            baseline_files, optimized_files = _collect_profile_role_files(pair_dir, suffix)

    if not baseline_files or not optimized_files:
        return None, None, None, None

    pair, selected_key, error = _select_profile_pair(
        baseline_files,
        optimized_files,
        pair_key=pair_key,
        label=label,
    )
    if error:
        return None, None, None, error
    if not pair:
        return None, None, None, None

    baseline_path, optimized_path = pair
    return baseline_path, optimized_path, selected_key or pair_key, None


def _top_kernel_summary(kernel_stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not kernel_stats:
        return None
    best_key = max(
        kernel_stats.keys(),
        key=lambda k: kernel_stats.get(k, {}).get("total_time_ns", 0),
    )
    payload = kernel_stats.get(best_key, {})
    return {
        "name": payload.get("full_name") or best_key,
        "total_time_ns": payload.get("total_time_ns", 0),
        "avg_ns": payload.get("avg_ns", 0),
        "instances": payload.get("instances", 0),
    }


def _summarize_ncu_side_by_side(ncu_comparison: Dict[str, Any]) -> Dict[str, Any]:
    if ncu_comparison.get("result_scope") == "range":
        comparison = ncu_comparison.get("range_comparison") or {}
        return {
            "result_scope": "range",
            "kernel": None,
            "metrics": list(comparison.get("metrics") or []),
            "range_summary": {
                "nvtx_range": comparison.get("nvtx_range"),
                "coverage_policy": comparison.get("coverage_policy"),
                "baseline_range_time_ms": comparison.get("baseline_range_time_ms"),
                "optimized_range_time_ms": comparison.get("optimized_range_time_ms"),
                "canonical_speedup_eligible": False,
                "timing_semantics": comparison.get("timing_semantics"),
            },
        }

    if "metrics" in ncu_comparison:
        metrics: List[Dict[str, Any]] = []
        for row in ncu_comparison.get("metrics", []):
            name = row.get("name")
            if not name:
                continue
            baseline_val = _safe_float(row.get("baseline"))
            optimized_val = _safe_float(row.get("optimized"))
            delta = None
            ratio = None
            if baseline_val is not None and optimized_val is not None:
                delta = optimized_val - baseline_val
                ratio = (optimized_val / baseline_val) if baseline_val != 0 else None
            metrics.append(
                {
                    "name": name,
                    "baseline": baseline_val,
                    "optimized": optimized_val,
                    "delta": delta,
                    "ratio": ratio,
                }
            )
        return {"kernel": None, "metrics": metrics}

    kernel_rows = ncu_comparison.get("kernel_comparison") or []
    if not kernel_rows:
        return {"kernel": None, "metrics": []}

    top_kernel = kernel_rows[0]
    metrics_payload: Dict[str, Any] = top_kernel.get("metrics") or {}
    metrics = []
    for name, payload in metrics_payload.items():
        metrics.append(
            {
                "name": name,
                "baseline": payload.get("baseline"),
                "optimized": payload.get("optimized"),
                "delta": payload.get("delta"),
                "ratio": payload.get("ratio"),
            }
        )
    return {"kernel": top_kernel.get("kernel"), "metrics": metrics}


def generate_side_by_side_report(
    profiles_dir: Path,
    pair_key: Optional[str] = None,
    report_dir: Optional[Path] = None,
    ncu_comparison: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate side-by-side Nsight Systems + Nsight Compute JSON report with narrative."""
    pair_health = assess_profile_pair_health(profiles_dir, pair_key=pair_key)
    if not profiles_dir.exists():
        return {"success": False, "error": f"profiles_dir not found: {profiles_dir}"}

    pair_dir, error = _select_pair_dir(profiles_dir, pair_key, label="side-by-side")
    if error:
        return {"success": False, **error, "pair_health": pair_health}
    search_dir = pair_dir or profiles_dir

    baseline_nsys, optimized_nsys, selected_key, nsys_error = _select_profile_pair_paths(
        search_dir, pair_key, ".nsys-rep", "nsys"
    )
    if nsys_error:
        return {"success": False, **nsys_error, "pair_health": pair_health}
    if not baseline_nsys or not optimized_nsys:
        return {
            "success": False,
            "error": "No baseline/optimized nsys profiles found",
            "pair_health": pair_health,
        }

    baseline_ncu, optimized_ncu, _, ncu_error = _select_profile_pair_paths(
        search_dir, pair_key, ".ncu-rep", "ncu"
    )
    if ncu_error:
        return {"success": False, **ncu_error, "pair_health": pair_health}
    if not baseline_ncu or not optimized_ncu:
        return {
            "success": False,
            "error": "No baseline/optimized ncu profiles found",
            "pair_health": pair_health,
        }

    baseline_api = _extract_nsys_cuda_api_stats(baseline_nsys)
    optimized_api = _extract_nsys_cuda_api_stats(optimized_nsys)
    if not baseline_api or not optimized_api:
        return {
            "success": False,
            "error": "Failed to extract nsys cuda_api_sum stats",
            "pair_health": pair_health,
        }

    baseline_kernel_stats = _extract_nsys_kernel_stats(baseline_nsys)
    optimized_kernel_stats = _extract_nsys_kernel_stats(optimized_nsys)
    if not baseline_kernel_stats or not optimized_kernel_stats:
        return {
            "success": False,
            "error": "Failed to extract nsys cuda_gpu_kern_sum stats",
            "pair_health": pair_health,
        }

    if ncu_comparison is None:
        ncu_comparison = compare_ncu_files(search_dir, pair_key=pair_key)
    if not ncu_comparison:
        return {
            "success": False,
            "error": "No comparable ncu metrics found",
            "pair_health": pair_health,
        }
    if ncu_comparison.get("error"):
        return {"success": False, **ncu_comparison, "pair_health": pair_health}

    ncu_summary = _summarize_ncu_side_by_side(ncu_comparison)

    api_total_baseline_ns = sum(v.get("total_time_ns", 0) for v in baseline_api.values())
    api_total_optimized_ns = sum(v.get("total_time_ns", 0) for v in optimized_api.values())

    api_priority = [
        "cudaMemcpy",
        "cudaMemcpyAsync",
        "cudaMalloc",
        "cudaFree",
        "cudaHostAlloc",
        "cudaFreeHost",
        "cudaDeviceSynchronize",
        "cudaLaunchKernel",
    ]
    api_rows: List[Dict[str, Any]] = []
    for name in api_priority:
        b = baseline_api.get(name)
        o = optimized_api.get(name)
        if not b and not o:
            api_rows.append(
                {
                    "name": name,
                    "baseline_total_ns": None,
                    "baseline_total_s": None,
                    "baseline_calls": None,
                    "baseline_avg_ns": None,
                    "baseline_avg_ms": None,
                    "optimized_total_ns": None,
                    "optimized_total_s": None,
                    "optimized_calls": None,
                    "optimized_avg_ns": None,
                    "optimized_avg_ms": None,
                    "delta_ns": None,
                    "delta_s": None,
                    "delta_pct": None,
                }
            )
            continue
        b_total_ns = (b or {}).get("total_time_ns") if b else None
        o_total_ns = (o or {}).get("total_time_ns") if o else None
        b_avg_ns = (b or {}).get("avg_ns") if b else None
        o_avg_ns = (o or {}).get("avg_ns") if o else None
        delta_ns = None
        delta_pct = None
        if isinstance(b_total_ns, (int, float)) and isinstance(o_total_ns, (int, float)):
            delta_ns = o_total_ns - b_total_ns
            delta_pct = ((o_total_ns - b_total_ns) / b_total_ns * 100) if b_total_ns else None
        api_rows.append(
            {
                "name": name,
                "baseline_total_ns": b_total_ns,
                "baseline_total_s": (b_total_ns / 1e9) if isinstance(b_total_ns, (int, float)) else None,
                "baseline_calls": (b or {}).get("num_calls") if b else None,
                "baseline_avg_ns": b_avg_ns,
                "baseline_avg_ms": (b_avg_ns / 1e6) if isinstance(b_avg_ns, (int, float)) else None,
                "optimized_total_ns": o_total_ns,
                "optimized_total_s": (o_total_ns / 1e9) if isinstance(o_total_ns, (int, float)) else None,
                "optimized_calls": (o or {}).get("num_calls") if o else None,
                "optimized_avg_ns": o_avg_ns,
                "optimized_avg_ms": (o_avg_ns / 1e6) if isinstance(o_avg_ns, (int, float)) else None,
                "delta_ns": delta_ns,
                "delta_s": (delta_ns / 1e9) if isinstance(delta_ns, (int, float)) else None,
                "delta_pct": delta_pct,
            }
        )

    base_kernel = _top_kernel_summary(baseline_kernel_stats)
    opt_kernel = _top_kernel_summary(optimized_kernel_stats)
    kernel_summary = None
    if base_kernel or opt_kernel:
        kernel_summary = {
            "name": (base_kernel or opt_kernel or {}).get("name", ""),
            "baseline_total_ns": (base_kernel or {}).get("total_time_ns", 0),
            "baseline_avg_ns": (base_kernel or {}).get("avg_ns", 0),
            "optimized_total_ns": (opt_kernel or {}).get("total_time_ns", 0),
            "optimized_avg_ns": (opt_kernel or {}).get("avg_ns", 0),
            "instances": (base_kernel or opt_kernel or {}).get("instances", 0),
        }

    narrative_parts: List[str] = []
    api_delta_candidates = [
        (row["name"], row.get("delta_s"))
        for row in api_rows
        if isinstance(row.get("delta_s"), (int, float))
    ]
    if api_delta_candidates:
        top_api = sorted(
            api_delta_candidates, key=lambda item: abs(item[1] or 0), reverse=True
        )[:3]
        api_desc = ", ".join(
            f"{name} {delta_s:+.3f}s" for name, delta_s in top_api if delta_s is not None
        )
        if api_desc:
            narrative_parts.append(f"Top CUDA API time deltas: {api_desc}.")

    if kernel_summary:
        b_total = kernel_summary.get("baseline_total_ns", 0)
        o_total = kernel_summary.get("optimized_total_ns", 0)
        if isinstance(b_total, (int, float)) and isinstance(o_total, (int, float)) and b_total:
            ratio = o_total / b_total
            narrative_parts.append(
                "Kernel time ratio: {ratio:.3f}x ({b_ms:.3f}ms -> {o_ms:.3f}ms).".format(
                    ratio=ratio,
                    b_ms=b_total / 1e6,
                    o_ms=o_total / 1e6,
                )
            )

    range_summary = ncu_summary.get("range_summary")
    if isinstance(range_summary, dict):
        nvtx_range = range_summary.get("nvtx_range") or "<unknown>"
        narrative_parts.append(
            f"NCU metrics cover the full selected NVTX range {nvtx_range!r}; "
            "the profiler range duration is descriptive and is not a benchmark speedup."
        )

    ncu_metrics = ncu_summary.get("metrics", [])
    metric_deltas: List[Tuple[str, float, Optional[float]]] = []
    for metric in ncu_metrics:
        name = metric.get("name")
        ratio = metric.get("ratio")
        delta = metric.get("delta")
        score = None
        if isinstance(ratio, (int, float)):
            score = abs(ratio - 1.0)
        elif isinstance(delta, (int, float)):
            score = abs(delta)
        if name and score is not None:
            metric_deltas.append((name, score, ratio))
    if metric_deltas:
        top_metrics = sorted(metric_deltas, key=lambda item: item[1], reverse=True)[:3]
        metric_desc = []
        for name, _, ratio in top_metrics:
            metric = next((m for m in ncu_metrics if m.get("name") == name), {})
            delta = metric.get("delta")
            if isinstance(ratio, (int, float)):
                metric_desc.append(f"{name} {ratio:.3f}x")
            elif isinstance(delta, (int, float)):
                metric_desc.append(f"{name} {delta:+.3f}")
        if metric_desc:
            narrative_parts.append("Top NCU metric deltas: " + ", ".join(metric_desc) + ".")

    narrative = " ".join(narrative_parts) if narrative_parts else "Profile comparison available."

    report_base = report_dir or (search_dir / "reports")
    report_base.mkdir(parents=True, exist_ok=True)
    slug_source = selected_key or _normalize_profile_name(baseline_nsys.name) or "profile"
    slug = re.sub(r"[^a-z0-9_]+", "_", slug_source.lower()).strip("_") or "profile"
    report_path = report_base / f"{slug}_side_by_side.json"

    side_by_side = {
        "sources": {
            "baseline_nsys": str(baseline_nsys),
            "optimized_nsys": str(optimized_nsys),
            "baseline_ncu": str(baseline_ncu),
            "optimized_ncu": str(optimized_ncu),
        },
        "nsys": {
            "api_total_baseline_ns": api_total_baseline_ns,
            "api_total_optimized_ns": api_total_optimized_ns,
            "api_rows": api_rows,
            "kernel_summary": kernel_summary,
        },
        "ncu": ncu_summary,
    }

    report_payload = {
        "narrative": narrative,
        "side_by_side": side_by_side,
    }
    report_path.write_text(json.dumps(report_payload, indent=2, default=str))

    return {
        "success": True,
        "report_json_path": str(report_path),
        "narrative": narrative,
        "side_by_side": side_by_side,
        "nsys_summary": side_by_side["nsys"],
        "ncu_summary": side_by_side["ncu"],
        "pair_health": pair_health,
    }


def _strip_profile_suffix(name: str) -> str:
    if name.endswith(".nsys-rep"):
        return name[:-9]
    if name.endswith(".ncu-rep"):
        return name[:-8]
    if name.endswith(".csv"):
        return name[:-4]
    if name.endswith(".sqlite"):
        return name[:-7]
    return name


def _snake_case(name: str) -> str:
    name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", name)
    name = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    return name


def _normalize_profile_name(name: str) -> str:
    name = _strip_profile_suffix(name)
    if name.startswith("nsys_"):
        name = name[5:]
    name = _snake_case(name)
    raw_tokens = re.split(r"[^A-Za-z0-9]+", name)
    tokens = []
    for token in raw_tokens:
        lowered = token.lower()
        if not lowered:
            continue
        if lowered in _ROLE_TOKENS:
            continue
        if lowered == "benchmark":
            continue
        tokens.append(lowered)

    normalized = "_".join(tokens)
    tokens = [token for token in normalized.split("_") if token]
    if tokens and len(tokens) % 2 == 0:
        half = len(tokens) // 2
        if tokens[:half] == tokens[half:]:
            normalized = "_".join(tokens[:half])
    return normalized


def _tokenize_profile_name(name: str) -> List[str]:
    normalized = _normalize_profile_name(name)
    tokens = [token for token in normalized.split("_") if token]
    return tokens


def _tokenize_raw_profile_name(name: str) -> List[str]:
    raw = _snake_case(_strip_profile_suffix(name))
    raw = re.sub(r"[^A-Za-z0-9]+", "_", raw)
    return [token.lower() for token in raw.split("_") if token]


def _path_name_candidates(path: Path) -> List[str]:
    names = [path.name]
    # Include nearby directory names so role detection still works when
    # filenames are generic but parent folders encode baseline/optimized.
    if path.parent:
        names.append(path.parent.name)
    if path.parent and path.parent.parent:
        names.append(path.parent.parent.name)
    return names


def _classify_profile_role(path: Path) -> Optional[str]:
    # Prefer explicit role markers in the filename itself. Parent directories
    # can contain "pair__optimized" while still holding both baseline and optimized files.
    filename_tokens = set(_tokenize_raw_profile_name(path.name))
    file_has_baseline = bool(filename_tokens & _BASELINE_ROLE_TOKENS)
    file_has_optimized = bool(filename_tokens & _OPTIMIZED_ROLE_TOKENS)
    if file_has_baseline and not file_has_optimized:
        return "baseline"
    if file_has_optimized and not file_has_baseline:
        return "optimized"

    baseline_hits = 0
    optimized_hits = 0
    for name in _path_name_candidates(path)[1:]:
        tokens = set(_tokenize_raw_profile_name(name))
        if tokens & _BASELINE_ROLE_TOKENS:
            baseline_hits += 1
        if tokens & _OPTIMIZED_ROLE_TOKENS:
            optimized_hits += 1
    if baseline_hits and not optimized_hits:
        return "baseline"
    if optimized_hits and not baseline_hits:
        return "optimized"
    return None


def _collect_profile_role_files(
    profiles_dir: Path,
    suffix: str,
    name_predicate: Optional[Callable[[Path], bool]] = None,
) -> Tuple[List[Path], List[Path]]:
    def _safe_mtime_ns(path: Path) -> int:
        try:
            return path.stat().st_mtime_ns
        except Exception:
            return 0

    def _capture_order(path: Path) -> Tuple[int, str]:
        return (_safe_mtime_ns(path), str(path.absolute()))

    files = list(profiles_dir.glob(f"*{suffix}"))
    files.extend(list(profiles_dir.rglob(f"*{suffix}")))

    # Avoid recursive self-noise from internal copy staging.
    files = [
        path
        for path in files
        if ".materialized_profiles" not in path.parts and ".staged_profile_pairs" not in path.parts
    ]

    # Materialize symlink artifacts up front so pairing/classification operates
    # on concrete files even when capture directories contain link-heavy layouts.
    materialized_files: List[Path] = []
    for path in files:
        if path.is_symlink():
            materialized = _materialize_profile_if_needed(path, root=profiles_dir)
            materialized_files.append(materialized)
        else:
            materialized_files.append(path)
    files = materialized_files

    deduped: Dict[str, Path] = {}
    for path in files:
        # Deduplicate by concrete discovered path only. Do not collapse by
        # resolved target, otherwise distinct role-tagged captures can merge.
        dedupe_key = str(path.absolute())
        deduped[dedupe_key] = path
    files = list(deduped.values())
    if name_predicate is not None:
        files = [path for path in files if name_predicate(path)]

    baseline_files: List[Path] = []
    optimized_files: List[Path] = []
    unclassified: List[Path] = []
    for path in files:
        role = _classify_profile_role(path)
        if role == "baseline":
            baseline_files.append(path)
        elif role == "optimized":
            optimized_files.append(path)
        else:
            unclassified.append(path)

    # Standalone-capture fallback: when exactly two files exist and role tokens
    # are absent, infer baseline/optimized from capture time to keep compare
    # tools usable without symlink-based layouts.
    if not baseline_files and not optimized_files and len(unclassified) == 2:
        ordered = sorted(unclassified, key=_capture_order)
        baseline_files = [ordered[0]]
        optimized_files = [ordered[1]]

    # Robust fallback: some symlink-heavy layouts can misclassify both files as
    # the same role via parent directory tokens. If there are exactly two files,
    # force deterministic baseline/optimized assignment by capture time.
    all_files = baseline_files + optimized_files + unclassified
    if len(all_files) == 2 and (not baseline_files or not optimized_files):
        ordered = sorted(all_files, key=_capture_order)
        baseline_files = [ordered[0]]
        optimized_files = [ordered[1]]
    return baseline_files, optimized_files


def _group_profile_pairs(
    baseline_files: List[Path],
    optimized_files: List[Path],
) -> Dict[str, Dict[str, List[Path]]]:
    grouped: Dict[str, Dict[str, List[Path]]] = {}

    def _key_from_path(path: Path) -> str:
        key = _normalize_profile_name(path.name)
        if not key:
            raw_tokens = [
                token
                for token in _tokenize_raw_profile_name(path.name)
                if token not in _ROLE_TOKENS and token not in {"benchmark", "nsys", "ncu"}
            ]
            key = "_".join(raw_tokens)
        if not key:
            key = "default"
        return key

    for path in baseline_files:
        key = _key_from_path(path)
        grouped.setdefault(key, {}).setdefault("baseline", []).append(path)
    for path in optimized_files:
        key = _key_from_path(path)
        grouped.setdefault(key, {}).setdefault("optimized", []).append(path)
    return {
        key: value
        for key, value in grouped.items()
        if value.get("baseline") and value.get("optimized")
    }


def _select_profile_pair(
    baseline_files: List[Path],
    optimized_files: List[Path],
    pair_key: Optional[str],
    label: str,
) -> tuple[Optional[tuple[Path, Path]], Optional[str], Optional[Dict[str, Any]]]:
    def _safe_mtime(path: Path) -> float:
        try:
            return float(path.stat().st_mtime)
        except Exception:
            return 0.0

    def _role_pair_fallback(
        baselines: List[Path],
        optimized: List[Path],
        requested_key: Optional[str],
    ) -> tuple[Path, Path]:
        # Token-aware fallback when canonical key grouping fails (e.g., standalone
        # captures with non-matching filenames). Prefer stronger name overlap,
        # then tighter capture-time proximity, then newest captures.
        key_tokens = set(_tokenize_profile_name(requested_key or ""))
        scored: List[Tuple[float, float, float, Path, Path]] = []
        for b in baselines:
            b_tokens = set(_tokenize_profile_name(b.name)) - _ROLE_TOKENS
            for o in optimized:
                o_tokens = set(_tokenize_profile_name(o.name)) - _ROLE_TOKENS
                overlap = len(b_tokens & o_tokens)
                union = len(b_tokens | o_tokens) or 1
                token_score = overlap / union
                if key_tokens:
                    key_score = (
                        len((b_tokens | o_tokens) & key_tokens) / max(len(key_tokens), 1)
                    )
                else:
                    key_score = 0.0
                dt_score = -abs(_safe_mtime(b) - _safe_mtime(o))
                scored.append((key_score, token_score, dt_score, b, o))
        if not scored:
            raise ValueError("no baseline/optimized files available")
        scored.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4]))
        _, _, _, baseline_choice, optimized_choice = scored[-1]
        return baseline_choice, optimized_choice

    if not baseline_files or not optimized_files:
        return None, None, None

    candidates = _group_profile_pairs(baseline_files, optimized_files)
    # Fallback: if role classification found a single baseline and optimized
    # capture but normalized names don't align, pair newest files directly.
    if not candidates and len(baseline_files) == 1 and len(optimized_files) == 1:
        return (baseline_files[0], optimized_files[0]), "single_role_pair", None
    if not candidates:
        fallback_baseline, fallback_optimized = _role_pair_fallback(
            baseline_files, optimized_files, pair_key
        )
        return (fallback_baseline, fallback_optimized), "role_fallback", None

    candidate_keys = sorted(candidates.keys())
    if pair_key is None:
        if len(candidate_keys) > 1:
            return (
                None,
                None,
                {
                    "error": (
                        f"Multiple {label} profile pairs found; provide --pair to select one."
                    ),
                    "candidates": candidate_keys,
                },
            )
        selected_key = candidate_keys[0]
        return (
            (
                max(candidates[selected_key]["baseline"], key=lambda p: p.stat().st_mtime),
                max(candidates[selected_key]["optimized"], key=lambda p: p.stat().st_mtime),
            ),
            selected_key,
            None,
        )

    key_tokens = set(_tokenize_profile_name(pair_key))
    pair_key_norm = _normalize_profile_name(pair_key) or pair_key.lower()

    best_key: Optional[str] = None
    best_score = -1.0
    for key in candidate_keys:
        candidate_tokens = set(_tokenize_profile_name(key))
        overlap = len(key_tokens & candidate_tokens)
        if overlap == 0 and pair_key_norm not in key:
            continue
        score = 0.0
        if key == pair_key_norm:
            score += 1.5
        if pair_key_norm in key:
            score += 1.0
        if overlap:
            score += (2.0 * overlap) / (len(key_tokens) + len(candidate_tokens))
        if score > best_score:
            best_score = score
            best_key = key

    if best_key is None:
        return (
            None,
            None,
            {
                "error": f"No matching {label} profile pair found for --pair={pair_key}.",
                "candidates": candidate_keys,
            },
        )

    return (
        (
            max(candidates[best_key]["baseline"], key=lambda p: p.stat().st_mtime),
            max(candidates[best_key]["optimized"], key=lambda p: p.stat().st_mtime),
        ),
        best_key,
        None,
    )


def _pair_key_from_dir(root: Path, pair_dir: Path) -> str:
    rel = pair_dir.relative_to(root)
    parts = [part.replace("pair__", "") for part in rel.parts]
    return "/".join(parts)


def _select_pair_dir(
    profiles_dir: Path,
    pair_key: Optional[str],
    label: str,
) -> tuple[Optional[Path], Optional[Dict[str, Any]]]:
    if profiles_dir.name.startswith("pair__"):
        return profiles_dir, None

    pair_dirs = {
        _pair_key_from_dir(profiles_dir, pair_dir): pair_dir
        for pair_dir in profiles_dir.rglob("pair__*")
        if pair_dir.is_dir()
    }
    if not pair_dirs:
        return None, None

    candidate_keys = sorted(pair_dirs.keys())
    if pair_key is None:
        if len(candidate_keys) > 1:
            return (
                None,
                {
                    "error": f"Multiple {label} profile pairs found; provide --pair to select one.",
                    "candidates": candidate_keys,
                },
            )
        return pair_dirs[candidate_keys[0]], None

    key_tokens = set(_tokenize_profile_name(pair_key.replace("/", "_")))
    best_key: Optional[str] = None
    best_score = -1.0
    for key in candidate_keys:
        candidate_tokens = set(_tokenize_profile_name(key.replace("/", "_")))
        overlap = len(key_tokens & candidate_tokens)
        if overlap == 0 and pair_key not in key:
            continue
        score = overlap
        if pair_key in key:
            score += 1.0
        if score > best_score:
            best_score = score
            best_key = key

    if best_key is None:
        return (
            None,
            {
                "error": f"No matching {label} profile pair found for --pair={pair_key}.",
                "candidates": candidate_keys,
            },
        )

    return pair_dirs[best_key], None


def assess_profile_pair_health(
    profiles_dir: Path,
    pair_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Assess whether a profile directory has complete baseline/optimized pairs."""
    pair_dir, pair_error = _select_pair_dir(profiles_dir, pair_key, label="profile health")
    if pair_error:
        return {
            "ok": False,
            "profiles_dir": str(profiles_dir),
            "pair_key": pair_key,
            "pair_dir": None,
            **pair_error,
        }

    search_dir = pair_dir or profiles_dir
    baseline_nsys, optimized_nsys = _collect_profile_role_files(search_dir, ".nsys-rep")
    baseline_ncu_rep, optimized_ncu_rep = _collect_profile_role_files(search_dir, ".ncu-rep")
    baseline_ncu_csv, optimized_ncu_csv = _collect_profile_role_files(
        search_dir,
        ".csv",
        name_predicate=lambda p: "ncu" in p.name.lower(),
    )

    has_nsys_pair = bool(baseline_nsys and optimized_nsys)
    has_ncu_rep_pair = bool(baseline_ncu_rep and optimized_ncu_rep)
    has_ncu_csv_pair = bool(baseline_ncu_csv and optimized_ncu_csv)
    has_any_ncu_pair = has_ncu_rep_pair or has_ncu_csv_pair
    missing: List[str] = []
    if not has_nsys_pair:
        missing.append("nsys_pair")
    if not has_any_ncu_pair:
        missing.append("ncu_pair")

    result = {
        "ok": not missing,
        "profiles_dir": str(profiles_dir),
        "pair_key": pair_key,
        "pair_dir": str(search_dir),
        "has_nsys_pair": has_nsys_pair,
        "has_ncu_rep_pair": has_ncu_rep_pair,
        "has_ncu_csv_pair": has_ncu_csv_pair,
        "has_any_ncu_pair": has_any_ncu_pair,
        "missing": missing,
        "counts": {
            "baseline_nsys": len(baseline_nsys),
            "optimized_nsys": len(optimized_nsys),
            "baseline_ncu_rep": len(baseline_ncu_rep),
            "optimized_ncu_rep": len(optimized_ncu_rep),
            "baseline_ncu_csv": len(baseline_ncu_csv),
            "optimized_ncu_csv": len(optimized_ncu_csv),
        },
        "suggested_next_steps": [
            "Capture baseline and optimized NSYS profiles to unlock timeline/flamegraph comparisons.",
            "Capture baseline and optimized NCU profiles (or NCU CSV exports) to unlock kernel metric comparisons.",
        ],
    }
    manifest_payload = {
        "schema_version": "1.0",
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "label": "pair_health",
        "pair_status": "complete" if result["ok"] else "partial",
        "pair_dir": str(search_dir),
        "profiles_dir": str(profiles_dir),
        "pair_key": pair_key,
        "availability": {
            "has_nsys_pair": has_nsys_pair,
            "has_ncu_rep_pair": has_ncu_rep_pair,
            "has_ncu_csv_pair": has_ncu_csv_pair,
        },
        "missing": missing,
        "counts": result["counts"],
        "sample_artifacts": {
            "baseline_nsys": str(baseline_nsys[0]) if baseline_nsys else None,
            "optimized_nsys": str(optimized_nsys[0]) if optimized_nsys else None,
            "baseline_ncu_rep": str(baseline_ncu_rep[0]) if baseline_ncu_rep else None,
            "optimized_ncu_rep": str(optimized_ncu_rep[0]) if optimized_ncu_rep else None,
            "baseline_ncu_csv": str(baseline_ncu_csv[0]) if baseline_ncu_csv else None,
            "optimized_ncu_csv": str(optimized_ncu_csv[0]) if optimized_ncu_csv else None,
        },
    }
    _write_pair_manifest(search_dir / _PAIR_MANIFEST_FILENAME, manifest_payload)
    return result


def _select_flamegraph_pair(
    profiles_dir: Path,
    pair_key: Optional[str] = None,
) -> tuple[Optional[tuple[Path, Path]], Optional[Dict[str, Any]]]:
    baseline_nsys, optimized_nsys = _collect_profile_role_files(profiles_dir, ".nsys-rep")

    if not baseline_nsys or not optimized_nsys:
        pair_dir, error = _select_pair_dir(profiles_dir, pair_key, label="nsys")
        if error:
            return None, error
        if not pair_dir:
            return None, None
        baseline_nsys, optimized_nsys = _collect_profile_role_files(pair_dir, ".nsys-rep")
        if not baseline_nsys or not optimized_nsys:
            return None, None

    pair, _, error = _select_profile_pair(
        baseline_nsys,
        optimized_nsys,
        pair_key=pair_key,
        label="nsys",
    )
    return pair, error


def generate_flamegraph_comparison(
    profiles_dir: Path,
    pair_key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Generate flame graph comparison data for baseline vs optimized profiles.
    
    Returns structured data for the FlameGraphComparison React component.
    """
    pair_health = assess_profile_pair_health(profiles_dir, pair_key=pair_key)
    pair, error = _select_flamegraph_pair(profiles_dir, pair_key=pair_key)
    if error:
        return {**error, "pair_health": pair_health}
    if pair is None:
        return None
    baseline_path, optimized_path = pair
    
    try:
        baseline_api = _extract_nsys_cuda_api_stats(baseline_path)
        optimized_api = _extract_nsys_cuda_api_stats(optimized_path)
        baseline_kernels = _extract_nsys_kernel_stats(baseline_path)
        optimized_kernels = _extract_nsys_kernel_stats(optimized_path)
        
        # Calculate total times
        baseline_total_ns = sum(s.get('total_time_ns', 0) for s in baseline_api.values())
        optimized_total_ns = sum(s.get('total_time_ns', 0) for s in optimized_api.values())
        
        # Calculate speedup
        baseline_total_ns_f = float(baseline_total_ns)
        optimized_total_ns_f = float(optimized_total_ns)
        if (
            not math.isfinite(baseline_total_ns_f)
            or not math.isfinite(optimized_total_ns_f)
            or baseline_total_ns_f <= 0
            or optimized_total_ns_f <= 0
        ):
            speedup = 1.0
        else:
            speedup = baseline_total_ns_f / optimized_total_ns_f
        
        # Build API breakdown for flame bars
        def build_api_bars(api_stats: Dict) -> List[Dict]:
            bars = []
            for name, stats in sorted(api_stats.items(), key=lambda x: -x[1].get('time_pct', 0)):
                bar_type = 'sync' if 'Sync' in name else \
                          'malloc' if 'Malloc' in name or 'Alloc' in name or 'Free' in name else \
                          'launch' if 'Launch' in name else \
                          'memcpy' if 'Memcpy' in name else \
                          'wait' if 'Wait' in name else 'other'
                bars.append({
                    'name': name,
                    'time_pct': stats.get('time_pct', 0),
                    'total_time_ns': stats.get('total_time_ns', 0),
                    'num_calls': stats.get('num_calls', 0),
                    'type': bar_type,
                })
            return bars[:10]  # Top 10 API calls
        
        def build_kernel_bars(kernel_stats: Dict) -> List[Dict]:
            bars = []
            for name, stats in sorted(kernel_stats.items(), key=lambda x: -x[1].get('time_pct', 0)):
                bars.append({
                    'name': name,
                    'time_pct': stats.get('time_pct', 0),
                    'total_time_ns': stats.get('total_time_ns', 0),
                    'instances': stats.get('instances', 0),
                })
            return bars[:8]  # Top 8 kernels
        
        # Key metrics for comparison
        baseline_sync_calls = sum(
            s.get('num_calls', 0) for n, s in baseline_api.items() 
            if 'Sync' in n
        )
        optimized_sync_calls = sum(
            s.get('num_calls', 0) for n, s in optimized_api.items() 
            if 'Sync' in n
        )
        
        baseline_device_sync = baseline_api.get('cudaDeviceSynchronize', {}).get('num_calls', 0)
        optimized_device_sync = optimized_api.get('cudaDeviceSynchronize', {}).get('num_calls', 0)
        
        optimized_wait_events = optimized_api.get('cudaStreamWaitEvent', {}).get('num_calls', 0)
        
        return {
            'baseline': {
                'file': baseline_path.name,
                'total_time_ms': baseline_total_ns / 1_000_000,
                'api_bars': build_api_bars(baseline_api),
                'kernel_bars': build_kernel_bars(baseline_kernels),
            },
            'optimized': {
                'file': optimized_path.name,
                'total_time_ms': optimized_total_ns / 1_000_000,
                'api_bars': build_api_bars(optimized_api),
                'kernel_bars': build_kernel_bars(optimized_kernels),
            },
            'speedup': round(speedup, 2),
            'metrics': {
                'baseline_sync_calls': baseline_sync_calls,
                'optimized_sync_calls': optimized_sync_calls,
                'sync_reduction_pct': round((1 - optimized_sync_calls / baseline_sync_calls) * 100, 1) if baseline_sync_calls > 0 else 0,
                'baseline_device_sync': baseline_device_sync,
                'optimized_device_sync': optimized_device_sync,
                'device_sync_reduction_pct': round((1 - optimized_device_sync / baseline_device_sync) * 100, 1) if baseline_device_sync > 0 else 0,
                'optimized_wait_events': optimized_wait_events,
            },
            'insight': _generate_optimization_insight(baseline_api, optimized_api),
            'pair_health': pair_health,
        }
    except Exception as exc:
        return {'error': str(exc), 'pair_health': pair_health}


def _generate_optimization_insight(baseline_api: Dict, optimized_api: Dict) -> str:
    """Generate a human-readable insight about the optimization."""
    insights = []
    
    # Check sync reduction
    baseline_sync = baseline_api.get('cudaStreamSynchronize', {}).get('num_calls', 0)
    optimized_sync = optimized_api.get('cudaStreamSynchronize', {}).get('num_calls', 0)
    if baseline_sync > 0 and optimized_sync < baseline_sync:
        reduction = (1 - optimized_sync / baseline_sync) * 100
        insights.append(f"Stream syncs reduced by {reduction:.0f}% ({baseline_sync} → {optimized_sync})")
    
    # Check device sync reduction
    baseline_dev = baseline_api.get('cudaDeviceSynchronize', {}).get('num_calls', 0)
    optimized_dev = optimized_api.get('cudaDeviceSynchronize', {}).get('num_calls', 0)
    if baseline_dev > 0 and optimized_dev < baseline_dev:
        reduction = (1 - optimized_dev / baseline_dev) * 100
        insights.append(f"Device syncs reduced by {reduction:.0f}% ({baseline_dev} → {optimized_dev})")
    
    # Check for stream wait events (pipelining indicator)
    wait_events = optimized_api.get('cudaStreamWaitEvent', {}).get('num_calls', 0)
    if wait_events > 0:
        insights.append(f"Uses {wait_events} stream wait events for lightweight coordination")
    
    return "; ".join(insights) if insights else "Profile comparison available"


def generate_recommendations_from_profiles(result: Dict[str, Any]) -> List[str]:
    """Generate recommendations based on profile comparison data."""
    recommendations: List[str] = []

    nsys = result.get("nsys_comparison", {})
    if nsys and isinstance(nsys, dict) and "metrics" in nsys:
        for metric in nsys.get("metrics", []):
            name = metric.get("name", "").lower()
            delta = metric.get("delta")

            if delta and "dram" in name and delta < -10:
                recommendations.append(
                    "Memory traffic reduced: Consider further optimization with TMA prefetching"
                )
            elif delta and "sm" in name and delta > 10:
                recommendations.append(
                    f"SM utilization improved by {delta:.1f}%: Good progress on compute efficiency"
                )

    ncu = result.get("ncu_comparison", {})
    if (
        ncu
        and isinstance(ncu, dict)
        and ncu.get("result_scope") != "range"
        and "metrics" in ncu
    ):
        for metric in ncu.get("metrics", []):
            name = metric.get("name", "").lower()
            if "occupancy" in name:
                try:
                    opt_val = float(str(metric.get("optimized", "0")).replace("%", ""))
                    if opt_val < 50:
                        recommendations.append(
                            f"Occupancy at {opt_val:.0f}%: Consider tuning block size or reducing register pressure"
                        )
                except (ValueError, TypeError):
                    pass

    if not recommendations and isinstance(ncu, dict) and ncu.get("result_scope") == "range":
        recommendations.append(
            "NCU metrics cover the selected range; collect a kernel-scoped report "
            "before making per-kernel tuning recommendations"
        )
    elif not recommendations:
        recommendations.append("Profile both baseline and optimized to get detailed recommendations")

    return recommendations
