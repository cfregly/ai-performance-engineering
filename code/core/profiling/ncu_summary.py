from __future__ import annotations

import csv
import hashlib
import io
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_NCU_NVTX_RANGE_COLUMNS = (
    "Id:Domain:Start/Stop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
    "thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg",
)
_NCU_METADATA_SUFFIX_FIELD_COUNT = 6
_NCU_IDENTITY_FIELDS = {
    "ID",
    "Kernel Name",
    "Block Size",
    "Grid Size",
    "Stream",
    "Device",
    "CC",
    "Context",
    "Process ID",
    "Process Name",
    "Host Name",
    *_NCU_NVTX_RANGE_COLUMNS,
}


def _parse_float(text: str) -> Optional[float]:
    raw = (text or "").strip().replace(",", "")
    if not raw:
        return None
    if raw.lower() in {"nan", "inf", "+inf", "-inf"}:
        return None
    if raw.endswith("%"):
        raw = raw[:-1].strip()
    try:
        return float(raw)
    except Exception:
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
    # Nsight Compute CSV usually uses base units; treat unknown as-is.
    return value


def _ncu_nvtx_range_metadata(record: Dict[str, str]) -> Tuple[Optional[str], str]:
    """Return the selected NVTX range name and the raw NCU identity field."""
    for key in _NCU_NVTX_RANGE_COLUMNS:
        raw = (record.get(key) or "").strip()
        if not raw:
            continue
        text = raw.strip().strip('"').strip()
        parts = [part.strip() for part in text.split(":")]
        if len(parts) <= _NCU_METADATA_SUFFIX_FIELD_COUNT + 1:
            return None, raw
        start = 2 if parts[0].isdigit() else 1
        end = len(parts) - _NCU_METADATA_SUFFIX_FIELD_COUNT
        name = ":".join(parts[start:end]).strip()
        return (name or None), raw
    return None, ""


def _is_ncu_range_record(record: Dict[str, str]) -> bool:
    """Classify an observed NCU row as a range result from row identity evidence."""
    kernel_name = (record.get("Kernel Name") or "").strip().casefold()
    nvtx_range, _ = _ncu_nvtx_range_metadata(record)
    return kernel_name == "range" and nvtx_range is not None


def _parse_record_metrics(
    header: List[str],
    units: Dict[str, str],
    record: Dict[str, str],
) -> Tuple[Dict[str, float], Optional[float], Optional[float]]:
    metrics_out: Dict[str, float] = {}
    time_avg_ms: Optional[float] = None
    time_sum_ms: Optional[float] = None
    for key in header:
        if key in _NCU_IDENTITY_FIELDS:
            continue
        val = _parse_float((record.get(key) or "").strip())
        if val is None:
            continue
        if key.startswith("gpu__time_duration"):
            val = _time_to_ms(val, units.get(key, ""))
        metrics_out[key] = val
        if key == "gpu__time_duration.avg":
            time_avg_ms = val
        if key == "gpu__time_duration.sum":
            time_sum_ms = val
    return metrics_out, time_avg_ms, time_sum_ms


def _range_metrics_from_model(metrics: Any) -> Dict[str, float]:
    """Map first-class NCU range metrics back to their metric identifiers."""
    payload = dict(getattr(metrics, "raw_metrics", {}) or {})
    field_map = {
        "gpu__time_duration.avg": "range_time_ms",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed": "l2_throughput_pct",
        "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy_pct",
    }
    for metric_name, field_name in field_map.items():
        value = getattr(metrics, field_name, None)
        if isinstance(value, (int, float)):
            payload[metric_name] = float(value)
    return payload


def _range_result_from_csv(
    *,
    path: Path,
    record: Dict[str, str],
    metrics_out: Dict[str, float],
    units: Dict[str, str],
    metrics_requested: List[str],
    command: Optional[List[str]],
    stderr: str,
    returncode: int,
) -> Dict[str, Any]:
    nvtx_range, nvtx_range_raw = _ncu_nvtx_range_metadata(record)
    range_time_ms = metrics_out.get("gpu__time_duration.avg")
    observed_metrics = sorted(metrics_out)
    provenance = {
        "schema_version": "1.0",
        "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "replay_mode": None,
        "result_scope": "range",
        "result_id": int((record.get("ID") or "0").strip()),
        "result_count": 1,
        "nvtx_range": nvtx_range,
        "nvtx_range_raw": nvtx_range_raw,
        "coverage_policy": "range_row_only_unverified",
        "requested_metrics": None,
        "observed_metrics": observed_metrics,
        "metric_units": {name: units.get(name, "") for name in observed_metrics},
        "session_capture": None,
        "source_format": "ncu_raw_csv",
    }
    return {
        "success": True,
        "report_path": str(path),
        "result_scope": "range",
        "range_summary": {
            "result_id": provenance["result_id"],
            "nvtx_range": nvtx_range,
            "nvtx_range_raw": nvtx_range_raw,
            "range_time_ms": range_time_ms,
            "metrics": metrics_out,
        },
        "coverage_policy": provenance["coverage_policy"],
        "provenance": provenance,
        "metrics_requested": list(metrics_requested),
        "command": command,
        "stderr": stderr if stderr else None,
        "returncode": returncode,
    }


def _parse_raw_csv(csv_text: str) -> Tuple[List[str], Dict[str, str], List[Dict[str, str]]]:
    reader = csv.reader(io.StringIO(csv_text))
    rows = list(reader)
    if not rows:
        return [], {}, []

    header = rows[0]
    units_row: List[str] = rows[1] if len(rows) > 1 else []
    units: Dict[str, str] = {}
    for idx, key in enumerate(header):
        units[key] = units_row[idx] if idx < len(units_row) else ""

    records: List[Dict[str, str]] = []
    for row in rows[2:]:
        if not row:
            continue
        record: Dict[str, str] = {}
        for idx, key in enumerate(header):
            record[key] = row[idx] if idx < len(row) else ""
        records.append(record)
    return header, units, records


def _default_metrics() -> List[str]:
    # Chosen for "what kernel should I tune next?" triage: time + utilization + occupancy + resource limits.
    return [
        "gpu__time_duration.avg",
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "launch__registers_per_thread",
        "launch__shared_mem_per_block",
        "launch__occupancy_limit_blocks",
        "launch__occupancy_limit_registers",
        "launch__occupancy_limit_shared_mem",
        "launch__occupancy_limit_warps",
    ]


def _ncu_import_raw_csv(report_path: Path, metrics: Iterable[str], timeout_seconds: int) -> Tuple[int, str, str, List[str]]:
    cmd = [
        "ncu",
        "--csv",
        "--page",
        "raw",
        "--metrics",
        ",".join(metrics),
        "--import",
        str(report_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_seconds)
    return proc.returncode, proc.stdout, proc.stderr, cmd


def summarize_ncu_report(
    report_path: Path,
    *,
    top_k: int = 10,
    metrics: Optional[List[str]] = None,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """Summarize an NCU report or exported CSV by its observed result scope.

    Supports:
    - `.ncu-rep` (imports via `ncu --import ... --page raw`)
    - `.csv` exported from `ncu --csv --page raw ...` (parsed directly)
    - Companion `<report>.csv` next to `.ncu-rep` (parsed directly when present)

    Kernel rows retain the top-k table contract. A selected NVTX range row is
    returned as a range summary and is never exposed through kernel fields.
    """
    path = Path(report_path)
    if not path.exists():
        return {"success": False, "error": f"NCU report not found: {path}", "report_path": str(path)}

    if top_k is None:
        top_k = 10
    try:
        top_k_int = int(top_k)
    except Exception:
        top_k_int = 10
    top_k_int = max(1, min(200, top_k_int))

    metrics_list = metrics or _default_metrics()
    metrics_list = [m.strip() for m in metrics_list if isinstance(m, str) and m.strip()]
    if not metrics_list:
        metrics_list = _default_metrics()

    csv_text = ""
    cmd: Optional[List[str]] = None
    stderr = ""
    returncode = 0

    # Prefer parsing exported CSV if user provides one (or if a companion exists).
    if path.suffix.lower() == ".csv":
        csv_text = path.read_text(encoding="utf-8", errors="replace")
    else:
        companion_csv = path.with_suffix(".csv")
        if companion_csv.exists():
            csv_text = companion_csv.read_text(encoding="utf-8", errors="replace")
        else:
            try:
                returncode, out, stderr, cmd = _ncu_import_raw_csv(path, metrics_list, int(timeout_seconds))
                csv_text = out
            except FileNotFoundError:
                return {"success": False, "error": "ncu not found on PATH", "report_path": str(path)}
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "error": f"ncu import timed out after {timeout_seconds}s",
                    "report_path": str(path),
                    "command": cmd,
                }
            except Exception as exc:
                return {"success": False, "error": str(exc), "report_path": str(path), "command": cmd}

    header, units, records = _parse_raw_csv(csv_text)
    if not header:
        return {
            "success": False,
            "error": "Empty or unparseable NCU CSV",
            "report_path": str(path),
            "command": cmd,
            "stderr": stderr,
            "returncode": returncode,
        }

    numeric_records = [
        record for record in records if (record.get("ID") or "").strip().isdigit()
    ]
    range_records = [record for record in numeric_records if _is_ncu_range_record(record)]
    kernel_records = [record for record in numeric_records if not _is_ncu_range_record(record)]
    if range_records:
        if kernel_records:
            return {
                "success": False,
                "error": "NCU CSV mixes range and kernel result rows; refusing ambiguous summary",
                "report_path": str(path),
                "result_scope": "ambiguous",
                "range_result_count": len(range_records),
                "kernel_result_count": len(kernel_records),
                "command": cmd,
                "stderr": stderr if stderr else None,
                "returncode": returncode,
            }
        if len(range_records) != 1:
            return {
                "success": False,
                "error": (
                    "NCU range summary requires exactly one selected range row; "
                    f"observed {len(range_records)}"
                ),
                "report_path": str(path),
                "result_scope": "range",
                "range_result_count": len(range_records),
                "command": cmd,
                "stderr": stderr if stderr else None,
                "returncode": returncode,
            }

        record = range_records[0]
        metrics_out, _, _ = _parse_record_metrics(header, units, record)
        if path.suffix.lower() == ".csv":
            return _range_result_from_csv(
                path=path,
                record=record,
                metrics_out=metrics_out,
                units=units,
                metrics_requested=metrics_list,
                command=cmd,
                stderr=stderr,
                returncode=returncode,
            )

        nvtx_range, _ = _ncu_nvtx_range_metadata(record)
        try:
            from core.profiling.metrics_extractor import inspect_ncu_app_range_report

            range_metrics, provenance = inspect_ncu_app_range_report(
                path,
                expected_nvtx_range=nvtx_range or "compute_kernel:profile",
                requested_metrics=None,
                timeout=int(timeout_seconds),
            )
        except Exception as exc:
            return {
                "success": False,
                "error": f"NCU range validation failed: {exc}",
                "report_path": str(path),
                "result_scope": "range",
                "command": cmd,
                "stderr": stderr if stderr else None,
                "returncode": returncode,
            }

        if (
            provenance.get("result_scope") != "range"
            or getattr(range_metrics, "kernel_time_ms", None) is not None
            or getattr(range_metrics, "range_time_ms", None) is None
        ):
            return {
                "success": False,
                "error": "NCU range inspector returned inconsistent scope or timing semantics",
                "report_path": str(path),
                "result_scope": "range",
                "command": cmd,
                "stderr": stderr if stderr else None,
                "returncode": returncode,
            }

        validated_metrics = _range_metrics_from_model(range_metrics)
        return {
            "success": True,
            "report_path": str(path),
            "result_scope": "range",
            "range_summary": {
                "result_id": provenance.get("result_id"),
                "nvtx_range": provenance.get("nvtx_range"),
                "nvtx_range_raw": provenance.get("nvtx_range_raw"),
                "range_time_ms": float(range_metrics.range_time_ms),
                "metrics": validated_metrics,
            },
            "coverage_policy": provenance.get("coverage_policy"),
            "provenance": provenance,
            "metrics_requested": provenance.get("requested_metrics", metrics_list),
            "command": cmd,
            "stderr": stderr if stderr else None,
            "returncode": returncode,
        }

    kernels: List[Dict[str, Any]] = []
    total_time_sum_ms = 0.0

    for record in kernel_records:
        raw_id = (record.get("ID") or "").strip()
        kernel_id = int(raw_id)
        kernel_name = (record.get("Kernel Name") or record.get("launch__kernel_name") or "").strip()
        if not kernel_name:
            kernel_name = "<unknown>"

        block_size = (record.get("Block Size") or "").strip()
        grid_size = (record.get("Grid Size") or "").strip()
        stream = (record.get("Stream") or "").strip()
        device = (record.get("Device") or "").strip()
        cc = (record.get("CC") or "").strip()

        metrics_out, time_avg_ms, time_sum_ms = _parse_record_metrics(
            header, units, record
        )

        # Derive an occupancy limiting factor hint when the raw launch limits are present.
        limit_fields = {
            "blocks": metrics_out.get("launch__occupancy_limit_blocks"),
            "registers": metrics_out.get("launch__occupancy_limit_registers"),
            "shared_mem": metrics_out.get("launch__occupancy_limit_shared_mem"),
            "warps": metrics_out.get("launch__occupancy_limit_warps"),
        }
        numeric_limits = {k: v for k, v in limit_fields.items() if isinstance(v, (int, float))}
        occupancy_limit_reason: Optional[str] = None
        if numeric_limits:
            min_val = min(numeric_limits.values())
            reasons = [k for k, v in numeric_limits.items() if v == min_val]
            if reasons:
                occupancy_limit_reason = ",".join(sorted(reasons))

        # Accumulate total time for percent attribution.
        if time_sum_ms is not None:
            total_time_sum_ms += time_sum_ms

        kernels.append(
            {
                "id": kernel_id,
                "kernel_name": kernel_name,
                "block_size": block_size,
                "grid_size": grid_size,
                "stream": stream,
                "device": device,
                "cc": cc,
                "time_avg_ms": time_avg_ms,
                "time_sum_ms": time_sum_ms,
                "occupancy_limit_reason": occupancy_limit_reason,
                "metrics": metrics_out,
            }
        )

    if not kernels:
        return {
            "success": False,
            "error": "No kernel rows found in NCU CSV (expected numeric ID rows)",
            "report_path": str(path),
            "command": cmd,
            "stderr": stderr,
            "returncode": returncode,
        }

    # Prefer sum time when present (closest to "top kernels by total time").
    has_sum = any(k.get("time_sum_ms") is not None for k in kernels)
    sort_key = "time_sum_ms" if has_sum else "time_avg_ms"

    def _safe_key(k: Dict[str, Any]) -> float:
        v = k.get(sort_key)
        return float(v) if isinstance(v, (int, float)) else 0.0

    kernels_sorted = sorted(kernels, key=_safe_key, reverse=True)
    kernels_top = kernels_sorted[:top_k_int]

    if total_time_sum_ms > 0:
        for k in kernels_top:
            t = k.get("time_sum_ms")
            if isinstance(t, (int, float)):
                k["time_pct"] = 100.0 * float(t) / total_time_sum_ms

    return {
        "success": True,
        "report_path": str(path),
        "result_scope": "kernel",
        "top_k": top_k_int,
        "sort_by": sort_key,
        "kernel_count": len(kernels_sorted),
        "total_time_sum_ms": total_time_sum_ms if total_time_sum_ms > 0 else None,
        "kernels": kernels_top,
        "metrics_requested": metrics_list,
        "command": cmd,
        "stderr": stderr if stderr else None,
        "returncode": returncode,
    }
