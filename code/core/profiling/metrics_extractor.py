"""Metrics extraction from profiling tools (nsys, ncu, proton, torch).

Provides functions for extracting metrics from profiling reports and returning Pydantic models.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Type

from core.profiling.profiler_config import MINIMAL_METRICS

try:
    from core.benchmark.models import NcuMetrics, NsysMetrics, ProtonMetrics, TorchMetrics
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    NsysMetrics: Any = None  # type: ignore[no-redef]
    NcuMetrics: Any = None  # type: ignore[no-redef]
    TorchMetrics: Any = None  # type: ignore[no-redef]
    ProtonMetrics: Any = None  # type: ignore[no-redef]


# Mapping of metric identifiers to natural language descriptions
NCU_METRIC_DESCRIPTIONS = {
    "gpu__time_duration.avg": "Kernel Execution Time",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "SM Compute Throughput (% of peak)",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "DRAM/HBM Memory Throughput (% of peak)",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "L2 Cache Throughput (% of peak)",
    "sm__sass_thread_inst_executed_op_fp32_pred_on.sum": "FP32 Instructions Executed (compute proxy)",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "Achieved Occupancy (% active warps)",
    "dram__sectors_read.sum": "DRAM Sectors Read",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum": "L1 Global Memory Load Sectors",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum": "L1 Global Memory Store Sectors",
    "l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum": "Shared Memory Bank Conflicts",
    "sm__inst_executed_pipe_tensor.sum": "Tensor Core Instructions Executed",
}

NCU_APP_RANGE_DEFAULT_METRICS = tuple(MINIMAL_METRICS)

_NCU_APP_RANGE_METRIC_UNITS = {
    "gpu__time_duration.avg": "ns",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "%",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "%",
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": "%",
    "sm__warps_active.avg.pct_of_peak_sustained_active": "%",
}

_NCU_APP_RANGE_LIMITING_FLAGS = (
    "--filter-mode",
    "--kernel-id",
    "--kernel-name",
    "--launch-count",
    "--launch-skip",
    "--launch-skip-before-match",
    "--nvtx-exclude",
    "--pm-sampling-interval",
    "--range-filter",
    "--sampling-interval",
    "--sampling-max-passes",
)

NSYS_METRIC_KEY_MAP = {
    "total_gpu_time": ["nsys_total_gpu_time_ms", "nsys_total_gpu_time"],
    "kernel_time": ["nsys_kernel_time_ms", "nsys_kernel_time"],
    "memory_throughput_gb_per_s": ["nsys_memory_throughput_gb_per_s"],
    "compute_throughput_gflops": ["nsys_compute_throughput_gflops"],
    "memory_bandwidth_utilization": ["nsys_memory_bandwidth_utilization_pct"],
    "compute_utilization": ["nsys_compute_utilization_pct"],
}


def get_ncu_metric_description(metric_key: str, fallback_to_key: bool = True) -> str:
    """Get natural language description for a metric key.
    
    Args:
        metric_key: The metric identifier (cryptic ID or clean name)
        fallback_to_key: If True, return the key itself if no description found
    
    Returns:
        Natural language description, or the key itself if not found and fallback_to_key=True
    """
    # First check if it's directly in our mapping
    if metric_key in NCU_METRIC_DESCRIPTIONS:
        return NCU_METRIC_DESCRIPTIONS[metric_key]
    
    # Try to find matching cryptic ID
    clean_key = metric_key.replace("ncu_", "").replace("_pct", "").replace("_ms", "")
    for cryptic_id, description in NCU_METRIC_DESCRIPTIONS.items():
        cryptic_parts = cryptic_id.replace("__", "_").replace(".", "_").split("_")
        key_parts = clean_key.split("_")
        
        # Check if significant parts match
        if len(set(cryptic_parts) & set(key_parts)) >= 2:
            return description
        if cryptic_id.replace("__", "_").replace(".", "_") in clean_key or clean_key in cryptic_id.replace("__", "_").replace(".", "_"):
            return description
    
    # If no match found and fallback is enabled, return a cleaned version of the key
    if fallback_to_key:
        cleaned = metric_key.replace("ncu_", "").replace("__", " ").replace("_", " ").replace(".", " ")
        return cleaned.title()
    
    return metric_key


def extract_nsys_metrics(nsys_rep_path: Path, timeout: int = 180) -> NsysMetrics:
    """Extract metrics from nsys report file.
    
    Args:
        nsys_rep_path: Path to .nsys-rep file
        timeout: Timeout for nsys stats command in seconds
        
    Returns:
        NsysMetrics (Pydantic) object with extracted metrics
    """
    if not PYDANTIC_AVAILABLE or NsysMetrics is None:
        raise ImportError("pydantic and NsysMetrics are required for extract_nsys_metrics")
    
    if not nsys_rep_path.exists():
        return NsysMetrics(total_gpu_time_ms=None, raw_metrics={}, schemaVersion="1.0")
    
    total_gpu_time_ms = None
    raw_metrics = {}
    
    # Try using nsys stats command
    try:
        result = subprocess.run(
            [
                "nsys",
                "stats",
                "--force-export=true",
                "--report",
                "cuda_gpu_sum",
                "--format",
                "csv",
                str(nsys_rep_path),
            ],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0:
            csv_metrics = _parse_nsys_csv(result.stdout)
            if "nsys_total_gpu_time_ms" in csv_metrics:
                total_gpu_time_ms = csv_metrics["nsys_total_gpu_time_ms"]
            # Store other metrics in raw_metrics
            for k, v in csv_metrics.items():
                if k != "nsys_total_gpu_time_ms":
                    clean_key = k.replace("nsys_", "")
                    raw_metrics[clean_key] = v
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Also try using the extract_nsys_summary module if available
    try:
        from core.profiling.extract_nsys_summary import harvest
        harvested = harvest(nsys_rep_path)
        
        # Convert harvested metrics to dict format
        for entry in harvested:
            metric_name = entry.get("metric", "")
            value_str = entry.get("value", "")
            if metric_name and value_str:
                try:
                    value = float(value_str.replace(",", "").replace("%", ""))
                    clean_name = metric_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                    raw_metrics[clean_name] = value
                except (ValueError, AttributeError):
                    pass
    except (ImportError, SystemExit, Exception):
        pass
    
    return NsysMetrics(total_gpu_time_ms=total_gpu_time_ms, raw_metrics=raw_metrics, schemaVersion="1.0")


def _invalid_ncu_app_range(report_path: Path, detail: str) -> ValueError:
    return ValueError(f"Invalid NCU app-range report {report_path}: {detail}")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _run_ncu_report_import(
    report_path: Path,
    *,
    page: str,
    timeout: int,
    metrics: Sequence[str] | None = None,
) -> str:
    command = ["ncu", "--csv", "--page", page, "--print-units", "base"]
    if page == "details":
        command.extend(["--print-metric-name", "name"])
    if metrics:
        command.extend(["--metrics", ",".join(metrics)])
    command.extend(["--import", str(report_path)])
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise _invalid_ncu_app_range(
            report_path,
            f"{page} import timed out after {timeout}s",
        ) from exc
    except FileNotFoundError as exc:
        raise _invalid_ncu_app_range(report_path, "ncu executable is unavailable") from exc
    except OSError as exc:
        raise _invalid_ncu_app_range(
            report_path,
            f"{page} import could not start: {exc}",
        ) from exc

    if result.returncode != 0:
        raise _invalid_ncu_app_range(
            report_path,
            f"{page} import exited with code {result.returncode}",
        )
    if not result.stdout.strip():
        raise _invalid_ncu_app_range(report_path, f"{page} import returned empty CSV")
    return result.stdout


def _ncu_csv_table(
    csv_text: str,
    *,
    required_headers: Sequence[str],
    reject_overlong_rows: bool = False,
) -> tuple[list[str], list[Dict[str, str]]]:
    try:
        rows = list(csv.reader(io.StringIO(csv_text)))
    except csv.Error:
        return [], []

    header_index = None
    header: list[str] = []
    required = set(required_headers)
    for index, row in enumerate(rows):
        candidate = [cell.strip() for cell in row]
        if required.issubset(candidate):
            header_index = index
            header = candidate
            break
    if header_index is None or len(set(header)) != len(header):
        return [], []

    records: list[Dict[str, str]] = []
    for row_number, row in enumerate(rows[header_index + 1 :], start=header_index + 2):
        if len(row) > len(header):
            if reject_overlong_rows:
                raise ValueError(
                    f"CSV row {row_number} has {len(row)} columns; expected at most {len(header)}"
                )
            continue
        if not any(cell.strip() for cell in row):
            continue
        if len(row) < len(header):
            row = [*row, *([""] * (len(header) - len(row)))]
        records.append(dict(zip(header, row, strict=True)))
    return header, records


def _ncu_csv_declares_range(csv_text: str) -> bool:
    _, records = _ncu_csv_table(
        csv_text,
        required_headers=("ID", "Kernel Name"),
    )
    return any(record.get("Kernel Name", "").strip() == "range" for record in records)


def _strip_ncu_range_export_wrapping(value: str) -> str:
    normalized = value.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] == '"':
        normalized = normalized[1:-1].strip()
    return normalized


def _ncu_cli_option_values(argv: Sequence[str], option: str) -> list[str]:
    values: list[str] = []
    for index, token in enumerate(argv):
        if token == option:
            if index + 1 >= len(argv) or argv[index + 1].startswith("--"):
                values.append("")
            else:
                values.append(argv[index + 1])
        elif token.startswith(f"{option}="):
            values.append(token.split("=", 1)[1])
    return values


def _ncu_cli_has_option(argv: Sequence[str], option: str) -> bool:
    return any(token == option or token.startswith(f"{option}=") for token in argv)


def _parse_ncu_app_range_session(
    session_csv: str,
    *,
    report_path: Path,
    expected_nvtx_range: str,
    requested_metrics: Sequence[str],
) -> Dict[str, Any]:
    try:
        rows = list(csv.reader(io.StringIO(session_csv)))
    except csv.Error as exc:
        raise _invalid_ncu_app_range(report_path, "session CSV is malformed") from exc

    commands = [
        row[1].strip()
        for row in rows
        if len(row) >= 2 and row[0].strip() == "Profiler Command Line" and row[1].strip()
    ]
    if len(commands) != 1:
        raise _invalid_ncu_app_range(
            report_path,
            f"session CSV contains {len(commands)} profiler command lines; expected 1",
        )
    try:
        argv = shlex.split(commands[0])
    except ValueError as exc:
        raise _invalid_ncu_app_range(report_path, "session profiler command is not valid argv") from exc

    replay_modes = _ncu_cli_option_values(argv, "--replay-mode")
    if replay_modes != ["app-range"]:
        raise _invalid_ncu_app_range(
            report_path,
            f"session replay mode is {replay_modes!r}; expected ['app-range']",
        )
    if not _ncu_cli_has_option(argv, "--nvtx"):
        raise _invalid_ncu_app_range(report_path, "session command does not enable NVTX selection")
    nvtx_values = _ncu_cli_option_values(argv, "--nvtx")
    if any(value.strip().lower() in {"0", "false", "no", "off"} for value in nvtx_values):
        raise _invalid_ncu_app_range(report_path, "session command disables NVTX selection")

    nvtx_includes = _ncu_cli_option_values(argv, "--nvtx-include")
    if nvtx_includes != [expected_nvtx_range]:
        raise _invalid_ncu_app_range(
            report_path,
            f"session NVTX include is {nvtx_includes!r}; expected [{expected_nvtx_range!r}]",
        )

    target_processes = _ncu_cli_option_values(argv, "--target-processes")
    if target_processes != ["all"]:
        raise _invalid_ncu_app_range(
            report_path,
            f"session target processes is {target_processes!r}; expected ['all']",
        )

    metric_options = _ncu_cli_option_values(argv, "--metrics")
    if len(metric_options) != 1:
        raise _invalid_ncu_app_range(
            report_path,
            f"session command contains {len(metric_options)} metric lists; expected 1",
        )
    session_metrics = [metric.strip() for metric in metric_options[0].split(",") if metric.strip()]
    if len(session_metrics) != len(set(session_metrics)) or set(session_metrics) != set(requested_metrics):
        raise _invalid_ncu_app_range(
            report_path,
            "session metric set does not exactly match the requested metrics",
        )

    limiting_flags = [
        flag for flag in _NCU_APP_RANGE_LIMITING_FLAGS if _ncu_cli_has_option(argv, flag)
    ]
    profile_from_start = _ncu_cli_option_values(argv, "--profile-from-start")
    if len(profile_from_start) > 1 or (
        profile_from_start
        and any(
            value.strip().lower() not in {"1", "true", "yes", "on"}
            for value in profile_from_start
        )
    ):
        limiting_flags.append("--profile-from-start=off")
    if limiting_flags:
        raise _invalid_ncu_app_range(
            report_path,
            f"session command limits range coverage with {limiting_flags!r}",
        )

    return {
        "replay_mode": "app-range",
        "nvtx_enabled": True,
        "nvtx_includes": list(nvtx_includes),
        "target_processes": "all",
        "metrics": list(session_metrics),
        "limiting_flags": [],
        "profile_from_start": profile_from_start[0] if profile_from_start else "default",
        "command_sha256": hashlib.sha256(commands[0].encode("utf-8")).hexdigest(),
    }


def inspect_ncu_app_range_report(
    report_path: Path,
    *,
    expected_nvtx_range: str = "compute_kernel:profile",
    requested_metrics: Sequence[str] | None = None,
    timeout: int = 300,
) -> tuple[NcuMetrics, Dict[str, Any]]:
    """Strictly inspect one NCU application-range report.

    This path accepts only report-carried app-range evidence. It never uses a
    companion CSV and keeps aggregate range duration distinct from kernel time.
    """
    if not PYDANTIC_AVAILABLE or NcuMetrics is None:
        raise ImportError("pydantic and NcuMetrics are required for NCU app-range inspection")

    report_path = Path(report_path)
    if not report_path.is_file():
        raise _invalid_ncu_app_range(report_path, "report file does not exist")
    try:
        report_stat = report_path.stat()
    except OSError as exc:
        raise _invalid_ncu_app_range(report_path, f"report could not be inspected: {exc}") from exc
    if timeout <= 0:
        raise _invalid_ncu_app_range(report_path, "timeout must be positive")
    if not expected_nvtx_range or expected_nvtx_range.strip() != expected_nvtx_range:
        raise _invalid_ncu_app_range(report_path, "expected NVTX range must be nonempty and trimmed")

    if requested_metrics is None:
        metric_names = NCU_APP_RANGE_DEFAULT_METRICS
    elif isinstance(requested_metrics, (str, bytes)):
        raise _invalid_ncu_app_range(report_path, "requested metrics must be a sequence of names")
    else:
        metric_names = tuple(str(metric).strip() for metric in requested_metrics)
    if not metric_names or any(not metric for metric in metric_names):
        raise _invalid_ncu_app_range(report_path, "requested metrics must be nonempty")
    if len(metric_names) != len(set(metric_names)):
        raise _invalid_ncu_app_range(report_path, "requested metrics contain duplicates")
    unsupported = sorted(set(metric_names) - set(_NCU_APP_RANGE_METRIC_UNITS))
    if unsupported:
        raise _invalid_ncu_app_range(
            report_path,
            f"requested metrics have no strict unit contract: {unsupported!r}",
        )
    if "gpu__time_duration.avg" not in metric_names:
        raise _invalid_ncu_app_range(report_path, "requested metrics omit range duration")
    if set(metric_names) != set(NCU_APP_RANGE_DEFAULT_METRICS) or len(metric_names) != len(
        NCU_APP_RANGE_DEFAULT_METRICS
    ):
        raise _invalid_ncu_app_range(
            report_path,
            "app-range qualification requires the five minimal metrics",
        )

    details_csv = _run_ncu_report_import(
        report_path,
        page="details",
        timeout=timeout,
        metrics=metric_names,
    )
    session_csv = _run_ncu_report_import(
        report_path,
        page="session",
        timeout=timeout,
    )

    try:
        header, records = _ncu_csv_table(
            details_csv,
            required_headers=(
                "ID",
                "Kernel Name",
                "Section Name",
                "Metric Name",
                "Metric Unit",
                "Metric Value",
            ),
            reject_overlong_rows=True,
        )
    except ValueError as exc:
        raise _invalid_ncu_app_range(report_path, f"details {exc}") from exc
    if not header:
        raise _invalid_ncu_app_range(report_path, "details CSV is missing the required header")
    range_headers = [name for name in header if name.startswith("Id:Domain:Start/Stop_Range:")]
    if len(range_headers) != 1:
        raise _invalid_ncu_app_range(
            report_path,
            f"details CSV contains {len(range_headers)} start/stop range columns; expected 1",
        )
    range_header = range_headers[0]

    metric_records = [
        record
        for record in records
        if record.get("Section Name", "").strip() == "Command line profiler metrics"
    ]
    if not metric_records:
        raise _invalid_ncu_app_range(report_path, "details CSV has no command-line metric rows")

    data_records = [record for record in records if record.get("ID", "").strip().isdigit()]
    if not data_records:
        raise _invalid_ncu_app_range(report_path, "details CSV has no numeric result records")
    result_ids = {record.get("ID", "").strip() for record in data_records}
    if len(result_ids) != 1:
        raise _invalid_ncu_app_range(
            report_path,
            f"details CSV contains {len(result_ids)} logical result IDs; expected 1",
        )
    result_id = next(iter(result_ids))
    if any(record.get("Kernel Name", "").strip() != "range" for record in data_records):
        raise _invalid_ncu_app_range(report_path, "details result scope is not exclusively 'range'")
    if any(record.get("ID", "").strip() != result_id for record in metric_records):
        raise _invalid_ncu_app_range(
            report_path,
            "details command-line metric rows do not belong to the numeric range result",
        )

    range_cells = {record.get(range_header, "") for record in data_records}
    if len(range_cells) != 1:
        raise _invalid_ncu_app_range(report_path, "details metric rows disagree on range identity")
    range_cell_raw = next(iter(range_cells))
    range_identity = _strip_ncu_range_export_wrapping(range_cell_raw)
    identity_prefix = f"{result_id}:<default domain>:"
    identity_suffix = ":none:none:none:none:none:none"
    if not range_identity.startswith(identity_prefix) or not range_identity.endswith(identity_suffix):
        raise _invalid_ncu_app_range(report_path, "details start/stop range encoding is unrecognized")
    observed_nvtx_range = range_identity[
        len(identity_prefix) : len(range_identity) - len(identity_suffix)
    ]
    if observed_nvtx_range != expected_nvtx_range:
        raise _invalid_ncu_app_range(
            report_path,
            f"details NVTX range is {observed_nvtx_range!r}; expected {expected_nvtx_range!r}",
        )

    by_metric: Dict[str, list[Dict[str, str]]] = {}
    for record in metric_records:
        by_metric.setdefault(record.get("Metric Name", "").strip(), []).append(record)
    if set(by_metric) != set(metric_names):
        missing = sorted(set(metric_names) - set(by_metric))
        extra = sorted(set(by_metric) - set(metric_names))
        raise _invalid_ncu_app_range(
            report_path,
            f"details metric set mismatch (missing={missing!r}, extra={extra!r})",
        )
    duplicate_metrics = sorted(name for name, rows in by_metric.items() if len(rows) != 1)
    if duplicate_metrics:
        raise _invalid_ncu_app_range(
            report_path,
            f"details command-line section has duplicate metrics: {duplicate_metrics!r}",
        )

    values: Dict[str, float] = {}
    units: Dict[str, str] = {}
    for metric_name in metric_names:
        record = by_metric[metric_name][0]
        unit = record.get("Metric Unit", "").strip()
        expected_unit = _NCU_APP_RANGE_METRIC_UNITS[metric_name]
        if unit != expected_unit:
            raise _invalid_ncu_app_range(
                report_path,
                f"metric {metric_name!r} uses unit {unit!r}; expected {expected_unit!r}",
            )
        value_text = record.get("Metric Value", "").strip().replace(",", "")
        try:
            value = float(value_text)
        except ValueError as exc:
            raise _invalid_ncu_app_range(
                report_path,
                f"metric {metric_name!r} has nonnumeric value {value_text!r}",
            ) from exc
        if not math.isfinite(value):
            raise _invalid_ncu_app_range(
                report_path,
                f"metric {metric_name!r} has a nonfinite value",
            )
        values[metric_name] = value
        units[metric_name] = unit
    if values["gpu__time_duration.avg"] <= 0:
        raise _invalid_ncu_app_range(report_path, "range duration must be positive")

    session_capture = _parse_ncu_app_range_session(
        session_csv,
        report_path=report_path,
        expected_nvtx_range=expected_nvtx_range,
        requested_metrics=metric_names,
    )
    try:
        report_sha256 = _sha256_path(report_path)
        final_report_stat = report_path.stat()
    except OSError as exc:
        raise _invalid_ncu_app_range(report_path, f"report could not be hashed: {exc}") from exc
    report_identity = (
        report_stat.st_dev,
        report_stat.st_ino,
        report_stat.st_size,
        report_stat.st_mtime_ns,
    )
    final_report_identity = (
        final_report_stat.st_dev,
        final_report_stat.st_ino,
        final_report_stat.st_size,
        final_report_stat.st_mtime_ns,
    )
    if report_identity != final_report_identity:
        raise _invalid_ncu_app_range(report_path, "report changed during inspection")

    metrics = NcuMetrics(
        kernel_time_ms=None,
        range_time_ms=values["gpu__time_duration.avg"] / 1e6,
        sm_throughput_pct=values.get("sm__throughput.avg.pct_of_peak_sustained_elapsed"),
        dram_throughput_pct=values.get(
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"
        ),
        l2_throughput_pct=values.get("lts__throughput.avg.pct_of_peak_sustained_elapsed"),
        occupancy_pct=values.get("sm__warps_active.avg.pct_of_peak_sustained_active"),
        raw_metrics={},
        schemaVersion="1.0",
    )
    provenance: Dict[str, Any] = {
        "schema_version": "1.0",
        "report_sha256": report_sha256,
        "report_bytes": final_report_stat.st_size,
        "replay_mode": "app-range",
        "result_scope": "range",
        "result_id": int(result_id),
        "result_count": 1,
        "nvtx_range": observed_nvtx_range,
        "nvtx_range_raw": range_cell_raw,
        "coverage_policy": "full_selected_nvtx_range",
        "constituent_kernels_enumerated": False,
        "duration_semantics": "ncu_aggregate_range",
        "requested_metrics": list(metric_names),
        "observed_metrics": list(metric_names),
        "metric_units": units,
        "session_capture": session_capture,
    }
    return metrics, provenance


def extract_ncu_metrics(ncu_rep_path: Path, timeout: int = 300) -> NcuMetrics:
    """Extract metrics from ncu report file.
    
    Args:
        ncu_rep_path: Path to .ncu-rep file
        timeout: Timeout for ncu command in seconds
        
    Returns:
        NcuMetrics (Pydantic) object with extracted metrics
    """
    if not PYDANTIC_AVAILABLE or NcuMetrics is None:
        raise ImportError("pydantic and NcuMetrics are required for extract_ncu_metrics")
    
    if not ncu_rep_path.exists():
        return NcuMetrics(
            kernel_time_ms=None,
            range_time_ms=None,
            sm_throughput_pct=None,
            dram_throughput_pct=None,
            l2_throughput_pct=None,
            occupancy_pct=None,
            raw_metrics={},
            schemaVersion="1.0"
        )
    
    kernel_time_ms = None
    sm_throughput_pct = None
    dram_throughput_pct = None
    l2_throughput_pct = None
    occupancy_pct = None
    raw_metrics = {}
    
    # Try using ncu CLI to export metrics
    try:
        # Use --page details (Metric Name/Unit/Value rows) so we can honor units
        # while keeping output small via --metrics filtering.
        metrics = list(NCU_APP_RANGE_DEFAULT_METRICS)
        result = subprocess.run(
            ["ncu", "--csv", "--page", "details", "--metrics", ",".join(metrics), "--import", str(ncu_rep_path)],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        if result.returncode == 0 and _ncu_csv_declares_range(result.stdout):
            try:
                range_metrics, _ = inspect_ncu_app_range_report(
                    ncu_rep_path,
                    requested_metrics=metrics,
                    timeout=timeout,
                )
                return range_metrics
            except ValueError:
                # A recognized range must pass the strict report-side contract.
                # Never reinterpret it as a kernel or qualify it from a sidecar.
                return NcuMetrics(
                    kernel_time_ms=None,
                    range_time_ms=None,
                    sm_throughput_pct=None,
                    dram_throughput_pct=None,
                    l2_throughput_pct=None,
                    occupancy_pct=None,
                    raw_metrics={},
                    schemaVersion="1.0",
                )
        if result.returncode == 0:
            csv_metrics = _parse_ncu_csv(result.stdout)
            kernel_time_ms, sm_throughput_pct, dram_throughput_pct, l2_throughput_pct, occupancy_pct, raw_metrics = _populate_ncu_metrics(csv_metrics)
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        pass
    
    # Also check for companion CSV file
    companion_csv = ncu_rep_path.with_suffix(".csv")
    if companion_csv.exists():
        try:
            csv_text = companion_csv.read_text()
            if _ncu_csv_declares_range(csv_text):
                return NcuMetrics(
                    kernel_time_ms=kernel_time_ms,
                    range_time_ms=None,
                    sm_throughput_pct=sm_throughput_pct,
                    dram_throughput_pct=dram_throughput_pct,
                    l2_throughput_pct=l2_throughput_pct,
                    occupancy_pct=occupancy_pct,
                    raw_metrics=raw_metrics,
                    schemaVersion="1.0",
                )
            csv_metrics = _parse_ncu_csv(csv_text)
            kt, sm, dram, l2, occ, raw = _populate_ncu_metrics(csv_metrics)
            if kernel_time_ms is None and kt is not None:
                kernel_time_ms = kt
            if sm_throughput_pct is None and sm is not None:
                sm_throughput_pct = sm
            if dram_throughput_pct is None and dram is not None:
                dram_throughput_pct = dram
            if l2_throughput_pct is None and l2 is not None:
                l2_throughput_pct = l2
            if occupancy_pct is None and occ is not None:
                occupancy_pct = occ
            raw_metrics.update(raw)
        except (ValueError, KeyError, OSError):
            pass  # CSV parsing failed or file error
    
    return NcuMetrics(
        kernel_time_ms=kernel_time_ms,
        range_time_ms=None,
        sm_throughput_pct=sm_throughput_pct,
        dram_throughput_pct=dram_throughput_pct,
        l2_throughput_pct=l2_throughput_pct,
        occupancy_pct=occupancy_pct,
        raw_metrics=raw_metrics,
        schemaVersion="1.0"
    )


def extract_proton_metrics(report_path: Path) -> ProtonMetrics:
    """Extract Proton kernel summaries from a JSON report."""
    if not PYDANTIC_AVAILABLE or ProtonMetrics is None:
        raise ImportError("pydantic and ProtonMetrics are required for extract_proton_metrics")
    
    if not report_path.exists():
        return ProtonMetrics(
            kernel_count=None,
            occupancy_limited_kernels=[],
            summary_stats={},
            kernel_summaries=[],
            schemaVersion="1.0",
        )
    
    try:
        data = json.loads(report_path.read_text())
    except Exception:
        # Return minimal object with parse failure note
        return ProtonMetrics(
            kernel_count=None,
            occupancy_limited_kernels=[],
            summary_stats={"parse_error": 1.0},
            kernel_summaries=[],
            schemaVersion="1.0",
        )
    
    kernel_entries = []
    if isinstance(data, dict):
        for key in ("kernels", "kernel_reports", "results"):
            if key in data and isinstance(data[key], list):
                kernel_entries = data[key]
                break
        if not kernel_entries and "data" in data and isinstance(data["data"], list):
            kernel_entries = data["data"]
    elif isinstance(data, list):
        kernel_entries = data
    
    summaries = []
    occupancy_limited: list[str] = []
    summary_stats: Dict[str, float] = {}
    
    def _maybe_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    
    for entry in kernel_entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name") or entry.get("kernel") or entry.get("kernel_name")
        regs = entry.get("registers_per_thread") or entry.get("regs_per_thread") or entry.get("registers")
        smem = entry.get("shared_memory_bytes") or entry.get("shared_memory") or entry.get("smem_bytes")
        blocks_per_sm = entry.get("blocks_per_sm") or entry.get("cta_per_sm")
        occupancy = entry.get("occupancy_pct") or entry.get("theoretical_occupancy") or entry.get("occupancy")
        time_ms = entry.get("time_ms") or entry.get("duration_ms") or entry.get("total_time_ms")
        tma_desc = entry.get("tma_descriptors") or entry.get("tma") or entry.get("mma_tma")
        
        reg_f = _maybe_float(regs)
        smem_f = _maybe_float(smem)
        occupancy_f = _maybe_float(occupancy)
        blocks_f = _maybe_float(blocks_per_sm)
        time_f = _maybe_float(time_ms)
        
        summaries.append(
            {
                "name": name,
                "regs_per_thread": reg_f,
                "shared_mem_bytes": smem_f,
                "blocks_per_sm": blocks_f,
                "occupancy_pct": occupancy_f,
                "time_ms": time_f,
                "tma_descriptors": tma_desc if isinstance(tma_desc, (int, float, str)) else None,
            }
        )
        
        if occupancy_f is not None and occupancy_f < 40.0:
            occupancy_limited.append(name or "unknown_kernel")
    
    if summaries:
        regs_max = max((s.get("regs_per_thread") for s in summaries if s.get("regs_per_thread") is not None), default=None)
        smem_max = max((s.get("shared_mem_bytes") for s in summaries if s.get("shared_mem_bytes") is not None), default=None)
        blocks_max = max((s.get("blocks_per_sm") for s in summaries if s.get("blocks_per_sm") is not None), default=None)
        time_max = max((s.get("time_ms") for s in summaries if s.get("time_ms") is not None), default=None)
        if regs_max is not None:
            summary_stats["max_regs_per_thread"] = regs_max
        if smem_max is not None:
            summary_stats["max_shared_mem_bytes"] = smem_max
        if blocks_max is not None:
            summary_stats["max_blocks_per_sm"] = blocks_max
        if time_max is not None:
            summary_stats["max_time_ms"] = time_max
    
    return ProtonMetrics(
        kernel_count=len(summaries),
        occupancy_limited_kernels=occupancy_limited,
        summary_stats=summary_stats,
        kernel_summaries=summaries,
        schemaVersion="1.0",
    )


def _parse_nsys_csv(csv_text: str) -> Dict[str, float]:
    """Parse nsys CSV output for timing and bandwidth metrics.
    
    NSYS CSV format has header row: "Metric,Value"
    Example:
        Metric,Value
        Total GPU Time,1234.56
        Memory Throughput GB/s,500.25
    
    Args:
        csv_text: CSV text from nsys stats command
        
    Returns:
        Dictionary of metric names to values
    """
    metrics = {}
    
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        return metrics

    # Format A (legacy): "Metric,Value" with rows like "Total GPU Time,123.45"
    header = [c.strip() for c in lines[0].split(",")]
    if len(header) >= 2 and header[0].strip().lower() == "metric":
        for line in lines[1:]:
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            metric_name = parts[0]
            value_str = parts[1]
            if not metric_name or not value_str:
                continue
            try:
                value = float(value_str)
            except ValueError:
                continue
            clean_name = (
                metric_name.lower()
                .replace(" ", "_")
                .replace("/", "_per_")
                .replace("(", "")
                .replace(")", "")
            )
            mapped_keys = NSYS_METRIC_KEY_MAP.get(clean_name)
            if not mapped_keys:
                continue
            for target in mapped_keys:
                metrics[target] = value
        return metrics

    # Format B (current): cuda_gpu_sum table CSV with header like:
    # "Time (%),Total Time (ns),Instances,...,Category,Operation"
    header_idx = None
    for idx, line in enumerate(lines):
        if "Total Time (ns)" in line and "Category" in line and "Operation" in line:
            header_idx = idx
            break
    if header_idx is None:
        return metrics

    table_lines = lines[header_idx:]
    try:
        reader = csv.DictReader(io.StringIO("\n".join(table_lines)))
        total_time_ns = 0.0
        kernel_time_ns = 0.0
        for row in reader:
            try:
                time_ns = float(row.get("Total Time (ns)", "") or 0.0)
            except ValueError:
                continue
            total_time_ns += time_ns
            if (row.get("Category") or "").strip() == "CUDA_KERNEL":
                kernel_time_ns += time_ns
        if total_time_ns > 0:
            metrics["nsys_total_gpu_time_ms"] = total_time_ns / 1e6
        if kernel_time_ns > 0:
            metrics["nsys_kernel_time_ms"] = kernel_time_ns / 1e6
    except Exception:
        return metrics
    
    return metrics


def _parse_ncu_csv(csv_text: str) -> Dict[str, float]:
    """Parse ncu CSV output for comprehensive roofline and performance metrics.
    
    NCU CSV format is "metric,value" per line with no header.
    Example:
        "gpu__time_duration.avg","10.500"
        "sm__throughput.avg.pct_of_peak_sustained_elapsed","85.25"
    
    Args:
        csv_text: CSV text from ncu export or companion CSV file
        
    Returns:
        Dictionary of metric identifiers to values
    """
    metrics: Dict[str, float] = {}

    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        return metrics

    # Format A (legacy): "metric","value" per line (no header)
    if len(lines) >= 1 and lines[0].count(",") == 1 and lines[0].startswith('"') and lines[0].endswith('"'):
        for line in lines:
            try:
                row = next(csv.reader(io.StringIO(line)))
            except Exception:
                continue
            if len(row) < 2:
                continue
            metric_name = (row[0] or "").strip()
            value_str = (row[1] or "").strip()
            if not metric_name or not value_str:
                continue
            try:
                metrics[metric_name] = float(value_str)
            except ValueError:
                continue
        return metrics

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
        # Nsight Compute CSV usually uses base units; treat unknown as-is.
        return value

    # Format B (current): header + rows (ncu --page details/raw --import ...)
    try:
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
    except Exception:
        return metrics

    fieldnames = reader.fieldnames or []
    has_metric_name = "Metric Name" in fieldnames and "Metric Value" in fieldnames

    # B1) Details page: one row per (kernel, metric) with units.
    if has_metric_name:
        per_kernel: Dict[str, Dict[str, float]] = {}
        per_kernel_time: Dict[str, float] = {}

        for row in reader:
            kernel_id = (row.get("ID") or "").strip()
            if not kernel_id.isdigit():
                continue
            metric_name = (row.get("Metric Name") or "").strip()
            metric_value_raw = (row.get("Metric Value") or "").strip()
            if not metric_name or not metric_value_raw:
                continue
            value = _parse_float(metric_value_raw)
            if value is None:
                continue
            unit = (row.get("Metric Unit") or "").strip()
            if metric_name.startswith("gpu__time_duration"):
                value = _time_to_ms(value, unit)
            per_kernel.setdefault(kernel_id, {})[metric_name] = value
            if metric_name == "gpu__time_duration.avg":
                per_kernel_time[kernel_id] = value

        if not per_kernel:
            return metrics

        # Choose the kernel with the highest avg duration (dominant kernel).
        best_kernel_id = max(per_kernel_time, key=per_kernel_time.get) if per_kernel_time else sorted(per_kernel.keys())[0]
        metrics.update(per_kernel[best_kernel_id])
        return metrics

    # B2) Raw page: one row per kernel with metrics as columns (no units).
    best_row: Optional[Dict[str, str]] = None
    best_time = -1.0
    for row in reader:
        row_id = (row.get("ID") or "").strip()
        if not row_id.isdigit():
            continue
        time_val = _parse_float((row.get("gpu__time_duration.avg") or "").strip() if row.get("gpu__time_duration.avg") else "")
        time_val_num = time_val if time_val is not None else 0.0
        if best_row is None or time_val_num > best_time:
            best_row = row
            best_time = time_val_num

    if best_row is None:
        return metrics

    for key, value_str in best_row.items():
        if not key:
            continue
        value = _parse_float(str(value_str))
        if value is None:
            continue
        # Heuristic: gpu__time_duration.* is usually printed in microseconds in CSV mode.
        if key.startswith("gpu__time_duration"):
            value = value / 1e3
        metrics[key] = value

    return metrics


def _populate_ncu_metrics(csv_metrics: Dict[str, float]) -> tuple:
    """Extract NcuMetrics fields from parsed CSV metrics.
    
    Args:
        csv_metrics: Dictionary of metric identifiers to values
        
    Returns:
        Tuple of (kernel_time_ms, sm_throughput_pct, dram_throughput_pct, l2_throughput_pct, occupancy_pct, raw_metrics)
    """
    kernel_time_ms = None
    sm_throughput_pct = None
    dram_throughput_pct = None
    l2_throughput_pct = None
    occupancy_pct = None
    raw_metrics = {}
    recognized_keys = set()
    
    # Map known metric IDs to fields
    kernel_key = "gpu__time_duration.avg"
    if kernel_key in csv_metrics:
        kernel_time_ms = csv_metrics[kernel_key]
        recognized_keys.add(kernel_key)
    
    sm_key = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    if sm_key in csv_metrics:
        sm_throughput_pct = csv_metrics[sm_key]
        recognized_keys.add(sm_key)
    
    dram_keys = [
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    ]
    for key in dram_keys:
        if key in csv_metrics:
            dram_throughput_pct = csv_metrics[key]
            recognized_keys.add(key)
            break
    
    l2_keys = [
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "l2__throughput.avg.pct_of_peak_sustained_elapsed",
    ]
    for key in l2_keys:
        if key in csv_metrics:
            l2_throughput_pct = csv_metrics[key]
            recognized_keys.add(key)
            break
    
    occupancy_key = "sm__warps_active.avg.pct_of_peak_sustained_active"
    if occupancy_key in csv_metrics:
        occupancy_pct = csv_metrics[occupancy_key]
        recognized_keys.add(occupancy_key)
    
    # Store all other metrics in raw_metrics
    for key, value in csv_metrics.items():
        if key in recognized_keys:
            continue
        clean_key = key.replace("ncu_", "") if key.startswith("ncu_") else key
        raw_metrics[clean_key] = value
    
    return (kernel_time_ms, sm_throughput_pct, dram_throughput_pct, l2_throughput_pct, occupancy_pct, raw_metrics)
