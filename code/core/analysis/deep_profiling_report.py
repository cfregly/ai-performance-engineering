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
        --hardware-profile b200 \\
        --output-json artifacts/runs/analysis/double_buffered_pipeline_analysis.json

The script understands the CSV format produced by either:
* `ncu --set roofline --csv ...`
* `python core/profiling/extract_ncu_metrics.py --example <name>`

It does not require GPU access to run; it analyses offline artifacts. Roofline
ceilings come from an exact device name plus compute capability recorded in the
artifact and validated against a reviewed profile, or from an explicit, exact,
reviewed ``--hardware-profile``. Compute capability alone cannot identify a GPU
SKU or its peaks. If neither source establishes an exact SKU, the report records
an unknown hardware state and omits ceiling-based classification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
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

from core.analysis.kernel_roofline import (
    VALIDATED_HARDWARE_PROFILES,
    ArchitectureSpecs,
    RooflineAnalyzer,
    get_architecture_specs_for_profile,
)
from core.benchmark.metrics import hardware_specs_for_device
from core.profiling.nsight_systems import NsightSystemsReportParser


class ConflictingMetricError(ValueError):
    """Raised when two artifacts assign different values to one launch metric."""


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
    capture_id: str | None = None
    source_artifacts: tuple[str, ...] = ()
    device_name: str | None = None
    compute_capability: tuple[int, int] | None = None
    artifact_identity_invalid: bool = False

    def get(self, *candidates: str) -> Optional[RawMetric]:
        for candidate in candidates:
            if candidate in self.metrics:
                return self.metrics[candidate]
        return None

    def get_value(self, *candidates: str) -> Optional[float]:
        entry = self.get(*candidates)
        return entry.value if entry else None


@dataclass
class _CaptureIdentityEvidence:
    pairs: set[tuple[str, tuple[int, int]]] = field(default_factory=set)
    device_names: set[str] = field(default_factory=set)
    capabilities: set[tuple[int, int]] = field(default_factory=set)
    invalid: bool = False
    incomplete: bool = False


@dataclass(frozen=True)
class HardwareSelection:
    """Hardware ceilings and their artifact-safe provenance."""

    profile: str
    provenance: str
    specs: ArchitectureSpecs | None
    compute_capability: str | None = None

    def as_dict(self) -> dict[str, object]:
        specs = self.specs
        return {
            "profile": self.profile,
            "provenance": self.provenance,
            "compute_capability": self.compute_capability,
            "specs": None
            if specs is None
            else {
                "name": specs.name,
                "peak_fp32_tflops": specs.peak_fp32_tflops,
                "peak_fp16_tflops": specs.peak_fp16_tflops,
                "peak_tensor_fp16_tflops": specs.peak_tensor_fp16_tflops,
                "peak_fp8_tflops": specs.peak_fp8_tflops,
                "peak_tf32_tflops": specs.peak_tf32_tflops,
                "memory_bandwidth_gbs": specs.memory_bandwidth_gbs,
                "cpu_gpu_bandwidth_gbs": specs.cpu_gpu_bandwidth_gbs,
                "profile_source": specs.profile_source,
            },
        }


@dataclass
class RooflineSummary:
    achieved_tflops: float
    achieved_bandwidth_gbs: float
    arithmetic_intensity: float
    compute_utilization_pct: Optional[float]
    memory_utilization_pct: float
    tmem_utilization_pct: Optional[float]
    l2_utilization_pct: Optional[float]
    binding: str
    is_memory_bound: bool
    is_compute_bound: bool
    is_tmem_bound: bool
    ridge_point: Optional[float]
    memory_bound_limit_tflops: float
    peak_tflops: Optional[float]
    peak_bandwidth_gbs: float


@dataclass
class Advisory:
    kernel: str
    capture_id: str | None
    source_artifacts: tuple[str, ...]
    device_name: str | None
    compute_capability: str | None
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
]

NCU_LAUNCH_ID_KEYS = ("ID", "Kernel ID", "Launch ID")
NCU_HOST_KEYS = ("Host Name", "Hostname", "Host")
NCU_PROCESS_KEYS = ("Process ID", "PID")
NCU_DEVICE_ID_KEYS = ("Device ID", "GPU ID")
NCU_DEVICE_NAME_KEYS = ("Device", "Device Name", "GPU Name")
NCU_COMPUTE_CAPABILITY_KEYS = ("CC", "Compute Capability")
NCU_CONTEXT_KEYS = ("Context", "Context ID")
NCU_STREAM_KEYS = ("Stream", "Stream ID")

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

HARDWARE_PROFILES = VALIDATED_HARDWARE_PROFILES
COMPUTE_CAPABILITY_MAJOR_KEYS = (
    "device__attribute_compute_capability_major",
    "gpu__compute_capability_major",
    "Compute Capability Major",
)
COMPUTE_CAPABILITY_MINOR_KEYS = (
    "device__attribute_compute_capability_minor",
    "gpu__compute_capability_minor",
    "Compute Capability Minor",
)
PROFILE_COMPUTE_CAPABILITIES = {
    "b200": (10, 0),
    "h100-sxm": (9, 0),
}
DEEP_PROFILING_REPORT_SCHEMA_VERSION = "deep_profiling_report.v1"


def resolve_hardware_selection(
    kernel_metrics: Iterable[KernelMetrics],
    declared_profile: str | None,
) -> HardwareSelection:
    """Resolve roofline ceilings without consulting the reporting host."""
    capabilities: set[tuple[int, int]] = set()
    incomplete_capabilities: set[int] = set()
    device_names: set[str] = set()
    capture_evidence: dict[str, _CaptureIdentityEvidence] = {}
    has_unscoped_metrics = False
    invalid_capability = False

    def capability_component(value: float) -> int | None:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(numeric) or numeric < 0 or not numeric.is_integer():
            return None
        return int(numeric)

    for metrics in kernel_metrics:
        device_name = metrics.device_name
        local_capability = metrics.compute_capability
        local_invalid = metrics.artifact_identity_invalid
        local_incomplete = False
        if device_name:
            device_names.add(device_name)
        if local_capability is not None:
            capabilities.add(local_capability)
        if metrics.artifact_identity_invalid:
            invalid_capability = True
        major_metric = metrics.get(*COMPUTE_CAPABILITY_MAJOR_KEYS)
        if major_metric is not None:
            major = capability_component(major_metric.value)
            if major is None:
                invalid_capability = True
                local_invalid = True
            else:
                minor_metric = metrics.get(*COMPUTE_CAPABILITY_MINOR_KEYS)
                if minor_metric is None:
                    incomplete_capabilities.add(major)
                    local_incomplete = True
                else:
                    minor = capability_component(minor_metric.value)
                    if minor is None:
                        invalid_capability = True
                        local_invalid = True
                    else:
                        metric_capability = (major, minor)
                        capabilities.add(metric_capability)
                        if (
                            local_capability is not None
                            and local_capability != metric_capability
                        ):
                            invalid_capability = True
                            local_invalid = True
                        else:
                            local_capability = metric_capability
        if metrics.capture_id is None:
            has_unscoped_metrics = True
            continue
        evidence = capture_evidence.setdefault(
            metrics.capture_id,
            _CaptureIdentityEvidence(),
        )
        if device_name is not None:
            evidence.device_names.add(device_name)
        if local_capability is not None:
            evidence.capabilities.add(local_capability)
        if device_name is not None and local_capability is not None:
            evidence.pairs.add((device_name, local_capability))
        evidence.invalid = evidence.invalid or local_invalid
        evidence.incomplete = evidence.incomplete or local_incomplete

    exact_artifact_profile: str | None = None
    validated_capture_profiles: list[str] = []
    if capture_evidence and not has_unscoped_metrics:
        for evidence in capture_evidence.values():
            pairs = evidence.pairs
            if evidence.invalid or evidence.incomplete or len(pairs) != 1:
                break
            device_name, capability = next(iter(pairs))
            if evidence.device_names != {device_name} or evidence.capabilities != {
                capability
            }:
                break
            try:
                reviewed = hardware_specs_for_device(device_name, capability)
            except ValueError:
                break
            matched_profile = next(
                (
                    candidate
                    for candidate in HARDWARE_PROFILES
                    if get_architecture_specs_for_profile(candidate).name == reviewed.name
                ),
                None,
            )
            if matched_profile is None:
                break
            validated_capture_profiles.append(matched_profile)
        if (
            len(validated_capture_profiles) == len(capture_evidence)
            and len(set(validated_capture_profiles)) == 1
        ):
            exact_artifact_profile = validated_capture_profiles[0]

    if declared_profile is not None:
        profile = declared_profile.strip().lower().replace("_", "-")
        specs = get_architecture_specs_for_profile(profile)
        expected_capability = PROFILE_COMPUTE_CAPABILITIES[profile]
        declared_values = [
            *(f"{major}.{minor}" for major, minor in sorted(capabilities)),
            *(f"{major}.x" for major in sorted(incomplete_capabilities)),
        ]
        artifact_capability = ",".join(declared_values) or None
        if invalid_capability:
            return HardwareSelection(
                profile=profile,
                provenance="explicit_cli_artifact_compute_capability_invalid",
                specs=None,
                compute_capability="invalid",
            )
        if incomplete_capabilities or len(capabilities) > 1 or len(device_names) > 1:
            return HardwareSelection(
                profile=profile,
                provenance="explicit_cli_artifact_compute_capability_ambiguous",
                specs=None,
                compute_capability=artifact_capability,
            )
        if capabilities and next(iter(capabilities)) != expected_capability:
            return HardwareSelection(
                profile=profile,
                provenance="explicit_cli_artifact_compute_capability_mismatch",
                specs=None,
                compute_capability=artifact_capability,
            )
        for device_name in device_names:
            try:
                reviewed = hardware_specs_for_device(device_name, expected_capability)
            except ValueError:
                return HardwareSelection(
                    profile=profile,
                    provenance="explicit_cli_artifact_device_identity_unvalidated",
                    specs=None,
                    compute_capability=artifact_capability,
                )
            if reviewed.name != specs.name:
                return HardwareSelection(
                    profile=profile,
                    provenance="explicit_cli_artifact_device_identity_mismatch",
                    specs=None,
                    compute_capability=artifact_capability,
                )
        return HardwareSelection(
            profile=profile,
            provenance=(
                "explicit_cli_artifact_compute_capability_compatible"
                if capabilities
                else "explicit_cli"
            ),
            specs=specs,
            compute_capability=artifact_capability,
        )

    if invalid_capability:
        return HardwareSelection(
            profile="unknown",
            provenance="source_ncu_compute_capability_invalid",
            specs=None,
            compute_capability="invalid",
        )

    if exact_artifact_profile is not None:
        major, minor = next(iter(capabilities))
        return HardwareSelection(
            profile=exact_artifact_profile,
            provenance="source_ncu_exact_device_identity",
            specs=get_architecture_specs_for_profile(exact_artifact_profile),
            compute_capability=f"{major}.{minor}",
        )

    if device_names:
        capability = None
        if len(capabilities) == 1:
            major, minor = next(iter(capabilities))
            capability = f"{major}.{minor}"
        return HardwareSelection(
            profile="unknown",
            provenance=(
                "source_ncu_device_identity_ambiguous"
                if len(device_names) > 1 or len(capabilities) > 1
                else "source_ncu_device_identity_unvalidated"
            ),
            specs=None,
            compute_capability=capability,
        )

    if len(capabilities) == 1 and not incomplete_capabilities:
        major, minor = next(iter(capabilities))
        capability = f"{major}.{minor}"
        return HardwareSelection(
            profile="unknown",
            provenance="source_ncu_compute_capability_only",
            specs=None,
            compute_capability=capability,
        )

    if capabilities or incomplete_capabilities:
        declared_values = [
            *(f"{major}.{minor}" for major, minor in sorted(capabilities)),
            *(f"{major}.x" for major in sorted(incomplete_capabilities)),
        ]
        declared = ",".join(declared_values)
        return HardwareSelection(
            profile="unknown",
            provenance=(
                "source_ncu_compute_capability_incomplete"
                if incomplete_capabilities and not capabilities
                else "source_ncu_compute_capability_ambiguous"
            ),
            specs=None,
            compute_capability=declared,
        )

    return HardwareSelection(
        profile="unknown",
        provenance="default_unknown",
        specs=None,
    )


def parse_float(text: str) -> Optional[float]:
    """Parse finite Nsight numbers without repairing malformed evidence."""
    if text is None:
        return None
    stripped = text.strip()
    if not stripped or stripped.lower() in {"n/a", "nan"}:
        return None
    # Nsight can append a human-readable unit in a separate parenthetical
    # field.  Strip only one whitespace-delimited suffix so corrupt values such
    # as ``1(foo)0`` cannot be silently repaired into valid evidence.
    stripped = re.sub(r"\s+\([^()]*\)\s*$", "", stripped, count=1)
    if stripped.endswith("%"):
        stripped = stripped[:-1]
    stripped = stripped.strip()
    plain_number = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
    grouped_number = r"[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][+-]?\d+)?"
    match = re.fullmatch(rf"(?:{plain_number}|{grouped_number})", stripped)
    if not match:
        return None
    try:
        value = float(match.group(0).replace(",", ""))
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def _paths_refer_to_same_file(first: Path, second: Path) -> bool:
    """Return true for lexical, symlink, or hard-link aliases."""
    first_path = first.expanduser().resolve()
    second_path = second.expanduser().resolve()
    if first_path == second_path:
        return True
    try:
        return os.path.samefile(first_path, second_path)
    except (FileNotFoundError, OSError):
        return False


def first_key(row: Dict[str, str], keys: Sequence[str]) -> Optional[str]:
    for key in keys:
        if key in row and row[key].strip():
            return row[key].strip()
    return None


def parse_compute_capability(text: str) -> tuple[int, int] | None:
    """Parse the explicit major.minor form emitted by Nsight Compute CSV."""
    match = re.fullmatch(r"(\d+)\.(\d+)", text.strip())
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _capture_namespace(path: Path) -> str:
    """Return a content-owned, non-path-disclosing identity for one artifact."""
    digest = hashlib.sha256()
    with path.expanduser().open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _capture_namespace_from_bytes(content: bytes) -> str:
    return f"sha256:{hashlib.sha256(content).hexdigest()}"


def parse_ncu_csv(
    path: Path,
    *,
    capture_id: str | None = None,
) -> Dict[str, KernelMetrics]:
    """Parse one Nsight Compute CSV without joining unrelated captures.

    Nsight launch, context, and stream identifiers restart for each capture.
    The default namespace therefore binds every parsed launch to its source
    artifact.  Callers may pass the same explicit ``capture_id`` only when
    multiple CSV files are known exports of the same capture.
    """
    kernels: Dict[str, KernelMetrics] = {}
    # Read one immutable byte snapshot so the capture digest and every parsed
    # metric describe the same artifact even if the source path is replaced.
    content = path.expanduser().read_bytes()
    capture = (
        capture_id.strip()
        if capture_id is not None
        else _capture_namespace_from_bytes(content)
    )
    if not capture:
        raise ValueError("capture_id must be non-empty when provided")
    source_artifact = str(path.expanduser().resolve())
    with io.StringIO(content.decode("utf-8-sig"), newline="") as fh:
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
            metric_name_raw = first_key(row, NCU_METRIC_KEYS)
            metric_value_raw = first_key(row, NCU_VALUE_KEYS)
            if not metric_name_raw or metric_value_raw is None:
                continue
            kernel_name = first_key(row, NCU_KERNEL_KEYS)
            if kernel_name is None:
                raise ValueError(
                    "Nsight Compute metric row is missing a non-empty kernel name"
                )
            launch_id = first_key(row, NCU_LAUNCH_ID_KEYS)
            if launch_id is None:
                raise ValueError(
                    "Nsight Compute metric row is missing a non-empty launch ID"
                )
            launch_identity = [f"capture={capture}"]
            for label, keys in (
                ("host", NCU_HOST_KEYS),
                ("process_id", NCU_PROCESS_KEYS),
                ("device_id", NCU_DEVICE_ID_KEYS),
                ("device", NCU_DEVICE_NAME_KEYS),
                ("launch_id", NCU_LAUNCH_ID_KEYS),
                ("context", NCU_CONTEXT_KEYS),
                ("stream", NCU_STREAM_KEYS),
            ):
                if (identity_value := first_key(row, keys)) is not None:
                    launch_identity.append(f"{label}={identity_value}")
            kernel_key = f"{kernel_name} [{','.join(launch_identity)}]"
            value = parse_float(metric_value_raw)
            if value is None:
                # Retain an invalid sentinel only for hardware identity metadata
                # so the resolver can distinguish malformed identity from absent
                # metadata. Other non-finite metrics are omitted fail-closed.
                if metric_name_raw.strip() in {
                    *COMPUTE_CAPABILITY_MAJOR_KEYS,
                    *COMPUTE_CAPABILITY_MINOR_KEYS,
                }:
                    value = math.nan
                else:
                    continue
            metric_name = metric_name_raw.strip()
            unit = first_key(row, NCU_UNIT_KEYS)
            section = first_key(row, NCU_SECTION_KEYS)
            device_name = first_key(row, NCU_DEVICE_NAME_KEYS)
            capability_raw = first_key(row, NCU_COMPUTE_CAPABILITY_KEYS)
            capability = (
                parse_compute_capability(capability_raw)
                if capability_raw is not None
                else None
            )
            kernel = kernels.setdefault(
                kernel_key,
                KernelMetrics(
                    kernel_key,
                    capture_id=capture,
                    source_artifacts=(source_artifact,),
                    device_name=device_name,
                    compute_capability=capability,
                    artifact_identity_invalid=(
                        capability_raw is not None and capability is None
                    ),
                ),
            )
            if device_name is not None:
                if kernel.device_name is not None and kernel.device_name != device_name:
                    raise ConflictingMetricError(
                        f"Conflicting device identity for {kernel_key!r}"
                    )
                kernel.device_name = device_name
            if capability is not None:
                if (
                    kernel.compute_capability is not None
                    and kernel.compute_capability != capability
                ):
                    raise ConflictingMetricError(
                        f"Conflicting compute capability for {kernel_key!r}"
                    )
                kernel.compute_capability = capability
            if capability_raw is not None and capability is None:
                kernel.artifact_identity_invalid = True
            next_metric = RawMetric(metric_name, value, unit, section)
            existing = kernel.metrics.get(metric_name)
            if existing is not None and existing != next_metric:
                raise ConflictingMetricError(
                    f"Conflicting duplicate NCU metric {metric_name!r} for {kernel_key!r}"
                )
            kernel.metrics[metric_name] = next_metric
    return kernels


def merge_kernel_metrics(dicts: Iterable[Dict[str, KernelMetrics]]) -> Dict[str, KernelMetrics]:
    result: Dict[str, KernelMetrics] = {}
    for kernel_map in dicts:
        for name, metrics in kernel_map.items():
            combined = result.setdefault(
                name,
                KernelMetrics(
                    name,
                    capture_id=metrics.capture_id,
                    source_artifacts=metrics.source_artifacts,
                    device_name=metrics.device_name,
                    compute_capability=metrics.compute_capability,
                    artifact_identity_invalid=metrics.artifact_identity_invalid,
                ),
            )
            for field_name in ("capture_id", "device_name", "compute_capability"):
                current = getattr(combined, field_name)
                incoming = getattr(metrics, field_name)
                if current is not None and incoming is not None and current != incoming:
                    raise ConflictingMetricError(
                        f"Conflicting {field_name} metadata for {name!r}"
                    )
                if current is None:
                    setattr(combined, field_name, incoming)
            combined.source_artifacts = tuple(
                dict.fromkeys((*combined.source_artifacts, *metrics.source_artifacts))
            )
            combined.artifact_identity_invalid = (
                combined.artifact_identity_invalid or metrics.artifact_identity_invalid
            )
            for metric_name, next_metric in metrics.metrics.items():
                existing = combined.metrics.get(metric_name)
                if existing is not None and existing != next_metric:
                    raise ConflictingMetricError(
                        f"Conflicting duplicate NCU metric {metric_name!r} for {name!r}"
                    )
                combined.metrics[metric_name] = next_metric
    return result


def metric_in_unit(entry: Optional[RawMetric]) -> Optional[float]:
    """Convert an explicitly recognized Nsight duration unit to milliseconds."""
    if entry is None:
        return None
    if isinstance(entry.value, bool) or not math.isfinite(entry.value) or entry.value < 0:
        return None
    unit = (entry.unit or "").strip().lower()
    millisecond_scales = {
        "ns": 1e-6,
        "nsecond": 1e-6,
        "nseconds": 1e-6,
        "nanosecond": 1e-6,
        "nanoseconds": 1e-6,
        "us": 1e-3,
        "µs": 1e-3,
        "μs": 1e-3,
        "usecond": 1e-3,
        "useconds": 1e-3,
        "microsecond": 1e-3,
        "microseconds": 1e-3,
        "ms": 1.0,
        "msecond": 1.0,
        "mseconds": 1.0,
        "millisecond": 1.0,
        "milliseconds": 1.0,
        "s": 1e3,
        "second": 1e3,
        "seconds": 1e3,
    }
    scale = millisecond_scales.get(unit)
    return None if scale is None else entry.value * scale


def pick_precision(metrics: KernelMetrics) -> str:
    aggregate_aliases = {
        "fp32": ("flop_count_sp", "flop_count_sp_sum"),
        "fp16": ("flop_count_hp", "flop_count_hp_sum", "flop_count_half_precision"),
        "fp8": ("flop_count_fp8",),
        "fp64": ("flop_count_dp", "flop_count_dp_sum"),
    }
    observed: set[str] = set()
    zero_aggregate_observed = False
    for precision, aliases in aggregate_aliases.items():
        entries = [metric for alias in aliases if (metric := metrics.get(alias)) is not None]
        if not entries:
            continue
        if any(
            isinstance(metric.value, bool)
            or not math.isfinite(metric.value)
            or metric.value < 0
            for metric in entries
        ):
            return "unknown"
        if any(metric.value > 0 for metric in entries):
            observed.add(precision)
        else:
            zero_aggregate_observed = True

    legacy_families = {
        "fp64": (
            "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
            "sm__sass_thread_inst_executed_op_dfma_pred_on.sum",
            "sm__sass_thread_inst_executed_op_dadd_pred_on.sum",
            "sm__sass_thread_inst_executed_op_dmul_pred_on.sum",
        ),
        "fp16": (
            "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
            "sm__sass_thread_inst_executed_op_hfma_pred_on.sum",
            "sm__sass_thread_inst_executed_op_hadd_pred_on.sum",
            "sm__sass_thread_inst_executed_op_hmul_pred_on.sum",
        ),
        "fp32": (
            "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
            "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
            # Retain older exports while preferring the documented SM-subpartition names.
            "sm__sass_thread_inst_executed_op_ffma_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fadd_pred_on.sum",
            "sm__sass_thread_inst_executed_op_fmul_pred_on.sum",
        ),
    }
    zero_complete_sass_observed = False
    for precision, names in legacy_families.items():
        for scope_names in (names[:3], names[3:]):
            entries = [metrics.get(name) for name in scope_names]
            present = [entry for entry in entries if entry is not None]
            if any(
                isinstance(entry.value, bool)
                or not math.isfinite(entry.value)
                or entry.value < 0
                for entry in present
            ):
                return "unknown"
            if any(entry.value > 0 for entry in present):
                observed.add(precision)
            elif len(present) == len(scope_names):
                zero_complete_sass_observed = True
    if len(observed) == 1:
        return next(iter(observed))
    if observed:
        return "mixed"
    return (
        "operation_free"
        if zero_aggregate_observed or zero_complete_sass_observed
        else "unknown"
    )


def compute_flops(metrics: KernelMetrics) -> Optional[float]:
    """Select one non-overlapping FLOP counter family by strict precedence.

    Aggregate ``flop_count_*`` counters are authoritative when present. Scalar
    SASS fallbacks are accepted only when FMA, add, and multiply counters are
    all present for one scope. Tensor-pipe instruction counts are excluded
    because an opcode and operation shape are required to convert them into
    FLOPs. Values are summed only within one complete representation family.
    """
    precision = pick_precision(metrics)
    if precision == "operation_free":
        return 0.0
    if precision in {"unknown", "mixed"}:
        return None
    aggregate_aliases = {
        "fp32": ("flop_count_sp", "flop_count_sp_sum"),
        "fp16": ("flop_count_hp", "flop_count_hp_sum", "flop_count_half_precision"),
        "fp8": ("flop_count_fp8",),
        "fp64": ("flop_count_dp", "flop_count_dp_sum"),
    }
    aggregates = [
        metric
        for alias in aggregate_aliases[precision]
        if (metric := metrics.get(alias)) is not None
    ]
    if aggregates:
        values = {metric.value for metric in aggregates}
        if len(values) != 1:
            return None
        return aggregates[0].value

    prefixes = ("smsp", "sm")
    operation_letters = {"fp64": "d", "fp16": "h", "fp32": "f"}
    letter = operation_letters.get(precision)
    if letter is None:
        return None
    for prefix in prefixes:
        family = (
            (f"{prefix}__sass_thread_inst_executed_op_{letter}fma_pred_on.sum", 2.0),
            (f"{prefix}__sass_thread_inst_executed_op_{letter}add_pred_on.sum", 1.0),
            (f"{prefix}__sass_thread_inst_executed_op_{letter}mul_pred_on.sum", 1.0),
        )
        values = [metrics.get_value(key) for key, _weight in family]
        if not all(value is not None for value in values):
            continue
        if not all(value is not None and math.isfinite(value) and value >= 0 for value in values):
            return None
        return sum(
            value * weight
            for value, (_key, weight) in zip(values, family, strict=True)
            if value is not None
        )
    return None


def _counter_value_in_bytes(
    metrics: KernelMetrics,
    key: str,
    unit_hint: str,
) -> float | None:
    metric = metrics.get(key)
    if metric is None:
        return None
    if (
        isinstance(metric.value, bool)
        or not math.isfinite(metric.value)
        or metric.value < 0
    ):
        return None
    unit = (metric.unit or unit_hint).strip().lower()
    byte_scales = {
        "byte": 1.0,
        "bytes": 1.0,
        "kbyte": 1e3,
        "kbytes": 1e3,
        "mbyte": 1e6,
        "mbytes": 1e6,
        "gbyte": 1e9,
        "gbytes": 1e9,
        "sector": 32.0,
        "sectors": 32.0,
    }
    scale = byte_scales.get(unit)
    return None if scale is None else metric.value * scale


def _complete_byte_family_total(
    metrics: KernelMetrics,
    family: tuple[tuple[str, str], ...],
) -> float | None:
    values = [
        _counter_value_in_bytes(metrics, key, unit_hint)
        for key, unit_hint in family
    ]
    if not all(value is not None for value in values):
        return None
    return sum(value for value in values if value is not None)


def compute_hbm_bytes(metrics: KernelMetrics) -> Optional[float]:
    """Return bytes from one complete DRAM counter family, or ``None``."""
    if metrics.get("dram__bytes.sum") is not None:
        return _counter_value_in_bytes(metrics, "dram__bytes.sum", "byte")

    dram_families = (
        (
            ("dram__bytes_read.sum", "byte"),
            ("dram__bytes_write.sum", "byte"),
        ),
        (
            ("gpu__dram_sectors_read.sum", "sector"),
            ("gpu__dram_sectors_write.sum", "sector"),
        ),
    )
    for family in dram_families:
        if any(metrics.get(key) is not None for key, _unit_hint in family):
            return _complete_byte_family_total(metrics, family)
    return None


def compute_bytes(metrics: KernelMetrics) -> Optional[float]:
    """Select one complete hierarchy-specific byte family by precedence.

    DRAM families have precedence, followed by L1 LSU and L1 TMA counters.
    This helper preserves lower-hierarchy diagnostics; HBM roofline analysis
    uses :func:`compute_hbm_bytes` and therefore never treats L1 traffic as HBM.
    """
    if (hbm_bytes := compute_hbm_bytes(metrics)) is not None:
        return hbm_bytes

    l1_families = (
        (
            ("l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum", "byte"),
            ("l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum", "byte"),
        ),
        (
            ("l1tex__t_bytes_pipe_lsu_mem_global_op_tma_ld.sum", "byte"),
            ("l1tex__t_bytes_pipe_lsu_mem_global_op_tma_st.sum", "byte"),
        ),
    )
    for family in l1_families:
        if (total := _complete_byte_family_total(metrics, family)) is not None:
            return total
    return None


def safe_pct(metric: RawMetric | float | None) -> Optional[float]:
    """Normalize unitless fractions while preserving explicitly percent values."""
    if metric is None:
        return None
    if isinstance(metric, RawMetric):
        value = metric.value
        unit = (metric.unit or "").strip().lower()
        if isinstance(value, bool) or not math.isfinite(value) or value < 0:
            return None
        if "%" in unit or unit in {"pct", "percent", "percentage"}:
            return value
    else:
        value = metric
    if isinstance(value, bool) or not math.isfinite(value) or value < 0:
        return None
    return value if value > 1.0 else value * 100.0


def determine_binding_roof(
    compute_pct: Optional[float],
    tmem_pct: Optional[float],
    dram_pct: Optional[float],
    l2_pct: Optional[float],
    threshold: float = 5.0,
) -> str:
    """Return the highest reported utilization domain, or ``unknown``."""
    reported = [
        (label, value)
        for label, value in (
            ("compute", compute_pct),
            ("tmem", tmem_pct),
            ("l2", l2_pct),
            ("dram", dram_pct),
        )
        if value is not None
    ]
    if not reported:
        return "unknown"
    best, best_value = reported[0]
    for label, value in reported[1:]:
        if value > best_value + threshold:
            best = label
            best_value = value
    return best


def derive_roofline(
    metrics: KernelMetrics,
    specs: ArchitectureSpecs | None,
) -> Tuple[Optional[RooflineSummary], Optional[float], Optional[float], Optional[float], str]:
    analyzer = RooflineAnalyzer(specs) if specs is not None else None
    precision = pick_precision(metrics)
    duration_metric = metrics.get(
        "gpu__time_duration.sum",
        "gpu__time_duration.avg",
        "Duration",
        "Kernel Duration"
    )
    duration_ms = metric_in_unit(duration_metric)
    flops = compute_flops(metrics)
    bytes_transferred = compute_hbm_bytes(metrics)
    compute_util_pct = safe_pct(metrics.get(
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "Compute (SM) Throughput",
        "Compute Pipe Throughput",
        "SM Throughput",
    ))
    memory_util_pct = safe_pct(metrics.get(
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "Memory Throughput",
        "Memory (Device) Throughput",
        "Device Memory Throughput",
    ))
    tmem_util_pct = safe_pct(metrics.get(
        "tmem__throughput.avg.pct_of_peak_sustained_elapsed",
        "Tensor Memory Throughput",
        "Tensor Memory (TMEM) Throughput",
    ))
    l2_util_pct = safe_pct(metrics.get(
        "l2__throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "L2 Throughput",
    ))
    binding = determine_binding_roof(compute_util_pct, tmem_util_pct, memory_util_pct, l2_util_pct)
    is_tmem_bound = binding == "tmem"
    roofline_summary: Optional[RooflineSummary] = None
    if (
        analyzer is not None
        and duration_ms is not None
        and flops is not None
        and bytes_transferred is not None
        and precision in {"fp32", "fp16", "fp8", "tf32"}
        and duration_ms > 0
        and flops >= 0
        and bytes_transferred > 0
    ):
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
    elif (
        analyzer is not None
        and duration_ms is not None
        and flops == 0.0
        and bytes_transferred is not None
        and precision == "operation_free"
        and duration_ms > 0
        and bytes_transferred > 0
    ):
        achieved_bandwidth_gbs = bytes_transferred / (duration_ms * 1e6)
        roofline_summary = RooflineSummary(
            achieved_tflops=0.0,
            achieved_bandwidth_gbs=achieved_bandwidth_gbs,
            arithmetic_intensity=0.0,
            compute_utilization_pct=None,
            memory_utilization_pct=(
                achieved_bandwidth_gbs / analyzer.specs.memory_bandwidth_gbs * 100.0
            ),
            tmem_utilization_pct=tmem_util_pct,
            l2_utilization_pct=l2_util_pct,
            binding=binding,
            is_memory_bound=True,
            is_compute_bound=False,
            is_tmem_bound=is_tmem_bound,
            ridge_point=None,
            memory_bound_limit_tflops=0.0,
            peak_tflops=None,
            peak_bandwidth_gbs=analyzer.specs.memory_bandwidth_gbs,
        )
    return roofline_summary, duration_ms, flops, bytes_transferred, precision


def build_advisory(
    metrics: KernelMetrics,
    specs: ArchitectureSpecs | None,
) -> Advisory:
    roofline_summary, duration_ms, flops, bytes_transferred, precision = derive_roofline(
        metrics, specs
    )
    sm_util = safe_pct(metrics.get(
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "Compute (SM) Throughput",
        "Compute Pipe Throughput",
        "SM Throughput",
    ))
    dram_util = safe_pct(metrics.get(
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "Memory Throughput",
        "Memory (Device) Throughput",
        "Device Memory Throughput",
    ))
    tmem_util = safe_pct(metrics.get(
        "tmem__throughput.avg.pct_of_peak_sustained_elapsed",
        "Tensor Memory Throughput",
        "Tensor Memory (TMEM) Throughput",
    ))
    occupancy = safe_pct(metrics.get("sm__warps_active.avg.pct_of_peak_sustained_active"))
    tensor_util = safe_pct(
        metrics.get(
            "smpp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
            "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed",
        )
    )
    warp_exec = safe_pct(
        metrics.get(
            "smsp__warp_execution_efficiency.avg.pct",
            "warp_execution_efficiency",
            "Warp Execution Efficiency",
        )
    )
    l2_hit = safe_pct(
        metrics.get(
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
        if (
            roofline_summary.ridge_point is not None
            and roofline_summary.arithmetic_intensity < roofline_summary.ridge_point * 0.8
        ):
            recommendations.append(
                "Arithmetic intensity below ridge point; restructure data movement to perform more FLOPs per byte."
            )
        if (
            roofline_summary.compute_utilization_pct is not None
            and roofline_summary.compute_utilization_pct < 40
        ):
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
        if specs is None:
            recommendations.append(
                "Roofline ceilings unknown; no ceiling-based bottleneck classification was performed."
            )
        elif roofline_summary is None:
            recommendations.append(
                "Roofline classification unavailable; measured duration, FLOP, and byte "
                "counters are required and utilization percentages are not substitutes."
            )
        else:
            recommendations.append("No major bottlenecks detected; kernel appears well optimised.")

    return Advisory(
        kernel=metrics.name,
        capture_id=metrics.capture_id,
        source_artifacts=metrics.source_artifacts,
        device_name=metrics.device_name,
        compute_capability=(
            None
            if metrics.compute_capability is None
            else f"{metrics.compute_capability[0]}.{metrics.compute_capability[1]}"
        ),
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
    if report_path is None:
        return None
    try:
        summary = NsightSystemsReportParser.summarize_report(
            str(report_path),
            top_k=top_k,
            print_summary=False,
        )
    except Exception as exc:  # pragma: no cover - nsys may be unavailable
        source = getattr(
            exc,
            "source",
            {
                "requested_path": str(report_path),
                "resolved_path": str(report_path.expanduser().resolve()),
                "sha256": None,
                "size_bytes": None,
                "status": "error",
                "error": str(exc),
            },
        )
        return {
            "source": source,
            "error": f"Failed to parse Nsight Systems report: {exc}",
        }
    return {
        "report": summary["report"],
        "source": summary["source"],
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
    compute_util = [
        adv.roofline.compute_utilization_pct
        for adv in advisories
        if adv.roofline and adv.roofline.compute_utilization_pct is not None
    ]
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
        "--hardware-profile",
        choices=HARDWARE_PROFILES,
        help=(
            "Explicit exact-SKU ceilings for roofline classification. When omitted, "
            "an exact artifact device name plus compute capability may select a "
            "reviewed profile; compute capability alone cannot select SKU peaks."
        ),
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
    if args.top_k <= 0:
        print("--top-k must be a positive integer.", file=sys.stderr)
        return 1
    if not args.ncu_csv:
        print("Provide at least one --ncu-csv export from Nsight Compute.", file=sys.stderr)
        return 1
    output_path = args.output_json.expanduser().resolve() if args.output_json else None
    if output_path is not None:
        input_paths = [path.expanduser().resolve() for path in args.ncu_csv]
        if args.nsys_report is not None:
            input_paths.append(args.nsys_report.expanduser().resolve())
        if any(_paths_refer_to_same_file(output_path, path) for path in input_paths):
            print(
                "--output-json must not alias an input artifact path.",
                file=sys.stderr,
            )
            return 1

    kernel_maps = []
    input_receipts: list[dict[str, object]] = []
    input_issue = False

    def write_terminal_error(error: str) -> None:
        if output_path is None:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": DEEP_PROFILING_REPORT_SCHEMA_VERSION,
                    "success": False,
                    "error": error,
                    "inputs": input_receipts,
                    "advisories": [],
                },
                indent=2,
            )
        )

    for csv_path in args.ncu_csv:
        receipt: dict[str, object] = {
            "requested_path": str(csv_path),
            "resolved_path": str(csv_path.expanduser().resolve()),
            "capture_id": None,
            "status": "pending",
            "kernel_count": 0,
            "error": None,
        }
        input_receipts.append(receipt)
        if not csv_path.exists():
            receipt["status"] = "missing"
            receipt["error"] = "file not found"
            input_issue = True
            print(f"[ncu-input] {csv_path}: status=missing", file=sys.stderr)
            continue
        try:
            parsed = parse_ncu_csv(csv_path)
        except ConflictingMetricError as exc:
            receipt["status"] = "conflict"
            receipt["error"] = str(exc)
            input_issue = True
            print(
                f"[ncu-input] {csv_path}: status=conflict ({exc})",
                file=sys.stderr,
            )
            continue
        except Exception as exc:
            receipt["status"] = "error"
            receipt["error"] = str(exc)
            input_issue = True
            print(
                f"[ncu-input] {csv_path}: status=error ({exc})",
                file=sys.stderr,
            )
            continue
        if not parsed:
            receipt["status"] = "empty_or_unrecognized"
            receipt["error"] = "no usable kernel metrics"
            input_issue = True
            print(
                f"[ncu-input] {csv_path}: status=empty_or_unrecognized",
                file=sys.stderr,
            )
            continue
        receipt["status"] = "parsed"
        receipt["kernel_count"] = len(parsed)
        capture_ids = {metrics.capture_id for metrics in parsed.values()}
        if len(capture_ids) != 1 or None in capture_ids:
            receipt["status"] = "error"
            receipt["error"] = "parsed kernels do not share one capture identity"
            input_issue = True
            print(
                f"[ncu-input] {csv_path}: status=error ({receipt['error']})",
                file=sys.stderr,
            )
            continue
        receipt["capture_id"] = next(iter(capture_ids))
        print(
            f"[ncu-input] {csv_path}: status=parsed kernels={len(parsed)}",
            file=sys.stderr,
        )
        kernel_maps.append(parsed)
    if not kernel_maps:
        error = "No Nsight Compute CSV input contained usable kernel metrics."
        print(
            error,
            file=sys.stderr,
        )
        write_terminal_error(error)
        return 2

    try:
        merged = merge_kernel_metrics(kernel_maps)
    except ConflictingMetricError as exc:
        print(f"[ncu-input] status=conflict_across_inputs ({exc})", file=sys.stderr)
        write_terminal_error(f"Conflicting metrics across inputs: {exc}")
        return 2
    hardware = resolve_hardware_selection(merged.values(), args.hardware_profile)
    hardware_conflict = args.hardware_profile is not None and hardware.specs is None
    if hardware_conflict:
        print(
            "Declared --hardware-profile conflicts with incomplete, invalid, or "
            f"mismatched artifact identity (provenance={hardware.provenance}).",
            file=sys.stderr,
        )
    advisories = sorted(
        (build_advisory(metrics, hardware.specs) for metrics in merged.values()),
        key=lambda adv: (
            -(adv.duration_ms if adv.duration_ms is not None else -math.inf),
            adv.kernel,
        ),
    )

    if hardware.specs is None:
        print(
            "Roofline hardware: unknown "
            f"(provenance={hardware.provenance}); ceiling classification omitted."
        )
    else:
        print(
            f"Roofline hardware: {hardware.specs.name} "
            f"(profile={hardware.profile}, provenance={hardware.provenance})"
        )

    if args.print_markdown:
        print("```markdown")
    print(render_table(advisories, args.top_k))
    if args.print_markdown:
        print("```")
    print()

    systems_summary = None
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

    if output_path is not None:
        nsys_failed = bool(systems_summary and "error" in systems_summary)
        summary = {
            "schema_version": DEEP_PROFILING_REPORT_SCHEMA_VERSION,
            "success": not (input_issue or hardware_conflict or nsys_failed),
            "inputs": input_receipts,
            "hardware": hardware.as_dict(),
            "advisories": [
                {
                    "kernel": adv.kernel,
                    "source": {
                        "capture_id": adv.capture_id,
                        "artifacts": list(adv.source_artifacts),
                        "device_name": adv.device_name,
                        "compute_capability": adv.compute_capability,
                    },
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
            summary["nsight_systems"] = systems_summary
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2))
        print(f"Wrote JSON summary to {output_path}")

    if hardware_conflict:
        return 4
    if input_issue or (systems_summary and "error" in systems_summary):
        return 3
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
