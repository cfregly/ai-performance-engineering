"""Read-only observations and recommendations for DataLoader configuration.

The diagnostics in this module deliberately never change process affinity.
Hardware observations retain their provenance and unknown state so callers do
not mistake an unavailable GPU-to-NUMA mapping for NUMA node zero.
"""

from __future__ import annotations

import os
import platform
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

ObservationStatus = Literal["observed", "not_observed", "unknown"]
GpuStatus = Literal["available", "unavailable", "unknown"]
NumaMappingStatus = Literal["known", "unknown", "not_applicable"]
RecommendationSource = Literal["default_static_policy", "explicit_provider_config"]
DATA_LOADING_ANALYSIS_SCHEMA_VERSION = "data_loading_analysis.v1"


@dataclass(frozen=True)
class CpuObservation:
    """CPU facts obtained without changing system state."""

    architecture: str | None
    logical_cpu_count: int | None
    model: str | None
    is_grace: bool | None
    grace_status: ObservationStatus
    provenance: str


@dataclass(frozen=True)
class GpuObservation:
    """Current CUDA device facts needed for NUMA lookup."""

    status: GpuStatus
    gpu_id: int | None
    device_count: int | None
    pci_bus_id: str | None
    provenance: str


@dataclass(frozen=True)
class AffinityObservation:
    """The process affinity observed at analysis time."""

    cpu_ids: tuple[int, ...] | None
    status: Literal["observed", "unknown"]
    provenance: str


@dataclass(frozen=True)
class NumaMappingObservation:
    """Observed GPU-to-NUMA mapping, including unknown/not-applicable states."""

    status: NumaMappingStatus
    numa_node: int | None
    provenance: str


@dataclass(frozen=True)
class DataLoaderRecommendationConfig:
    """Explicit policy used to produce deterministic DataLoader kwargs."""

    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    prefetch_factor: int | None = 2
    persistent_workers: bool = True

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int:
            raise TypeError("batch_size must be a non-bool integer")
        if type(self.num_workers) is not int:
            raise TypeError("num_workers must be a non-bool integer")
        if self.prefetch_factor is not None and type(self.prefetch_factor) is not int:
            raise TypeError("prefetch_factor must be a non-bool integer or None")
        if type(self.pin_memory) is not bool:
            raise TypeError("pin_memory must be a bool")
        if type(self.persistent_workers) is not bool:
            raise TypeError("persistent_workers must be a bool")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")
        if self.prefetch_factor is not None and self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive when configured")
        if self.num_workers == 0 and self.prefetch_factor is not None:
            raise ValueError("prefetch_factor requires num_workers > 0")
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError("persistent_workers requires num_workers > 0")

    def as_kwargs(self) -> dict:
        """Return the configured policy without hardware-dependent rewriting."""
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs


@dataclass(frozen=True)
class DataLoadingAnalysis:
    """Read-only observations plus configured DataLoader recommendations."""

    cpu: CpuObservation
    gpu: GpuObservation
    affinity: AffinityObservation
    gpu_numa: NumaMappingObservation
    recommendations: DataLoaderRecommendationConfig
    recommendation_source: RecommendationSource

    def to_dict(self) -> dict:
        current_affinity = (
            list(self.affinity.cpu_ids) if self.affinity.cpu_ids is not None else None
        )
        if self.gpu_numa.status == "known":
            mapping_note = (
                f"GPU {self.gpu.gpu_id} maps to observed NUMA node "
                f"{self.gpu_numa.numa_node}; this analysis did not change affinity"
            )
        elif self.gpu_numa.status == "not_applicable":
            mapping_note = "No CUDA device was observed; GPU NUMA placement is not applicable"
        else:
            mapping_note = "GPU NUMA placement is unknown and was not inferred"

        return {
            "schema_version": DATA_LOADING_ANALYSIS_SCHEMA_VERSION,
            "analysis_mode": "read_only",
            "success": True,
            "gpu_id": self.gpu.gpu_id,
            "dataloader_kwargs": self.recommendations.as_kwargs(),
            "cpu": {
                # Preserve the prior response names where the value is observed.
                "is_grace": self.cpu.is_grace,
                "cpu_arch": self.cpu.architecture,
                "cpu_count": None,
                "cpu_threads": self.cpu.logical_cpu_count,
                "memory_gb": None,
                "numa_nodes": None,
                "gpus": self.gpu.device_count,
                "cpu_model": self.cpu.model,
                "grace_status": self.cpu.grace_status,
                "provenance": self.cpu.provenance,
            },
            "numa_node": self.gpu_numa.numa_node,
            "gpu_numa_mapping": {
                "status": self.gpu_numa.status,
                "gpu_id": self.gpu.gpu_id,
                "numa_node": self.gpu_numa.numa_node,
                "pci_bus_id": self.gpu.pci_bus_id,
                "provenance": self.gpu_numa.provenance,
            },
            # `cpu_affinity` is retained for existing clients, but now reports
            # only the current observation rather than an affinity just applied.
            "cpu_affinity": current_affinity,
            "current_cpu_affinity": current_affinity,
            "current_affinity_status": self.affinity.status,
            "current_affinity_provenance": self.affinity.provenance,
            "affinity_applied": False,
            "recommendation_provenance": {
                "source": self.recommendation_source,
                "hardware_adaptive": False,
            },
            "observation_provenance": {
                "cpu": self.cpu.provenance,
                "gpu": self.gpu.provenance,
                "gpu_numa_mapping": self.gpu_numa.provenance,
                "current_cpu_affinity": self.affinity.provenance,
            },
            "notes": [
                (
                    "DataLoader values use the provider's explicit configuration; "
                    "validate them with the workload"
                    if self.recommendation_source == "explicit_provider_config"
                    else "DataLoader values use unmeasured static defaults; validate and "
                    "configure them for the workload"
                ),
                mapping_note,
                "No CPU affinity was changed by this analysis",
            ],
        }


class DataLoadingAnalysisProvider(Protocol):
    """Provider seam consumed by the shared CLI/engine/MCP implementation."""

    def analyze(self) -> DataLoadingAnalysis:
        """Return read-only observations and recommendations."""
        ...


CpuObserver = Callable[[], CpuObservation]
GpuObserver = Callable[[], GpuObservation]
AffinityObserver = Callable[[], AffinityObservation]
NumaObserver = Callable[[GpuObservation], NumaMappingObservation]


class ReadOnlyDataLoadingAnalysisProvider:
    """Default provider built from read-only host and CUDA observations."""

    def __init__(
        self,
        recommendations: DataLoaderRecommendationConfig | None = None,
        *,
        cpu_observer: CpuObserver | None = None,
        gpu_observer: GpuObserver | None = None,
        affinity_observer: AffinityObserver | None = None,
        numa_observer: NumaObserver | None = None,
    ) -> None:
        self._recommendation_source: RecommendationSource = (
            "explicit_provider_config"
            if recommendations is not None
            else "default_static_policy"
        )
        self._recommendations = (
            recommendations if recommendations is not None else DataLoaderRecommendationConfig()
        )
        self._cpu_observer = cpu_observer if cpu_observer is not None else observe_cpu
        self._gpu_observer = gpu_observer if gpu_observer is not None else observe_current_gpu
        self._affinity_observer = (
            affinity_observer if affinity_observer is not None else observe_current_affinity
        )
        self._numa_observer = (
            numa_observer if numa_observer is not None else observe_gpu_numa_mapping
        )

    def analyze(self) -> DataLoadingAnalysis:
        """Collect observations without applying affinity or other tuning."""
        cpu = self._cpu_observer()
        gpu = self._gpu_observer()
        affinity = self._affinity_observer()
        gpu_numa = self._numa_observer(gpu)
        return DataLoadingAnalysis(
            cpu=cpu,
            gpu=gpu,
            affinity=affinity,
            gpu_numa=gpu_numa,
            recommendations=self._recommendations,
            recommendation_source=self._recommendation_source,
        )


def observe_cpu() -> CpuObservation:
    """Observe CPU identity; only an explicit Grace marker establishes Grace."""
    architecture = platform.machine().strip() or None
    logical_cpu_count = os.cpu_count()
    cpuinfo_path = Path("/proc/cpuinfo")
    try:
        cpuinfo = cpuinfo_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        cpuinfo = ""

    cpuinfo_fields: dict[str, list[str]] = {}
    for line in cpuinfo.splitlines():
        key, separator, value = line.partition(":")
        normalized_key = key.strip().lower()
        candidate = value.strip()
        if (
            separator
            and candidate
            and normalized_key
            in {
                "model name",
                "hardware",
                "processor",
            }
        ):
            cpuinfo_fields.setdefault(normalized_key, []).append(candidate)

    model: str | None = None
    for field in ("model name", "hardware", "processor"):
        for candidate in cpuinfo_fields.get(field, []):
            if field == "processor" and candidate.isdecimal():
                continue
            model = candidate
            break
        if model is not None:
            break

    if cpuinfo:
        explicit_grace = re.search(
            r"\bnvidia\s+grace\b|\bgrace\s+(?:cpu|processor)\b",
            cpuinfo,
            flags=re.IGNORECASE,
        )
        is_grace: bool | None = True if explicit_grace else None
        grace_status: ObservationStatus = "observed" if explicit_grace else "not_observed"
        provenance = "platform.machine, os.cpu_count, /proc/cpuinfo"
    else:
        is_grace = None
        grace_status = "unknown"
        provenance = "platform.machine, os.cpu_count; /proc/cpuinfo unavailable"

    return CpuObservation(
        architecture=architecture,
        logical_cpu_count=logical_cpu_count,
        model=model,
        is_grace=is_grace,
        grace_status=grace_status,
        provenance=provenance,
    )


def _normalize_pci_bus_id(properties: object) -> str | None:
    raw_bus = getattr(properties, "pci_bus_id", None)
    if isinstance(raw_bus, str):
        candidate = raw_bus.strip().lower()
        if re.fullmatch(r"(?:[0-9a-f]{4}|[0-9a-f]{8}):[0-9a-f]{2}:[0-9a-f]{2}\.\d", candidate):
            if candidate.startswith("00000000:"):
                candidate = candidate[4:]
            return candidate

    domain = getattr(properties, "pci_domain_id", None)
    device = getattr(properties, "pci_device_id", None)
    if all(isinstance(value, int) and value >= 0 for value in (domain, raw_bus, device)):
        return f"{domain:04x}:{raw_bus:02x}:{device:02x}.0"
    return None


def observe_current_gpu() -> GpuObservation:
    """Observe the current CUDA device without running a workload."""
    try:
        import torch
    except (ImportError, OSError) as exc:
        return GpuObservation(
            status="unknown",
            gpu_id=None,
            device_count=None,
            pci_bus_id=None,
            provenance=f"torch import unavailable: {type(exc).__name__}",
        )

    try:
        if not torch.cuda.is_available():
            return GpuObservation(
                status="unavailable",
                gpu_id=None,
                device_count=0,
                pci_bus_id=None,
                provenance="torch.cuda.is_available=false",
            )
        gpu_id = int(torch.cuda.current_device())
        device_count = int(torch.cuda.device_count())
        properties = torch.cuda.get_device_properties(gpu_id)
        return GpuObservation(
            status="available",
            gpu_id=gpu_id,
            device_count=device_count,
            pci_bus_id=_normalize_pci_bus_id(properties),
            provenance="torch.cuda current device and device properties",
        )
    except Exception as exc:
        return GpuObservation(
            status="unknown",
            gpu_id=None,
            device_count=None,
            pci_bus_id=None,
            provenance=f"torch.cuda observation failed: {type(exc).__name__}",
        )


def observe_current_affinity() -> AffinityObservation:
    """Read process CPU affinity when the host exposes it."""
    reader = getattr(os, "sched_getaffinity", None)
    if reader is None:
        return AffinityObservation(
            cpu_ids=None,
            status="unknown",
            provenance="os.sched_getaffinity unavailable",
        )
    try:
        cpu_ids = tuple(sorted(int(cpu_id) for cpu_id in reader(0)))
    except (OSError, TypeError, ValueError) as exc:
        return AffinityObservation(
            cpu_ids=None,
            status="unknown",
            provenance=f"os.sched_getaffinity failed: {type(exc).__name__}",
        )
    return AffinityObservation(
        cpu_ids=cpu_ids,
        status="observed",
        provenance="os.sched_getaffinity(0)",
    )


def observe_gpu_numa_mapping(gpu: GpuObservation) -> NumaMappingObservation:
    """Read a GPU NUMA node from sysfs; never infer it from the GPU index."""
    if gpu.status == "unavailable":
        return NumaMappingObservation(
            status="not_applicable",
            numa_node=None,
            provenance="no CUDA device observed",
        )
    if gpu.status != "available":
        return NumaMappingObservation(
            status="unknown",
            numa_node=None,
            provenance="CUDA device observation unavailable",
        )
    if gpu.pci_bus_id is None:
        return NumaMappingObservation(
            status="unknown",
            numa_node=None,
            provenance="CUDA device PCI bus ID unavailable",
        )

    numa_path = Path("/sys/bus/pci/devices") / gpu.pci_bus_id / "numa_node"
    try:
        raw_value = numa_path.read_text(encoding="utf-8").strip()
        numa_node = int(raw_value)
    except (OSError, ValueError) as exc:
        return NumaMappingObservation(
            status="unknown",
            numa_node=None,
            provenance=f"{numa_path} unavailable: {type(exc).__name__}",
        )
    if numa_node < 0:
        return NumaMappingObservation(
            status="unknown",
            numa_node=None,
            provenance=f"{numa_path} reported no NUMA node",
        )
    return NumaMappingObservation(
        status="known",
        numa_node=numa_node,
        provenance=str(numa_path),
    )


__all__ = [
    "AffinityObservation",
    "CpuObservation",
    "DATA_LOADING_ANALYSIS_SCHEMA_VERSION",
    "DataLoaderRecommendationConfig",
    "DataLoadingAnalysis",
    "DataLoadingAnalysisProvider",
    "GpuStatus",
    "GpuObservation",
    "NumaMappingStatus",
    "NumaMappingObservation",
    "ObservationStatus",
    "ReadOnlyDataLoadingAnalysisProvider",
    "observe_cpu",
    "observe_current_affinity",
    "observe_current_gpu",
    "observe_gpu_numa_mapping",
]
