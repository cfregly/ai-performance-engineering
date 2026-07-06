#!/usr/bin/env python3
"""Dynamic hardware capability detection and guards."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
PROBE_FILE = ARTIFACTS_DIR / "hardware_capabilities.json"
PROBE_SCRIPT = REPO_ROOT / "core" / "scripts" / "utilities" / "probe_hardware_capabilities.py"

__all__ = [
    "HardwareCapabilities",
    "TMALimits",
    "ClusterCapabilities",
    "detect_capabilities",
    "refresh_capability_cache",
    "all_capability_records",
    "format_capability_report",
    "ensure_tma_box_supported",
    "ensure_dsmem_supported",
]


@dataclass(frozen=True)
class TMALimits:
    max_1d_elements: int
    max_2d_width: int
    max_2d_height: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_1d_elements": self.max_1d_elements,
            "max_2d_width": self.max_2d_width,
            "max_2d_height": self.max_2d_height,
        }


@dataclass(frozen=True)
class ClusterCapabilities:
    supports_clusters: bool
    has_dsmem: bool
    max_cluster_size: int
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "supports_clusters": self.supports_clusters,
            "has_dsmem": self.has_dsmem,
            "max_cluster_size": self.max_cluster_size,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class HardwareCapabilities:
    key: str
    name: str
    architecture: str
    compute_capability: str
    sm_version: str
    device_name: str
    total_memory_gb: float
    l2_cache_kb: Optional[float]
    num_sms: int
    warp_size: int
    max_threads_per_block: int
    max_threads_per_sm: int
    max_shared_mem_per_block: int
    max_shared_mem_per_sm: int
    tensor_cores: str
    features: List[str]
    memory_bandwidth_tbps: Optional[float]
    max_unified_memory_tb: Optional[int]
    nvlink_c2c: bool
    grace_coherence: bool
    tma_supported: bool
    tma_compiler_supported: bool
    tma_limits: TMALimits
    cluster: ClusterCapabilities
    driver_version: Optional[str]
    cuda_runtime_version: Optional[str]
    notes: List[str]

    def to_dict(self) -> Dict[str, Any]:
        data = self.__dict__.copy()
        data["tma_limits"] = self.tma_limits.to_dict()
        data["cluster"] = self.cluster.to_dict()
        return data

    @property
    def tma_ready(self) -> bool:
        return self.tma_supported and self.tma_compiler_supported


_probe_cache: Optional[Dict[str, Any]] = None


def _torch_cuda_available() -> bool:
    if torch is None:
        return False
    try:
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _run_probe_if_needed(*, force: bool = False) -> None:
    if PROBE_FILE.exists() and not force:
        return
    if not _torch_cuda_available():
        return

    # Fail loudly if the probe script doesn't exist - no silent fallbacks
    if not PROBE_SCRIPT.exists():
        raise FileNotFoundError(
            f"Hardware probe script not found: {PROBE_SCRIPT}\n"
            f"Expected location: {PROBE_SCRIPT.resolve()}\n"
            f"This is a critical configuration error - fix the path or restore the file."
        )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    # Run the probe script - let errors propagate, no silent fallbacks
    subprocess.run([sys.executable, str(PROBE_SCRIPT)], check=True, timeout=180)

    if not PROBE_FILE.exists():
        raise RuntimeError(
            f"Probe script ran but did not create expected output: {PROBE_FILE}\n"
            f"The probe script may have failed silently."
        )


def _read_probe_devices() -> List[Dict[str, Any]]:
    if not PROBE_FILE.exists():
        return []
    with PROBE_FILE.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    devices = data.get("devices", [])
    # Backward compatibility: older stub used {"by_index": {...}}
    if not devices and "by_index" in data:
        devices = list(data["by_index"].values())
    return list(devices)


def _load_probe_data() -> Dict[str, Any]:
    global _probe_cache
    if _probe_cache is None:
        _run_probe_if_needed()
        devices = _read_probe_devices()
        if not devices and _torch_cuda_available():
            _run_probe_if_needed(force=True)
            devices = _read_probe_devices()
        by_index = {entry["device_index"]: entry for entry in devices}
        by_key: Dict[str, Any] = {}
        for entry in devices:
            by_key.setdefault(entry["key"], entry)
        _probe_cache = {"by_index": by_index, "by_key": by_key}
    return _probe_cache


def refresh_capability_cache() -> None:
    global _probe_cache
    _probe_cache = None


def _entry_to_capabilities(entry: Dict[str, Any]) -> HardwareCapabilities:
    tma_limits = TMALimits(
        max_1d_elements=int(entry["tma"].get("max_1d", 0) or 0),
        max_2d_width=int(entry["tma"].get("max_2d_width", 0) or 0),
        max_2d_height=int(entry["tma"].get("max_2d_height", 0) or 0),
    )
    cluster_info = entry.get("cluster", {})
    cluster = ClusterCapabilities(
        supports_clusters=bool(cluster_info.get("supports_clusters", False)),
        has_dsmem=bool(cluster_info.get("has_dsmem", False)),
        max_cluster_size=int(cluster_info.get("max_cluster_size", 1)),
        notes=cluster_info.get("notes"),
    )
    tensor_cores = entry.get("tensor_cores") or (
        "5th Gen" if entry.get("architecture", "").startswith("blackwell") else "Hopper/Ampere"
    )
    tma_supported_flag = bool(entry.get("tma", {}).get("supported", False))
    tma_compiler_supported_flag = bool(entry.get("tma", {}).get("compiler_support", False))
    return HardwareCapabilities(
        key=entry["key"],
        name=entry.get("name", entry["key"]),
        architecture=entry.get("architecture", "other"),
        compute_capability=entry.get("compute_capability", entry["key"].replace("sm_", "")),
        sm_version=entry["key"],
        device_name=entry.get("name", entry["key"]),
        total_memory_gb=float(entry.get("total_memory_gb", 0.0)),
        l2_cache_kb=(entry.get("l2_cache_bytes") or 0) / 1024.0,
        num_sms=int(entry.get("num_sms", 0)),
        warp_size=int(entry.get("warp_size", 32)),
        max_threads_per_block=int(entry.get("max_threads_per_block", 1024)),
        max_threads_per_sm=int(entry.get("max_threads_per_sm", 0)),
        max_shared_mem_per_block=int(entry.get("max_shared_mem_per_block", 48 * 1024)),
        max_shared_mem_per_sm=int(entry.get("max_shared_mem_per_sm", 0)),
        tensor_cores=tensor_cores,
        features=list(entry.get("features", [])),
        memory_bandwidth_tbps=entry.get("memory_bandwidth_tbps"),
        max_unified_memory_tb=entry.get("max_unified_memory_tb"),
        nvlink_c2c=bool(entry.get("nvlink_c2c")),
        grace_coherence=bool(entry.get("grace_coherence")),
        tma_supported=tma_supported_flag,
        tma_compiler_supported=tma_compiler_supported_flag,
        tma_limits=tma_limits,
        cluster=cluster,
        driver_version=entry.get("driver_version"),
        cuda_runtime_version=entry.get("cuda_runtime_version"),
        notes=list(entry.get("notes", [])),
    )


def detect_capabilities(device_index: int = 0) -> Optional[HardwareCapabilities]:
    data = _load_probe_data()
    entry = data["by_index"].get(device_index)
    if entry is None and torch is not None and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device_index)
        entry = data["by_key"].get(f"sm_{props.major}{props.minor}")
    if entry is None:
        return None
    return _entry_to_capabilities(entry)


def all_capability_records() -> Dict[str, Dict[str, Any]]:
    data = _load_probe_data()
    return {key: value for key, value in data["by_key"].items()}


def format_capability_report(capabilities: Optional[HardwareCapabilities] = None) -> str:
    cap = capabilities or detect_capabilities()
    if cap is None:
        return "CUDA not available on this system."
    lines = [
        f"GPU Name: {cap.device_name}",
        f"Compute Capability: {cap.compute_capability} ({cap.sm_version})",
        f"Architecture: {cap.name} ({cap.architecture})",
        f"Total Memory: {cap.total_memory_gb:.2f} GB",
        f"SMs: {cap.num_sms}",
        f"Warp Size: {cap.warp_size}",
        f"Max Threads per Block: {cap.max_threads_per_block}",
        f"Max Shared Memory per Block: {cap.max_shared_mem_per_block / 1024:.1f} KB",
    ]
    if cap.l2_cache_kb:
        lines.append(f"L2 Cache: {cap.l2_cache_kb:.0f} KB")
    if cap.memory_bandwidth_tbps:
        lines.append(f"Memory Bandwidth: {cap.memory_bandwidth_tbps:.2f} TB/s")
    lines.append(f"TMA Supported: {'yes' if cap.tma_supported else 'no'}")
    if cap.tma_supported:
        lines.append(
            f"  Compiler Support: {'yes' if cap.tma_compiler_supported else 'no'} "
            f"(limits: 1D={cap.tma_limits.max_1d_elements}, "
            f"2D={cap.tma_limits.max_2d_width}x{cap.tma_limits.max_2d_height})"
        )
    lines.append(
        f"Thread Block Clusters: {'yes' if cap.cluster.supports_clusters else 'no'} "
        f"(DSMEM={'yes' if cap.cluster.has_dsmem else 'no'})"
    )
    if cap.driver_version:
        lines.append(f"NVIDIA Driver: {cap.driver_version}")
    if cap.cuda_runtime_version:
        lines.append(f"CUDA Runtime (PyTorch): {cap.cuda_runtime_version}")
    if cap.features:
        lines.append("Features: " + ", ".join(cap.features))
    if cap.notes:
        lines.append("Notes:")
        lines.extend([f"  - {note}" for note in cap.notes])
    return "\n".join(lines)


def ensure_tma_box_supported(
    box_shape: Sequence[int],
    *,
    capability: Optional[HardwareCapabilities] = None,
    description: str = "TMA descriptor",
) -> TMALimits:
    cap = capability or detect_capabilities()
    if cap is None:
        raise RuntimeError("SKIPPED: CUDA device unavailable for TMA operations.")

    if not cap.tma_supported:
        raise RuntimeError(f"SKIPPED: {description} requires Tensor Memory Accelerator (unsupported on {cap.device_name}).")
    if not cap.tma_compiler_supported:
        raise RuntimeError(
            f"SKIPPED: {description} requires tensormap instructions, but the current toolkit/driver refuses them on {cap.sm_version}."
        )

    dims = tuple(int(x) for x in box_shape)
    if len(dims) == 0 or len(dims) > 2:
        raise ValueError("box_shape must describe a 1D or 2D TMA transfer.")

    max_1d = cap.tma_limits.max_1d_elements
    max_width = cap.tma_limits.max_2d_width
    max_height = cap.tma_limits.max_2d_height

    if len(dims) == 1:
        if dims[0] > max_1d:
            raise RuntimeError(
                f"SKIPPED: {description} requests {dims[0]} elements but {cap.sm_version} only supports up to {max_1d}."
            )
    else:
        width, height = dims
        if width > max_width or height > max_height:
            raise RuntimeError(
                f"SKIPPED: {description} requests {width}x{height} but {cap.sm_version} only supports up to {max_width}x{max_height}."
            )
    return cap.tma_limits


def ensure_dsmem_supported(
    *,
    capability: Optional[HardwareCapabilities] = None,
    require_clusters: bool = True,
    description: str = "Distributed shared memory",
) -> ClusterCapabilities:
    cap = capability or detect_capabilities()
    if cap is None:
        raise RuntimeError("SKIPPED: CUDA hardware unavailable for cluster kernels.")

    if require_clusters and not cap.cluster.supports_clusters:
        raise RuntimeError(
            f"SKIPPED: {description} requires thread block clusters, but {cap.device_name} does not expose them."
        )
    if not cap.cluster.has_dsmem:
        hint = cap.cluster.notes or "Driver disabled DSMEM; upgrade or enable compute optimizer."
        raise RuntimeError(f"SKIPPED: {description} requires DSMEM ({hint}).")
    return cap.cluster
