"""Lightweight GPU telemetry helpers (temperature, power, utilization).

This module is intentionally fail-fast: when CUDA is available, NVML is
required and any NVML errors propagate as exceptions.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
import time
from typing import Any, Dict, List, Optional

import torch

try:
    import pynvml  # type: ignore
    _PYNVML_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as exc:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore
    _PYNVML_IMPORT_ERROR = exc

_NVML_INITIALIZED = False
_TELEMETRY_CACHE: Dict[int, tuple[float, Dict[str, Optional[float | str]]]] = {}
# Cache telemetry briefly to avoid repeated NVML calls inside tight loops.
# Kept fixed to avoid environment-dependent behavior.
_TELEMETRY_TTL_SEC = 1.0


def _ensure_nvml_initialized() -> None:
    """Initialize NVML once per process."""
    global _NVML_INITIALIZED
    if _NVML_INITIALIZED:
        return
    if pynvml is None:
        raise RuntimeError(
            "query_gpu_telemetry requires pynvml (nvidia-ml-py) when CUDA is available."
        ) from _PYNVML_IMPORT_ERROR
    pynvml.nvmlInit()  # type: ignore[attr-defined]
    _NVML_INITIALIZED = True


def _resolve_physical_gpu_index(logical_index: int) -> int:
    """Map a logical CUDA device index through numeric visibility tokens.

    When CUDA_VISIBLE_DEVICES is set, PyTorch reports logical indices starting
    at zero. UUID and MIG tokens require identity-based NVML lookup and are
    intentionally rejected here instead of silently selecting the wrong GPU.
    """
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not visible:
        return logical_index
    tokens = [token.strip() for token in visible.split(",") if token.strip()]
    if not tokens:
        return logical_index
    if logical_index < 0 or logical_index >= len(tokens):
        raise RuntimeError(
            f"CUDA_VISIBLE_DEVICES={visible!r} does not include logical index {logical_index}"
        )
    token = tokens[logical_index]
    try:
        physical_index = int(token)
    except ValueError as exc:
        raise RuntimeError(
            "Cannot resolve a nonnumeric CUDA_VISIBLE_DEVICES entry without "
            f"the CUDA device UUID: {token!r}"
        ) from exc
    if physical_index < 0:
        raise RuntimeError(f"Invalid physical GPU index in CUDA_VISIBLE_DEVICES: {token!r}")
    return physical_index


def normalize_gpu_uuid(value: object) -> Optional[str]:
    """Give bare CUDA and GPU-prefixed NVML UUIDs the same identity.

    PyTorch can stringify a full-device UUID without NVML's ``GPU-`` prefix.
    Preserve MIG and other opaque identity formats for exact NVML lookup.
    """
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    normalized = str(value).strip()
    if not normalized:
        return None
    candidate = normalized[4:] if normalized.startswith("GPU-") else normalized
    try:
        return f"GPU-{uuid.UUID(candidate)}"
    except ValueError:
        return normalized


def resolve_nvml_device_handle(
    nvml_module: Any,
    logical_index: int,
    *,
    cuda_device_uuid: object = None,
) -> Any:
    """Resolve the NVML handle for a logical CUDA device by exact identity."""
    normalized_uuid = normalize_gpu_uuid(cuda_device_uuid)
    if normalized_uuid is not None:
        return nvml_module.nvmlDeviceGetHandleByUUID(normalized_uuid)
    physical_index = _resolve_physical_gpu_index(logical_index)
    return nvml_module.nvmlDeviceGetHandleByIndex(physical_index)


def _coerce_metric_value(value: object) -> Optional[float | str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def query_gpu_telemetry(
    device_index: Optional[int] = None, *, force_refresh: bool = False
) -> Dict[str, Optional[float | str]]:
    """Return a snapshot of GPU telemetry (temperature, power, utilization, errors).

    Args:
        device_index: Logical CUDA device index. Defaults to the current device.

    Returns:
        Dict with temperature/power/utilization/error data (values may be None when unavailable).
    """
    metrics: Dict[str, Optional[float | str]] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gpu_index": None,
        "temperature_gpu_c": None,
        "temperature_memory_c": None,
        "power_draw_w": None,
        "fan_speed_pct": None,
        "utilization_gpu_pct": None,
        "utilization_memory_pct": None,
        "graphics_clock_mhz": None,
        "memory_clock_mhz": None,
        "nvlink_tx_gbps": None,
        "nvlink_rx_gbps": None,
        "nvlink_tx_bytes_total": None,
        "nvlink_rx_bytes_total": None,
        "nvlink_link_count": None,
        "nvlink_status": "unknown",
        # ECC and error metrics
        "ecc_errors_corrected": None,
        "ecc_errors_uncorrected": None,
        "retired_pages_sbe": None,
        "retired_pages_dbe": None,
        # PCIe metrics
        "pcie_tx_bytes": None,
        "pcie_rx_bytes": None,
        "pcie_replay_counter": None,
        "pcie_generation": None,
        "pcie_link_width": None,
        # Performance state
        "performance_state": None,
        "throttle_reasons": None,
    }

    if not torch.cuda.is_available():
        return metrics

    logical_index = device_index if device_index is not None else torch.cuda.current_device()
    metrics["gpu_index"] = logical_index

    cache_key = int(logical_index)
    cached = _TELEMETRY_CACHE.get(cache_key)
    now = time.monotonic()
    if (
        (not force_refresh)
        and cached
        and _TELEMETRY_TTL_SEC > 0
        and (now - cached[0]) < _TELEMETRY_TTL_SEC
    ):
        return dict(cached[1])

    nvml_metrics = _query_via_nvml(logical_index)
    metrics.update(nvml_metrics)
    metrics_copy = {key: _coerce_metric_value(value) for key, value in metrics.items()}
    if _TELEMETRY_TTL_SEC > 0:
        _TELEMETRY_CACHE[cache_key] = (now, metrics_copy)
    return metrics_copy


def _query_via_nvml(logical_index: int) -> Dict[str, Optional[float]]:
    _ensure_nvml_initialized()
    if pynvml is None:
        raise RuntimeError("Internal error: pynvml unavailable after initialization.")
    properties = torch.cuda.get_device_properties(logical_index)
    handle = resolve_nvml_device_handle(
        pynvml,
        logical_index,
        cuda_device_uuid=getattr(properties, "uuid", None),
    )

    temp_gpu = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)  # type: ignore[attr-defined]
    temp_mem: Optional[float] = None
    if hasattr(pynvml, "NVML_TEMPERATURE_MEMORY"):
        temp_mem = float(
            pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_MEMORY)  # type: ignore[attr-defined]
        )

    power_draw_mw = pynvml.nvmlDeviceGetPowerUsage(handle)  # type: ignore[attr-defined]
    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[attr-defined]
    graphics_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)  # type: ignore[attr-defined]
    memory_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)  # type: ignore[attr-defined]
    app_graphics_clock = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_SM)  # type: ignore[attr-defined]
    app_memory_clock = pynvml.nvmlDeviceGetApplicationsClock(handle, pynvml.NVML_CLOCK_MEM)  # type: ignore[attr-defined]

    metrics: Dict[str, Optional[float]] = {
        "temperature_gpu_c": float(temp_gpu),
        "temperature_memory_c": temp_mem,
        "power_draw_w": float(power_draw_mw) / 1000.0,
        "fan_speed_pct": None,  # nvmlDeviceGetFanSpeed may raise NotSupported on datacenter GPUs
        "utilization_gpu_pct": float(utilization.gpu),  # type: ignore[attr-defined]
        "utilization_memory_pct": float(utilization.memory),  # type: ignore[attr-defined]
        "graphics_clock_mhz": float(graphics_clock),
        "memory_clock_mhz": float(memory_clock),
        "applications_clock_sm_mhz": float(app_graphics_clock),
        "applications_clock_memory_mhz": float(app_memory_clock),
        # NVLink counters/utilization are not universally supported; keep keys but do not query here.
        "nvlink_tx_gbps": None,
        "nvlink_rx_gbps": None,
        "nvlink_tx_bytes_total": None,
        "nvlink_rx_bytes_total": None,
        "nvlink_link_count": None,
        "nvlink_status": "not_collected",
        # ECC/retired-page APIs may raise NotSupported; keep keys but do not query here.
        "ecc_errors_corrected": None,
        "ecc_errors_uncorrected": None,
        "retired_pages_sbe": None,
        "retired_pages_dbe": None,
    }

    # PCIe metrics (supported on modern datacenter GPUs; fail-fast on NVML errors).
    pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_TX_BYTES)  # type: ignore[attr-defined]
    pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(handle, pynvml.NVML_PCIE_UTIL_RX_BYTES)  # type: ignore[attr-defined]
    metrics["pcie_tx_bytes"] = float(pcie_tx) * 1024.0  # KB/s to bytes/s
    metrics["pcie_rx_bytes"] = float(pcie_rx) * 1024.0
    metrics["pcie_replay_counter"] = float(pynvml.nvmlDeviceGetPcieReplayCounter(handle))  # type: ignore[attr-defined]
    metrics["pcie_generation"] = float(pynvml.nvmlDeviceGetCurrPcieLinkGeneration(handle))  # type: ignore[attr-defined]
    metrics["pcie_link_width"] = float(pynvml.nvmlDeviceGetCurrPcieLinkWidth(handle))  # type: ignore[attr-defined]

    # Performance state + throttle reasons.
    metrics["performance_state"] = float(pynvml.nvmlDeviceGetPerformanceState(handle))  # type: ignore[attr-defined]
    metrics["throttle_reasons"] = float(pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle))  # type: ignore[attr-defined]

    return metrics


def format_gpu_telemetry(metrics: Dict[str, Optional[float]]) -> str:
    """Return a human-readable summary string for telemetry."""
    if not metrics:
        return "GPU telemetry unavailable"

    parts = []
    temp = metrics.get("temperature_gpu_c")
    if temp is not None:
        parts.append(f"temp={temp:.1f}°C")
    power = metrics.get("power_draw_w")
    if power is not None:
        parts.append(f"power={power:.1f}W")
    util = metrics.get("utilization_gpu_pct")
    if util is not None:
        parts.append(f"util={util:.0f}%")
    mem_util = metrics.get("utilization_memory_pct")
    if mem_util is not None:
        parts.append(f"mem_util={mem_util:.0f}%")
    clock = metrics.get("graphics_clock_mhz")
    if clock is not None:
        parts.append(f"clock={clock:.0f}MHz")
    app_clock = metrics.get("applications_clock_sm_mhz")
    app_mem_clock = metrics.get("applications_clock_memory_mhz")
    if app_clock is not None:
        if app_mem_clock is not None:
            parts.append(f"app_clock={app_clock:.0f}/{app_mem_clock:.0f}MHz")
        else:
            parts.append(f"app_clock={app_clock:.0f}MHz")
    fan = metrics.get("fan_speed_pct")
    if fan is not None:
        parts.append(f"fan={fan:.0f}%")
    
    # PCIe metrics
    pcie_gen = metrics.get("pcie_generation")
    pcie_width = metrics.get("pcie_link_width")
    if pcie_gen is not None and pcie_width is not None:
        parts.append(f"pcie=Gen{int(pcie_gen)}x{int(pcie_width)}")
    
    # Error indicators (only show if non-zero)
    ecc_corrected = metrics.get("ecc_errors_corrected")
    if ecc_corrected is not None and ecc_corrected > 0:
        parts.append(f"ecc_corr={int(ecc_corrected)}")
    ecc_uncorrected = metrics.get("ecc_errors_uncorrected")
    if ecc_uncorrected is not None and ecc_uncorrected > 0:
        parts.append(f"ecc_uncorr={int(ecc_uncorrected)}⚠")
    
    # Throttle indicator
    throttle = metrics.get("throttle_reasons")
    if throttle is not None and throttle > 0:
        parts.append(f"throttled⚠")
    
    if not parts:
        return "GPU telemetry unavailable"
    return ", ".join(parts)


def get_throttle_reason_names(throttle_bitmask: int) -> List[str]:
    """Decode throttle reasons bitmask into human-readable names."""
    if pynvml is None or throttle_bitmask == 0:
        return []
    
    reasons = []
    # NVML throttle reason constants
    reason_map = {
        getattr(pynvml, "nvmlClocksThrottleReasonGpuIdle", 0x0000000000000001): "idle",
        getattr(pynvml, "nvmlClocksThrottleReasonApplicationsClocksSetting", 0x0000000000000002): "app_clocks",
        getattr(pynvml, "nvmlClocksThrottleReasonSwPowerCap", 0x0000000000000004): "sw_power_cap",
        getattr(pynvml, "nvmlClocksThrottleReasonHwSlowdown", 0x0000000000000008): "hw_slowdown",
        getattr(pynvml, "nvmlClocksThrottleReasonSyncBoost", 0x0000000000000010): "sync_boost",
        getattr(pynvml, "nvmlClocksThrottleReasonSwThermalSlowdown", 0x0000000000000020): "sw_thermal",
        getattr(pynvml, "nvmlClocksThrottleReasonHwThermalSlowdown", 0x0000000000000040): "hw_thermal",
        getattr(pynvml, "nvmlClocksThrottleReasonHwPowerBrakeSlowdown", 0x0000000000000080): "power_brake",
        getattr(pynvml, "nvmlClocksThrottleReasonDisplayClockSetting", 0x0000000000000100): "display_clocks",
    }
    
    for mask, name in reason_map.items():
        if mask and (throttle_bitmask & mask):
            reasons.append(name)
    
    return reasons
