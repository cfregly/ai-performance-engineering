"""Utilities for querying CUDA hardware feature support."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

try:
    import torch
except ImportError:  # pragma: no cover - torch always installed in benchmarks
    torch = None  # type: ignore[assignment]

from core.harness.hardware_capabilities import detect_capabilities

# Global flag for forcing pipeline API (set via CLI flag, not env var)
_force_pipeline: bool = False


def set_force_pipeline(force: bool) -> None:
    """Set the force pipeline flag.
    
    Args:
        force: If True, force enable pipeline kernels even on compute capability 12.0+.
    """
    global _force_pipeline
    _force_pipeline = force


@lru_cache(maxsize=1)
def _get_compute_capability() -> Tuple[int, int]:
    """Return (major, minor) compute capability, defaulting to (0, 0)."""
    cap = detect_capabilities()
    if cap is None:
        return (0, 0)
    parts = cap.compute_capability.split(".")
    try:
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
    except ValueError:
        major, minor = 0, 0
    return major, minor


def pipeline_support_status() -> Tuple[bool, str]:
    """Return (supported, reason) for CUDA Pipeline API availability."""
    major, minor = _get_compute_capability()
    if major == 0 and minor == 0:
        return False, "CUDA device not available"
    if major >= 8:
        return True, f"compute capability {major}.{minor} (>= 8.0)"
    return False, f"requires compute capability >= 8.0, found {major}.{minor}"


def supports_cuda_pipeline_api() -> bool:
    """Shortcut for pipeline API availability."""
    return pipeline_support_status()[0]


def tma_support_status() -> Tuple[bool, str]:
    """Return (supported, reason) for Tensor Memory Accelerator availability."""
    cap = detect_capabilities()
    if cap is None:
        return False, "CUDA device not available"
    if not cap.tma_supported:
        return False, f"TMA unavailable on {cap.device_name} ({cap.compute_capability})"
    if not cap.tma_compiler_supported:
        return False, (
            f"TMA instructions are disabled on {cap.device_name} ({cap.sm_version}); "
            "CUDA 13.0 refuses tensormap operands for this architecture."
        )
    return True, f"{cap.device_name} ({cap.compute_capability}) exposes TMA"


def supports_tensor_memory_accelerator() -> bool:
    """Shortcut for Tensor Memory Accelerator availability."""
    return tma_support_status()[0]


def blackwell_tma_support_status() -> Tuple[bool, str]:
    """Return (supported, reason) for Blackwell-ready TMA pipelines."""
    cap = detect_capabilities()
    if cap is None:
        return False, "CUDA device not available"
    # Capability-probe labels: "blackwell" is SM 10.x below 10.3 and SM 11.x,
    # "blackwell_ultra" is SM 10.3, "blackwell_consumer" is SM 12.0 (GeForce RTX 50
    # and RTX PRO) and "grace_blackwell" is SM 12.1 (GB10).
    if cap.architecture not in {"blackwell", "blackwell_ultra", "blackwell_consumer", "grace_blackwell"}:
        return False, f"requires a Blackwell-family GPU (SM 10.x, 11.x or 12.x), found {cap.architecture}"
    if not cap.tma_ready:
        return False, (
            f"{cap.device_name} ({cap.sm_version}) hardware supports TMA but the "
            "current CUDA toolchain does not emit tcgen05/tensormap instructions."
        )
    return True, f"{cap.name} ({cap.compute_capability})"


def pipeline_runtime_allowed() -> Tuple[bool, str]:
    """Return (allowed, reason) for actually running pipeline kernels safely.
    
    This function checks if the CUDA Pipeline API can be safely used at runtime.
    On compute capability 12.0+ (Blackwell), pipeline kernels are disabled by
    default due to known instability issues. This can be overridden via the
    --force-pipeline CLI flag (or set_force_pipeline() function).
    
    Returns:
        Tuple of (allowed: bool, reason: str) indicating whether pipeline
        kernels can be safely run and the reason for the decision.
    """
    supported, reason = pipeline_support_status()
    if not supported:
        return False, reason
    # Check global flag (set via --force-pipeline CLI flag)
    force = _force_pipeline
    major, minor = _get_compute_capability()
    if major >= 12 and not force:
        return True, (
            f"compute capability {major}.{minor} (automatic enablement on Blackwell/GB series)"
        )
    return True, f"compute capability {major}.{minor}"
