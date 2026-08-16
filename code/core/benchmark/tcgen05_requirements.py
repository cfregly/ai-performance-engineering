"""Shared helpers for tcgen05-capable benchmarks."""

from __future__ import annotations

from collections.abc import Callable

import torch

try:  # Ensure TORCH_CUDA_ARCH_LIST gets clamped to sm_120 on GB10.
    import arch_config  # noqa: F401
except ImportError:  # pragma: no cover - optional during import bootstrap
    arch_config = None  # type: ignore[assignment]

from core.benchmark.blackwell_requirements import (  # noqa: E402, I001
    ensure_blackwell_tma_supported,
)


SUPPORTED_TCGEN05_CAPABILITIES = frozenset({(10, 0), (10, 3)})


def ensure_tcgen05_capability_supported(
    capability: tuple[int, int],
    *,
    module_name: str = "tcgen05 kernel",
) -> None:
    """Raise a SKIPPED error unless an exact, validated tcgen05 target is visible."""
    if capability in SUPPORTED_TCGEN05_CAPABILITIES:
        return

    major, minor = capability
    sm_version = f"sm_{major}{minor}"
    if major < 10:
        reason = "requires SM100-class Tensor Cores"
    elif capability == (12, 0):
        reason = "has no natively validated SM120 implementation"
    elif capability == (12, 1):
        reason = "is not supported on sm_121 (GB10)"
    else:
        reason = f"does not support {sm_version}. Validated targets are sm_100 and sm_103"
    raise RuntimeError(f"SKIPPED: {module_name} {reason}.")


def ensure_tcgen05_supported(
    loader: Callable[[], object] | None = None,
    *,
    module_name: str = "tcgen05 kernel",
) -> None:
    """Raise a SKIPPED error if tcgen05 kernels cannot run."""
    ensure_blackwell_tma_supported(module_name)
    ensure_tcgen05_capability_supported(
        torch.cuda.get_device_capability(),
        module_name=module_name,
    )
    if loader is None:
        return
    try:
        loader()
    except RuntimeError as exc:
        raise RuntimeError(f"SKIPPED: {module_name} unavailable ({exc})") from exc


def check_tcgen05_support(
    loader: Callable[[], object] | None = None,
    *,
    module_name: str = "tcgen05 kernel",
) -> tuple[bool, str | None]:
    """Return (is_supported, reason) without raising on SKIPPED errors."""
    try:
        ensure_tcgen05_supported(loader=loader, module_name=module_name)
        return True, None
    except RuntimeError as exc:
        message = str(exc)
        if "SKIPPED" not in message:
            unsupported_markers = (
                "No CUDA GPUs are available",
                "Found no NVIDIA driver",
                "CUDA driver version is insufficient",
                "Torch not compiled with CUDA enabled",
            )
            if any(marker in message for marker in unsupported_markers):
                return False, f"SKIPPED: {module_name} unavailable ({message})"
        if "SKIPPED" not in message:
            raise
        return False, message
