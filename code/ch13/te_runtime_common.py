"""Shared runtime initialization helpers for Transformer Engine benchmarks."""

from __future__ import annotations

import ctypes
from functools import lru_cache
from pathlib import Path

import torch

from core.env import apply_env_defaults

_TORCH_CUDA_LIBS = (
    "libtorch_cuda.so",
    "libtorch_cuda_linalg.so",
    "libtorch_nvshmem.so",
    "libc10_cuda.so",
)

_TE_PRECISION_OUTPUT_TOLERANCES: dict[str, tuple[float, float]] = {
    "prediction": (0.4, 1.0),
    "parameter.fc1.weight": (0.001, 0.00075),
    "parameter.fc1.bias": (0.001, 0.00005),
    "parameter.fc2.weight": (0.001, 0.00075),
    "parameter.fc2.bias": (0.001, 0.00005),
}


def get_te_precision_output_tolerances() -> dict[str, tuple[float, float]]:
    """Return the calibrated full-output policy for the TE2.18 precision pair."""
    return dict(_TE_PRECISION_OUTPUT_TOLERANCES)


@lru_cache(maxsize=1)
def ensure_te_runtime_initialized() -> None:
    """Apply runtime defaults only when a TE-backed benchmark actually runs."""
    apply_env_defaults()
    torch_lib_dir = Path(torch.__file__).resolve().parent / "lib"
    for name in _TORCH_CUDA_LIBS:
        candidate = torch_lib_dir / name
        if candidate.exists():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
