"""CUDA extension loader for the B200 kernel optimization candidates."""

from __future__ import annotations

from pathlib import Path

import torch

from core.utils.extension_loader_template import load_cuda_extension_v2

_PACKAGE_ROOT = Path(__file__).resolve().parent


def load_optimization_extension():
    """Compile the direct cuBLASLt plan and cache-policy copy candidate."""
    if not torch.cuda.is_available():
        raise RuntimeError("The B200 optimization extension requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) != (10, 0):
        raise RuntimeError(
            "The B200 optimization extension requires compute capability 10.0, "
            f"got {major}.{minor}"
        )
    return load_cuda_extension_v2(
        name=f"arithmetic_intensity_b200_optimization_sm{major}{minor}",
        sources=[
            _PACKAGE_ROOT / "optimization_bindings.cpp",
            _PACKAGE_ROOT / "optimization_kernels.cu",
        ],
        extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++20"],
        extra_cflags=["-O3", "-std=c++20"],
        extra_ldflags=["-lcublasLt", "-lcublas"],
    )
