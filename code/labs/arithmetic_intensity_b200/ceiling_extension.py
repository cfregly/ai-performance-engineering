"""CUDA extension loader for the bounded B200 ceiling diagnostic."""

from __future__ import annotations

from pathlib import Path

import torch

from core.utils.extension_loader_template import load_cuda_extension_v2

_PACKAGE_ROOT = Path(__file__).resolve().parent


def load_ceiling_extension():
    """Compile and load the local diagnostic kernels for the visible GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("The B200 ceiling extension requires CUDA")
    major, minor = torch.cuda.get_device_capability()
    return load_cuda_extension_v2(
        name=f"arithmetic_intensity_b200_ceiling_sm{major}{minor}",
        sources=[_PACKAGE_ROOT / "ceiling_kernels.cu"],
        extra_cuda_cflags=["-O3", "-lineinfo", "-std=c++20"],
        extra_cflags=["-O3", "-std=c++20"],
    )
