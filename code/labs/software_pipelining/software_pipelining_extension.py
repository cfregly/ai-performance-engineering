"""CUDA extension loader for the software-pipelining lab."""

from __future__ import annotations

from pathlib import Path

from core.utils.extension_loader_template import load_cuda_extension_v2

_EXTENSION_DIR = Path(__file__).resolve().parent


def load_software_pipelining_extension():
    """Load the local CUDA kernels that compare serialized vs pipelined tiling."""

    return load_cuda_extension_v2(
        name="software_pipelining_kernels_ext",
        sources=[_EXTENSION_DIR / "software_pipelining_kernels.cu"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        extra_cflags=["-O3", "-std=c++20"],
    )
