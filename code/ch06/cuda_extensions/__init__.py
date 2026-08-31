"""CUDA extension loader for ch06 kernels."""

from pathlib import Path

from core.profiling.nvtx_stub import ensure_nvtx_stub
from core.utils.extension_loader_template import load_cuda_extension_v2

NVTX_CFLAG = "-DENABLE_NVTX_PROFILING"

_EXTENSION_DIR = Path(__file__).parent
_COMMON_HEADERS = _EXTENSION_DIR.parent.parent / "core" / "common" / "headers"


def _cuda_flags() -> list[str]:
    return ["-lineinfo", f"-I{_COMMON_HEADERS}", NVTX_CFLAG]


def _nvtx_ldflags() -> list[str]:
    """Resolve the build dependency only inside a tracked extension loader."""
    stub_library = ensure_nvtx_stub()
    return [f"-L{stub_library.parent}", "-lnvToolsExt"]


def load_coalescing_extension():
    """Load the coalescing kernels CUDA extension."""
    return load_cuda_extension_v2(
        name="coalescing_kernels",
        sources=[_EXTENSION_DIR / "coalescing_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=_nvtx_ldflags(),
    )


def load_bank_conflicts_extension():
    """Load the bank conflicts kernels CUDA extension."""
    return load_cuda_extension_v2(
        name="bank_conflicts_kernels",
        sources=[_EXTENSION_DIR / "bank_conflicts_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=_nvtx_ldflags(),
    )


def load_ilp_extension():
    """Load the ILP kernels CUDA extension."""
    return load_cuda_extension_v2(
        name="ilp_kernels",
        sources=[_EXTENSION_DIR / "ilp_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=_nvtx_ldflags(),
    )


def load_launch_bounds_extension():
    """Load the launch bounds CUDA extension."""
    return load_cuda_extension_v2(
        name="launch_bounds_kernels",
        sources=[_EXTENSION_DIR / "launch_bounds_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=_nvtx_ldflags(),
    )
