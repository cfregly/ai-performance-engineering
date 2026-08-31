"""CUDA extension loader for ch12 kernels."""

from pathlib import Path

from core.profiling.nvtx_stub import ensure_nvtx_stub
from core.utils.extension_loader_template import load_cuda_extension_v2

NVTX_CFLAG = "-DENABLE_NVTX_PROFILING"
NVCC_ALLOW_UNSUPPORTED = "--allow-unsupported-compiler"

_EXTENSION_DIR = Path(__file__).parent
_COMMON_HEADERS = _EXTENSION_DIR.parent.parent / "core" / "common" / "headers"


def _cuda_flags() -> list[str]:
    return ["-lineinfo", f"-I{_COMMON_HEADERS}", NVTX_CFLAG, NVCC_ALLOW_UNSUPPORTED]


def _nvtx_ldflags() -> list[str]:
    """Resolve the build dependency only inside a tracked extension loader."""
    stub_library = ensure_nvtx_stub()
    return [f"-L{stub_library.parent}", "-lnvToolsExt"]


def load_kernel_fusion_extension():
    """Load the kernel fusion CUDA extension."""
    return load_cuda_extension_v2(
        name="kernel_fusion_kernels",
        sources=[_EXTENSION_DIR / "kernel_fusion_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=_nvtx_ldflags(),
    )


def load_graph_bandwidth_extension():
    """Load the graph bandwidth CUDA extension."""
    return load_cuda_extension_v2(
        name="graph_bandwidth_kernels",
        sources=[_EXTENSION_DIR / "graph_bandwidth_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=_nvtx_ldflags(),
    )


def load_work_queue_extension():
    """Load the work queue CUDA extension."""
    return load_cuda_extension_v2(
        name="work_queue_kernels",
        sources=[_EXTENSION_DIR / "work_queue_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=_nvtx_ldflags(),
    )


def load_cuda_graphs_extension():
    """Load the CUDA graphs extension."""
    return load_cuda_extension_v2(
        name="cuda_graphs_kernels",
        sources=[_EXTENSION_DIR / "cuda_graphs_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=_nvtx_ldflags(),
    )


def load_bias_relu_residual_extension():
    """Load CUDA kernels for bias+ReLU and residual-add fusion benchmark."""
    return load_cuda_extension_v2(
        name="bias_relu_residual_kernels",
        sources=[_EXTENSION_DIR / "bias_relu_residual_kernels.cu"],
        extra_cuda_cflags=_cuda_flags(),
        extra_ldflags=_nvtx_ldflags(),
    )
