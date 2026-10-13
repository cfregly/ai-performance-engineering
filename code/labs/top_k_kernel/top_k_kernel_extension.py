"""CUTLASS CUDA extension loader for the grouped Top-K selection kernel lab."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from core.utils.extension_loader_template import load_cuda_extension

_EXTENSION_DIR = Path(__file__).parent


@lru_cache()
def load_top_k_kernel_extension():
    """Load the CUTLASS SM80 tensor-op GEMM helper that scores the forward CUDA Top-K path."""

    return load_cuda_extension(
        extension_name="top_k_kernel_cutlass_ext",
        cuda_source_file=str(_EXTENSION_DIR / "top_k_kernel_cuda.cu"),
        extra_cuda_cflags=[
            "-O3",
            "-lineinfo",
            "--use_fast_math",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "-std=c++17",
        ],
    )
