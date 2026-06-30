"""CUDA extension loader for the Ch13 fused matmul epilogue."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from core.utils.extension_loader_template import load_cuda_extension


@lru_cache(None)
def load_matmul_epilogue_extension():
    """Compile and return the fused bias/ReLU/residual/scale epilogue extension."""
    return load_cuda_extension(
        extension_name="ch13_matmul_epilogue_ext",
        cuda_source_file=str(Path(__file__).with_name("matmul_epilogue_extension.cu")),
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
    )


__all__ = ["load_matmul_epilogue_extension"]
