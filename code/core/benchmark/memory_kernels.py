"""CUDA kernels for memory access pattern benchmarks."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from core.utils.extension_loader_template import load_cuda_extension


@lru_cache()
def _load_module():
    package_root = Path(__file__).resolve().parent
    source = package_root / "cuda" / "memory_patterns_extension.cu"
    extra_flags = ["-O3", "--use_fast_math", "-std=c++20", "--extended-lambda"]
    return load_cuda_extension(
        extension_name="memory_patterns_ext",
        cuda_source_file=str(source),
        extra_cuda_cflags=extra_flags,
    )


def coalesced_copy(src: torch.Tensor, dst: torch.Tensor) -> None:
    """Launch the coalesced copy kernel."""
    _load_module().coalesced_copy(src, dst)


def uncoalesced_copy(src: torch.Tensor, dst: torch.Tensor, stride: int) -> None:
    """Launch the uncoalesced copy kernel with the provided stride."""
    _load_module().uncoalesced_copy(src, dst, stride)


def bank_conflict_transpose(src: torch.Tensor, dst: torch.Tensor, padded: bool) -> None:
    """Launch the transpose kernel with/without padding to expose bank conflicts."""
    _load_module().bank_conflict_transpose(src, dst, padded)
