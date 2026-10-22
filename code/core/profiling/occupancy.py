"""Bindings for occupancy CUDA kernels."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from core.utils.extension_loader_template import load_cuda_extension


@lru_cache()
def _load_module():
    repo_root = Path(__file__).resolve().parent.parent
    source = repo_root / "profiling" / "cuda" / "occupancy_extension.cu"
    extra_flags = ["-O3", "--use_fast_math", "-std=c++20"]
    return load_cuda_extension(
        extension_name="occupancy_ext",
        cuda_source_file=str(source),
        extra_cuda_cflags=extra_flags,
    )


def run_low_occupancy(input_tensor: torch.Tensor, output_tensor: torch.Tensor, work_iters: int) -> None:
    _load_module().run_low_occupancy(input_tensor, output_tensor, int(work_iters))


def run_high_occupancy(input_tensor: torch.Tensor, output_tensor: torch.Tensor, work_iters: int) -> None:
    _load_module().run_high_occupancy(input_tensor, output_tensor, int(work_iters))
