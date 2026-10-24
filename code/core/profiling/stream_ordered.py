"""Stream-ordered allocator CUDA extension bindings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from core.utils.extension_loader_template import load_cuda_extension


@lru_cache()
def _load_module():
    repo_root = Path(__file__).resolve().parent.parent
    source = repo_root / "profiling" / "cuda" / "stream_ordered_extension.cu"
    extra_flags = ["-O3", "--use_fast_math", "-std=c++20"]
    return load_cuda_extension(
        extension_name="stream_ordered_ext",
        cuda_source_file=str(source),
        extra_cuda_cflags=extra_flags,
    )


def load_stream_ordered_module():
    """Load and return the stream-ordered allocator extension module."""
    return _load_module()


def run_standard_allocator(elements: int, iterations: int = 5) -> None:
    """Execute the cudaMalloc baseline workload."""
    _load_module().run_standard_allocator(int(elements), int(iterations))


def run_stream_ordered_allocator(elements: int, iterations: int = 5) -> None:
    """Execute the cudaMallocAsync (stream-ordered) workload."""
    _load_module().run_stream_ordered_allocator(int(elements), int(iterations))


def run_standard_allocator_capture(elements: int, iterations: int = 5):
    """Execute the cudaMalloc baseline workload and return a small output slice."""
    return _load_module().run_standard_allocator_capture(int(elements), int(iterations))


def run_stream_ordered_allocator_capture(elements: int, iterations: int = 5):
    """Execute the cudaMallocAsync workload and return a small output slice."""
    return _load_module().run_stream_ordered_allocator_capture(int(elements), int(iterations))
