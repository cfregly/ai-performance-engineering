"""Optimized FP16 gradient all-reduce (preallocated full-buffer compression, multi-GPU)."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch04.gradient_compression_common import (
    GradientCompressionBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    bench = GradientCompressionBenchmark(
        compression="fp16",
        equivalence_group="ch04_gradient_compression_fp16",
        output_tolerance=(1e-3, 1e-2),
        tensor_size_mb=1024,
        multi_gpu=True,
        use_prealloc_buffers=True,
        bucket_mb=0,
        variant="optimized",
        pair_compression="fp16",
    )
    return attach_benchmark_metadata(bench, __file__)

