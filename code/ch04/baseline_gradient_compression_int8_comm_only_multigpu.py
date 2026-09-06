"""Baseline FP32 gradient all-reduce (communication-only INT8 compression, multi-GPU)."""

from __future__ import annotations

from core.harness.benchmark_harness import BaseBenchmark

from ch04.gradient_compression_common import (
    GradientCompressionBenchmark,
    attach_benchmark_metadata,
)


def get_benchmark() -> BaseBenchmark:
    bench = GradientCompressionBenchmark(
        compression="none",
        equivalence_group="ch04_gradient_compression_int8_comm_only",
        output_tolerance=(1e-1, 5e-1),
        tensor_size_mb=1024,
        multi_gpu=True,
        comm_only=True,
        variant="baseline",
        pair_compression="int8",
    )
    return attach_benchmark_metadata(bench, __file__)
