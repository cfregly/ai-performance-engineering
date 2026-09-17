"""Optimized: safe adaptation of fast.cu's vectorized int32 reduction."""

from __future__ import annotations

import torch

from core.harness.benchmark_harness import BaseBenchmark
from labs.fast_cu.h100_common import Int32ReductionBenchmarkBase


class OptimizedInt32ReductionBenchmark(Int32ReductionBenchmarkBase):
    """Run the upstream-derived int4 reduction without its cross-CTA reset race."""

    nvtx_label = "fast_cu_optimized_int32_reduction"

    def setup(self) -> None:
        self._setup_tensors()

    def benchmark_fn(self) -> None:
        if self.extension is None or self.input is None or self._output_buffer is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        with torch.inference_mode(), self._nvtx_range(self.nvtx_label):
            self.extension.run_safe_vectorized_reduction(self.input, self._output_buffer)
            self.output = self._output_buffer


def get_benchmark() -> BaseBenchmark:
    return OptimizedInt32ReductionBenchmark()
