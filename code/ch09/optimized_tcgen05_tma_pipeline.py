"""Optimized tcgen05 matmul with 2-stage TMA pipeline."""

from __future__ import annotations

import torch

from core.benchmark.tcgen05_matmul_base import Tcgen05MatmulBenchmarkBase
from core.benchmark.tcgen05_requirements import ensure_tcgen05_supported
from core.common.tcgen05 import load_tcgen05_pipelined_module
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedTcgen05TmaPipelineBenchmark(Tcgen05MatmulBenchmarkBase):
    """Double-buffered tcgen05 matmul optimized for Chapter 9."""

    shared_dim = 2048
    nvtx_label = "optimized_tcgen05_tma_pipeline"

    def __init__(self) -> None:
        super().__init__()
        self.extension = None

    def setup(self) -> None:
        ensure_tcgen05_supported(
            loader=load_tcgen05_pipelined_module,
            module_name="ch09 tcgen05 TMA pipeline",
        )
        super().setup()
        if self.extension is None:
            self.extension = load_tcgen05_pipelined_module()

    def benchmark_fn(self) -> None:
        if self.extension is None or self.matrix_a is None or self.matrix_b is None:
            raise RuntimeError("Inputs or extension not initialized")
        with self._nvtx_range(self.nvtx_label):
            with torch.inference_mode():
                self.output = self.extension.matmul_tcgen05_pipelined(self.matrix_a, self.matrix_b)


def get_benchmark() -> BaseBenchmark:
    return OptimizedTcgen05TmaPipelineBenchmark()

