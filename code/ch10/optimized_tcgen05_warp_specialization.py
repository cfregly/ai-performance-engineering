"""Optimized tcgen05 matmul using a warp-specialized TMA pipeline."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.tcgen05_matmul_base import Tcgen05MatmulBenchmarkBase
from core.benchmark.tcgen05_requirements import ensure_tcgen05_supported
from core.common.tcgen05 import load_tcgen05_warp_specialized_module
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedTcgen05WarpSpecializationBenchmark(Tcgen05MatmulBenchmarkBase):
    """Chapter 10 optimized: tcgen05 GEMM with warp specialization."""

    matrix_rows = 16384
    matrix_cols = 16384
    shared_dim = 2048
    nvtx_label = "optimized_tcgen05_warp_specialization"

    def __init__(self) -> None:
        super().__init__()
        self.extension: Optional[object] = None

    def setup(self) -> None:
        ensure_tcgen05_supported(
            loader=load_tcgen05_warp_specialized_module,
            module_name="ch10 tcgen05 warp specialization",
        )
        super().setup()
        if self.extension is None:
            self.extension = load_tcgen05_warp_specialized_module()

    def benchmark_fn(self) -> None:
        if self.extension is None or self.matrix_a is None or self.matrix_b is None:
            raise RuntimeError("Inputs or extension not initialized")
        with self._nvtx_range(self.nvtx_label):
            with torch.inference_mode():
                self.output = self.extension.matmul_tcgen05_warp_specialized(self.matrix_a, self.matrix_b)


def get_benchmark() -> BaseBenchmark:
    return OptimizedTcgen05WarpSpecializationBenchmark()

