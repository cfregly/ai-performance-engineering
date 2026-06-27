"""Baseline tcgen05 matmul with single-stage TMA loads (no pipelining)."""

from __future__ import annotations

import torch

from core.benchmark.tcgen05_matmul_base import Tcgen05MatmulBenchmarkBase
from core.benchmark.tcgen05_requirements import ensure_tcgen05_supported
from core.common.tcgen05 import load_tcgen05_basic_module
from core.harness.benchmark_harness import BaseBenchmark


class BaselineTcgen05TmaPipelineBenchmark(Tcgen05MatmulBenchmarkBase):
    """Single-stage tcgen05 matmul baseline for Chapter 9."""

    shared_dim = 2048
    nvtx_label = "baseline_tcgen05_tma_pipeline"

    def __init__(self) -> None:
        super().__init__()
        self.extension = None

    def setup(self) -> None:
        ensure_tcgen05_supported(
            loader=load_tcgen05_basic_module,
            module_name="ch09 tcgen05 TMA pipeline",
        )
        super().setup()
        if self.extension is None:
            self.extension = load_tcgen05_basic_module()

    def benchmark_fn(self) -> None:
        if self.extension is None or self.matrix_a is None or self.matrix_b is None:
            raise RuntimeError("Inputs or extension not initialized")
        with self._nvtx_range(self.nvtx_label):
            with torch.inference_mode():
                self.output = self.extension.matmul_tcgen05_basic(self.matrix_a, self.matrix_b)


def get_benchmark() -> BaseBenchmark:
    return BaselineTcgen05TmaPipelineBenchmark()

