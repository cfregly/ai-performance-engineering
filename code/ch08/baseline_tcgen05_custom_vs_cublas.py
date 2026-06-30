"""Baseline side of the tcgen05-versus-cuBLAS bridge comparison."""

from __future__ import annotations

import torch

from ch08.tcgen05_custom_vs_cublas_benchmark_base import Tcgen05CustomVsCublasBase
from core.harness.benchmark_harness import BaseBenchmark


class BaselineTcgen05CustomVsCublasBenchmark(Tcgen05CustomVsCublasBase):
    """Vendor cuBLAS reference side of the comparison pair."""

    nvtx_label = "baseline_tcgen05_custom_vs_cublas"

    def setup(self) -> None:
        super().setup()
        self._cublas_output_buffer = self._new_output_buffer()

    def benchmark_fn(self) -> None:
        if self.matrix_a is None or self.matrix_b is None:
            raise RuntimeError("Inputs not initialized")
        with torch.inference_mode(), self._nvtx_range(self.nvtx_label):
            self.output = self._run_cublas_reference()


def get_benchmark() -> BaseBenchmark:
    return BaselineTcgen05CustomVsCublasBenchmark()
