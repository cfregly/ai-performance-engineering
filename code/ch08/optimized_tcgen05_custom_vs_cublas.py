"""Optimized side of the tcgen05-versus-cuBLAS bridge comparison."""

from __future__ import annotations

import torch

from ch08.tcgen05_custom_vs_cublas_benchmark_base import Tcgen05CustomVsCublasBase
from core.benchmark.tcgen05_requirements import ensure_tcgen05_supported
from core.common.tcgen05 import load_tiling_tcgen05_module
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedTcgen05CustomVsCublasBenchmark(Tcgen05CustomVsCublasBase):
    """Custom tcgen05 kernel side of the comparison pair."""

    nvtx_label = "optimized_tcgen05_custom_vs_cublas"

    def __init__(self) -> None:
        super().__init__()
        self.extension = None
        self.matrix_b_t = None
        self._tcgen05_output_buffer = None

    def setup(self) -> None:
        ensure_tcgen05_supported(
            loader=load_tiling_tcgen05_module,
            module_name="ch08 tcgen05 tiling vs cuBLAS",
        )
        super().setup()
        if self.matrix_b is None:
            raise RuntimeError("Input matrices not initialized")
        if self.extension is None:
            self.extension = load_tiling_tcgen05_module()
        self.matrix_b_t = self.matrix_b.t().contiguous()
        self._tcgen05_output_buffer = torch.empty(
            self.matrix_rows,
            self.matrix_cols,
            device=self.device,
            dtype=torch.float32,
        )

    def benchmark_fn(self) -> None:
        if (
            self.extension is None
            or self.matrix_a is None
            or self.matrix_b_t is None
            or self._tcgen05_output_buffer is None
        ):
            raise RuntimeError("Inputs or extension not initialized")
        with self._nvtx_range(self.nvtx_label):
            with torch.inference_mode():
                self.output = self.extension.matmul_tiling_tcgen05_pretransposed_out(
                    self.matrix_a,
                    self.matrix_b_t,
                    self._tcgen05_output_buffer,
                )

    def teardown(self) -> None:
        self.matrix_b_t = None
        self._tcgen05_output_buffer = None
        self.extension = None
        super().teardown()


def get_benchmark() -> BaseBenchmark:
    return OptimizedTcgen05CustomVsCublasBenchmark()
