"""Optimized tiling benchmark that targets tcgen05 tensor cores."""

from __future__ import annotations

from ch08.tiling_benchmark_base_tcgen05 import TilingBenchmarkBaseTCGen05


class OptimizedTilingBenchmarkTCGen05(TilingBenchmarkBaseTCGen05):
    """Runs the SM100 tcgen05 GEMM."""

    nvtx_label = "optimized_tiling_tcgen05"

    def __init__(self) -> None:
        super().__init__()
        self.matrix_b_t = None

    def setup(self) -> None:
        super().setup()
        if self.matrix_b is None:
            raise RuntimeError("Input matrices not initialized")
        self.matrix_b_t = self.matrix_b.t().contiguous()

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.matrix_a is not None
        assert self.matrix_b_t is not None
        result = self.extension.matmul_tiling_tcgen05_pretransposed(self.matrix_a, self.matrix_b_t)
        self.output = result

    def teardown(self) -> None:
        self.matrix_b_t = None
        super().teardown()


def get_benchmark() -> OptimizedTilingBenchmarkTCGen05:
    return OptimizedTilingBenchmarkTCGen05()

