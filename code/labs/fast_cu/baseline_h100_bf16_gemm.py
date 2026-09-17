"""Baseline: cuBLAS BF16 GEMM for the fast.cu H100 comparison."""

from __future__ import annotations

import torch

from core.harness.benchmark_harness import BaseBenchmark
from labs.fast_cu.h100_common import H100Bf16GemmBenchmarkBase


class BaselineH100Bf16GemmBenchmark(H100Bf16GemmBenchmarkBase):
    """Compute the full A @ B.T output with cuBLAS on the active stream."""

    nvtx_label = "fast_cu_baseline_h100_bf16_gemm"

    def setup(self) -> None:
        self._setup_tensors()
        if (
            self.extension is None
            or self.matrix_a is None
            or self.matrix_b is None
            or self._physical_output is None
        ):
            raise RuntimeError("H100 GEMM setup did not initialize required state")
        # Handle creation and its internal allocations are steady-state setup.
        self.extension.prepare_cublas_gemm(self.matrix_a, self.matrix_b, self._physical_output)

    def benchmark_fn(self) -> None:
        if (
            self.extension is None
            or self.matrix_a is None
            or self.matrix_b is None
            or self._physical_output is None
            or self._logical_output is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")
        with torch.inference_mode(), self._nvtx_range(self.nvtx_label):
            self.extension.run_cublas_gemm(self.matrix_a, self.matrix_b, self._physical_output)
            self.output = self._logical_output


def get_benchmark() -> BaseBenchmark:
    return BaselineH100Bf16GemmBenchmark()
