"""Optimized: pranjalssh/fast.cu's latest H100 BF16 WGMMA kernel."""

from __future__ import annotations

import torch

from core.harness.benchmark_harness import BaseBenchmark
from labs.fast_cu.h100_common import H100Bf16GemmBenchmarkBase


class OptimizedH100Bf16GemmBenchmark(H100Bf16GemmBenchmarkBase):
    """Run matmul_12 with the repository benchmark harness."""

    nvtx_label = "fast_cu_optimized_h100_bf16_gemm"

    def setup(self) -> None:
        self._setup_tensors()
        if (
            self.extension is None
            or self.matrix_a is None
            or self.matrix_b is None
            or self._physical_output is None
        ):
            raise RuntimeError("H100 GEMM setup did not initialize required state")
        # TMA descriptors, the Hilbert schedule, device scratch, and the dynamic
        # shared-memory attribute are prepared before the steady-state timer.
        self.extension.prepare_upstream_gemm(self.matrix_a, self.matrix_b, self._physical_output)

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
            self.extension.run_upstream_gemm(self.matrix_a, self.matrix_b, self._physical_output)
            self.output = self._logical_output


def get_benchmark() -> BaseBenchmark:
    return OptimizedH100Bf16GemmBenchmark()
