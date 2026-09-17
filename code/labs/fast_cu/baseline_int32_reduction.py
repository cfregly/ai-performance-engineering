"""Baseline: CUB int32 reduction for the fast.cu comparison."""

from __future__ import annotations

import torch

from core.harness.benchmark_harness import BaseBenchmark
from labs.fast_cu.h100_common import Int32ReductionBenchmarkBase


class BaselineInt32ReductionBenchmark(Int32ReductionBenchmarkBase):
    """Reduce the complete int32 input with CUB on the active stream."""

    nvtx_label = "fast_cu_baseline_int32_reduction"

    def __init__(self) -> None:
        super().__init__()
        self._temp_storage: torch.Tensor | None = None

    def setup(self) -> None:
        self._setup_tensors()
        if self.extension is None or self.input is None or self._output_buffer is None:
            raise RuntimeError("int32 reduction setup did not initialize required state")
        storage_bytes = self.extension.cub_temp_storage_bytes(self.input, self._output_buffer)
        if storage_bytes <= 0:
            raise RuntimeError(f"CUB returned invalid temporary storage size: {storage_bytes}")
        self._temp_storage = torch.empty((storage_bytes,), device=self.device, dtype=torch.uint8)

    def benchmark_fn(self) -> None:
        if (
            self.extension is None
            or self.input is None
            or self._output_buffer is None
            or self._temp_storage is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")
        with torch.inference_mode(), self._nvtx_range(self.nvtx_label):
            self.extension.run_cub_reduction(self.input, self._output_buffer, self._temp_storage)
            self.output = self._output_buffer

    def teardown(self) -> None:
        self._temp_storage = None
        super().teardown()


def get_benchmark() -> BaseBenchmark:
    return BaselineInt32ReductionBenchmark()
