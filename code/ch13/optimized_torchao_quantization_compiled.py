"""Informational compound variant: torchao INT8 plus torch.compile.

This target is intentionally noncanonical. It combines quantization and
compilation to demonstrate the stacked effect without claiming the speedup as a
quantization-only comparison.
"""

from __future__ import annotations

from typing import Optional

import torch

from ch13.optimized_torchao_quantization import OptimizedTorchAOQuantizationBenchmark
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedTorchAOQuantizationCompiledBenchmark(OptimizedTorchAOQuantizationBenchmark):
    """Compound informational variant: quantized model plus compiled execution."""

    def __init__(self) -> None:
        super().__init__()
        self.compiled_model = None

    def setup(self) -> None:
        super().setup()
        if self.model is None or self.data is None:
            raise RuntimeError("Model/data not initialized")
        self.compiled_model = torch.compile(self.model, mode="max-autotune")
        for _ in range(3):
            with torch.inference_mode():
                _ = self.compiled_model(self.data)

    def benchmark_fn(self) -> None:
        if self.compiled_model is None or self.data is None:
            raise RuntimeError("Compiled model/data not initialized")
        with self._nvtx_range("optimized_torchao_quantization_compiled"):
            with torch.inference_mode():
                self.output = self.compiled_model(self.data)
        if self._verify_input is None or self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def teardown(self) -> None:
        self.compiled_model = None
        super().teardown()

    def validate_result(self) -> Optional[str]:
        if self.compiled_model is None:
            return "Compiled model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedTorchAOQuantizationCompiledBenchmark()
