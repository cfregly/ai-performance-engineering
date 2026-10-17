"""Python harness wrapper for optimized_fp4_hardware_kernel.cu (Chapter 19).

This benchmark participates in the harness as a *comparable* optimized variant
for the FP4 hardware-kernel example. It shares the full-output verification
wrapper defined in baseline_fp4_hardware_kernel.py.

Pairs with: baseline_fp4_hardware_kernel.py
"""

from __future__ import annotations

from ch19.baseline_fp4_hardware_kernel import _FP4HardwareKernelBinaryBenchmark
from core.harness.benchmark_harness import BaseBenchmark


class OptimizedFP4HardwareKernelBenchmark(_FP4HardwareKernelBinaryBenchmark):
    """Runs optimized_fp4_hardware_kernel.cu (cuBLASLt NVFP4 tensor cores)."""

    def __init__(self) -> None:
        super().__init__(
            binary_name="optimized_fp4_hardware_kernel",
            friendly_name="Optimized Fp4 Hardware Kernel",
            variant="optimized_cublaslt_nvfp4",
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedFP4HardwareKernelBenchmark()
