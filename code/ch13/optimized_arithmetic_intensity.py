"""optimized_high_ai.py - High arithmetic intensity optimization (optimized).

Compute-bound kernel with high arithmetic intensity.
Many compute operations relative to memory operations.
Optimized for maximum FLOPs per byte accessed.

Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

from pathlib import Path

import torch


from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class OptimizedArithmeticIntensityBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """High arithmetic intensity optimization - compute-bound."""
    
    def __init__(self):
        super().__init__()
        self.A: torch.Tensor | None = None
        self.B: torch.Tensor | None = None
        self.C: torch.Tensor | None = None
        self._verify_output_buffer: torch.Tensor | None = None
        self.M = 2048
        self.K = 2048
        self.N = 2048
        tokens = self.M * self.N
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors for compute-bound operation."""

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Smaller tensors but with high compute intensity
        self.A = torch.randn(self.M, self.K, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.K, self.N, device=self.device, dtype=torch.float32)
        self.C = torch.empty(self.M, self.N, device=self.device, dtype=torch.float32)
        self._verify_output_buffer = torch.empty_like(self.C)

        # Warmup high-AI matmul so autotuning occurs before measurement.
        self._fast_matmul()
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def _fast_matmul(self) -> None:
        assert self.A is not None and self.B is not None and self.C is not None
        # Use out= parameter to avoid allocation overhead on each iteration
        torch.matmul(self.A, self.B, out=self.C)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - high arithmetic intensity (compute-bound)."""
        with torch.inference_mode(), self._nvtx_range("optimized_arithmetic_intensity"):
            if self.A is None or self.B is None or self.C is None:
                raise RuntimeError("Benchmark not initialized")
            # High arithmetic intensity: full fused matmul, single launch.
            self._fast_matmul()
        if self.A is None or self.B is None or self.C is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.C is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.C)
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B},
            output=self._verify_output_buffer,
            batch_size=self.M,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        """Cleanup."""
        self.A = None
        self.B = None
        self.C = None
        self._verify_output_buffer = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=25,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=None,
            reduced_precision_time_ms=getattr(self, '_last_elapsed_ms', None),
            precision_type="fp32",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.C is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedArithmeticIntensityBenchmark()
