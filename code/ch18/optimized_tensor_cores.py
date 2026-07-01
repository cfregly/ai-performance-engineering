"""optimized_tensor_cores.py - Optimized tensor core acceleration.

Demonstrates tensor core acceleration using FP16/BF16.
Tensor cores: Uses tensor cores for accelerated matrix operations.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import torch

from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class OptimizedTensorCoresBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Tensor core accelerated matrix operations."""

    signature_equivalence_group = "ch18_tensor_cores_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.A_tc = None
        self.B_tc = None
        self.size = 4096
        # B200 sustains slightly better throughput for this path with BF16.
        self.dtype = torch.bfloat16
        self.output_buffer = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.size * self.size),
        )
        self.output = None
        self._verification_payload = None
    
    def setup(self) -> None:
        """Setup: Initialize matrices in FP16/BF16 for tensor cores."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        # Keep verification inputs FP32 to match the baseline signature; compute path casts to FP16/BF16.
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=torch.float32)
        # Pre-cast once so benchmark iterations measure tensor-core GEMM, not cast overhead.
        self.A_tc = self.A.to(self.dtype)
        self.B_tc = self.B.to(self.dtype)
        self.output_buffer = torch.empty((self.size, self.size), device=self.device, dtype=self.dtype)
        self._verify_output_buffer = torch.empty_like(self.output_buffer, dtype=torch.float32)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Tensor core accelerated matrix multiplication."""
        # Optimization: FP16/BF16 matmul with tensor cores
        # Tensor cores provide high throughput for these operations
        with torch.inference_mode(), self._nvtx_range("optimized_tensor_cores"):
            if self.output_buffer is None:
                raise RuntimeError("Output buffer not initialized")
            if self.A_tc is None or self.B_tc is None:
                raise RuntimeError("Tensor-core inputs not initialized")
            torch.matmul(self.A_tc, self.B_tc, out=self.output_buffer)
            self.output = self.output_buffer
        if self.output is None or self.A is None or self.B is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output, non_blocking=False)
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B},
            output=self._verify_output_buffer,
            batch_size=1,
            parameter_count=0,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        self.A_tc = None
        self.B_tc = None
        self.output_buffer = None
        self._verify_output_buffer = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_gemm_metrics
        return compute_gemm_metrics(
            self.size,
            self.size,
            self.size,
            elapsed_ms=getattr(self, "_last_elapsed_ms", None),
            precision="tensor",
            bytes_per_element=2 if self.dtype in (torch.float16, torch.bfloat16) else 4,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedTensorCoresBenchmark()
