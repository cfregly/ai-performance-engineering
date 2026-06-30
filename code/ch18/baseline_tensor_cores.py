"""baseline_tensor_cores.py - Baseline without tensor core acceleration.

Demonstrates matrix operations without tensor core acceleration.
Tensor cores: This baseline uses standard FP32 operations without tensor cores.
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


class BaselineTensorCoresBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: FP32 matrix operations without tensor cores."""

    signature_equivalence_group = "ch18_tensor_cores_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.size = 4096
        self.output_buffer = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.size * self.size),
        )
        self.output = None
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.size * self.size),
        )
    
    def setup(self) -> None:
        """Setup: Initialize matrices in FP32."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        # Baseline: FP32 operations without tensor cores
        # Tensor cores accelerate FP16/BF16 matrix operations
        # This baseline uses FP32 which doesn't use tensor cores
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=torch.float32)
        self.output_buffer = torch.empty((self.size, self.size), device=self.device, dtype=torch.float32)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: FP32 matrix multiplication without tensor cores."""
        # Baseline: FP32 matmul without tensor cores
        # Tensor cores accelerate FP16/BF16 operations
        with torch.inference_mode(), self._nvtx_range("baseline_tensor_cores"):
            if self.output_buffer is None:
                raise RuntimeError("Output buffer not initialized")
            torch.matmul(self.A, self.B, out=self.output_buffer)
            self.output = self.output_buffer
        if self.output is None or self.A is None or self.B is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B},
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.5, 5.0),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            backend_policy="fp32_strict",
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_gemm_metrics
        return compute_gemm_metrics(
            self.size,
            self.size,
            self.size,
            elapsed_ms=getattr(self, "_last_elapsed_ms", None),
            precision="fp32",
            bytes_per_element=4,
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
    return BaselineTensorCoresBenchmark()
