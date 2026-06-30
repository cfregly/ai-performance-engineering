"""baseline_add.py - Sequential addition benchmark (baseline)."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineAddBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Many tiny kernel launches to illustrate overhead."""
    
    def __init__(self):
        super().__init__()
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.C: Optional[torch.Tensor] = None
        self.N = 10_000  # Small enough to complete, but still demonstrates the problem
        self._index_range = range(self.N)
        # Kernel launch overhead benchmark - fixed input size
        tokens = self.N
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors (excluded from timing)."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.A = torch.arange(self.N, dtype=torch.float32, device=self.device)
        self.B = 2 * self.A
        self.C = torch.empty_like(self.A)
        self._index_range = range(self.N)
    
    def benchmark_fn(self) -> None:
        """Sequential loop launches N tiny kernels."""
        assert self.A is not None and self.B is not None and self.C is not None
        with torch.inference_mode(), self._nvtx_range("baseline_add_sequential"):
            for i in self._index_range:
                self.C[i] = self.A[i] + self.B[i]

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B},
            output=self.C.detach(),
            batch_size=self.N,
            parameter_count=0,
            output_tolerance=(1e-5, 1e-5),
        )
    
    def teardown(self) -> None:
        """Cleanup (excluded from timing)."""
        self.A = None
        self.B = None
        self.C = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_kernel_fundamentals_metrics
        return compute_kernel_fundamentals_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            num_iterations=1,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.C is None:
            return "Result tensor C not initialized"
        if self.A is None:
            return "Input tensor A not initialized"
        if self.B is None:
            return "Input tensor B not initialized"
        if self.C.shape != self.A.shape or self.C.shape != self.B.shape:
            return f"Shape mismatch: A={self.A.shape}, B={self.B.shape}, C={self.C.shape}"
        if not torch.isfinite(self.C).all():
            return "Result tensor C contains non-finite values"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineAddBenchmark()
