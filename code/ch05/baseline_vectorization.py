"""baseline_vectorization.py - Baseline without vectorization in storage I/O context."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineVectorizationBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Scalar operations to contrast with vectorized paths."""
    
    def __init__(self):
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._chunk_sum_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.N = 1_000_000
        self.chunk = 4096
        self._chunk_views: list[torch.Tensor] = []
        # Computation benchmark - jitter check not applicable
        tokens = self.N
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize data."""
        torch.manual_seed(42)
        self.data = torch.randn(self.N, device=self.device)
        self._output_buffer = torch.empty(1, device=self.device)
        self._chunk_sum_buffer = torch.empty_like(self._output_buffer)
        self._verify_output_buffer = torch.empty_like(self._output_buffer)
        self._chunk_views = list(self.data.split(self.chunk))
    
    def benchmark_fn(self) -> None:
        """Benchmark: Chunked reductions to simulate scalar-style overhead."""
        assert self.data is not None
        with torch.inference_mode(), self._nvtx_range("baseline_vectorization"):
            result = self._output_buffer
            chunk_sum = self._chunk_sum_buffer
            if result is None or chunk_sum is None:
                raise RuntimeError("Output buffer not initialized")
            result.zero_()
            for chunk_view in self._chunk_views:
                torch.sum(chunk_view, dim=0, keepdim=True, out=chunk_sum)
                result.add_(chunk_sum)
        self.output = result

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"data": self.data},
            output=self._verify_output_buffer,
            batch_size=self.data.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        self._output_buffer = None
        self._chunk_sum_buffer = None
        self._verify_output_buffer = None
        self._chunk_views = []
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Report the scalar chunking strategy instead of fake storage metrics."""
        from ch05.metrics_common import compute_vectorization_metrics

        return compute_vectorization_metrics(
            num_elements=self.N,
            chunk_elements=self.chunk,
            is_vectorized=False,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineVectorizationBenchmark()
