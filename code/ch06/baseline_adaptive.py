"""baseline_adaptive.py - Baseline without adaptive optimization."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineAdaptiveBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: static configuration (no runtime adaptation)."""
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._chunk_views: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.N = 4_000_000
        self.static_chunk = 2048
        # Chunked processing benchmark - fixed input size
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
    
    def setup(self) -> None:
        """Setup: Initialize with static configuration."""
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = None
        self._output_buffer = torch.empty_like(self.input)
        self._chunk_views = [
            (self.input[start:end], self._output_buffer[start:end])
            for start in range(0, self.N, self.static_chunk)
            for end in (min(start + self.static_chunk, self.N),)
        ]
        self._synchronize()

    def _transform(self, tensor: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        """Shared math used by both baseline and optimized variants."""
        torch.mul(tensor, 1.75, out=out)
        out.add_(0.1)
        F.silu(out, inplace=True)
        return out
    
    def benchmark_fn(self) -> None:
        """Benchmark: static configuration operations."""
        assert self.input is not None
        assert self._output_buffer is not None and self._output_buffer.shape == self.input.shape
        if not self._chunk_views:
            raise RuntimeError("setup() must initialize chunk views")
        with torch.inference_mode(), self._nvtx_range("baseline_adaptive"):
            for window, out_window in self._chunk_views:
                self._transform(window, out_window)
            self.output = self._output_buffer

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output.detach(),
            batch_size=self.N,
            parameter_count=0,
            output_tolerance=(1e-4, 1e-4),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        self._output_buffer = None
        self._chunk_views = []
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_kernel_fundamentals_metrics
        metrics = compute_kernel_fundamentals_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            num_iterations=1,
        )
        metrics.update(
            {
                "adaptive.chunk_size": float(self.static_chunk),
                "adaptive.chunk_count": float((self.N + self.static_chunk - 1) // self.static_chunk),
            }
        )
        return metrics

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineAdaptiveBenchmark()
