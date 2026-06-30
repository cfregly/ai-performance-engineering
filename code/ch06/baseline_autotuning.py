"""baseline_autotuning.py - Baseline without autotuning."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineAutotuningBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: fixed parameters without autotuning."""
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._chunk_views: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.N = 4_000_000
        self.block_size = 2048  # Fixed micro-chunk
        # Autotuning benchmark - fixed input size
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = None
        self._output_buffer = torch.empty_like(self.input)
        self._chunk_views = [
            (self.input[start:end], self._output_buffer[start:end])
            for start in range(0, self.N, self.block_size)
            for end in (min(start + self.block_size, self.N),)
        ]
        self._synchronize()

    def _transform(self, tensor: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        torch.mul(tensor, 1.75, out=out)
        out.add_(0.1)
        F.silu(out, inplace=True)
        return out
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with fixed parameters."""
        assert self.input is not None
        assert self._output_buffer is not None and self._output_buffer.shape == self.input.shape
        if not self._chunk_views:
            raise RuntimeError("setup() must initialize chunk views")
        with torch.inference_mode(), self._nvtx_range("baseline_autotuning"):
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
        return compute_kernel_fundamentals_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            num_iterations=1,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineAutotuningBenchmark()
