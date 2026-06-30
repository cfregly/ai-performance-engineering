"""optimized_bank_conflicts.py - Padding solution to eliminate bank conflicts (optimized)."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch06.cuda_extensions import load_bank_conflicts_extension


class OptimizedBankConflictsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Padding solution - no bank conflicts (uses CUDA extension)."""
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self.N = 8_000_000
        self._extension = None
        self.repeats = 8
        self._repeat_range = range(self.repeats)
        # Bank conflicts benchmark - fixed input size to demonstrate shared memory patterns
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * self.repeats),
        )
    
    def setup(self) -> None:
        """Initialize tensors and load CUDA extension."""
        self._extension = load_bank_conflicts_extension()
        
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = None
        self._output_buffer = torch.empty_like(self.input)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: padding eliminates bank conflicts."""
        assert self._extension is not None and self.input is not None
        assert self._output_buffer is not None and self._output_buffer.shape == self.input.shape
        with torch.inference_mode(), self._nvtx_range("bank_conflicts_optimized"):
            for _ in self._repeat_range:
                self._extension.bank_conflicts_padded(self._output_buffer, self.input)
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
        """Clean up resources."""
        self.input = None
        self.output = None
        self._output_buffer = None
        torch.cuda.empty_cache()

    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=120,  # CUDA extension compilation can take time
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
        if self.input is None:
            return "Input tensor not initialized"
        if self.output.shape != self.input.shape:
            return f"Shape mismatch: input={self.input.shape}, output={self.output.shape}"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedBankConflictsBenchmark()
