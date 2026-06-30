"""optimized_elementwise_ilp.py - Independent operations for high ILP."""

from __future__ import annotations

from typing import Optional

import torch

from ch06.cuda_extensions import load_ilp_extension
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedElementwiseILPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Independent operations and unrolling for high ILP.
    
    Does the same amount of work as baseline (repeat count) for fair comparison.
    The ILP benefit is measured per-operation - both baseline and optimized do
    the same number of kernel launches, but the optimized kernel has better ILP.
    """
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self._buf0: Optional[torch.Tensor] = None
        self._buf1: Optional[torch.Tensor] = None
        self._output_view0: Optional[torch.Tensor] = None
        self._output_view1: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.N = 50_000_000
        self._extension = None
        self.repeats = 10  # Same as baseline for fair comparison
        self._repeat_range = range(self.repeats)
        # ILP benchmark - fixed input size to measure instruction-level parallelism
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * self.repeats),
        )
    
    def setup(self) -> None:
        """Initialize tensors and load CUDA extension."""
        self._extension = load_ilp_extension()
        
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Keep magnitudes small so the dependent square chain remains finite.
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32) * 0.1
        self._buf0 = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self._buf1 = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self._output_view0 = self._buf0[:1024]
        self._output_view1 = self._buf1[:1024]
        self._verify_output_buffer = torch.empty_like(self._output_view0)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: independent operations (high ILP).
        
        Same iteration count as baseline for fair comparison and output verification.
        """
        assert self._extension is not None and self.input is not None and self._buf0 is not None and self._buf1 is not None
        assert self._output_view0 is not None and self._output_view1 is not None
        with torch.inference_mode(), self._nvtx_range("elementwise_ilp_optimized"):
            src: torch.Tensor = self.input
            buf0: torch.Tensor = self._buf0
            buf1: torch.Tensor = self._buf1
            dst: torch.Tensor = buf0
            for _ in self._repeat_range:
                self._extension.unrolled_ilp(dst, src)
                src, dst = dst, (buf1 if dst is buf0 else buf0)

        self.output = self._output_view0 if src is buf0 else self._output_view1
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self._verify_output_buffer,
            batch_size=self.N,
            parameter_count=0,
            output_tolerance=(1e-4, 1e-4),
        )
    
    def teardown(self) -> None:
        """Clean up resources."""
        self.input = None
        self._buf0 = None
        self._buf1 = None
        self._output_view0 = None
        self._output_view1 = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
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
        if self.output.shape[0] != 1024:
            return f"Output shape mismatch: expected 1024, got {self.output.shape[0]}"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedElementwiseILPBenchmark()
