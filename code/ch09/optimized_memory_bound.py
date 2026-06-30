"""optimized_memory_bound.py - Memory-bound kernel optimized via buffer reuse.

Optimization strategy:
- Keep math identical to the baseline (`t = t * 1.0001 + 0.0001`).
- Use in-place ops on a per-iteration working buffer to avoid allocating
  intermediate tensors on every repeat.
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
class OptimizedMemoryBoundBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Keeps data resident on the GPU and reduces allocation overhead."""
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.N = 16_777_216  # Same size as baseline (~64 MB)
        self.repeats = 64
        self._repeat_range = range(self.repeats)
        self.output: Optional[torch.Tensor] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._compiled_run = None
        # Memory-bound benchmark - fixed dimensions for roofline analysis
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * self.repeats),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.data = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.output_buffer = torch.empty_like(self.data)
        self._verify_output_buffer = torch.empty(4096, device=self.device, dtype=torch.float32)
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
        if not hasattr(torch, "compile"):
            raise RuntimeError("torch.compile is required for the fused memory-bound optimization.")

        def fused_kernel(inp: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
            out.copy_(inp)
            for _ in self._repeat_range:
                out.mul_(1.0001).add_(0.0001)
            return out

        self._compiled_run = torch.compile(fused_kernel, mode="reduce-overhead", fullgraph=True)
        # Warmup to trigger compilation outside the timed region.
        _ = self._compiled_run(self.data, self.output_buffer)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Fused operations (high AI)."""
        if self.data is None:
            raise RuntimeError("Data tensor not initialized")
        if self.output_buffer is None:
            raise RuntimeError("Output buffer not initialized")
        if self._compiled_run is None:
            raise RuntimeError("Compiled kernel not initialized")
        with torch.inference_mode(), self._nvtx_range("memory_bound"):
            self.output = self._compiled_run(self.data, self.output_buffer)
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self.data is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        # Keep verification lightweight: slice the large output tensor.
        output_slice = self.output[: self._verify_output_buffer.numel()].detach()
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"tensor": self.data},
            output=self._verify_output_buffer,
            batch_size=self.data.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-1, 1e-1),
        )

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        self.output = None
        self.output_buffer = None
        self._verify_output_buffer = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
            timing_method="wall_clock",
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=float(getattr(self, 'total_flops', getattr(self, 'N', 1024) * 2)),
            total_bytes=float(getattr(self, 'N', 1024) * 4 * 2),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            precision="fp32",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}, got {self.data.shape[0]}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedMemoryBoundBenchmark()
