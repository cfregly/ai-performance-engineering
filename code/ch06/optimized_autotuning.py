"""optimized_autotuning.py - Optimized with autotuning."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# Import arch_config to apply Triton patch for sm_12x support
try:
    import arch_config  # noqa: F401
except ImportError:
    pass  # Continue if arch_config not available

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedAutotuningBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: uses autotuning to find optimal parameters."""
    
    def __init__(self):
        super().__init__()
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self.N = 4_000_000
        self.candidates = [1024, 2048, 4096, 8192]
        self.optimal_chunk: Optional[int] = None
        self._chunk_views: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.timer_results: List[Tuple[int, float]] = []
        # Autotuning benchmark - fixed input size
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors and perform autotuning."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = None
        self._output_buffer = torch.empty_like(self.input)
        scratch = torch.empty_like(self.input)
        self.optimal_chunk = self._autotune_chunk_size(scratch)
        self._chunk_views = self._build_chunk_views(self.optimal_chunk)
        self._synchronize()

    def _transform(self, tensor: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        torch.mul(tensor, 1.75, out=out)
        out.add_(0.1)
        F.silu(out, inplace=True)
        return out

    def _build_chunk_views(self, chunk: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
        assert self.input is not None and self._output_buffer is not None
        views: list[tuple[torch.Tensor, torch.Tensor]] = []
        for offset in range(0, self.N, chunk):
            span = min(chunk, self.N - offset)
            end = offset + span
            views.append((self.input[offset:end], self._output_buffer[offset:end]))
        return views

    def _autotune_chunk_size(self, scratch: torch.Tensor) -> int:
        """Benchmark several staging chunk sizes using baseline timers."""
        best = None
        best_time = float("inf")
        for chunk in self.candidates:
            total_ms = 0.0
            trials = 3
            for _ in range(trials):
                start = self._record_start()
                for offset in range(0, self.N, chunk):
                    span = min(chunk, self.N - offset)
                    window = self.input[offset : offset + span]
                    self._transform(window, scratch[offset : offset + span])
                self._synchronize()
                total_ms += self._record_stop(start)
            avg_ms = total_ms / trials
            self.timer_results.append((chunk, avg_ms))
            if avg_ms < best_time:
                best_time = avg_ms
                best = chunk
        assert best is not None
        return best
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with autotuned parameters."""
        assert self.input is not None and self.optimal_chunk is not None
        assert self._output_buffer is not None and self._output_buffer.shape == self.input.shape
        if not self._chunk_views:
            raise RuntimeError("setup() must initialize chunk views")
        with torch.inference_mode(), self._nvtx_range("optimized_autotuning"):
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
    return OptimizedAutotuningBenchmark()
