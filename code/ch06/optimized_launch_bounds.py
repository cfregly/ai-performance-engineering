"""optimized_launch_bounds.py - Kernel with launch bounds annotation (optimized)."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch06.cuda_extensions import load_launch_bounds_extension


class OptimizedLaunchBoundsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Kernel with launch bounds annotation (optimized)."""
    
    def __init__(self):
        super().__init__()
        self.input_data: Optional[torch.Tensor] = None
        self.output_data: Optional[torch.Tensor] = None
        self.N = 1024 * 1024  # 1M elements
        # Match the baseline batching so the launch-bounds delta rises above
        # harness/Python overhead on single-GPU portable runs.
        self.kernel_launches_per_timed_call = 96
        self._extension = None
        # Launch bounds benchmark - fixed input size
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N * self.kernel_launches_per_timed_call),
        )
    
    def setup(self) -> None:
        """Initialize tensors and load CUDA extension."""
        self._extension = load_launch_bounds_extension()
        
        torch.manual_seed(42)
        self.input_data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        self.output_data = torch.empty(self.N, dtype=torch.float32, device=self.device)
        self._extension.launch_bounds_optimized(self.input_data, self.output_data, 8)
        self._synchronize()
        torch.manual_seed(42)
        self.input_data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        self.output_data = torch.empty(self.N, dtype=torch.float32, device=self.device)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: kernel with launch bounds."""
        assert self._extension is not None and self.input_data is not None and self.output_data is not None
        with torch.inference_mode(), self._nvtx_range("optimized_launch_bounds"):
            self._extension.launch_bounds_optimized(
                self.input_data,
                self.output_data,
                self.kernel_launches_per_timed_call,
            )

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self.input_data},
            output=self.output_data.detach(),
            batch_size=self.N,
            parameter_count=0,
            output_tolerance=(1e-4, 1e-4),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input_data = None
        self.output_data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,
            warmup=5,
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
            num_iterations=self.kernel_launches_per_timed_call,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input_data is None or self.output_data is None:
            return "Data tensors not initialized"
        if self.input_data.shape[0] != self.N or self.output_data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}"
        if not torch.isfinite(self.output_data).all():
            return "Output contains non-finite values"
        return None



def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedLaunchBoundsBenchmark()
