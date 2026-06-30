"""optimized_cuda_graphs.py - CUDA graph replay (optimized)."""

from __future__ import annotations

import torch

from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

from ch12.cuda_extensions import load_cuda_graphs_extension


class OptimizedCudaGraphsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """CUDA graph replay - reduced launch overhead through graph optimization (uses CUDA extension)."""
    
    def __init__(self):
        super().__init__()
        self.data = None
        self.N = 1 << 10  # Smaller buffers to make launch overhead dominant
        self.iterations = 32000  # More iterations to emphasize launch overhead
        self._extension = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N * self.iterations),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""

        self._extension = load_cuda_graphs_extension()
        
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        # Warm up graph replay so capture/instantiation happens before measurement.
        self._extension.graph_replay(self.data, self.iterations)
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.data = torch.linspace(0.0, 1.0, self.N, dtype=torch.float32, device=self.device)
        self._verify_input = self.data.detach().clone()
        self._verify_output_buffer = torch.empty_like(self.data)
    
    def benchmark_fn(self) -> None:
        """Benchmark: CUDA graph replay."""
        with torch.inference_mode(), self._nvtx_range("cuda_graphs"):
            self._extension.graph_replay(self.data, self.iterations)
        if self.data is None or self._verify_input is None:
            raise RuntimeError("Data or verification input not initialized")

    def capture_verification_payload(self) -> None:
        if self.data is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        self._verify_output_buffer.copy_(self.data)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        self._verify_input = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=120,  # CUDA extension compilation can take time
            measurement_timeout_seconds=120,
            ncu_replay_mode="application",
            nsys_timeout_seconds=1200,
            nsys_preset_override="light",
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return structural graph replay metrics without invented launch timings."""
        from ch12.graph_metrics_common import compute_ch12_workload_metrics

        return compute_ch12_workload_metrics(
            uses_cuda_graph=True,
            num_iterations=self.iterations,
            workload_elements=float(self.N),
        )
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedCudaGraphsBenchmark()
