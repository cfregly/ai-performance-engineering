"""optimized_graph_bandwidth.py - CUDA graphs for bandwidth measurement (optimized)."""

from __future__ import annotations

import torch


from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

# Import CUDA extension
from ch12.cuda_extensions import load_graph_bandwidth_extension


class OptimizedGraphBandwidthBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """CUDA graphs - measures bandwidth within graphs (uses CUDA extension)."""
    
    def __init__(self):
        super().__init__()
        self.src = None
        self.dst = None
        # Match baseline configuration for a fair comparison.
        self.N = 1 << 13
        self.iterations = 64_000
        self._extension = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N * self.iterations),
        )
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
    
    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""
        # Load CUDA extension (will compile on first call)
        # CUDA extensions may fail to compile or load on some hardware/driver combinations
        try:
            self._extension = load_graph_bandwidth_extension()
        except Exception as e:
            # If extension fails to load (compilation error, missing dependencies, etc.),
            # raise a clear error that will be caught by test harness and marked as hardware limitation
            raise RuntimeError(
                f"CUDA extension failed to load/compile: {e}. "
                f"This may indicate hardware/driver incompatibility or missing CUDA toolkit components."
            ) from e
        
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.src = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.dst = torch.empty_like(self.src)
        torch.cuda.synchronize(self.device)
        # Dry run so CUDA graph capture happens before timing.
        self._extension.graph_kernel(self.dst, self.src, self.iterations)
        torch.cuda.synchronize()
        self._verify_input = self.src.detach().clone()
        self._verify_output_buffer = torch.empty_like(self.dst)
    
    def benchmark_fn(self) -> None:
        """Benchmark: CUDA graph kernel."""
        with torch.inference_mode(), self._nvtx_range("optimized_graph_bandwidth_graph"):
            # Call CUDA extension with graph kernel
            self._extension.graph_kernel(self.dst, self.src, self.iterations)
        if self._verify_input is None or self.dst is None:
            raise RuntimeError("Verification input/output not initialized")

    def capture_verification_payload(self) -> None:
        if self.dst is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        self._verify_output_buffer.copy_(self.dst)
        self._set_verification_payload(
            inputs={"src": self._verify_input},
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
        self.src = None
        self.dst = None
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
            timing_method="wall_clock",
            full_device_sync=True,
            ncu_replay_mode="application",
            nsys_timeout_seconds=1200,
            nsys_preset_override="light",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return structural graph-bandwidth metrics without invented launch timings."""
        from ch12.graph_metrics_common import compute_ch12_workload_metrics

        return compute_ch12_workload_metrics(
            uses_cuda_graph=True,
            num_iterations=self.iterations,
            workload_elements=float(self.N),
            bytes_per_iteration=float(self.N * 4 * 2),
        )
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.dst is None:
            return "Destination tensor not initialized"
        if self.src is None:
            return "Source tensor not initialized"
        if self.dst.shape[0] != self.N:
            return f"Destination size mismatch: expected {self.N}, got {self.dst.shape[0]}"
        if not torch.isfinite(self.dst).all():
            return "Destination tensor contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedGraphBandwidthBenchmark()
