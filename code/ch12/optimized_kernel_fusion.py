"""optimized_kernel_fusion.py - Fused elementwise kernel (optimized)."""

from __future__ import annotations

import torch


from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.verification import simple_signature
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

# Import CUDA extension
from ch12.cuda_extensions import load_kernel_fusion_extension


class OptimizedKernelFusionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Fused kernel - single memory round trip (uses CUDA extension)."""
    
    def __init__(self):
        super().__init__()
        self.data = None
        self._verify_input = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.N = 16_000_000  # Larger size to be memory-bound
        self.iterations = 10
        self._extension = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N * self.iterations),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""

        # Load CUDA extension (will compile on first call)
        self._extension = load_kernel_fusion_extension()
        
        torch.manual_seed(42)
        self.data = torch.arange(self.N, dtype=torch.float32, device=self.device)
        self._verify_input = self.data.detach().clone()
        torch.cuda.synchronize(self.device)
        # Dry run so first-launch kernel fusion overhead is prepaid
        self._extension.fused_kernel(self.data, 1)
        torch.cuda.synchronize()
        torch.manual_seed(42)
        self.data = torch.arange(self.N, dtype=torch.float32, device=self.device)
        self._verify_output_buffer = torch.empty_like(self.data)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Fused kernel (single memory round trip)."""
        with torch.inference_mode(), self._nvtx_range("kernel_fusion"):
            # Call CUDA extension with fused kernel
            self._extension.fused_kernel(self.data, self.iterations)
        if self.data is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.data is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        self._verify_output_buffer.copy_(self.data)
        self._set_verification_payload(
            inputs={"data": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self.N,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
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
            iterations=5,  # Fewer iterations since kernels run internally
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=120,  # CUDA extension compilation can take 60-90 seconds
            ncu_replay_mode="application",
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_input_signature(self) -> dict:
        return simple_signature(
            batch_size=self.N,
            dtype="float32",
            elements=self.N,
            iterations=self.iterations,
        ).to_dict()
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return workload metrics for the fused kernel path."""
        from ch12.graph_metrics_common import compute_ch12_workload_metrics

        metrics = compute_ch12_workload_metrics(
            uses_cuda_graph=False,
            num_iterations=self.iterations,
            workload_elements=float(self.N),
        )
        metrics["cuda_runtime.kernel_fusion_enabled"] = 1.0
        return metrics
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
    return OptimizedKernelFusionBenchmark()
