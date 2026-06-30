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
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

# Import CUDA extension
from ch12.cuda_extensions import load_kernel_fusion_extension


class OptimizedKernelFusionDedicatedStreamBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Fused kernel - single memory round trip (uses CUDA extension)."""
    
    def __init__(self):
        super().__init__()
        self.data = None
        self._verify_input: Optional[torch.Tensor] = None
        self.N = 16_000_000  # Larger size to be memory-bound
        self.iterations = 10
        self._extension = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N * self.iterations),
        )
        self._stream = None
        self._prefetch_touch: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._enable_nvtx = False
    
    def setup(self) -> None:
        """Setup: Initialize tensors, stream, and load CUDA extension.

        Uses a dedicated non-blocking stream for the fused kernel and
        performs a dry run plus a prefetch-style touch to put the tensor
        into a steady-state on Blackwell.
        """
        # Load CUDA extension (will compile on first call)
        self._extension = load_kernel_fusion_extension()
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False

        # Create a dedicated non-blocking stream for this benchmark if not created.
        if self._stream is None:
            self._stream = torch.cuda.Stream(device=self.device, priority=0)

        # Initialize deterministic input on the device.
        torch.manual_seed(42)
        self.data = torch.arange(self.N, dtype=torch.float32, device=self.device)
        self._prefetch_touch = torch.empty((), device=self.device, dtype=self.data.dtype)
        self._verify_input = self.data.detach().clone()

        # Perform a dry run on the dedicated stream to pay compilation and
        # any internal setup costs up front so later benchmark iterations
        # run from a stable first-launch state.
        with torch.cuda.stream(self._stream):
            self._extension.fused_kernel(self.data, 1)

        # Ensure the warmup is complete before we reset data.
        self._stream.synchronize()

        # Reset data so benchmark iterations always start from same values.
        torch.manual_seed(42)
        self.data = torch.arange(self.N, dtype=torch.float32, device=self.device)
        self._verify_output_buffer = torch.empty_like(self.data)

        # Prefetch-style touch on the same stream to fault in pages and
        # encourage the driver to keep this region resident.
        with torch.cuda.stream(self._stream):
            torch.sum(self.data, dim=0, out=self._prefetch_touch)

        self._stream.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Fused kernel (single memory round trip) on a dedicated stream.

        Running on a dedicated non-blocking stream improves overlap and
        avoids unintended synchronization with unrelated work on the
        default stream, which helps approach steady-state bandwidth on
        Blackwell.
        """
        stream = self._stream if self._stream is not None else torch.cuda.current_stream(device=self.device)

        with nvtx_range("kernel_fusion", enable=self._enable_nvtx):
            # Enqueue the fused kernel on our dedicated stream.
            with torch.cuda.stream(stream):
                self._extension.fused_kernel(self.data, self.iterations)

        torch.cuda.current_stream(device=self.device).wait_stream(stream)

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
        """Teardown: Clean up resources and destroy the dedicated stream."""
        self.data = None
        self._verify_input = None
        self._prefetch_touch = None
        self._verify_output_buffer = None
        # Explicitly delete the stream so that future benchmarks start from
        # a clean state and to avoid holding onto resources.
        self._stream = None
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
        """Return workload metrics for the fused LLM kernel path."""
        from ch12.graph_metrics_common import compute_ch12_workload_metrics

        metrics = compute_ch12_workload_metrics(
            uses_cuda_graph=False,
            num_iterations=self.iterations,
            workload_elements=float(self.N),
        )
        metrics["cuda_runtime.kernel_fusion_enabled"] = 1.0
        metrics["cuda_runtime.dedicated_stream"] = 1.0
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
    return OptimizedKernelFusionDedicatedStreamBenchmark()
