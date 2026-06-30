"""optimized_kernel_launches.py - CUDA Graphs optimization."""

from __future__ import annotations

import torch

from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class OptimizedKernelLaunchesBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark implementation following BaseBenchmark."""
    
    def __init__(self):
        super().__init__()
        self.x_input = None
        self.work_a = None
        self.work_b = None
        self.graph = None
        self.graph_output = None
        self.size = (1024, 1024)
        self.iterations = 1000
        tokens = self.size[0] * self.size[1]
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        # Kernel launch benchmark - fixed dimensions for consistent overhead measurement
    
    def setup(self) -> None:
        """Setup: initialize tensor and capture CUDA graph."""
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.x_input = torch.randn(*self.size, device=self.device, dtype=dtype)
        self.work_a = torch.empty_like(self.x_input)
        
        for _ in range(10):
            x_warmup = self.x_input
            for _ in range(self.iterations):
                x_warmup = x_warmup + 1.0
                x_warmup = x_warmup * 0.99
                x_warmup = torch.relu(x_warmup)
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph):
            if self.work_a is None or self.x_input is None:
                raise RuntimeError("CUDA graph buffers missing")
            torch.add(self.x_input, 1.0, out=self.work_a)
            torch.mul(self.work_a, 0.99, out=self.work_a)
            torch.clamp_min(self.work_a, 0.0, out=self.work_a)
            for _ in range(self.iterations - 1):
                torch.add(self.work_a, 1.0, out=self.work_a)
                torch.mul(self.work_a, 0.99, out=self.work_a)
                torch.clamp_min(self.work_a, 0.0, out=self.work_a)
        self.graph_output = self.work_a
        self._verify_input = self.x_input.detach().clone()
        self._verify_output_buffer = torch.empty_like(self.graph_output)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        with torch.inference_mode(), self._nvtx_range("kernel_launches"):
            self.graph.replay()
            self.output = self.graph_output
        if self._verify_input is None or self.graph_output is None:
            raise RuntimeError("Verification input or captured output missing")
        dtype = self._verify_input.dtype
        self._payload_dtype = dtype

    def capture_verification_payload(self) -> None:
        if self.graph_output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        dtype = self._payload_dtype
        self._verify_output_buffer.copy_(self.graph_output)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-4, 1e-4),
        )

    def teardown(self) -> None:
        """Cleanup."""
        del self.x_input, self.work_a, self.graph, self.graph_output
        self._verify_input = None
        self._verify_output_buffer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=30,
            warmup=5,
            ncu_replay_mode="application",
            nsys_timeout_seconds=1200,
            nsys_preset_override="light",
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return structural graph-capture metrics without invented launch-overhead numbers."""
        from ch12.graph_metrics_common import compute_ch12_workload_metrics

        return compute_ch12_workload_metrics(
            uses_cuda_graph=True,
            num_iterations=self.iterations,
            workload_elements=float(self.size[0] * self.size[1]),
            num_nodes=3 * self.iterations,
        )
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.x_input is None:
            return "Input tensor x_input not initialized"
        if self.graph is None:
            return "CUDA graph not initialized"
        if self.graph_output is None:
            return "Graph output not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedKernelLaunchesBenchmark()
