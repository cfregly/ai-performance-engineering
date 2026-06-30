"""baseline_kernel_launches.py - Many small kernel launches (baseline).

Demonstrates performance issue: many sequential kernel launches with overhead.
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import torch

from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


def many_small_ops_regular(x: torch.Tensor, iterations: int = 100) -> torch.Tensor:
    """Run many small operations WITHOUT CUDA graphs (many kernel launches).
    
    Each operation launches a separate kernel:
    - x + 1.0 → kernel launch 1
    - x * 0.99 → kernel launch 2
    - torch.relu(x) → kernel launch 3
    
    Total: 3 * iterations kernel launches = high overhead!
    """
    for _ in range(iterations):
        x = x + 1.0
        x = x * 0.99
        x = torch.relu(x)
    return x


class BaselineKernelLaunchesBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark implementation following BaseBenchmark."""
    
    def __init__(self):
        super().__init__()
        self.x = None
        self.output = None
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
        """Setup: initialize tensor."""
        # Use bfloat16 for GPU performance
        dtype = torch.bfloat16 if self.device.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
        self.x = torch.randn(*self.size, device=self.device, dtype=dtype)
        self._verify_input = self.x.detach()
        self._verify_output_buffer = torch.empty_like(self.x)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        with torch.inference_mode(), self._nvtx_range("kernel_launches"):
            self.output = many_small_ops_regular(self.x, self.iterations)
        if self._verify_input is None:
            raise RuntimeError("Verification input not initialized")
        dtype = self._verify_input.dtype
        self._payload_dtype = dtype

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_output_buffer.copy_(self.output)
        dtype = self._payload_dtype
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
        self.x = None
        self.output = None
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
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return structural launch metrics without invented launch-overhead numbers."""
        from ch12.graph_metrics_common import compute_ch12_workload_metrics

        return compute_ch12_workload_metrics(
            uses_cuda_graph=False,
            num_iterations=self.iterations,
            workload_elements=float(self.size[0] * self.size[1]),
            num_nodes=3 * self.iterations,
        )
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.x is None:
            return "Input tensor not initialized"
        if self.output is None:
            return "Output tensor not initialized"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return BaselineKernelLaunchesBenchmark()
