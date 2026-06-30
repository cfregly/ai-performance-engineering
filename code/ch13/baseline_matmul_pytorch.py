"""baseline_matmul_pytorch.py - Eager PyTorch matmul baseline.

Uses the same FP16 compute dtype as the optimized benchmark so the comparison
isolates compilation/fusion rather than a precision change.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch


from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

resolve_device = partial(require_cuda_device, "CUDA required for ch13")


def baseline_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Keep the baseline epilogue unfused while reusing the preallocated output."""
    torch.mm(A, B, out=out)
    out.add_(bias)
    torch.relu_(out)
    out.add_(residual)
    out.mul_(scale)
    return out


class BaselineMatmulPyTorchBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """PyTorch matmul baseline - sequential unfused operations."""

    signature_equivalence_group = "ch13_matmul_pytorch_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.C = None
        self._verify_output_buffer = None
        self.bias = None
        self.residual = None
        self.scale = 0.125
        # Use larger matrices to show more difference
        self.m = 4096
        self.n = 4096
        self.k = 4096
        tokens = self.m * self.n
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        # Register workload metadata at init time for compliance check
        bytes_per_iter = (self.m * self.k + self.k * self.n + self.m * self.n * 3) * 2
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
            bytes_per_iteration=float(bytes_per_iter),
        )
    
    def setup(self) -> None:
        """Setup: Initialize matrices."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        self.C = torch.empty(self.m, self.n, device=self.device, dtype=torch.float16)
        self._verify_output_buffer = torch.empty_like(self.C)
        self.bias = torch.randn(self.m, self.n, device=self.device, dtype=torch.float16)
        self.residual = torch.randn(self.m, self.n, device=self.device, dtype=torch.float16)
        
        # Warmup
        baseline_matmul(self.A, self.B, self.bias, self.residual, self.C, self.scale)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - PyTorch matmul."""
        assert self.A is not None and self.B is not None and self.bias is not None and self.residual is not None and self.C is not None
        with torch.inference_mode(), self._nvtx_range("baseline_matmul_pytorch"):
            # Standard PyTorch matrix multiplication
            self.C = baseline_matmul(
                self.A,
                self.B,
                self.bias,
                self.residual,
                self.C,
                self.scale,
            )

    def capture_verification_payload(self) -> None:
        if self.C is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.C)
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B, "bias": self.bias, "residual": self.residual},
            output=self._verify_output_buffer,
            batch_size=self.m,
            parameter_count=0,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        """Cleanup."""
        del self.A, self.B, self.C, self.bias, self.residual
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=180,  # torch.compile compilation can take 60-120 seconds
        )
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_last_elapsed_ms', None),
            reduced_precision_time_ms=None,
            precision_type="fp16",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.bias is None or self.residual is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineMatmulPyTorchBenchmark()
