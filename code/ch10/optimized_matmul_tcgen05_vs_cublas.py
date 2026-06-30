"""Optimized side of the tcgen05-versus-cuBLAS matmul comparison.

CHAPTER 10 CONTEXT: This demonstrates that vendor libraries like cuBLAS
are highly optimized through years of engineering and represent the
production-ready path for GEMM operations.

Compare against: baseline_matmul_tcgen05_vs_cublas.py (custom tcgen05 kernel)
This comparison shows the performance gap between educational custom kernels
and production-ready vendor libraries.

LESSON: Custom kernels are valuable for learning tensor core programming
and for specialized optimization opportunities, but cuBLAS/cuDNN should
be the default choice for production workloads.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.common.device_utils import require_cuda_device
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.tcgen05_requirements import check_tcgen05_support


class OptimizedMatmulTCGen05Benchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: PyTorch/cuBLAS matmul (production-ready)."""

    def __init__(self) -> None:
        super().__init__()
        available, reason = check_tcgen05_support(
            loader=None,
            module_name="ch10 matmul tcgen05 kernels",
        )
        self._tcgen05_available = available
        self._skip_reason = reason or "SKIPPED: tcgen05 matmul unavailable"
        self.device = require_cuda_device("CUDA required for ch10")
        self.dtype = torch.float16
        # Match the educational tcgen05 baseline dimensions.
        self.n = 8192
        self.size = self.n  # For compatibility
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self._B_t: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.register_workload_metadata(bytes_per_iteration=float(self.n * self.n * 2 * 3))

    def setup(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)
        self._B_t = self.B.transpose(0, 1)
        self.output = torch.empty(self.size, self.size, device=self.device, dtype=self.dtype)
        self._verify_output_buffer = torch.empty(
            min(128, self.size),
            min(256, self.size),
            device=self.device,
            dtype=torch.float32,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        assert self.A is not None and self._B_t is not None and self.output is not None
        with self._nvtx_range("optimized_matmul_tcgen05_vs_cublas_cublas"):
            with torch.inference_mode():
                # Match baseline math: baseline kernel treats B as (N, K) and computes A @ B^T
                torch.mm(self.A, self._B_t, out=self.output)
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.A is None or self.B is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B},
            output=self._verify_output_buffer,
            batch_size=self.size,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(5e-2, 5e-2),
        )

    def teardown(self) -> None:
        self.A = None
        self.B = None
        self._B_t = None
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Report the actual tcgen05 comparison workload."""
        from core.benchmark.metrics import compute_gemm_metrics

        return compute_gemm_metrics(
            m=self.size,
            n=self.size,
            k=self.size,
            precision="fp16",
            bytes_per_element=2,
        )

    def validate_result(self) -> Optional[str]:
        if not self._tcgen05_available:
            return self._skip_reason
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> OptimizedMatmulTCGen05Benchmark:
    return OptimizedMatmulTCGen05Benchmark()
