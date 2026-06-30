"""Optimized matmul benchmark: Pipelined tcgen05 kernel variant.

CHAPTER 10 CONTEXT: Uses an advanced pipelined tcgen05 kernel from the
custom_vs_cublas lab to overlap compute and memory via double-buffering.

Key optimizations over basic tcgen05:
1. Double-buffered shared memory for async prefetch
2. Overlapped TMA loads with MMA compute
3. Reduced barrier waits (no-wait pipeline pattern)

Compare against:
- optimized_matmul_tcgen05_vs_cublas.py (cuBLAS) - The library reference path
- baseline_matmul_tcgen05_vs_cublas.py (custom tcgen05) - Single-stage custom kernel

EDUCATIONAL VALUE: Shows the progression from basic tensor core kernel
to a pipelined implementation with overlap.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.common.device_utils import require_cuda_device
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.tcgen05_requirements import check_tcgen05_support
from labs.custom_vs_cublas.tcgen05_loader import load_tcgen05_no_wait_module


class OptimizedMatmulTCGen05PipelinedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Pipelined tcgen05 kernel with double-buffering.
    """

    def __init__(self) -> None:
        super().__init__()
        available, reason = check_tcgen05_support(
            loader=None,
            module_name="ch10 matmul tcgen05 kernels",
        )
        self._tcgen05_available = available
        self._skip_reason = reason or "SKIPPED: tcgen05 matmul unavailable"
        # Match baseline for fair comparison (larger size reduces CPU overhead noise).
        self.n = 12288
        self.size = self.n
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self._placeholder_kernel = False
        self._warned_placeholder = False
        self.output: Optional[torch.Tensor] = None
        self._module = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.register_workload_metadata(bytes_per_iteration=float(self.n * self.n * 2 * 3))

    def _resolve_device(self) -> torch.device:
        return require_cuda_device("CUDA required for ch10")

    def setup(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        dtype = torch.float16
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=dtype)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=dtype)
        self._module = load_tcgen05_no_wait_module()
        self._verify_output_buffer = torch.empty(
            min(128, self.size),
            min(256, self.size),
            device=self.device,
            dtype=torch.float32,
        )

    def benchmark_fn(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        assert self.A is not None and self.B is not None
        assert self._module is not None
        with self._nvtx_range("optimized_matmul_tcgen05_pipelined"):
            with torch.inference_mode():
                self.output = self._module.matmul_tcgen05_no_wait(self.A, self.B)
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
        self._module = None
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Report the actual tcgen05 GEMM workload for the no-wait pipeline."""
        from core.benchmark.metrics import compute_gemm_metrics

        metrics = compute_gemm_metrics(
            m=self.size,
            n=self.size,
            k=self.size,
            precision="fp16",
            bytes_per_element=2,
        )
        metrics["gemm.uses_tcgen05"] = 1.0
        metrics["gemm.pipeline_stages"] = 2.0
        metrics["gemm.no_wait_pipeline"] = 1.0
        return metrics

    def validate_result(self) -> Optional[str]:
        if not self._tcgen05_available:
            return self._skip_reason
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> OptimizedMatmulTCGen05PipelinedBenchmark:
    return OptimizedMatmulTCGen05PipelinedBenchmark()
