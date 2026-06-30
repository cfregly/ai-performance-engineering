"""Baseline matmul benchmark: Single-stage tcgen05 kernel.

CHAPTER 10 CONTEXT: This is the basic tcgen05 kernel without software
pipelining. Compare against the pipelined tcgen05 variant to see how
double-buffering and overlap help.

WHY THIS IS SLOWER:
1. Single-stage (no overlap between load/compute)
2. No double-buffered shared memory
3. Fewer scheduling optimizations
"""

from __future__ import annotations

from typing import Optional

import torch

from ch10.matmul_extension_tcgen05 import load_matmul_tcgen05_module
from core.common.device_utils import require_cuda_device
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.tcgen05_requirements import check_tcgen05_support


class BaselineMatmulTCGen05PipelinedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Single-stage tcgen05 baseline (no pipelining)."""

    def __init__(self) -> None:
        super().__init__()
        available, reason = check_tcgen05_support(
            loader=None,
            module_name="ch10 matmul tcgen05 kernels",
        )
        self._tcgen05_available = available
        self._skip_reason = reason or "SKIPPED: tcgen05 matmul unavailable"
        self.dtype = torch.float16
        # Increase workload to reduce CPU overhead in timing cross-validation.
        self.n = 12288
        self.size = self.n
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.module = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.register_workload_metadata(bytes_per_iteration=float(self.n * self.n * 2 * 3))

    def _resolve_device(self) -> torch.device:
        return require_cuda_device("CUDA required for ch10")

    def setup(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        if self.module is None:
            self.module = load_matmul_tcgen05_module()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)
        self._verify_output_buffer = torch.empty(
            min(128, self.size),
            min(256, self.size),
            device=self.device,
            dtype=torch.float32,
        )

    def benchmark_fn(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        assert self.A is not None and self.B is not None and self.module is not None
        with self._nvtx_range("baseline_matmul_tcgen05_single_stage"):
            with torch.inference_mode():
                self.output = self.module.matmul_tcgen05(self.A, self.B)
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
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Report the actual tcgen05 GEMM workload for the single-stage variant."""
        from core.benchmark.metrics import compute_gemm_metrics

        metrics = compute_gemm_metrics(
            m=self.size,
            n=self.size,
            k=self.size,
            precision="fp16",
            bytes_per_element=2,
        )
        metrics["gemm.uses_tcgen05"] = 1.0
        metrics["gemm.pipeline_stages"] = 1.0
        metrics["gemm.no_wait_pipeline"] = 0.0
        return metrics

    def validate_result(self) -> Optional[str]:
        if not self._tcgen05_available:
            return self._skip_reason
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaselineMatmulTCGen05PipelinedBenchmark:
    return BaselineMatmulTCGen05PipelinedBenchmark()
