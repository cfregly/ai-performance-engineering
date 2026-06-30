"""baseline_cublas.py - Naive FP32 matmul without TF32 tensor-core acceleration."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineCublasBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """
    Baseline: FP32 matmul with TF32 disabled.

    Demonstrates the cost of ignoring tensor-core friendly settings before we
    introduce a pure cuBLAS/TF32 path in the optimized example.
    """

    allow_cpu = True
    signature_equivalence_group = "ch02_cublas_tf32"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.m = 2048
        self.n = 2048
        self.k = 2048
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.C: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._last_elapsed_ms: Optional[float] = None
        tokens = self.m * self.n
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Allocate FP32 matrices, disable TF32, and warm cuBLAS identically."""
        if self.device.type != "cuda":
            raise RuntimeError("SKIPPED: cuBLAS benchmark requires CUDA")
        # Seed FIRST for deterministic verification
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        self.C = torch.empty(self.m, self.n, device=self.device, dtype=torch.float32)
        self._verify_output_buffer = torch.empty_like(self.C)

        # Mirror the optimized setup warmup so both paths hit cuBLAS/Lt
        # heuristics with the same pre-timed launch history.
        for _ in range(10):
            _ = torch.matmul(self.A, self.B)

    def benchmark_fn(self) -> None:
        """Plain cuBLAS FP32 matmul."""
        assert self.A is not None and self.B is not None and self.C is not None
        with torch.inference_mode(), self._nvtx_range("baseline_cublas_fp32"):
            torch.mm(self.A, self.B, out=self.C)

        if self.C is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if (
            self.A is None
            or self.B is None
            or self.C is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("benchmark_fn() must be called before verification")
        self._verify_output_buffer.copy_(self.C)
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B},
            output=self._verify_output_buffer,
            batch_size=self.A.shape[0],
            parameter_count=0,
            precision_flags={
                # Baseline runs pure FP32 math; signature stays consistent with optimized path for fair comparison.
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(1e-2, 1e-1),
        )

    def teardown(self) -> None:
        """Free benchmark tensors."""
        self.A = None
        self.B = None
        self.C = None
        self._verify_output_buffer = None
        self._last_elapsed_ms = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5, backend_policy="fp32_strict")

    def get_workload_metadata(self):
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return real GEMM workload metrics for the measured matmul."""
        from core.benchmark.metrics import compute_gemm_metrics
        return compute_gemm_metrics(
            self.m,
            self.n,
            self.k,
            elapsed_ms=self._last_elapsed_ms,
            precision="fp32",
            bytes_per_element=4,
        )

    def validate_result(self) -> Optional[str]:
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineCublasBenchmark()
