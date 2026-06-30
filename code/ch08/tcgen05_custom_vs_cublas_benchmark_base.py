"""Shared base for the tcgen05 custom-versus-cuBLAS comparison pair."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class Tcgen05CustomVsCublasBase(VerificationPayloadMixin, BaseBenchmark):
    """Common setup for the Chapter 8 tcgen05-versus-library bridge comparison."""

    nvtx_label = "tcgen05_custom_vs_cublas"
    matrix_rows = 8192
    matrix_cols = 8192
    shared_dim = 512
    tensor_dtype = torch.float16
    output_tolerance = (0.1, 1.0)

    def __init__(self) -> None:
        super().__init__()
        self.matrix_a: Optional[torch.Tensor] = None
        self.matrix_b: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._cublas_output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count = 0
        bytes_per_iter = (
            (self.matrix_rows * self.shared_dim)
            + (self.shared_dim * self.matrix_cols)
            + (self.matrix_rows * self.matrix_cols)
        ) * 2.0
        self.register_workload_metadata(bytes_per_iteration=float(bytes_per_iter))

    def _resolve_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA required for ch08 tcgen05 custom vs cuBLAS benchmark")

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.matrix_a = torch.randn(
            self.matrix_rows,
            self.shared_dim,
            device=self.device,
            dtype=self.tensor_dtype,
        ).contiguous()
        self.matrix_b = torch.randn(
            self.shared_dim,
            self.matrix_cols,
            device=self.device,
            dtype=self.tensor_dtype,
        ).contiguous()
        self._verify_output_buffer = self._new_output_buffer()
        self.output = None
        self.parameter_count = int(self.matrix_a.numel() + self.matrix_b.numel())
        self._synchronize()

    def _new_output_buffer(self) -> torch.Tensor:
        return torch.empty(
            self.matrix_rows,
            self.matrix_cols,
            device=self.device,
            dtype=self.tensor_dtype,
        )

    def _run_cublas_reference(self) -> torch.Tensor:
        if self.matrix_a is None or self.matrix_b is None or self._cublas_output_buffer is None:
            raise RuntimeError("Inputs not initialized")
        torch.mm(self.matrix_a, self.matrix_b, out=self._cublas_output_buffer)
        return self._cublas_output_buffer

    def capture_verification_payload(self) -> None:
        if (
            self.matrix_a is None
            or self.matrix_b is None
            or self.output is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"matrix_a": self.matrix_a, "matrix_b": self.matrix_b},
            output=self._verify_output_buffer,
            batch_size=self.matrix_rows,
            parameter_count=int(self.parameter_count),
            precision_flags={
                "fp16": self.tensor_dtype == torch.float16,
                "bf16": self.tensor_dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=self.output_tolerance,
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        return None

    def teardown(self) -> None:
        self.matrix_a = None
        self.matrix_b = None
        self.output = None
        self._cublas_output_buffer = None
        self._verify_output_buffer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=48,
            warmup=8,
            timing_method="wall_clock",
        )

    def get_custom_metrics(self) -> Optional[dict]:
        bytes_per_iter = (
            (self.matrix_rows * self.shared_dim)
            + (self.shared_dim * self.matrix_cols)
            + (self.matrix_rows * self.matrix_cols)
        ) * 2.0
        flops = 2.0 * self.matrix_rows * self.shared_dim * self.matrix_cols
        return {
            "tcgen05_custom_vs_cublas.matrix_m": float(self.matrix_rows),
            "tcgen05_custom_vs_cublas.matrix_k": float(self.shared_dim),
            "tcgen05_custom_vs_cublas.matrix_n": float(self.matrix_cols),
            "tcgen05_custom_vs_cublas.flops": float(flops),
            "tcgen05_custom_vs_cublas.bytes_transferred": float(bytes_per_iter),
            "story.comparison_pair": 1.0,
            "story.chapter_native_exemplar": 0.0,
            "story.bridge_to_ch09": 1.0,
        }

    def get_optimization_goal(self) -> str:
        """Keep the bridge-comparison pair runnable without headline speed gating."""
        return "comparison"
