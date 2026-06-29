"""baseline_precisionfp8_pad_inner_matmul.py - FP32 matmul baseline with non-multiple-of-16 K."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class BaselinePrecisionFP8PadInnerMatmulBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """FP32 matmul baseline for pad_inner_dim matmul comparison."""

    signature_equivalence_group = "ch13_precisionfp8_pad_inner_matmul"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.a: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self.m = 8192
        self.k = 8200
        self.n = 8192
        self.input_scale = 0.25
        tokens = self.m * self.k
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self.a = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32) * self.input_scale
        self.b = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32) * self.input_scale
        self._verify_output_buffer = torch.empty(
            min(128, self.m),
            min(256, self.n),
            device=self.device,
            dtype=torch.float32,
        )
        self.parameter_count = self.k * self.n
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def benchmark_fn(self) -> None:
        if self.a is None or self.b is None:
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("baseline_precisionfp8_pad_inner_matmul"):
            with torch.inference_mode():
                out = torch.matmul(self.a, self.b)
                self.output = out
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.a is None or self.b is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("Benchmark not configured")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"a": self.a, "b": self.b},
            output=self._verify_output_buffer,
            batch_size=self.a.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.25, 2.0),
        )

    def teardown(self) -> None:
        del self.a, self.b
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            backend_policy="fp32_strict",
        )


def get_benchmark() -> BaseBenchmark:
    return BaselinePrecisionFP8PadInnerMatmulBenchmark()
