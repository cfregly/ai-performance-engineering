"""optimized_precisionfp8_pad_inner_matmul.py - FP8 matmul with pad_inner_dim."""

from __future__ import annotations

from typing import Optional

import torch

try:
    from torchao.float8.config import ScalingGranularity, e4m3_dtype
    from torchao.float8.float8_training_tensor import (
        GemmInputRole,
        LinearMMConfig,
        ScaledMMConfig,
        hp_tensor_and_scale_to_float8,
    )
    from torchao.float8.float8_utils import tensor_to_scale
except Exception as exc:  # pragma: no cover
    ScalingGranularity = None  # type: ignore[assignment]
    e4m3_dtype = None  # type: ignore[assignment]
    GemmInputRole = None  # type: ignore[assignment]
    LinearMMConfig = None  # type: ignore[assignment]
    ScaledMMConfig = None  # type: ignore[assignment]
    hp_tensor_and_scale_to_float8 = None  # type: ignore[assignment]
    tensor_to_scale = None  # type: ignore[assignment]
    TORCHAO_IMPORT_ERROR = exc
else:
    TORCHAO_IMPORT_ERROR = None

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class OptimizedPrecisionFP8PadInnerMatmulBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """FP8 matmul using pad_inner_dim to handle non-multiple-of-16 K."""

    signature_equivalence_group = "ch13_precisionfp8_pad_inner_matmul"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.a: Optional[torch.Tensor] = None
        self.b: Optional[torch.Tensor] = None
        self.scale_a: Optional[torch.Tensor] = None
        self.scale_b: Optional[torch.Tensor] = None
        self.mm_config: Optional[LinearMMConfig] = None
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
        if TORCHAO_IMPORT_ERROR is not None:
            raise RuntimeError(
                f"SKIPPED: torchao is required for {self.__class__.__name__}: {TORCHAO_IMPORT_ERROR}"
            )
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

        self.scale_a = tensor_to_scale(
            self.a,
            e4m3_dtype,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=-1,
        ).float()
        self.scale_b = tensor_to_scale(
            self.b,
            e4m3_dtype,
            scaling_granularity=ScalingGranularity.AXISWISE,
            axiswise_dim=0,
        ).float()
        scaled_mm_config = ScaledMMConfig(
            emulate=False,
            use_fast_accum=True,
            fp8_output=False,
            pad_inner_dim=True,
        )
        self.mm_config = LinearMMConfig(scaled_mm_config, scaled_mm_config, scaled_mm_config)
        self.parameter_count = self.k * self.n
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def benchmark_fn(self) -> None:
        if self.a is None or self.b is None or self.scale_a is None or self.scale_b is None or self.mm_config is None:
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("optimized_precisionfp8_pad_inner_matmul"):
            with torch.inference_mode():
                a_fp8 = hp_tensor_and_scale_to_float8(
                    self.a,
                    self.scale_a,
                    e4m3_dtype,
                    self.mm_config,
                    GemmInputRole.INPUT,
                    axiswise_dim=-1,
                )
                b_fp8 = hp_tensor_and_scale_to_float8(
                    self.b,
                    self.scale_b,
                    e4m3_dtype,
                    self.mm_config,
                    GemmInputRole.WEIGHT,
                    axiswise_dim=0,
                )
                out = a_fp8 @ b_fp8
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
                "fp8": True,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.25, 2.0),
        )

    def teardown(self) -> None:
        del self.a, self.b, self.scale_a, self.scale_b, self.mm_config
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )


def get_benchmark() -> BaseBenchmark:
    return OptimizedPrecisionFP8PadInnerMatmulBenchmark()
