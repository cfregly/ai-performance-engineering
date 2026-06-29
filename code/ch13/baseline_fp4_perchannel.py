"""baseline_fp4_perchannel.py - Naive FP4 per-tensor quantization baseline."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ch13.fp4_perchannel_common import build_fp4_perchannel_reference_tensors
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class FP4PerTensorLinear(nn.Module):
    """Linear layer with naive per-tensor FP4 quantization."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.fp4_max = 6.0  # E2M1 range approx [-6, 6]
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_amax = x.abs().max()
        input_scale = torch.clamp(input_amax / self.fp4_max, min=1e-6)
        x_q = torch.clamp(x / input_scale, -self.fp4_max, self.fp4_max).round()

        weight_amax = self.weight.abs().max()
        weight_scale = torch.clamp(weight_amax / self.fp4_max, min=1e-6)
        weight_q = torch.clamp(self.weight / weight_scale, -self.fp4_max, self.fp4_max).round()

        output_q = torch.nn.functional.linear(x_q, weight_q, bias=None)
        output = output_q
        output.mul_(input_scale)
        output.mul_(weight_scale)

        if self.bias is not None:
            output.add_(self.bias)
        return output


class NaiveFP4MLP(nn.Module):
    """Two-layer MLP with naive FP4 per-tensor quantization."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = FP4PerTensorLinear(hidden_dim, hidden_dim * 2, bias=True)
        self.fc2 = FP4PerTensorLinear(hidden_dim * 2, hidden_dim, bias=True)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        return self.fc2(x)


def _init_linear_weights(linear: nn.Module, weight: torch.Tensor, bias: torch.Tensor) -> None:
    with torch.inference_mode():
        linear.weight.copy_(weight)
        if linear.bias is not None:
            linear.bias.copy_(bias)


class BaselineFP4PerTensorBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: naive per-tensor FP4 quantization."""

    signature_equivalence_group = "ch13_fp4_perchannel"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.batch_size = 1024
        self.hidden_dim = 8192
        self.dtype = torch.float32
        self.parameter_count: int = 0
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        tokens = self.batch_size * self.hidden_dim
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

        self.model = NaiveFP4MLP(hidden_dim=self.hidden_dim).to(self.device, dtype=self.dtype).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        w1, b1, w2, b2, inputs = build_fp4_perchannel_reference_tensors(
            hidden_dim=self.hidden_dim,
            batch_size=self.batch_size,
            device=self.device,
            dtype=self.dtype,
        )
        _init_linear_weights(self.model.fc1, w1, b1)
        _init_linear_weights(self.model.fc2, w2, b2)

        self.inputs = inputs
        self._verify_input = self.inputs.detach().clone()
        self._verify_output_buffer = torch.empty(
            min(128, self.batch_size),
            min(256, self.hidden_dim),
            device=self.device,
            dtype=torch.float32,
        )

        for _ in range(3):
            with torch.inference_mode():
                _ = self.model(self.inputs)

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Benchmark not initialized")
        with torch.inference_mode():
            self.output = self.model(self.inputs)
        if self._verify_input is None or self.output is None:
            raise RuntimeError("Verification input/output not initialized")

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(2.0, 20.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.output = None
        self._verify_input = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineFP4PerTensorBenchmark()
