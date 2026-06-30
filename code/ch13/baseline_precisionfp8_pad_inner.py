"""baseline_precisionfp8_pad_inner.py - FP32 baseline with non-multiple-of-16 input dim."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class SimpleModel(nn.Module):
    """Two-layer MLP with a non-multiple-of-16 input dimension."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselinePrecisionFP8PadInnerBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """FP32 precision baseline for pad_inner_dim comparison (forward-only)."""

    signature_equivalence_group = "ch13_precisionfp8_pad_inner"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.model = None
        self.inputs = None
        self.output = None
        self.batch_size = 4096
        self.input_dim = 8200
        self.hidden_dim = 8192
        self.output_dim = 8192
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        tokens = self.batch_size * self.input_dim
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

        self.model = SimpleModel(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
        ).to(self.device).train()
        self.inputs = torch.randn(self.batch_size, self.input_dim, device=self.device, dtype=torch.float32)
        self._verify_input = self.inputs.detach().clone()
        self._verify_output_buffer = torch.empty(
            min(128, self.batch_size),
            min(256, self.output_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None or self._verify_input is None:
            raise RuntimeError("Benchmark not configured")
        with self._nvtx_range("baseline_precisionfp8_pad_inner"):
            with torch.inference_mode():
                benchmark_out = self.model(self.inputs)
                self.output = benchmark_out
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

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
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.25, 2.0),
        )

    def teardown(self) -> None:
        del self.model, self.inputs
        self.output = None
        self._verify_input = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            backend_policy="fp32_strict",
        )

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_last_elapsed_ms', None),
            reduced_precision_time_ms=None,
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselinePrecisionFP8PadInnerBenchmark()
