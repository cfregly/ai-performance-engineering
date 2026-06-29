"""optimized_precisionfp8_rowwise_gw_hp.py - torchao FP8 training benchmark (rowwise_with_gw_hp)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

try:
    from torchao.float8.config import Float8LinearConfig, Float8LinearRecipeName
    from torchao.float8.float8_linear_utils import convert_to_float8_training
except Exception as exc:  # pragma: no cover
    Float8LinearConfig = None  # type: ignore[assignment]
    Float8LinearRecipeName = None  # type: ignore[assignment]
    convert_to_float8_training = None  # type: ignore[assignment]
    TORCHAO_IMPORT_ERROR = exc
else:
    TORCHAO_IMPORT_ERROR = None

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class SimpleModel(nn.Module):
    """Two-layer MLP used for torchao Float8Linear runs."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class OptimizedFP8RowwiseGWHpBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized FP8 path using torchao rowwise_with_gw_hp recipe."""

    signature_equivalence_group = "ch13_precisionfp8_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.inputs_fp16: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.targets_fp16: Optional[torch.Tensor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_input_fp16: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self.batch_size = 8192
        self.hidden_dim = 8192
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
        if TORCHAO_IMPORT_ERROR is not None:
            raise RuntimeError(
                f"SKIPPED: torchao is required for {self.__class__.__name__}: {TORCHAO_IMPORT_ERROR}"
            )
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        model = SimpleModel(hidden_dim=self.hidden_dim).to(self.device).half().train()
        fp8_config = Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.ROWWISE_WITH_GW_HP)
        model = convert_to_float8_training(model, config=fp8_config)

        self.inputs = torch.randn(
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self.targets = torch.randn(
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self._verify_input = self.inputs.detach().clone()
        self._verify_input_fp16 = self._verify_input.to(torch.float16)
        self._verify_output_buffer = torch.empty(
            min(128, self.batch_size),
            min(256, self.hidden_dim),
            device=self.device,
            dtype=torch.float32,
        )
        self.inputs_fp16 = self.inputs.to(torch.float16)
        self.targets_fp16 = self.targets.to(torch.float16)

        self.model = model
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()

        for _ in range(5):
            self._train_step()
        self._synchronize()
        self.optimizer.zero_grad(set_to_none=True)
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def _train_step(self) -> None:
        assert self.model and self.inputs_fp16 is not None and self.targets_fp16 is not None
        assert self.optimizer and self.criterion is not None
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(self.inputs_fp16)
        loss = self.criterion(outputs, self.targets_fp16)
        loss.backward()
        self.optimizer.step()

    def benchmark_fn(self) -> None:
        if self._verify_input is None or self._verify_input_fp16 is None:
            raise RuntimeError("Verification input not initialized")
        with self._nvtx_range("optimized_precisionfp8_rowwise_gw_hp"):
            self._train_step()
            self.output = None

    def capture_verification_payload(self) -> None:
        if self.model is None or self._verify_input is None or self._verify_input_fp16 is None or self._verify_output_buffer is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        with torch.inference_mode():
            verify_out = self.model(self._verify_input_fp16)
            output_slice = verify_out[
                : self._verify_output_buffer.shape[0],
                : self._verify_output_buffer.shape[1],
            ]
            self._verify_output_buffer.copy_(output_slice)
            self.output = self._verify_output_buffer
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": True,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.25, 2.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.inputs_fp16 = None
        self.targets = None
        self.targets_fp16 = None
        self.optimizer = None
        self.criterion = None
        self.output = None
        self._verify_input = None
        self._verify_input_fp16 = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=None,
            reduced_precision_time_ms=getattr(self, '_last_elapsed_ms', None),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedFP8RowwiseGWHpBenchmark()
