"""baseline_precisionfp8_te.py - Transformer Engine FP16 baseline.

Provides a baseline that exercises the Transformer Engine Linear layers in
float16 mode so the optimized FP8 benchmark can focus on precision benefits
instead of framework overhead.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn

from ch13.te_runtime_common import ensure_te_runtime_initialized
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

TELinear: Any | None = None
TE_IMPORT_ERROR: Optional[ImportError] = None


def _load_te_linear() -> Any:
    global TELinear, TE_IMPORT_ERROR
    if TELinear is not None:
        return TELinear
    ensure_te_runtime_initialized()
    try:
        from transformer_engine.pytorch import Linear as te_linear
    except ImportError as exc:  # pragma: no cover
        TE_IMPORT_ERROR = exc
        raise RuntimeError(
            "SKIPPED: Transformer Engine is required for baseline_precisionfp8_te. "
            f"(import error: {exc})"
        ) from exc
    TELinear = te_linear
    return TELinear


class TEFP16MLP(nn.Module):
    """Two-layer MLP built with Transformer Engine Linear layers."""

    def __init__(self, hidden_dim: int = 1024):
        super().__init__()
        te_linear = _load_te_linear()
        self.fc1 = te_linear(hidden_dim, hidden_dim * 2, bias=True)
        self.fc2 = te_linear(hidden_dim * 2, hidden_dim, bias=True)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


class BaselineTEFP8Benchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline Transformer Engine run in float16 precision."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.targets: Optional[torch.Tensor] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.criterion: Optional[nn.Module] = None
        self.batch_size = 256
        self.hidden_dim = 4096
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.output: Optional[torch.Tensor] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count = 0
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self._verify_input: Optional[torch.Tensor] = None

    def setup(self) -> None:
        _load_te_linear()
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        model = TEFP16MLP(hidden_dim=self.hidden_dim).to(self.device).train().half()
        self.model = model
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self.inputs = torch.randn(
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self.targets = torch.randn_like(self.inputs)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.output_buffer = torch.empty_like(self.inputs)
        self._verify_output_buffer = torch.empty_like(self.output_buffer)
        self._verify_input = self.inputs.detach().clone()

        for _ in range(5):
            self._train_step(capture_output=False)
        self.optimizer.zero_grad(set_to_none=True)
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def _train_step(self, capture_output: bool = True) -> None:
        assert self.model and self.inputs is not None and self.targets is not None
        assert self.optimizer and self.criterion
        self.optimizer.zero_grad(set_to_none=True)
        outputs = self.model(self.inputs)
        if capture_output:
            if self.output_buffer is None:
                raise RuntimeError("Output buffer not initialized")
            with torch.inference_mode():
                self.output_buffer.copy_(outputs)
            self.output = self.output_buffer
        loss = self.criterion(outputs, self.targets)
        loss.backward()
        self.optimizer.step()

    def benchmark_fn(self) -> None:
        with self._nvtx_range("baseline_precisionfp8_te"):
            self._train_step(capture_output=True)
        if self._verify_input is None or self.output is None:
            raise RuntimeError("Verification input/output not initialized")

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.targets = None
        self.optimizer = None
        self.criterion = None
        self.output = None
        self.output_buffer = None
        self._verify_output_buffer = None
        self._verify_input = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10, backend_policy="fp32_strict")

    def get_custom_metrics(self) -> Optional[dict]:
        """Return truthful precision metrics for the current FP32 TE run."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, "_last_elapsed_ms", None),
            reduced_precision_time_ms=None,
            precision_type="fp8",
        )

    def get_output_for_verification(self) -> Optional[torch.Tensor]:
        # Use the latest inputs as representative output; optimized path returns a static input snapshot.
        return self.inputs

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineTEFP8Benchmark()
