"""Optimized control-flow benchmark for Chapter 14.

Optimized behavior:
- Rewrites data-dependent Python branch as tensor control flow (`torch.where`).
- Uses torch.compile with fullgraph enabled to enforce single-graph capture.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
from core.utils.compile_utils import compile_model
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class TensorizedControlFlowBlock(nn.Module):
    """Small MLP with tensorized branch (capturable)."""

    def __init__(self, hidden_size: int = 2048, depth: int = 3):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(depth)])

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        x = a
        for layer in self.layers:
            x = F.gelu(layer(x))

        # Optimized path: tensor control flow stays in graph capture.
        b_eff = torch.where(b.sum() < 0, -b, b)
        return x * b_eff


class OptimizedGraphBreakControlFlowBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: tensorized control flow with full-graph capture."""

    signature_equivalence_group = "ch14_graph_break_control_flow"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        self.batch_size = 1024
        self.hidden_size = 2048
        self.depth = 3
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model: Optional[nn.Module] = None
        self.compiled_model: Optional[nn.Module] = None
        self.a: Optional[torch.Tensor] = None
        self.b_pos: Optional[torch.Tensor] = None
        self.b_neg: Optional[torch.Tensor] = None
        self._last_b: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._step = 0
        self.parameter_count = 0

        tokens = self.batch_size * self.hidden_size
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.model = TensorizedControlFlowBlock(
            hidden_size=self.hidden_size,
            depth=self.depth,
        ).to(self.device, dtype=self.dtype).eval()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        self.compiled_model = compile_model(
            self.model,
            mode="max-autotune",
            fullgraph=True,  # Optimized: require a single capturable graph
            dynamic=False,
            backend="inductor",
        )

        self.a = torch.randn(
            self.batch_size,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        base_b = torch.randn(
            self.batch_size,
            self.hidden_size,
            device=self.device,
            dtype=self.dtype,
        )
        self.b_pos = base_b.abs() + 1e-2
        self.b_neg = -self.b_pos
        self._last_b = self.b_neg

        # Warm up both branch directions to validate dynamic predicate capture.
        for i in range(20):
            b = self.b_pos if (i % 2 == 0) else self.b_neg
            with torch.inference_mode():
                _ = self.compiled_model(self.a, b)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.compiled_model is None or self.a is None or self.b_pos is None or self.b_neg is None:
            raise RuntimeError("Benchmark not initialized")

        b = self.b_pos if (self._step % 2 == 0) else self.b_neg
        self._step += 1

        with torch.inference_mode(), self._nvtx_range("optimized_graph_break_control_flow"):
            self._last_b = b
            self.output = self.compiled_model(self.a, b)
        if self.output is None or self._last_b is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        if self.compiled_model is None or self.a is None or self.b_neg is None:
            raise RuntimeError("Benchmark not initialized for verification")
        with torch.inference_mode():
            verify_output = self.compiled_model(self.a, self.b_neg)
        self._set_verification_payload(
            inputs={"a": self.a, "b": self.b_neg},
            output=verify_output.float(),
            batch_size=self.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.compiled_model = None
        self.a = None
        self.b_pos = None
        self.b_neg = None
        self._last_b = None
        self.output = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=40,
            warmup=10,
            setup_timeout_seconds=900,
            measurement_timeout_seconds=900,
            use_subprocess=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return None

    def validate_result(self) -> Optional[str]:
        if self.compiled_model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedGraphBreakControlFlowBenchmark()

