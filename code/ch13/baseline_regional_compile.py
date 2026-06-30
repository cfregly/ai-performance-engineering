"""baseline_regional_compile.py - Whole-block torch.compile baseline.

Compiles the entire Transformer block on the same BF16 workload used by the
regional benchmark. This keeps precision fixed so the comparison isolates
whole-graph versus regional compilation rather than BF16 versus FP32.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.utils import compile_utils as _compile_utils_patch  # noqa: F401


class MLP(nn.Module):
    """Standard MLP used in the Transformer block."""

    def __init__(self, hidden: int, mlp_hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden, mlp_hidden)
        self.fc2 = nn.Linear(mlp_hidden, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x


class TinyTransformerBlock(nn.Module):
    """Small Transformer block to demonstrate full-graph compilation."""

    def __init__(self, hidden: int = 1024, num_heads: int = 8, mlp_hidden: int = 4096):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden,
            num_heads=num_heads,
            batch_first=True,
        )
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = MLP(hidden, mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = residual + attn_out

        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        return x


class BaselineFullGraphCompileBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """
    Baseline: torch.compile over the entire block for each sequence bucket.
    """

    signature_equivalence_group = "ch13_regional_compile_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)

    def __init__(self):
        super().__init__()
        # Larger workload to amortize compile overhead
        self.hidden = 2048
        self.num_heads = 16
        self.mlp_hidden = 16384
        self.batch_size = 16
        self.sequence_schedule: List[int] = [256, 512, 1024, 1536]
        self._step = 0

        self.model: Optional[nn.Module] = None
        self.compiled_model = None
        self.inputs: Dict[int, torch.Tensor] = {}
        self._verify_x: Optional[torch.Tensor] = None
        self._verify_output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None

        max_tokens = self.batch_size * max(self.sequence_schedule) * self.hidden
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(max_tokens),
        )
        self.output = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(max_tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = TinyTransformerBlock(
            hidden=self.hidden,
            num_heads=self.num_heads,
            mlp_hidden=self.mlp_hidden,
        ).to(self.device, dtype=torch.bfloat16).eval()
        self.compiled_model = torch.compile(
            self.model,
            backend="inductor",
            dynamic=True,
            mode="reduce-overhead",
        )

        for seq in self.sequence_schedule:
            self.inputs[seq] = torch.randn(
                self.batch_size,
                seq,
                self.hidden,
                device=self.device,
                dtype=torch.bfloat16,
            )
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            min(128, max(self.sequence_schedule)),
            self.hidden,
            device=self.device,
            dtype=torch.float32,
        )

        with torch.inference_mode():
            for _ in range(5):
                for seq in self.sequence_schedule:
                    _ = self.compiled_model(self.inputs[seq])

        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def _next_sequence_length(self) -> int:
        seq = self.sequence_schedule[self._step % len(self.sequence_schedule)]
        self._step += 1
        return seq

    def benchmark_fn(self) -> None:
        if self.model is None or self.compiled_model is None:
            raise RuntimeError("Model not initialized")

        seq_len = self._next_sequence_length()
        x = self.inputs[seq_len]

        with torch.inference_mode(), self._nvtx_range("baseline_full_graph_compile"):
            self.output = self.compiled_model(x)
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        if self._verify_x is None:
            self._verify_x = x
            self._verify_output = self.output

    def capture_verification_payload(self) -> None:
        if self._verify_x is None or self._verify_output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        x = self._verify_x
        verify_output = self._verify_output_buffer[
            : self._verify_output.shape[0],
            : min(self._verify_output.shape[1], self._verify_output_buffer.shape[1]),
            :,
        ]
        output_slice = self._verify_output[:, : verify_output.shape[1], :]
        verify_output.copy_(output_slice)
        self._set_verification_payload(
            inputs={"input": x},
            output=verify_output,
            batch_size=self.batch_size,
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1.0, 10.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.compiled_model = None
        self.inputs.clear()
        self.output = None
        self._verify_x = None
        self._verify_output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=300,
            measurement_timeout_seconds=300,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_last_elapsed_ms', None),
            reduced_precision_time_ms=None,
            precision_type="bf16",
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineFullGraphCompileBenchmark()
