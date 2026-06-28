"""optimized_regional_triton.py - Regional compilation with TorchInductor (Triton).

Optimized for Chapter 14 regional compilation:
- Keep the full Transformer block eager to avoid recompiling attention for each
  sequence bucket.
- Compile only the MLP hot region with TorchInductor (which lowers to Triton on
  NVIDIA GPUs) to enable fusion and reduce compile churn.

This keeps the math identical to the baseline while shifting compilation to the
region where it pays off most.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin


class MLP(nn.Module):
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
    """Small block to stress compile churn across sequence buckets."""

    def __init__(self, hidden: int = 1024, num_heads: int = 8, mlp_hidden: int = 4096):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(embed_dim=hidden, num_heads=num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(hidden)
        self.mlp = MLP(hidden, mlp_hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        attn_out.add_(residual)
        x = attn_out

        residual = x
        x = self.ln2(x)
        mlp_out = self.mlp(x)
        mlp_out.add_(residual)
        return mlp_out


class OptimizedRegionalTritonBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: compile only the MLP region with Inductor (Triton)."""

    def __init__(self):
        super().__init__()
        self.hidden = 1536
        self.num_heads = 12
        self.mlp_hidden = 12288
        self.batch_size = 8
        self.sequence_schedule: List[int] = [128, 256, 384, 512]
        self._step = 0
        self.model: Optional[nn.Module] = None
        self._compiled_model: Optional[nn.Module] = None
        self.inputs: Dict[int, torch.Tensor] = {}
        max_tokens = self.batch_size * max(self.sequence_schedule) * self.hidden
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(max_tokens),
        )
        self.output = None
        self._last_input: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self._verification_payload = None
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

        # Regional compilation: compile ONLY the MLP module. Inductor will
        # generate Triton kernels for eligible fusion patterns in this region.
        self.model.mlp = torch.compile(
            self.model.mlp,
            backend="inductor",
            fullgraph=False,
            dynamic=True,
            mode="max-autotune",
        )
        self._compiled_model = self.model

        for seq in self.sequence_schedule:
            self.inputs[seq] = torch.randn(
                self.batch_size,
                seq,
                self.hidden,
                device=self.device,
                dtype=torch.bfloat16,
            )
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        # Warm every sequence bucket in setup so the timed path measures steady
        # state regional compilation, not first-bucket Inductor/autotune churn.
        with torch.inference_mode():
            for _ in range(3):
                for seq in self.sequence_schedule:
                    _ = self._compiled_model(self.inputs[seq])

        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def _next_sequence_length(self) -> int:
        seq = self.sequence_schedule[self._step % len(self.sequence_schedule)]
        self._step += 1
        return seq

    def benchmark_fn(self) -> None:
        if self._compiled_model is None:
            raise RuntimeError("Model not initialized")
        seq_len = self._next_sequence_length()
        x = self.inputs[seq_len]
        with torch.inference_mode(), self._nvtx_range("optimized_regional_triton"):
            self._last_input = x
            self.output = self._compiled_model(x)
        if self.output is None or self._last_input is None:
            raise RuntimeError("benchmark_fn() must produce output")
        self._payload_dtype = self._last_input.dtype

    def capture_verification_payload(self) -> None:
        dtype = self._payload_dtype
        if self.output is None or self._last_input is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        self._set_verification_payload(
            inputs={"input": self._last_input},
            output=self.output.detach(),
            batch_size=self._last_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        self.model = None
        self._compiled_model = None
        self.inputs.clear()
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=10,  # Warmup amortizes Inductor compile + autotune per bucket
            adaptive_iterations=False,  # Stateful seq schedule must run fixed counts for verification
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=300,
            measurement_timeout_seconds=300,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_triton_metrics

        return compute_triton_metrics(
            num_elements=getattr(self, "N", getattr(self, "num_elements", 1024)),
            elapsed_ms=getattr(self, "_last_elapsed_ms", None),
            block_size=getattr(self, "BLOCK_SIZE", 1024),
            num_warps=getattr(self, "num_warps", 4),
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedRegionalTritonBenchmark()
