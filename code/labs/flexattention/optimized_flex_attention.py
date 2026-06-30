"""Optimized FlexAttention lab: torch.compile + block sparsity."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.flexattention.flexattention_common import (
    build_flex_attention_inputs,
    compute_attention_workload_metrics,
    make_relative_bias_score_mod,
    resolve_device,
)

DEFAULT_COMPILE_MODE = "reduce-overhead"
COMPILE_MODE = os.getenv("TORCH_COMPILE_MODE", DEFAULT_COMPILE_MODE)


class CompiledFlexAttention(nn.Module):
    """Minimal module to allow torch.compile to fuse FlexAttention."""

    def __init__(self, block_mask, rel_bias: torch.Tensor):
        super().__init__()
        self.block_mask = block_mask
        self.register_buffer("rel_bias", rel_bias)
        self.score_mod = make_relative_bias_score_mod(self.rel_bias)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return flex_attention(
            q,
            k,
            v,
            score_mod=self.score_mod,
            block_mask=self.block_mask,
        )


class OptimizedFlexAttentionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Compiled FlexAttention with the blog's DSL hooks."""

    allow_cpu = True

    def __init__(self) -> None:
        super().__init__()
        self.dtype = torch.bfloat16
        self.seq_len = 1024
        self.batch = 2
        self.heads = 8
        self.head_dim = 64
        self.block_size = 64
        self.doc_span = 256
        self.inputs = None
        self.module: Optional[nn.Module] = None
        self.compiled = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.device = resolve_device()
        torch._inductor.config.triton.cudagraphs = True
        torch._inductor.config.max_autotune = True

        self.inputs = build_flex_attention_inputs(
            batch=self.batch,
            heads=self.heads,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            doc_span=self.doc_span,
            block_size=self.block_size,
            dtype=self.dtype,
            device=self.device,
        )
        self._verify_output_buffer = torch.empty_like(self.inputs.q, dtype=torch.float32)

        self.module = CompiledFlexAttention(
            block_mask=self.inputs.block_mask,
            rel_bias=self.inputs.rel_bias,
        ).to(self.device)
        self.compiled = torch.compile(self.module, mode=COMPILE_MODE, fullgraph=True, dynamic=None)

        with torch.inference_mode():
            _ = self.compiled(self.inputs.q, self.inputs.k, self.inputs.v)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None or self.compiled is None:
            raise RuntimeError("FlexAttention module is not initialized")

        with self._nvtx_range("flexattention_compiled"):
            with torch.inference_mode():
                self.output = self.compiled(
                    self.inputs.q,
                    self.inputs.k,
                    self.inputs.v,
                )
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={
                "q": self.inputs.q.detach(),
                "k": self.inputs.k.detach(),
                "v": self.inputs.v.detach(),
            },
            output=self._verify_output_buffer,
            batch_size=self.batch,
            parameter_count=0,
            precision_flags={"bf16": True, "fp16": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None
        self.module = None
        self.compiled = None
        self.output = None
        self._verify_output_buffer = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=10,  # Required for FlexAttention + torch.compile JIT
            measurement_timeout_seconds=60,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return attention-specific workload metrics."""
        return compute_attention_workload_metrics(
            batch=self.batch,
            heads=self.heads,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            doc_span=self.doc_span,
            dtype=self.dtype,
        )

    def validate_result(self) -> Optional[str]:
        if self.inputs is None or self.compiled is None:
            return "FlexAttention module is not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
    return OptimizedFlexAttentionBenchmark()
