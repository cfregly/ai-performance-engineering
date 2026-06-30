"""Baseline FlexAttention lab: uncompiled CuTe DSL path."""

from __future__ import annotations

from typing import Optional

import torch
from torch.nn.attention.flex_attention import flex_attention

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.flexattention.flexattention_common import (
    build_flex_attention_inputs,
    compute_attention_workload_metrics,
    make_relative_bias_score_mod,
    resolve_device,
)


class BaselineFlexAttentionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Eager FlexAttention using the DSL score_mod + block_mask."""

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
        self.score_mod = None
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
        self.score_mod = make_relative_bias_score_mod(self.inputs.rel_bias)
        self._verify_output_buffer = torch.empty_like(self.inputs.q, dtype=torch.float32)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None or self.score_mod is None:
            raise RuntimeError("FlexAttention inputs are not initialized")

        with self._nvtx_range("flexattention_baseline"):
            with torch.inference_mode():
                self.output = flex_attention(
                    self.inputs.q,
                    self.inputs.k,
                    self.inputs.v,
                    score_mod=self.score_mod,
                    block_mask=self.inputs.block_mask,
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
        self.score_mod = None
        self.output = None
        self._verify_output_buffer = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=10,  # Required for FlexAttention JIT compilation
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
        if self.inputs is None or self.score_mod is None:
            return "FlexAttention inputs are not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
    return BaselineFlexAttentionBenchmark()
