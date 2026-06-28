"""Optimized speculative decoding: draft proposals + batched target verification.

This benchmark models the core speculative decoding speedup:
  1) Use a small draft model to propose K tokens.
  2) Verify those K tokens in a single target-model forward pass (batch on K).
  3) Accept matching draft tokens; on mismatch, fall back to the target token.

The final generated token sequence must match the baseline greedy decode.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

from labs.speculative_decode.speculative_decode_common import (
    TokenMLP,
    build_draft_from_target,
    default_workload,
    scale_tail_dims_,
)


class OptimizedSpeculativeDecodeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Spec decode loop: draft + batched target verification."""

    # This Python reference benchmark intentionally uses host-visible control
    # flow for the accept/reject boundary.
    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def __init__(self) -> None:
        super().__init__()

        # Use FP32 for deterministic argmax stability across launch/shape variations.
        self.workload = default_workload(dtype=torch.float32)

        self.target_model: Optional[TokenMLP] = None
        self.draft_model: Optional[TokenMLP] = None
        self.input_ids: Optional[torch.Tensor] = None
        self._output_ids: Optional[torch.Tensor] = None
        self._draft_ids: Optional[torch.Tensor] = None
        self._verify_prev: Optional[torch.Tensor] = None
        self._accept_prefix: Optional[torch.Tensor] = None
        self._accept_count: Optional[torch.Tensor] = None
        self._draft_next_values: Optional[torch.Tensor] = None
        self._draft_next_tokens: Optional[torch.Tensor] = None
        self._target_next_values: Optional[torch.Tensor] = None
        self._target_next_tokens: Optional[torch.Tensor] = None
        self._matches: Optional[torch.Tensor] = None
        self._draft_logits: Optional[torch.Tensor] = None
        self._target_logits: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0

        self._metrics: Dict[str, float] = {}

        tokens = float(self.workload.total_tokens)
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        wl = self.workload
        self.target_model = TokenMLP(
            vocab_size=wl.vocab_size,
            hidden_size=wl.target_hidden,
            num_layers=wl.target_layers,
            device=self.device,
            dtype=wl.dtype,
        ).eval()
        scale_tail_dims_(self.target_model, wl.draft_hidden, wl.tail_scale)
        self._payload_parameter_count = sum(p.numel() for p in self.target_model.parameters())

        # Deterministic starting token. Must be created BEFORE draft init so it
        # matches the baseline.
        self.input_ids = torch.randint(0, wl.vocab_size, (1, 1), device=self.device, dtype=torch.int64)

        self._output_ids = torch.empty((1, wl.total_tokens + 1), device=self.device, dtype=torch.int64)
        self._draft_ids = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.int64)
        self._verify_prev = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.int64)
        self._accept_prefix = torch.empty(wl.speculative_k, device=self.device, dtype=torch.int32)
        self._accept_count = torch.empty((), device=self.device, dtype=torch.int32)
        self._draft_next_values = torch.empty((1,), device=self.device, dtype=wl.dtype)
        self._draft_next_tokens = torch.empty((1,), device=self.device, dtype=torch.long)
        self._target_next_values = torch.empty((1, wl.speculative_k), device=self.device, dtype=wl.dtype)
        self._target_next_tokens = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.long)
        self._matches = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.bool)
        self._draft_logits = torch.empty((1, 1, wl.vocab_size), device=self.device, dtype=wl.dtype)
        self._target_logits = torch.empty((1, wl.speculative_k, wl.vocab_size), device=self.device, dtype=wl.dtype)

        self.draft_model = build_draft_from_target(self.target_model, wl.draft_hidden)
        self.output = None
        self._metrics = {}
        self._synchronize()

    def benchmark_fn(self) -> None:
        if (
            self.target_model is None
            or self.draft_model is None
            or self.input_ids is None
            or self._output_ids is None
            or self._draft_ids is None
            or self._verify_prev is None
            or self._accept_prefix is None
            or self._accept_count is None
            or self._draft_next_values is None
            or self._draft_next_tokens is None
            or self._target_next_values is None
            or self._target_next_tokens is None
            or self._matches is None
            or self._draft_logits is None
            or self._target_logits is None
        ):
            raise RuntimeError("Benchmark not initialized")

        wl = self.workload
        out = self._output_ids
        out[:, 0] = self.input_ids[:, 0]

        draft_tokens = 0
        accepted_draft = 0
        rounds = 0

        with torch.inference_mode():
            pos = 0
            while pos < wl.total_tokens:
                rounds += 1
                remaining = wl.total_tokens - pos
                k = wl.speculative_k if remaining >= wl.speculative_k else remaining

                # Draft: propose k tokens sequentially.
                prev = out[:, pos : pos + 1]
                for j in range(k):
                    logits_d = self.draft_model.forward_into(prev, self._draft_logits)
                    torch.max(logits_d[:, 0, :], dim=-1, out=(self._draft_next_values, self._draft_next_tokens))
                    self._draft_ids[:, j].copy_(self._draft_next_tokens)
                    next_d = self._draft_next_tokens
                    prev = next_d.view(1, 1)

                draft_tokens += int(k)

                # Verify: compute target predictions for the k steps in one call.
                self._verify_prev[:, 0] = out[:, pos]
                if k > 1:
                    self._verify_prev[:, 1:k] = self._draft_ids[:, : k - 1]

                logits_t = self.target_model.forward_into(self._verify_prev[:, :k], self._target_logits[:, :k])
                target_values = self._target_next_values[:, :k]
                target_next = self._target_next_tokens[:, :k]
                torch.max(logits_t, dim=-1, out=(target_values, target_next))
                matches = self._matches[:, :k]
                torch.eq(target_next, self._draft_ids[:, :k], out=matches)

                accept_prefix = self._accept_prefix[:k]
                torch.cumprod(matches[0], dim=0, dtype=torch.int32, out=accept_prefix)
                torch.sum(accept_prefix, dim=0, out=self._accept_count)
                accept_k = int(self._accept_count.item())

                if accept_k == k:
                    out[:, pos + 1 : pos + k + 1] = self._draft_ids[:, :k]
                    accepted_draft += int(k)
                    pos += k
                else:
                    if accept_k > 0:
                        out[:, pos + 1 : pos + accept_k + 1] = self._draft_ids[:, :accept_k]
                        accepted_draft += int(accept_k)
                    out[:, pos + accept_k + 1] = target_next[:, accept_k]
                    pos += accept_k + 1

        self.output = out
        self._metrics = {
            "speculative.draft_tokens": float(draft_tokens),
            "speculative.accepted_draft_tokens": float(accepted_draft),
            "speculative.acceptance_rate_pct": (accepted_draft / max(draft_tokens, 1)) * 100.0,
            "speculative.rounds": float(rounds),
        }

    def capture_verification_payload(self) -> None:
        if self.input_ids is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=self.output.float(),
            batch_size=1,
            parameter_count=self._payload_parameter_count,
            precision_flags={"bf16": False, "fp16": False, "fp8": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.target_model = None
        self.draft_model = None
        self.input_ids = None
        self._output_ids = None
        self._draft_ids = None
        self._verify_prev = None
        self._accept_prefix = None
        self._accept_count = None
        self._draft_next_values = None
        self._draft_next_tokens = None
        self._target_next_values = None
        self._target_next_tokens = None
        self._matches = None
        self._draft_logits = None
        self._target_logits = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return dict(self._metrics)

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if self.output.shape[-1] != self.workload.total_tokens + 1:
            return "Unexpected output shape"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedSpeculativeDecodeBenchmark()
