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
    accept_prefix_length,
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
        self._input_token_view: Optional[torch.Tensor] = None
        self._output_ids: Optional[torch.Tensor] = None
        self._output_step_views: list[torch.Tensor] = []
        self._output_token_views: list[torch.Tensor] = []
        self._output_write_views: list[list[torch.Tensor]] = []
        self._output_verify_views: list[list[torch.Tensor]] = []
        self._view_counts: tuple[int, ...] = ()
        self._expected_view_counts: tuple[int, ...] = ()
        self._accept_prefix: Optional[torch.Tensor] = None
        self._accept_count_device: Optional[torch.Tensor] = None
        self._accept_count_host: Optional[torch.Tensor] = None
        self._accept_count_device_scalar: Optional[torch.Tensor] = None
        self._accept_count_host_scalar: Optional[torch.Tensor] = None
        self._accept_all_device: Optional[torch.Tensor] = None
        self._accept_all_host: Optional[torch.Tensor] = None
        self._accept_all_host_scalar: Optional[torch.Tensor] = None
        self._draft_next_values: Optional[torch.Tensor] = None
        self._target_next_values: Optional[torch.Tensor] = None
        self._target_next_tokens: Optional[torch.Tensor] = None
        self._matches: Optional[torch.Tensor] = None
        self._draft_logits: Optional[torch.Tensor] = None
        self._draft_logits_next: Optional[torch.Tensor] = None
        self._target_logits: Optional[torch.Tensor] = None
        self._target_logits_views: list[torch.Tensor] = []
        self._target_value_views: list[torch.Tensor] = []
        self._target_token_views: list[torch.Tensor] = []
        self._target_token_column_views: list[torch.Tensor] = []
        self._match_views: list[torch.Tensor] = []
        self._accept_prefix_views: list[torch.Tensor] = []
        self._accept_prefix_row_views: list[torch.Tensor] = []
        self._speculation_step_ranges: list[range] = []
        self._draft_forward_buffers: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._target_forward_buffers: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._forward_buffer_counts: tuple[int, int] = ()
        self._expected_forward_buffer_counts: tuple[int, int] = ()
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0

        self._metrics: Dict[str, float] = {
            "speculative.draft_tokens": 0.0,
            "speculative.accepted_draft_tokens": 0.0,
            "speculative.acceptance_rate_pct": 0.0,
            "speculative.rounds": 0.0,
        }

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
        self._input_token_view = self.input_ids[:, 0]

        self._output_ids = torch.empty((1, wl.total_tokens + 1), device=self.device, dtype=torch.int64)
        self._verify_output_buffer = torch.empty_like(self._output_ids, dtype=torch.float32)
        self._output_step_views = [
            self._output_ids[:, token_idx : token_idx + 1] for token_idx in range(wl.total_tokens + 1)
        ]
        self._output_token_views = [
            self._output_ids[:, token_idx] for token_idx in range(wl.total_tokens + 1)
        ]
        self._output_write_views = [
            [
                self._output_ids[:, start + 1 : start + length + 1]
                for start in range(wl.total_tokens - length + 1)
            ]
            for length in range(1, wl.speculative_k + 1)
        ]
        self._output_verify_views = [
            [
                self._output_ids[:, start : start + length]
                for start in range(wl.total_tokens - length + 1)
            ]
            for length in range(1, wl.speculative_k + 1)
        ]
        self._accept_prefix = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.int64)
        self._accept_count_device = torch.empty((1,), device=self.device, dtype=torch.int64)
        self._accept_count_host = torch.empty(
            (1,),
            dtype=torch.int64,
            device="cpu",
            pin_memory=torch.cuda.is_available(),
        )
        self._accept_count_device_scalar = self._accept_count_device[0]
        self._accept_count_host_scalar = self._accept_count_host[0]
        self._accept_all_device = torch.empty((1,), device=self.device, dtype=torch.bool)
        self._accept_all_host = torch.empty(
            (1,),
            dtype=torch.bool,
            device="cpu",
            pin_memory=torch.cuda.is_available(),
        )
        self._accept_all_host_scalar = self._accept_all_host[0]
        # torch.max requires value and index outputs with matching strides.
        self._draft_next_values = torch.empty_strided(
            (1,),
            self._output_token_views[0].stride(),
            device=self.device,
            dtype=wl.dtype,
        )
        self._target_next_values = torch.empty((1, wl.speculative_k), device=self.device, dtype=wl.dtype)
        self._target_next_tokens = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.long)
        self._matches = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.bool)
        self._draft_logits = torch.empty((1, 1, wl.vocab_size), device=self.device, dtype=wl.dtype)
        self._draft_logits_next = self._draft_logits[:, 0, :]
        self._target_logits = torch.empty((1, wl.speculative_k, wl.vocab_size), device=self.device, dtype=wl.dtype)
        self._target_logits_views = [self._target_logits[:, :k] for k in range(1, wl.speculative_k + 1)]
        self._target_value_views = [self._target_next_values[:, :k] for k in range(1, wl.speculative_k + 1)]
        self._target_token_views = [self._target_next_tokens[:, :k] for k in range(1, wl.speculative_k + 1)]
        self._target_token_column_views = [
            self._target_next_tokens[:, token_idx] for token_idx in range(wl.speculative_k)
        ]
        self._match_views = [self._matches[:, :k] for k in range(1, wl.speculative_k + 1)]
        self._accept_prefix_views = [
            self._accept_prefix[:, :k] for k in range(1, wl.speculative_k + 1)
        ]
        self._accept_prefix_row_views = [
            self._accept_prefix[0, :k] for k in range(1, wl.speculative_k + 1)
        ]
        self._speculation_step_ranges = [range(k) for k in range(1, wl.speculative_k + 1)]
        self._view_counts = (
            len(self._output_step_views),
            len(self._output_token_views),
            len(self._output_write_views),
            len(self._output_verify_views),
            len(self._target_logits_views),
            len(self._target_value_views),
            len(self._target_token_views),
            len(self._target_token_column_views),
            len(self._match_views),
            len(self._accept_prefix_views),
            len(self._accept_prefix_row_views),
            len(self._speculation_step_ranges),
        )
        self._expected_view_counts = (
            wl.total_tokens + 1,
            wl.total_tokens + 1,
            wl.speculative_k,
            wl.speculative_k,
            wl.speculative_k,
            wl.speculative_k,
            wl.speculative_k,
            wl.speculative_k,
            wl.speculative_k,
            wl.speculative_k,
            wl.speculative_k,
            wl.speculative_k,
        )

        self.draft_model = build_draft_from_target(self.target_model, wl.draft_hidden)
        self._draft_forward_buffers = self.draft_model.prepare_forward_buffers(
            1,
            device=self.device,
            dtype=wl.dtype,
        )
        self._target_forward_buffers = [
            self.target_model.prepare_forward_buffers(k, device=self.device, dtype=wl.dtype)
            for k in range(1, wl.speculative_k + 1)
        ]
        self._forward_buffer_counts = (1, len(self._target_forward_buffers))
        self._expected_forward_buffer_counts = (1, wl.speculative_k)
        self.output = None
        for key in self._metrics:
            self._metrics[key] = 0.0
        self._synchronize()

    def benchmark_fn(self) -> None:
        if (
            self.target_model is None
            or self.draft_model is None
            or self.input_ids is None
            or self._input_token_view is None
            or self._output_ids is None
            or self._accept_prefix is None
            or self._accept_count_device is None
            or self._accept_count_host is None
            or self._accept_count_device_scalar is None
            or self._accept_count_host_scalar is None
            or self._accept_all_device is None
            or self._accept_all_host is None
            or self._accept_all_host_scalar is None
            or self._draft_next_values is None
            or self._target_next_values is None
            or self._target_next_tokens is None
            or self._matches is None
            or self._draft_logits is None
            or self._draft_logits_next is None
            or self._target_logits is None
            or self._view_counts != self._expected_view_counts
            or self._draft_forward_buffers is None
            or self._forward_buffer_counts != self._expected_forward_buffer_counts
        ):
            raise RuntimeError("Benchmark not initialized")

        wl = self.workload
        out = self._output_ids
        input_token_view = self._input_token_view
        draft_forward_into_prepared = self.draft_model.forward_into_prepared_unchecked
        target_forward_into_prepared = self.target_model.forward_into_prepared_unchecked
        draft_forward_buffers = self._draft_forward_buffers
        target_forward_buffers = self._target_forward_buffers
        output_step_views = self._output_step_views
        output_token_views = self._output_token_views
        output_write_views = self._output_write_views
        output_verify_views = self._output_verify_views
        target_logits_views = self._target_logits_views
        target_value_views = self._target_value_views
        target_token_views = self._target_token_views
        target_token_column_views = self._target_token_column_views
        match_views = self._match_views
        accept_prefix_views = self._accept_prefix_views
        accept_prefix_row_views = self._accept_prefix_row_views
        accept_count_device_scalar = self._accept_count_device_scalar
        accept_count_host_scalar = self._accept_count_host_scalar
        accept_all_device = self._accept_all_device
        accept_all_host = self._accept_all_host
        accept_all_host_scalar = self._accept_all_host_scalar
        speculation_step_ranges = self._speculation_step_ranges
        draft_next_values = self._draft_next_values
        draft_logits = self._draft_logits
        draft_logits_next = self._draft_logits_next
        output_token_views[0].copy_(input_token_view)

        draft_tokens = 0
        accepted_draft = 0
        rounds = 0

        with torch.inference_mode():
            pos = 0
            while pos < wl.total_tokens:
                rounds += 1
                remaining = wl.total_tokens - pos
                k = wl.speculative_k if remaining >= wl.speculative_k else remaining
                view_idx = k - 1
                speculation_step_range = speculation_step_ranges[view_idx]

                # Draft: propose k tokens sequentially.
                prev = output_step_views[pos]
                for j in speculation_step_range:
                    draft_forward_into_prepared(prev, draft_logits, draft_forward_buffers)
                    output_token = output_token_views[pos + j + 1]
                    torch.max(draft_logits_next, dim=-1, out=(draft_next_values, output_token))
                    prev = output_step_views[pos + j + 1]

                draft_tokens += int(k)

                # Verify: compute target predictions for the k steps in one call.
                draft_window = output_write_views[view_idx][pos]
                logits_t = target_forward_into_prepared(
                    output_verify_views[view_idx][pos],
                    target_logits_views[view_idx],
                    target_forward_buffers[view_idx],
                )
                target_values = target_value_views[view_idx]
                target_next = target_token_views[view_idx]
                torch.max(logits_t, dim=-1, out=(target_values, target_next))
                matches = match_views[view_idx]
                torch.eq(target_next, draft_window, out=matches)
                accept_prefix_length(matches, accept_count_device_scalar)
                accept_count_host_scalar.copy_(accept_count_device_scalar, non_blocking=False)
                accept_k = int(accept_count_host_scalar)

                if accept_k == k:
                    accepted_draft += int(k)
                    pos += k
                else:
                    if accept_k > 0:
                        accepted_draft += int(accept_k)
                    output_token_views[pos + accept_k + 1].copy_(
                        target_token_column_views[accept_k]
                    )
                    pos += accept_k + 1

        self.output = out
        if draft_tokens:
            acceptance_rate = accepted_draft / draft_tokens
        else:
            acceptance_rate = 0.0
        self._metrics["speculative.draft_tokens"] = float(draft_tokens)
        self._metrics["speculative.accepted_draft_tokens"] = float(accepted_draft)
        self._metrics["speculative.acceptance_rate_pct"] = acceptance_rate * 100.0
        self._metrics["speculative.rounds"] = float(rounds)

    def capture_verification_payload(self) -> None:
        if self.input_ids is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output, non_blocking=False)
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=self._verify_output_buffer,
            batch_size=1,
            parameter_count=self._payload_parameter_count,
            precision_flags={"bf16": False, "fp16": False, "fp8": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.target_model = None
        self.draft_model = None
        self.input_ids = None
        self._input_token_view = None
        self._output_ids = None
        self._output_step_views = []
        self._output_token_views = []
        self._output_write_views = []
        self._output_verify_views = []
        self._view_counts = ()
        self._expected_view_counts = ()
        self._accept_prefix = None
        self._accept_count_device = None
        self._accept_count_host = None
        self._accept_count_device_scalar = None
        self._accept_count_host_scalar = None
        self._accept_all_device = None
        self._accept_all_host = None
        self._accept_all_host_scalar = None
        self._draft_next_values = None
        self._target_next_values = None
        self._target_next_tokens = None
        self._matches = None
        self._draft_logits = None
        self._draft_logits_next = None
        self._target_logits = None
        self._target_logits_views = []
        self._target_value_views = []
        self._target_token_views = []
        self._target_token_column_views = []
        self._match_views = []
        self._accept_prefix_views = []
        self._accept_prefix_row_views = []
        self._speculation_step_ranges = []
        self._draft_forward_buffers = None
        self._target_forward_buffers = []
        self._forward_buffer_counts = ()
        self._expected_forward_buffer_counts = ()
        self.output = None
        self._verify_output_buffer = None
        for key in self._metrics:
            self._metrics[key] = 0.0
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
