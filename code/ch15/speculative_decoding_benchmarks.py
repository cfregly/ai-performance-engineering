"""Shared harness logic for Chapter 15 speculative decoding benchmarks."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.wrapper_utils import attach_benchmark_metadata as attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

from ch15.speculative_decoding_common import (
    TokenMLP,
    accept_prefix_length,
    build_draft_from_target,
    default_workload,
    resolve_speculative_decode_dtype,
    scale_tail_dims_,
)


class SpeculativeDecodingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Parameterized greedy or speculative decode benchmark."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def __init__(self, *, use_speculative: bool, label: str) -> None:
        super().__init__()
        self.use_speculative = bool(use_speculative)
        self.label = label

        self.workload = default_workload(dtype=resolve_speculative_decode_dtype())

        self.target_model: Optional[TokenMLP] = None
        self.draft_model: Optional[TokenMLP] = None
        self.input_ids: Optional[torch.Tensor] = None
        self._input_token_view: Optional[torch.Tensor] = None
        self._output_ids: Optional[torch.Tensor] = None
        self._output_step_views: list[torch.Tensor] = []
        self._output_token_views: list[torch.Tensor] = []
        self._output_write_views: list[list[torch.Tensor]] = []
        self._output_verify_views: list[list[torch.Tensor]] = []
        self._decode_token_range = range(0)
        self._accept_prefix: Optional[torch.Tensor] = None
        self._accept_count_device: Optional[torch.Tensor] = None
        self._accept_count_host: Optional[torch.Tensor] = None
        self._accept_count_device_scalar: Optional[torch.Tensor] = None
        self._accept_count_host_scalar: Optional[torch.Tensor] = None
        self._accept_all_device: Optional[torch.Tensor] = None
        self._accept_all_host: Optional[torch.Tensor] = None
        self._accept_all_host_scalar: Optional[torch.Tensor] = None
        self._greedy_next_values: Optional[torch.Tensor] = None
        self._greedy_logits: Optional[torch.Tensor] = None
        self._greedy_logits_next: Optional[torch.Tensor] = None
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
        self._view_counts: tuple[int, ...] = ()
        self._expected_view_counts: tuple[int, ...] = ()
        self._greedy_forward_buffers: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._draft_forward_buffers: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._target_forward_buffers: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._forward_buffer_counts: tuple[int, ...] = ()
        self._expected_forward_buffer_counts: tuple[int, ...] = ()
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._metrics: Dict[str, float] = {}
        self._payload_parameter_count = 0

        tokens = self.workload.total_tokens * 1.0
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

        self.input_ids = torch.randint(
            0,
            wl.vocab_size,
            (1, 1),
            device=self.device,
            dtype=torch.int64,
        )
        self._input_token_view = self.input_ids[:, 0]
        self._output_ids = torch.empty((1, wl.total_tokens + 1), device=self.device, dtype=torch.int64)
        self._verify_output_buffer = torch.empty_like(self._output_ids, dtype=torch.float32)
        self._output_step_views = [
            self._output_ids[:, token_idx : token_idx + 1] for token_idx in range(wl.total_tokens + 1)
        ]
        self._output_token_views = [
            self._output_ids[:, token_idx] for token_idx in range(wl.total_tokens + 1)
        ]
        self._decode_token_range = range(wl.total_tokens)
        # torch.max requires value and index outputs with matching strides.
        self._greedy_next_values = torch.empty_strided(
            (1,),
            self._output_token_views[0].stride(),
            device=self.device,
            dtype=wl.dtype,
        )
        self._greedy_logits = torch.empty((1, 1, wl.vocab_size), device=self.device, dtype=wl.dtype)
        self._greedy_logits_next = self._greedy_logits[:, 0, :]
        self._greedy_forward_buffers = self.target_model.prepare_forward_buffers(
            1,
            device=self.device,
            dtype=wl.dtype,
        )
        self.output = None
        self._metrics = {}

        if not self.use_speculative:
            self.draft_model = None
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
            self._output_write_views = []
            self._output_verify_views = []
            self._target_logits_views = []
            self._target_value_views = []
            self._target_token_views = []
            self._target_token_column_views = []
            self._match_views = []
            self._accept_prefix_views = []
            self._accept_prefix_row_views = []
            self._speculation_step_ranges = []
            self._view_counts = (
                len(self._output_step_views),
                len(self._output_token_views),
            )
            self._expected_view_counts = (
                wl.total_tokens + 1,
                wl.total_tokens + 1,
            )
            self._draft_forward_buffers = None
            self._target_forward_buffers = []
            self._forward_buffer_counts = (1, 0, 0)
            self._expected_forward_buffer_counts = (1, 0, 0)
            self._synchronize()
            return

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
        if self.target_model is None:
            raise RuntimeError("Target model not initialized")
        self.draft_model = build_draft_from_target(self.target_model, wl.draft_hidden)
        # Preserve the existing compile gate for environments that compare compiled module forwards.
        # The decode hot path below uses prepared helper methods so logits and hidden workspaces stay
        # caller-owned even when the modules are wrapped.
        import os as _os
        if _os.getenv("AISP_SPEC_COMPILE", "1").strip().lower() in {"1", "true", "on", "yes"}:
            try:
                import torch._dynamo as _dynamo

                _dynamo.reset()
            except Exception:
                pass
            self.target_model = torch.compile(self.target_model, mode="reduce-overhead", dynamic=False)
            self.draft_model = torch.compile(self.draft_model, mode="reduce-overhead", dynamic=False)
        self._draft_forward_buffers = self.draft_model.prepare_forward_buffers(
            1,
            device=self.device,
            dtype=wl.dtype,
        )
        self._target_forward_buffers = [
            self.target_model.prepare_forward_buffers(k, device=self.device, dtype=wl.dtype)
            for k in range(1, wl.speculative_k + 1)
        ]
        self._forward_buffer_counts = (
            1,
            1,
            len(self._target_forward_buffers),
        )
        self._expected_forward_buffer_counts = (
            1,
            1,
            wl.speculative_k,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.use_speculative:
            self._run_speculative_decode()
            return
        self._run_greedy_decode()

    def _run_greedy_decode(self) -> None:
        if (
            self.target_model is None
            or self.input_ids is None
            or self._input_token_view is None
            or self._output_ids is None
            or self._greedy_next_values is None
            or self._greedy_logits is None
            or self._greedy_logits_next is None
            or self._greedy_forward_buffers is None
            or self._view_counts != self._expected_view_counts
            or self._forward_buffer_counts != self._expected_forward_buffer_counts
        ):
            raise RuntimeError("Benchmark not initialized")

        out = self._output_ids
        input_token_view = self._input_token_view
        target_forward_into_prepared = self.target_model.forward_into_prepared_unchecked
        target_logits = self._greedy_logits
        target_logits_next = self._greedy_logits_next
        greedy_next_values = self._greedy_next_values
        greedy_forward_buffers = self._greedy_forward_buffers
        output_step_views = self._output_step_views
        output_token_views = self._output_token_views
        output_token_views[0].copy_(input_token_view)

        with self._nvtx_range(self.label):
            with torch.inference_mode():
                for t in self._decode_token_range:
                    target_forward_into_prepared(output_step_views[t], target_logits, greedy_forward_buffers)
                    output_token = output_token_views[t + 1]
                    torch.max(target_logits_next, dim=-1, out=(greedy_next_values, output_token))

        self.output = out

    def _run_speculative_decode(self) -> None:
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
            or self._draft_forward_buffers is None
            or self._forward_buffer_counts != self._expected_forward_buffer_counts
            or self._view_counts != self._expected_view_counts
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

        with self._nvtx_range(self.label):
            with torch.inference_mode():
                pos = 0
                while pos < wl.total_tokens:
                    rounds += 1
                    remaining = wl.total_tokens - pos
                    k = wl.speculative_k if remaining >= wl.speculative_k else remaining
                    view_idx = k - 1
                    speculation_step_range = speculation_step_ranges[view_idx]

                    prev = output_step_views[pos]
                    for j in speculation_step_range:
                        draft_forward_into_prepared(prev, draft_logits, draft_forward_buffers)
                        output_token = output_token_views[pos + j + 1]
                        torch.max(draft_logits_next, dim=-1, out=(draft_next_values, output_token))
                        prev = output_step_views[pos + j + 1]

                    draft_tokens += int(k)

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
        self._metrics = {
            "speculative.draft_tokens": float(draft_tokens),
            "speculative.accepted_draft_tokens": float(accepted_draft),
            "speculative.acceptance_rate_pct": (accepted_draft / max(draft_tokens, 1)) * 100.0,
            "speculative.rounds": float(rounds),
        }

    def capture_verification_payload(self) -> None:
        if self.input_ids is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output, non_blocking=False)
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=self._verify_output_buffer,
            batch_size=1,
            parameter_count=self._payload_parameter_count,
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
        self._decode_token_range = range(0)
        self._accept_prefix = None
        self._accept_count_device = None
        self._accept_count_host = None
        self._accept_count_device_scalar = None
        self._accept_count_host_scalar = None
        self._accept_all_device = None
        self._accept_all_host = None
        self._accept_all_host_scalar = None
        self._greedy_next_values = None
        self._greedy_logits = None
        self._greedy_logits_next = None
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
        self._view_counts = ()
        self._expected_view_counts = ()
        self._greedy_forward_buffers = None
        self._draft_forward_buffers = None
        self._target_forward_buffers = []
        self._forward_buffer_counts = ()
        self._expected_forward_buffer_counts = ()
        self.output = None
        self._verify_output_buffer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        metrics = dict(self._metrics) if self._metrics else {}
        metrics.update(
            {
                "speculative.dtype_bf16": 1.0 if self.workload.dtype == torch.bfloat16 else 0.0,
                "speculative.dtype_fp16": 1.0 if self.workload.dtype == torch.float16 else 0.0,
            }
        )
        return metrics

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if self.output.shape[-1] != self.workload.total_tokens + 1:
            return "Unexpected output shape"
        return None
