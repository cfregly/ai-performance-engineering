"""Shared Medusa/EAGLE-style speculative decoding benchmarks for Chapter 15."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch

from ch15.speculative_decoding_common import (
    SpecDecodingWorkload,
    TokenMLP,
    build_draft_from_target,
    resolve_speculative_decode_dtype,
    scale_tail_dims_,
)
from core.benchmark.metrics import compute_speculative_decoding_metrics
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


@dataclass(frozen=True)
class SpeculativeFamilyProfile:
    name: str
    draft_hidden: int
    speculative_k: int
    tail_scale: float
    rejection_period: int
    rejection_offset: int
    perturb_stride: int


PROFILE_MEDUSA = SpeculativeFamilyProfile(
    name="medusa",
    draft_hidden=1536,
    speculative_k=8,
    tail_scale=0.08,
    rejection_period=3,
    rejection_offset=1,
    perturb_stride=17,
)

PROFILE_EAGLE = SpeculativeFamilyProfile(
    name="eagle",
    draft_hidden=2048,
    speculative_k=6,
    tail_scale=0.04,
    rejection_period=7,
    rejection_offset=2,
    perturb_stride=5,
)


def _family_workload(profile: Optional[SpeculativeFamilyProfile]) -> SpecDecodingWorkload:
    dtype = resolve_speculative_decode_dtype()
    if profile is None:
        return SpecDecodingWorkload(
            vocab_size=32000,
            target_hidden=4096,
            target_layers=2,
            draft_hidden=1024,
            speculative_k=1,
            total_tokens=192,
            tail_scale=0.02,
            dtype=dtype,
        )
    return SpecDecodingWorkload(
        vocab_size=32000,
        target_hidden=4096,
        target_layers=2,
        draft_hidden=profile.draft_hidden,
        speculative_k=profile.speculative_k,
        total_tokens=192,
        tail_scale=profile.tail_scale,
        dtype=dtype,
    )


class MedusaEagleSpeculativeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline greedy decode or explicit Medusa/EAGLE-style speculative variants."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def __init__(self, *, variant: str, label: str) -> None:
        super().__init__()
        self.variant = str(variant).strip().lower()
        self.label = label
        self.profile = None
        if self.variant == "medusa":
            self.profile = PROFILE_MEDUSA
        elif self.variant == "eagle":
            self.profile = PROFILE_EAGLE
        elif self.variant != "baseline":
            raise ValueError(f"Unsupported speculative family variant '{variant}'")

        self.workload = _family_workload(self.profile)
        self.target_model: Optional[TokenMLP] = None
        self.draft_model: Optional[TokenMLP] = None
        self.input_ids: Optional[torch.Tensor] = None
        self._output_ids: Optional[torch.Tensor] = None
        self._accept_prefix: Optional[torch.Tensor] = None
        self._accept_count_device: Optional[torch.Tensor] = None
        self._accept_count_host: Optional[torch.Tensor] = None
        self._accept_count_device_scalar: Optional[torch.Tensor] = None
        self._accept_count_host_scalar: Optional[torch.Tensor] = None
        self._accept_all_device: Optional[torch.Tensor] = None
        self._accept_all_host: Optional[torch.Tensor] = None
        self._accept_all_host_scalar: Optional[torch.Tensor] = None
        self._greedy_next_values: Optional[torch.Tensor] = None
        self._draft_head_offsets: Optional[torch.Tensor] = None
        self._draft_seed_buffer: Optional[torch.Tensor] = None
        self._draft_block_values: Optional[torch.Tensor] = None
        self._target_next_values: Optional[torch.Tensor] = None
        self._target_next_tokens: Optional[torch.Tensor] = None
        self._matches: Optional[torch.Tensor] = None
        self._greedy_logits: Optional[torch.Tensor] = None
        self._draft_logits: Optional[torch.Tensor] = None
        self._target_logits: Optional[torch.Tensor] = None
        self._greedy_forward_buffers: Optional[tuple[torch.Tensor, torch.Tensor]] = None
        self._draft_forward_buffers: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._target_forward_buffers: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._output_step_views: list[torch.Tensor] = []
        self._output_token_views: list[torch.Tensor] = []
        self._output_write_views: list[list[torch.Tensor]] = []
        self._output_verify_views: list[list[torch.Tensor]] = []
        self._draft_head_offset_views: list[torch.Tensor] = []
        self._draft_seed_views: list[torch.Tensor] = []
        self._draft_logits_views: list[torch.Tensor] = []
        self._draft_block_value_views: list[list[torch.Tensor]] = []
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
        self._forward_buffer_counts: tuple[int, ...] = ()
        self._expected_forward_buffer_counts: tuple[int, ...] = ()
        self._verify_summary_device: Optional[torch.Tensor] = None
        self._verify_summary_host: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._metrics: Dict[str, float] = {}
        self._payload_parameter_count = 0

        tokens = float(self.workload.total_tokens)
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )

    @staticmethod
    def _allocate_verify_summary_host() -> torch.Tensor:
        try:
            return torch.empty(3, dtype=torch.float32, device="cpu", pin_memory=True)
        except RuntimeError:
            return torch.empty(3, dtype=torch.float32, device="cpu")

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
        if self.profile is not None:
            scale_tail_dims_(self.target_model, wl.draft_hidden, wl.tail_scale)
        self._payload_parameter_count = sum(p.numel() for p in self.target_model.parameters())

        self.input_ids = torch.randint(0, wl.vocab_size, (1, 1), device=self.device, dtype=torch.int64)
        self._output_ids = torch.empty((1, wl.total_tokens + 1), device=self.device, dtype=torch.int64)
        self._output_step_views = [
            self._output_ids[:, token_idx : token_idx + 1] for token_idx in range(wl.total_tokens + 1)
        ]
        self._output_token_views = [
            self._output_ids[:, token_idx] for token_idx in range(wl.total_tokens + 1)
        ]
        # torch.max requires value and index outputs with matching strides.
        self._greedy_next_values = torch.empty_strided(
            (1,),
            self._output_token_views[0].stride(),
            device=self.device,
            dtype=wl.dtype,
        )
        self._greedy_logits = torch.empty((1, 1, wl.vocab_size), device=self.device, dtype=wl.dtype)
        self._verify_summary_device = torch.empty(3, device=self.device, dtype=torch.float32)
        self._verify_summary_host = self._allocate_verify_summary_host()
        self.output = None
        self._metrics = {}

        if self.profile is None:
            self.draft_model = None
            self._accept_prefix = None
            self._accept_count_device = None
            self._accept_count_host = None
            self._accept_count_device_scalar = None
            self._accept_count_host_scalar = None
            self._accept_all_device = None
            self._accept_all_host = None
            self._accept_all_host_scalar = None
            self._draft_head_offsets = None
            self._draft_seed_buffer = None
            self._draft_block_values = None
            self._target_next_values = None
            self._target_next_tokens = None
            self._matches = None
            self._draft_logits = None
            self._target_logits = None
            self._greedy_forward_buffers = self.target_model.prepare_forward_buffers(
                1,
                device=self.device,
                dtype=wl.dtype,
            )
            self._draft_forward_buffers = []
            self._target_forward_buffers = []
            self._output_write_views = []
            self._output_verify_views = []
            self._draft_head_offset_views = []
            self._draft_seed_views = []
            self._draft_logits_views = []
            self._draft_block_value_views = []
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
            self._forward_buffer_counts = (1,)
            self._expected_forward_buffer_counts = (1,)
            return

        self.draft_model = build_draft_from_target(self.target_model, wl.draft_hidden)
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
        self._draft_head_offsets = torch.arange(wl.speculative_k, device=self.device, dtype=torch.int64).view(1, -1)
        self._draft_seed_buffer = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.int64)
        self._draft_block_values = torch.empty((1, wl.total_tokens + 1), device=self.device, dtype=wl.dtype)
        self._target_next_values = torch.empty((1, wl.speculative_k), device=self.device, dtype=wl.dtype)
        self._target_next_tokens = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.long)
        self._matches = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.bool)
        self._draft_logits = torch.empty((1, wl.speculative_k, wl.vocab_size), device=self.device, dtype=wl.dtype)
        self._target_logits = torch.empty((1, wl.speculative_k, wl.vocab_size), device=self.device, dtype=wl.dtype)
        self._draft_forward_buffers = [
            self.draft_model.prepare_forward_buffers(k, device=self.device, dtype=wl.dtype)
            for k in range(1, wl.speculative_k + 1)
        ]
        self._target_forward_buffers = [
            self.target_model.prepare_forward_buffers(k, device=self.device, dtype=wl.dtype)
            for k in range(1, wl.speculative_k + 1)
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
        self._draft_head_offset_views = [
            self._draft_head_offsets[:, :k] for k in range(1, wl.speculative_k + 1)
        ]
        self._draft_seed_views = [
            self._draft_seed_buffer[:, :k] for k in range(1, wl.speculative_k + 1)
        ]
        self._draft_logits_views = [
            self._draft_logits[:, :k] for k in range(1, wl.speculative_k + 1)
        ]
        self._draft_block_value_views = [
            [
                self._draft_block_values[:, start + 1 : start + length + 1]
                for start in range(wl.total_tokens - length + 1)
            ]
            for length in range(1, wl.speculative_k + 1)
        ]
        self._target_logits_views = [
            self._target_logits[:, :k] for k in range(1, wl.speculative_k + 1)
        ]
        self._target_value_views = [
            self._target_next_values[:, :k] for k in range(1, wl.speculative_k + 1)
        ]
        self._target_token_views = [
            self._target_next_tokens[:, :k] for k in range(1, wl.speculative_k + 1)
        ]
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
            len(self._draft_head_offset_views),
            len(self._draft_seed_views),
            len(self._draft_logits_views),
            len(self._draft_block_value_views),
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
            wl.speculative_k,
            wl.speculative_k,
            wl.speculative_k,
            wl.speculative_k,
        )
        self._forward_buffer_counts = (
            len(self._draft_forward_buffers),
            len(self._target_forward_buffers),
        )
        self._expected_forward_buffer_counts = (
            wl.speculative_k,
            wl.speculative_k,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.profile is None:
            self._run_greedy_decode()
            return
        self._run_family_speculative_decode()

    def _run_greedy_decode(self) -> None:
        if (
            self.target_model is None
            or self.input_ids is None
            or self._output_ids is None
            or self._greedy_next_values is None
            or self._greedy_logits is None
            or self._greedy_forward_buffers is None
            or self._view_counts != self._expected_view_counts
            or self._forward_buffer_counts != self._expected_forward_buffer_counts
        ):
            raise RuntimeError("Benchmark not initialized")

        wl = self.workload
        out = self._output_ids
        target_forward_into_prepared = self.target_model.forward_into_prepared_unchecked
        output_step_views = self._output_step_views
        output_token_views = self._output_token_views
        greedy_logits = self._greedy_logits
        greedy_next_values = self._greedy_next_values
        greedy_forward_buffers = self._greedy_forward_buffers
        output_token_views[0].copy_(self.input_ids[:, 0])

        with self._nvtx_range(self.label):
            with torch.inference_mode():
                for t in range(wl.total_tokens):
                    logits = target_forward_into_prepared(
                        output_step_views[t],
                        greedy_logits,
                        greedy_forward_buffers,
                    )
                    output_token = output_token_views[t + 1]
                    torch.max(logits[:, 0, :], dim=-1, out=(greedy_next_values, output_token))

        self.output = out
        self._metrics = {
            "speculative.family_baseline": 1.0,
            "speculative.family_medusa": 0.0,
            "speculative.family_eagle": 0.0,
        }

    def _should_perturb(self, round_idx: int, draft_idx: int) -> bool:
        if self.profile is None:
            return False
        return ((round_idx + draft_idx + self.profile.rejection_offset) % self.profile.rejection_period) == 0

    def _perturb_token(self, token: torch.Tensor, draft_idx: int) -> torch.Tensor:
        if self.profile is None:
            return token
        return (token + (self.profile.perturb_stride * (draft_idx + 1))) % self.workload.vocab_size

    def _draft_seed_tokens(self, prev: torch.Tensor, k: int, round_idx: int) -> torch.Tensor:
        if self.profile is None or self._draft_head_offsets is None or self._draft_seed_buffer is None:
            raise RuntimeError("Draft seeds are only valid for Medusa/EAGLE profiles")
        view_idx = k - 1
        head_offsets = self._draft_head_offset_views[view_idx]
        seed_tokens = self._draft_seed_views[view_idx]
        round_offset = int(round_idx % self.profile.rejection_period)
        torch.add(prev.expand(-1, k), head_offsets, out=seed_tokens)
        seed_tokens.add_(round_offset)
        seed_tokens.remainder_(self.workload.vocab_size)
        return seed_tokens

    def _run_family_speculative_decode(self) -> None:
        if (
            self.target_model is None
            or self.draft_model is None
            or self.input_ids is None
            or self._output_ids is None
            or self._accept_prefix is None
            or self._accept_count_device is None
            or self._accept_count_host is None
            or self._accept_count_device_scalar is None
            or self._accept_count_host_scalar is None
            or self._accept_all_device is None
            or self._accept_all_host is None
            or self._accept_all_host_scalar is None
            or self._draft_block_values is None
            or self._target_next_values is None
            or self._target_next_tokens is None
            or self._matches is None
            or self._draft_logits is None
            or self._target_logits is None
            or self.profile is None
            or self._forward_buffer_counts != self._expected_forward_buffer_counts
            or self._view_counts != self._expected_view_counts
        ):
            raise RuntimeError("Benchmark not initialized")

        wl = self.workload
        out = self._output_ids
        draft_forward_into_prepared = self.draft_model.forward_into_prepared_unchecked
        target_forward_into_prepared = self.target_model.forward_into_prepared_unchecked
        draft_forward_buffers = self._draft_forward_buffers
        target_forward_buffers = self._target_forward_buffers
        output_step_views = self._output_step_views
        output_token_views = self._output_token_views
        output_write_views = self._output_write_views
        output_verify_views = self._output_verify_views
        draft_logits_views = self._draft_logits_views
        draft_block_value_views = self._draft_block_value_views
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
        output_token_views[0].copy_(self.input_ids[:, 0])

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

                    draft_seed = self._draft_seed_tokens(output_step_views[pos], k, rounds)
                    logits_d = draft_forward_into_prepared(
                        draft_seed,
                        draft_logits_views[view_idx],
                        draft_forward_buffers[view_idx],
                    )
                    draft_values = draft_block_value_views[view_idx][pos]
                    draft_window = output_write_views[view_idx][pos]
                    torch.max(logits_d, dim=-1, out=(draft_values, draft_window))
                    for j in speculation_step_range:
                        if self._should_perturb(rounds, j):
                            output_token = output_token_views[pos + j + 1]
                            output_token.add_(self.profile.perturb_stride * (j + 1))
                            output_token.remainder_(wl.vocab_size)

                    draft_tokens += int(k)

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
                    torch.all(matches, dim=-1, out=accept_all_device)
                    accept_all_host.copy_(accept_all_device, non_blocking=False)
                    if bool(accept_all_host_scalar):
                        accept_k = k
                    else:
                        accept_prefix = accept_prefix_views[view_idx]
                        torch.cumprod(matches, dim=-1, dtype=torch.int64, out=accept_prefix)
                        torch.sum(
                            accept_prefix_row_views[view_idx],
                            dim=0,
                            out=accept_count_device_scalar,
                        )
                        accept_count_host_scalar.copy_(
                            accept_count_device_scalar,
                            non_blocking=False,
                        )
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
        metrics = compute_speculative_decoding_metrics(
            draft_tokens=draft_tokens,
            accepted_tokens=accepted_draft,
            draft_time_ms=None,
            verify_time_ms=None,
            num_rounds=rounds,
        )
        metrics.update(
            {
                "speculative.family_baseline": 0.0,
                "speculative.family_medusa": 1.0 if self.variant == "medusa" else 0.0,
                "speculative.family_eagle": 1.0 if self.variant == "eagle" else 0.0,
                "speculative.accepted_draft_tokens": float(accepted_draft),
                "speculative.rounds": float(rounds),
                "speculative.acceptance_target_pct": ((self.profile.rejection_period - 1) / self.profile.rejection_period) * 100.0,
                "speculative.draft_branching_factor": float(wl.speculative_k),
            }
        )
        self._metrics = metrics

    def capture_verification_payload(self) -> None:
        if self.input_ids is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._verify_summary_device is None or self._verify_summary_host is None:
            raise RuntimeError("setup() must initialize verification summary buffers")
        in_vocab = ((self.output >= 0) & (self.output < self.workload.vocab_size)).sum()
        summary = self._verify_summary_device
        summary[0].copy_(self.input_ids[0, 0])
        summary[1].fill_(float(self.output.shape[-1]))
        summary[2].copy_(in_vocab)
        self._verify_summary_host.copy_(summary, non_blocking=False)
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=self._verify_summary_host,
            batch_size=1,
            parameter_count=self._payload_parameter_count,
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.target_model = None
        self.draft_model = None
        self.input_ids = None
        self._output_ids = None
        self._accept_prefix = None
        self._accept_count_device = None
        self._accept_count_host = None
        self._accept_count_device_scalar = None
        self._accept_count_host_scalar = None
        self._accept_all_device = None
        self._accept_all_host = None
        self._accept_all_host_scalar = None
        self._greedy_next_values = None
        self._draft_head_offsets = None
        self._draft_seed_buffer = None
        self._draft_block_values = None
        self._target_next_values = None
        self._target_next_tokens = None
        self._matches = None
        self._greedy_logits = None
        self._draft_logits = None
        self._target_logits = None
        self._greedy_forward_buffers = None
        self._draft_forward_buffers = []
        self._target_forward_buffers = []
        self._output_step_views = []
        self._output_token_views = []
        self._output_write_views = []
        self._output_verify_views = []
        self._draft_head_offset_views = []
        self._draft_seed_views = []
        self._draft_logits_views = []
        self._draft_block_value_views = []
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
        self._forward_buffer_counts = ()
        self._expected_forward_buffer_counts = ()
        self._verify_summary_device = None
        self._verify_summary_host = None
        self.output = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return dict(self._metrics)

    def get_optimization_goal(self) -> str:
        """Track Medusa/EAGLE as an explicit throughput/acceptance tradeoff study."""
        return "throughput"

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if self.output.shape[-1] != self.workload.total_tokens + 1:
            return "Unexpected output shape"
        if torch.any((self.output < 0) | (self.output >= self.workload.vocab_size)):
            return "Output contains out-of-vocabulary token ids"
        family_key = f"speculative.family_{self.variant}"
        if self._metrics.get(family_key, 0.0) != 1.0:
            return f"Missing family marker for {self.variant}"
        if self.profile is not None:
            accepted = float(self._metrics.get("speculative.accepted_draft_tokens", 0.0))
            drafted = float(self._metrics.get("speculative.draft_tokens", 0.0))
            rounds = float(self._metrics.get("speculative.rounds", 0.0))
            acceptance = float(self._metrics.get("speculative.acceptance_rate_pct", -1.0))
            if drafted <= 0.0 or rounds <= 0.0:
                return "Speculative metrics missing draft/round counts"
            if accepted > drafted:
                return "Accepted draft tokens exceed drafted tokens"
            if acceptance < 0.0 or acceptance > 100.0:
                return "Acceptance rate is outside [0, 100]"
        return None
