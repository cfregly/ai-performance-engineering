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
        self._draft_ids: Optional[torch.Tensor] = None
        self._verify_prev: Optional[torch.Tensor] = None
        self._accept_prefix: Optional[torch.Tensor] = None
        self._accept_count: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._metrics: Dict[str, float] = {}

        tokens = float(self.workload.total_tokens)
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
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
        if self.profile is not None:
            scale_tail_dims_(self.target_model, wl.draft_hidden, wl.tail_scale)

        self.input_ids = torch.randint(0, wl.vocab_size, (1, 1), device=self.device, dtype=torch.int64)
        self._output_ids = torch.empty((1, wl.total_tokens + 1), device=self.device, dtype=torch.int64)
        self.output = None
        self._metrics = {}

        if self.profile is None:
            self.draft_model = None
            self._draft_ids = None
            self._verify_prev = None
            self._accept_prefix = None
            self._accept_count = None
            return

        self.draft_model = build_draft_from_target(self.target_model, wl.draft_hidden)
        self._draft_ids = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.int64)
        self._verify_prev = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.int64)
        self._accept_prefix = torch.empty(wl.speculative_k, device=self.device, dtype=torch.int32)
        self._accept_count = torch.empty((), device=self.device, dtype=torch.int32)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.profile is None:
            self._run_greedy_decode()
            return
        self._run_family_speculative_decode()

    def _run_greedy_decode(self) -> None:
        if self.target_model is None or self.input_ids is None or self._output_ids is None:
            raise RuntimeError("Benchmark not initialized")

        wl = self.workload
        out = self._output_ids
        out[:, 0] = self.input_ids[:, 0]

        with self._nvtx_range(self.label):
            with torch.no_grad():
                for t in range(wl.total_tokens):
                    logits = self.target_model(out[:, t : t + 1])
                    out[:, t + 1] = logits[:, 0, :].argmax(dim=-1)

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
        if self.profile is None:
            raise RuntimeError("Draft seeds are only valid for Medusa/EAGLE profiles")
        head_offsets = torch.arange(k, device=prev.device, dtype=torch.int64).view(1, k)
        round_offset = int(round_idx % self.profile.rejection_period)
        return (prev.expand(-1, k) + head_offsets + round_offset) % self.workload.vocab_size

    def _run_family_speculative_decode(self) -> None:
        if (
            self.target_model is None
            or self.draft_model is None
            or self.input_ids is None
            or self._output_ids is None
            or self._draft_ids is None
            or self._verify_prev is None
            or self._accept_prefix is None
            or self._accept_count is None
            or self.profile is None
        ):
            raise RuntimeError("Benchmark not initialized")

        wl = self.workload
        out = self._output_ids
        out[:, 0] = self.input_ids[:, 0]

        draft_tokens = 0
        accepted_draft = 0
        rounds = 0

        with self._nvtx_range(self.label):
            with torch.no_grad():
                pos = 0
                while pos < wl.total_tokens:
                    rounds += 1
                    remaining = wl.total_tokens - pos
                    k = wl.speculative_k if remaining >= wl.speculative_k else remaining

                    draft_seed = self._draft_seed_tokens(out[:, pos : pos + 1], k, rounds)
                    logits_d = self.draft_model(draft_seed)
                    draft_block = logits_d.argmax(dim=-1)
                    for j in range(k):
                        next_d = draft_block[:, j]
                        if self._should_perturb(rounds, j):
                            next_d = self._perturb_token(next_d, j)
                        self._draft_ids[:, j] = next_d

                    draft_tokens += int(k)

                    self._verify_prev[:, 0] = out[:, pos]
                    if k > 1:
                        self._verify_prev[:, 1:k] = self._draft_ids[:, : k - 1]

                    logits_t = self.target_model(self._verify_prev[:, :k])
                    target_next = logits_t.argmax(dim=-1)

                    matches = target_next.eq(self._draft_ids[:, :k])
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
        parameter_count = 0
        if self.target_model is not None:
            parameter_count = sum(p.numel() for p in self.target_model.parameters())
        in_vocab = ((self.output >= 0) & (self.output < self.workload.vocab_size)).sum()
        verify_summary = torch.stack(
            (
                self.input_ids[0, 0].to(torch.float32),
                torch.full((), float(self.output.shape[-1]), device=self.output.device, dtype=torch.float32),
                in_vocab.to(torch.float32),
            )
        ).detach().cpu()
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=verify_summary,
            batch_size=1,
            parameter_count=int(parameter_count),
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
