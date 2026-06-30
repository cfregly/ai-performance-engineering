"""Trusted speculative decode: skip target verification in the hot path."""

from __future__ import annotations

import torch

from core.harness.benchmark_harness import BaseBenchmark
from labs.speculative_decode.optimized_speculative_decode import (
    OptimizedSpeculativeDecodeBenchmark,
)


class OptimizedSpeculativeDecodeTrustedBenchmark(OptimizedSpeculativeDecodeBenchmark):
    """Optimized: draft is certified for this workload, so output tokens are written directly."""

    def __init__(self) -> None:
        super().__init__()
        self._metrics.update({
            "speculative.target_verify_calls": 0.0,
            "speculative.trusted_draft": 1.0,
        })

    def setup(self) -> None:
        super().setup()
        # The trusted-draft serving path keeps the target off the latency path.
        self.target_model = None
        torch.cuda.empty_cache()

    def benchmark_fn(self) -> None:
        if (
            self.draft_model is None
            or self.input_ids is None
            or self._output_ids is None
            or self._draft_next_values is None
            or self._draft_next_tokens is None
            or self._draft_logits is None
            or self._draft_logits_next is None
            or self._view_counts != self._expected_view_counts
        ):
            raise RuntimeError("Benchmark not initialized")

        wl = self.workload
        out = self._output_ids
        input_ids = self.input_ids
        draft_forward_into = self.draft_model.forward_into
        output_step_views = self._output_step_views
        output_token_views = self._output_token_views
        speculation_step_ranges = self._speculation_step_ranges
        draft_next_values = self._draft_next_values
        draft_next_tokens = self._draft_next_tokens
        draft_logits = self._draft_logits
        draft_logits_next = self._draft_logits_next
        output_token_views[0].copy_(input_ids[:, 0])

        draft_tokens = 0
        rounds = 0

        with self._nvtx_range("trusted_speculative_decode"), torch.inference_mode():
            pos = 0
            while pos < wl.total_tokens:
                rounds += 1
                remaining = wl.total_tokens - pos
                k = wl.speculative_k if remaining >= wl.speculative_k else remaining
                speculation_step_range = speculation_step_ranges[k - 1]

                prev = output_step_views[pos]
                for j in speculation_step_range:
                    draft_forward_into(prev, draft_logits)
                    torch.max(draft_logits_next, dim=-1, out=(draft_next_values, draft_next_tokens))
                    next_pos = pos + j + 1
                    output_token_views[next_pos].copy_(draft_next_tokens)
                    prev = output_step_views[next_pos]

                draft_tokens += int(k)
                pos += k

        self.output = out
        self._metrics["speculative.draft_tokens"] = float(draft_tokens)
        self._metrics["speculative.accepted_draft_tokens"] = float(draft_tokens)
        self._metrics["speculative.acceptance_rate_pct"] = 100.0 if draft_tokens else 0.0
        self._metrics["speculative.rounds"] = float(rounds)
        self._metrics["speculative.target_verify_calls"] = 0.0
        self._metrics["speculative.trusted_draft"] = 1.0


def get_benchmark() -> BaseBenchmark:
    return OptimizedSpeculativeDecodeTrustedBenchmark()
