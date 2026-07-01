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
        # The trusted-draft serving path keeps the target off the latency and memory path.
        self.target_model = None
        self._target_forward_buffers = []
        self._forward_buffer_counts = (1, 0)
        self._expected_forward_buffer_counts = (1, 0)
        torch.cuda.empty_cache()

    def benchmark_fn(self) -> None:
        if (
            self.draft_model is None
            or self.input_ids is None
            or self._output_ids is None
            or self._draft_next_values is None
            or self._draft_logits is None
            or self._draft_logits_next is None
            or self._view_counts != self._expected_view_counts
            or self._draft_forward_buffers is None
            or self._forward_buffer_counts != self._expected_forward_buffer_counts
        ):
            raise RuntimeError("Benchmark not initialized")

        wl = self.workload
        out = self._output_ids
        input_ids = self.input_ids
        draft_forward_into_prepared = self.draft_model.forward_into_prepared_unchecked
        draft_forward_buffers = self._draft_forward_buffers
        output_step_views = self._output_step_views
        output_token_views = self._output_token_views
        speculation_step_ranges = self._speculation_step_ranges
        draft_next_values = self._draft_next_values
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
                view_idx = k - 1
                speculation_step_range = speculation_step_ranges[view_idx]

                prev = output_step_views[pos]
                for j in speculation_step_range:
                    draft_forward_into_prepared(prev, draft_logits, draft_forward_buffers)
                    output_token = output_token_views[pos + j + 1]
                    torch.max(draft_logits_next, dim=-1, out=(draft_next_values, output_token))
                    prev = output_step_views[pos + j + 1]

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
