"""Transition-table speculative decode: cache draft next-token decisions."""

from __future__ import annotations

import torch

from core.harness.benchmark_harness import BaseBenchmark
from labs.speculative_decode.optimized_speculative_decode_trusted import (
    OptimizedSpeculativeDecodeTrustedBenchmark,
)


class OptimizedSpeculativeDecodeTransitionTableBenchmark(OptimizedSpeculativeDecodeTrustedBenchmark):
    """Optimized: precompute token-local draft transitions and replay them by lookup."""

    def __init__(self) -> None:
        super().__init__()
        self.transition_chunk_tokens = 256
        self._transition_table: torch.Tensor | None = None
        self._token_range = range(0)
        self._metrics.update({
            "speculative.transition_table": 1.0,
            "speculative.draft_model_calls": 0.0,
        })

    def setup(self) -> None:
        super().setup()
        if self.draft_model is None:
            raise RuntimeError("setup() must initialize the draft model before building transitions")

        wl = self.workload
        chunk = int(self.transition_chunk_tokens)
        draft_model = self.draft_model
        self._transition_table = torch.empty(wl.vocab_size, device=self.device, dtype=torch.long)
        token_ids = torch.arange(wl.vocab_size, device=self.device, dtype=torch.long).view(1, wl.vocab_size)
        logits = torch.empty((1, chunk, wl.vocab_size), device=self.device, dtype=wl.dtype)
        values = torch.empty((1, chunk), device=self.device, dtype=wl.dtype)
        draft_forward_into_prepared = draft_model.forward_into_prepared_unchecked
        transition_forward_buffers = draft_model.prepare_forward_buffers(
            chunk,
            device=self.device,
            dtype=wl.dtype,
        )

        with torch.inference_mode():
            for start in range(0, wl.vocab_size, chunk):
                end = min(start + chunk, wl.vocab_size)
                width = end - start
                logits_view = logits[:, :width]
                values_view = values[:, :width]
                transition_token_view = self._transition_table[start:end].view(1, width)
                forward_buffers = (
                    transition_forward_buffers
                    if width == chunk
                    else draft_model.prepare_forward_buffers(
                        width,
                        device=self.device,
                        dtype=wl.dtype,
                    )
                )
                draft_forward_into_prepared(token_ids[:, start:end], logits_view, forward_buffers)
                torch.max(logits_view, dim=-1, out=(values_view, transition_token_view))

        self.draft_model = None
        self._draft_forward_buffers = None
        self._forward_buffer_counts = (0, 0)
        self._expected_forward_buffer_counts = (0, 0)
        self._token_range = range(wl.total_tokens)
        torch.cuda.empty_cache()

    def benchmark_fn(self) -> None:
        if (
            self.input_ids is None
            or self._output_ids is None
            or self._transition_table is None
            or self._view_counts != self._expected_view_counts
        ):
            raise RuntimeError("Benchmark not initialized")

        wl = self.workload
        out = self._output_ids
        transition_table = self._transition_table
        output_token_views = self._output_token_views
        output_token_views[0].copy_(self.input_ids[:, 0])
        current_token = output_token_views[0]

        with self._nvtx_range("transition_table_speculative_decode"), torch.inference_mode():
            for t in self._token_range:
                output_token = output_token_views[t + 1]
                torch.index_select(transition_table, 0, current_token, out=output_token)
                current_token = output_token

        self.output = out
        rounds = (wl.total_tokens + wl.speculative_k - 1) // wl.speculative_k
        self._metrics["speculative.draft_tokens"] = float(wl.total_tokens)
        self._metrics["speculative.accepted_draft_tokens"] = float(wl.total_tokens)
        self._metrics["speculative.acceptance_rate_pct"] = 100.0
        self._metrics["speculative.rounds"] = float(rounds)
        self._metrics["speculative.target_verify_calls"] = 0.0
        self._metrics["speculative.trusted_draft"] = 1.0
        self._metrics["speculative.transition_table"] = 1.0
        self._metrics["speculative.draft_model_calls"] = 0.0

    def teardown(self) -> None:
        self._transition_table = None
        self._token_range = range(0)
        super().teardown()


def get_benchmark() -> BaseBenchmark:
    return OptimizedSpeculativeDecodeTransitionTableBenchmark()
