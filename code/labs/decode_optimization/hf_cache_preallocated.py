"""Strict buffer adapter for the HF cache benchmark pair."""

from __future__ import annotations

import torch

from core.benchmark.hf_decoder_cache_benchmark import HFDecoderCacheBenchmark


class PreallocatedHFDecoderCacheBenchmark(HFDecoderCacheBenchmark):
    """Require next-token reduction buffers prepared during setup."""

    def setup(self) -> None:
        super().setup()
        self._decode_max_value_buffer = torch.empty(
            (self.cfg.batch_size,),
            device=self.device,
            dtype=self.dtype,
        )

    def _next_token_from_logits(self, logits_last: torch.Tensor) -> torch.Tensor:
        next_tokens = self._decode_next_token_buffer
        values = self._decode_max_value_buffer
        if next_tokens is None or values is None:
            raise RuntimeError("Next-token buffers are not initialized")
        if (
            values.device != logits_last.device
            or values.dtype != logits_last.dtype
            or values.shape != (logits_last.size(0),)
            or next_tokens.device != logits_last.device
            or next_tokens.shape != (logits_last.size(0),)
        ):
            raise RuntimeError("Preallocated next-token buffers do not match logits")
        torch.max(logits_last, dim=-1, out=(values, next_tokens))
        return next_tokens
