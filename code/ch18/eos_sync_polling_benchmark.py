"""Chapter-local EOS benchmark with setup-owned decode workspaces."""

from __future__ import annotations

import torch

from core.benchmark.hf_decoder_cache_benchmark import (
    HFDecoderCacheBenchmark,
    HFDecoderCacheConfig,
)


class PreallocatedEOSSyncPollingBenchmark(HFDecoderCacheBenchmark):
    """Keep the EOS policy lesson while excluding allocator fallback from timing."""

    def __init__(self, cfg: HFDecoderCacheConfig) -> None:
        super().__init__(cfg)

    def setup(self) -> None:
        super().setup()
        self._decode_max_value_buffer = torch.empty(
            self.cfg.batch_size,
            dtype=self.dtype,
            device=self.device,
        )

    def _next_token_from_logits(self, logits_last: torch.Tensor) -> torch.Tensor:
        token_ids = self._decode_next_token_buffer
        values = self._decode_max_value_buffer
        if token_ids is None or values is None:
            raise RuntimeError("setup() must initialize decode selection buffers")
        if (
            token_ids.device != logits_last.device
            or values.device != logits_last.device
            or values.dtype != logits_last.dtype
            or token_ids.numel() != logits_last.size(0)
            or values.numel() != logits_last.size(0)
        ):
            raise RuntimeError("Decode selection buffers do not match the logits workload")
        torch.max(logits_last, dim=-1, out=(values, token_ids))
        return token_ids
