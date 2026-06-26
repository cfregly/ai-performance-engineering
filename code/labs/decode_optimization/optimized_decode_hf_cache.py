"""Optimized HF decode loop: static KV cache + compiled decode + batched EOS polling.

Inspired by:
https://chaimrand.medium.com/optimizing-token-generation-in-pytorch-decoder-models-8e63b5a5fc80
"""

from __future__ import annotations

from core.benchmark.hf_decoder_cache_benchmark import (
    HFDecoderCacheBenchmark,
    HFDecoderCacheConfig,
    attach_benchmark_metadata,
)


def get_benchmark() -> HFDecoderCacheBenchmark:
    cfg = HFDecoderCacheConfig(
        batch_size=4,
        prompt_tokens=128,
        decode_tokens=128,
        hidden_size=256,
        num_layers=4,
        num_heads=4,
        cache_mode="static",
        compile_decode_step=True,
        eos_sync_mode="async_streamed",
        eos_poll_interval=8,
        stop_on_all_done=False,
        iterations=6,
        warmup=6,
        label="optimized_decode_hf_cache",
    )
    return attach_benchmark_metadata(HFDecoderCacheBenchmark(cfg), __file__)  # type: ignore[return-value]

