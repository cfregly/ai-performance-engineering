"""Optimized EOS polling in decode loop: async stream-polled host checks.

Inspired by:
https://chaimrand.medium.com/optimizing-token-generation-in-pytorch-decoder-models-8e63b5a5fc80
"""

from __future__ import annotations

from core.benchmark.hf_decoder_cache_benchmark import (  # noqa: E402
    HFDecoderCacheBenchmark,
    HFDecoderCacheConfig,
    attach_benchmark_metadata,
)
from ch18.eos_sync_polling_benchmark import PreallocatedEOSSyncPollingBenchmark


def get_benchmark() -> HFDecoderCacheBenchmark:
    cfg = HFDecoderCacheConfig(
        batch_size=4,
        prompt_tokens=128,
        decode_tokens=192,
        hidden_size=192,
        num_layers=3,
        num_heads=3,
        cache_mode="static",
        compile_decode_step=True,
        eos_sync_mode="async_streamed",
        # Poll less frequently to amortize host synchronization overhead.
        eos_poll_interval=96,
        stop_on_all_done=False,
        iterations=8,
        warmup=8,
        label="optimized_eos_sync_polling",
    )
    benchmark = PreallocatedEOSSyncPollingBenchmark(cfg)
    return attach_benchmark_metadata(benchmark, __file__)  # type: ignore[return-value]

