"""Optimized: reuse setup-time prefill state for a static prefix."""

from __future__ import annotations

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata


def get_benchmark() -> DecodeBenchmark:
    """Prefix-cache-style path that skips recurring static-prefix prefill compute."""
    cfg = DecodeConfig(
        batch_size=64,
        prompt_tokens=2048,
        decode_tokens=1,
        prefetch_batches=1,
        host_payload_mb=0,
        hidden_size=256,
        use_pinned_host=True,
        use_copy_stream=False,
        use_compute_stream=False,
        use_cuda_graphs=False,
        use_torch_compile=False,
        reuse_device_prompt=True,
        reuse_prefill_state=True,
        label="optimized_decode_prefix_state_cache",
        iterations=10,
        warmup=8,
    )
    return attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)
