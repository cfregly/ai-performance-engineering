"""Baseline for guided decode: score the full vocabulary before candidate filtering."""

from __future__ import annotations

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata


def get_benchmark() -> DecodeBenchmark:
    """Full-vocab logits baseline for constrained candidate-token decoding."""
    cfg = DecodeConfig(
        batch_size=32,
        prompt_tokens=128,
        decode_tokens=64,
        prefetch_batches=1,
        host_payload_mb=0,
        hidden_size=512,
        vocab_size=131072,
        candidate_vocab_size=1,
        candidate_logits_only=False,
        use_pinned_host=True,
        use_copy_stream=False,
        use_compute_stream=False,
        use_cuda_graphs=False,
        use_torch_compile=False,
        label="baseline_decode_candidate_logits",
        iterations=10,
        warmup=8,
    )
    return attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)
