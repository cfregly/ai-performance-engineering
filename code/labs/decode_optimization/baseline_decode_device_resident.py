"""Baseline for device-resident decode: stage prompt-side payload every iteration."""

from __future__ import annotations

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata


def get_benchmark() -> DecodeBenchmark:
    """Recurring H2D staging baseline for the device-resident serving pair."""
    cfg = DecodeConfig(
        batch_size=64,
        prompt_tokens=2048,
        decode_tokens=16,
        prefetch_batches=1,
        host_payload_mb=512,
        hidden_size=256,
        use_pinned_host=True,
        use_copy_stream=False,
        use_compute_stream=False,
        use_cuda_graphs=False,
        use_torch_compile=False,
        reuse_device_prompt=False,
        label="baseline_decode_device_resident",
        iterations=10,
        warmup=8,
    )
    return attach_benchmark_metadata(DecodeBenchmark(cfg), __file__)
