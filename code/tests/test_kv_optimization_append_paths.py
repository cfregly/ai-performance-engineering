from __future__ import annotations

import pytest
import torch

from labs.kv_optimization.baseline_kv_standard import BaselineKVStandard
from labs.kv_optimization.optimized_kv_standard import OptimizedKVFP8Compressed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for KV append parity")
def test_baseline_active_layer_append_matches_per_layer_append() -> None:
    bench = BaselineKVStandard(
        batch_size=2,
        num_layers=5,
        num_heads=4,
        head_dim=16,
        max_seq_length=8,
        active_layers=3,
        num_decode_steps=2,
    )
    bench.setup()
    try:
        k = bench._generated_k_steps[0]
        v = bench._generated_v_steps[0]
        assert k is not None
        assert v is not None

        for layer_idx in range(bench.active_layers):
            bench.append_kv(layer_idx, k, v, pos=0)
        bench.append_active_layers(k, v, pos=1)

        per_layer = bench.kv_cache[:, : bench.active_layers, :, :, 0, :]
        batched = bench.kv_cache[:, : bench.active_layers, :, :, 1, :]
        assert torch.equal(per_layer, batched)
    finally:
        bench.teardown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for KV append parity")
def test_fp8_active_layer_append_matches_per_layer_append() -> None:
    bench = OptimizedKVFP8Compressed(
        batch_size=2,
        num_layers=5,
        num_heads=4,
        head_dim=16,
        max_seq_length=8,
        active_layers=3,
        num_decode_steps=2,
    )
    bench.setup()
    try:
        k = bench._generated_k_steps[0]
        v = bench._generated_v_steps[0]
        assert k is not None
        assert v is not None

        for layer_idx in range(bench.active_layers):
            bench.append_kv(layer_idx, k, v, pos=0)
        bench.append_active_layers(k, v, pos=1)

        per_layer = bench.kv_cache[:, : bench.active_layers, :, :, 0, :]
        batched = bench.kv_cache[:, : bench.active_layers, :, :, 1, :]
        assert torch.equal(per_layer, batched)
        assert torch.equal(
            bench.k_scales[: bench.active_layers, 0],
            bench.k_scales[: bench.active_layers, 1],
        )
        assert torch.equal(
            bench.v_scales[: bench.active_layers, 0],
            bench.v_scales[: bench.active_layers, 1],
        )
    finally:
        bench.teardown()
