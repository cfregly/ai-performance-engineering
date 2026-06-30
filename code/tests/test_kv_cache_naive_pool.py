from __future__ import annotations

import torch

from ch13.optimized_kv_cache_naive_pool import OptimizedKVCache, SimpleAttentionLayer


def test_optimized_kv_cache_reuses_released_slot() -> None:
    cache = OptimizedKVCache(
        max_seq_len=4,
        batch_size=1,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    cache.allocate("first")
    cache.allocate("second")
    first = cache.allocated_caches["first"]
    second = cache.allocated_caches["second"]
    cache.free("first")
    cache.allocate("third")
    reused = cache.allocated_caches["third"]

    assert {first, second} == {0, 1}
    assert reused == first


def test_simple_attention_layer_writes_only_active_prefix() -> None:
    layer = SimpleAttentionLayer(hidden_dim=8, num_heads=2, head_dim=4, dtype=torch.float32)
    cache = OptimizedKVCache(
        max_seq_len=4,
        batch_size=1,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )

    cache.allocate("req")
    cache_pair = cache.cache_pool[cache.allocated_caches["req"]][0]
    cache_pair[0].zero_()
    cache_pair[1].zero_()

    out = layer(torch.randn(1, 1, 8), cache, "req", layer_idx=0, cache_pos=0)

    assert out.shape == (1, 1, 8)
    assert torch.count_nonzero(cache_pair[0][:, :, 0, :]).item() > 0
    assert torch.count_nonzero(cache_pair[0][:, :, 1:, :]).item() == 0
    assert torch.count_nonzero(cache_pair[1][:, :, 0, :]).item() > 0
    assert torch.count_nonzero(cache_pair[1][:, :, 1:, :]).item() == 0


def test_simple_attention_layer_reuses_qkv_projection_buffer_in_inference() -> None:
    layer = SimpleAttentionLayer(hidden_dim=8, num_heads=2, head_dim=4, dtype=torch.float32)
    cache = OptimizedKVCache(
        max_seq_len=4,
        batch_size=1,
        num_layers=1,
        num_heads=2,
        head_dim=4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    cache.allocate("req")

    with torch.inference_mode():
        first = layer(torch.randn(1, 1, 8), cache, "req", layer_idx=0, cache_pos=0)
        first_ptr = layer._qkv_buffer.data_ptr()
        second = layer(torch.randn(1, 1, 8), cache, "req", layer_idx=0, cache_pos=1)

    assert first.shape == (1, 1, 8)
    assert second.shape == (1, 1, 8)
    assert layer._qkv_buffer.data_ptr() == first_ptr
    assert layer._qkv_weight_t is not None
