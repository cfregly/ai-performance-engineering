from __future__ import annotations

import inspect

import torch

from core.optimization.paged_attention import PagedAttentionConfig, PagedKVCache


def test_paged_kv_cache_uses_empty_pages_and_page_chunk_reads() -> None:
    allocate_source = inspect.getsource(PagedKVCache._allocate_page)
    get_kv_source = inspect.getsource(PagedKVCache.get_kv)

    assert "torch.empty(" in allocate_source
    assert "torch.zeros(" not in allocate_source
    assert "while pos < length:" in get_kv_source
    assert "offset : offset + 1" not in get_kv_source

    cfg = PagedAttentionConfig(
        batch_size=1,
        page_size=2,
        num_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    cache = PagedKVCache(cfg)

    expected_k = []
    expected_v = []
    for pos in range(5):
        k = torch.tensor([[[[float(pos), float(pos + 10)]]]])
        v = torch.tensor([[[[float(pos + 20), float(pos + 30)]]]])
        cache.write(pos, k, v)
        expected_k.append(k)
        expected_v.append(v)

    k_out, v_out = cache.get_kv(5)

    assert len(cache.k_pages) == 3
    torch.testing.assert_close(k_out, torch.cat(expected_k, dim=1))
    torch.testing.assert_close(v_out, torch.cat(expected_v, dim=1))
