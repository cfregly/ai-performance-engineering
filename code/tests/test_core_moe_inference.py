from __future__ import annotations

import torch

from core.optimization.moe_inference import MoEFeedForwardSortedDispatch


def test_sorted_dispatch_reuses_flat_token_id_cache_on_cpu() -> None:
    torch.manual_seed(123)
    layer = MoEFeedForwardSortedDispatch(
        hidden=8,
        ffn=16,
        num_experts=4,
        top_k=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    x = torch.randn(2, 3, 8)

    out1 = layer(x)
    cache1 = layer._token_ids_cache
    assert cache1.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]

    out2 = layer(x)
    cache2 = layer._token_ids_cache

    assert cache2.data_ptr() == cache1.data_ptr()
    torch.testing.assert_close(out2, out1)

    _ = layer(torch.randn(1, 2, 8))

    assert layer._token_ids_cache.numel() == 4
    assert layer._token_ids_cache.data_ptr() != cache1.data_ptr()
