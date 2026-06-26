"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

import torch
from nanochat.engine import Engine, KVCache
from nanochat.gpt import apply_rotary_emb

def test_kv_cache_resize():
    """
    The KV cache was not resized correctly, more information here:
    https://github.com/karpathy/nanochat/pull/186
    This test reproduces the issue and will be merged alongside the fix.
    """

    batch_size = 2
    num_heads = 3
    seq_len = 4
    head_dim = 5
    num_layers = 6

    kv_cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=num_layers
    )

    # Insert a single token with a distinct fill value to all layers
    def insert_token(token_idx):
        for layer_idx in range(num_layers):
            k = torch.full((batch_size, num_heads, 1, head_dim), fill_value=float(token_idx), dtype=torch.float32)
            v = torch.full((batch_size, num_heads, 1, head_dim), fill_value=float(token_idx * 100), dtype=torch.float32)
            kv_cache.insert_kv(layer_idx, k, v)

    # Insert 4 tokens (fills the initial seq_len=4)
    for i in range(4):
        insert_token(i)

    # Record the original state of the cache
    original_cache = kv_cache.kv_cache.clone()
    original_seq_len = original_cache.shape[4]

    # Insert the 5th token, which will trigger a resize
    insert_token(4)
    # Verify that the cache actually resized
    new_seq_len = kv_cache.kv_cache.shape[4]
    assert new_seq_len > original_seq_len, f"Cache did not resize: original seq_len={original_seq_len}, new seq_len={new_seq_len}"

    # Verify that the original 4 tokens are still intact after resize
    for layer_idx in range(num_layers):
        for token_idx in range(4):
            # Check that resized cache matches expected values
            expected_k = float(token_idx)
            expected_v = float(token_idx * 100)
            actual_k = kv_cache.kv_cache[layer_idx, 0, :, :, token_idx, :]
            actual_v = kv_cache.kv_cache[layer_idx, 1, :, :, token_idx, :]
            assert (actual_k == expected_k).all(), f"Layer {layer_idx}, token {token_idx}: key corrupted, expected {expected_k}"
            assert (actual_v == expected_v).all(), f"Layer {layer_idx}, token {token_idx}: value corrupted, expected {expected_v}"
            # And that the original cache matches resized cache
            original_k = original_cache[layer_idx, 0, :, :, token_idx, :]
            original_v = original_cache[layer_idx, 1, :, :, token_idx, :]
            assert (actual_k == original_k).all(), f"Layer {layer_idx}, token {token_idx}: key doesn't match original"
            assert (actual_v == original_v).all(), f"Layer {layer_idx}, token {token_idx}: value doesn't match original"


def test_sample_batch_tokens_batches_uniform_sampling(monkeypatch):
    calls = []

    def fake_sample_next_token(logits, rng, temperature=1.0, top_k=None):
        calls.append((logits.shape[0], temperature, top_k))
        return torch.arange(10, 10 + logits.shape[0], dtype=torch.long).view(-1, 1)

    monkeypatch.setitem(
        Engine._sample_batch_tokens.__globals__, "sample_next_token", fake_sample_next_token
    )
    logits = torch.zeros((4, 8), dtype=torch.float32)
    active_mask = torch.tensor([True, False, True, True])

    tokens = Engine._sample_batch_tokens(
        object(),
        logits,
        rng=None,
        temperatures=[0.7, 0.7, 0.7, 0.7],
        top_ks=[4, 4, 4, 4],
        active_mask=active_mask,
        pad_id=0,
    )

    assert calls == [(3, 0.7, 4)]
    assert tokens == [10, 0, 11, 12]


def test_sample_batch_tokens_preserves_mixed_sampling_fallback(monkeypatch):
    calls = []

    def fake_sample_next_token(logits, rng, temperature=1.0, top_k=None):
        calls.append((logits.shape[0], temperature, top_k))
        return torch.tensor([[len(calls) + 20]], dtype=torch.long)

    monkeypatch.setitem(
        Engine._sample_batch_tokens.__globals__, "sample_next_token", fake_sample_next_token
    )
    logits = torch.zeros((3, 8), dtype=torch.float32)
    active_mask = torch.tensor([True, True, False])

    tokens = Engine._sample_batch_tokens(
        object(),
        logits,
        rng=None,
        temperatures=[0.7, 0.9, 0.7],
        top_ks=[4, 4, 4],
        active_mask=active_mask,
        pad_id=0,
    )

    assert calls == [(1, 0.7, 4), (1, 0.9, 4)]
    assert tokens == [21, 22, 0]


def test_apply_rotary_emb_inference_matches_reference():
    x = torch.randn(2, 3, 4, 8, dtype=torch.float32)
    cos = torch.randn(1, 3, 1, 4, dtype=torch.float32)
    sin = torch.randn(1, 3, 1, 4, dtype=torch.float32)
    x1, x2 = x[..., :4], x[..., 4:]
    expected = torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3).to(x.dtype)

    with torch.inference_mode():
        actual = apply_rotary_emb(x, cos, sin)

    torch.testing.assert_close(actual, expected)


def test_apply_rotary_emb_training_path_keeps_gradients():
    x = torch.randn(2, 3, 4, 8, dtype=torch.float32, requires_grad=True)
    cos = torch.randn(1, 3, 1, 4, dtype=torch.float32)
    sin = torch.randn(1, 3, 1, 4, dtype=torch.float32)

    out = apply_rotary_emb(x, cos, sin)
    out.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
