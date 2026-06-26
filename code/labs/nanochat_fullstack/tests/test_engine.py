"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

import torch
from types import SimpleNamespace
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


class _TinyTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode_special(self, token):
        return {
            "<|python_start|>": 100,
            "<|python_end|>": 101,
            "<|output_start|>": 102,
            "<|output_end|>": 103,
            "<|assistant_end|>": 104,
        }[token]


class _TinyForwardModel:
    def __init__(self):
        self.config = SimpleNamespace(
            n_kv_head=1,
            n_head=1,
            n_embd=4,
            n_layer=1,
            sequence_len=8,
            vocab_size=16,
            enable_persistent_decode=False,
            use_cuda_graphs=False,
            use_persistent_decode_kernel=False,
        )

    def get_device(self):
        return torch.device("cpu")

    def forward(self, ids, kv_cache=None, attention_mask=None, token_mask=None):
        if kv_cache is not None and kv_cache.kv_cache is None:
            batch_size, seq_len = ids.shape
            k = torch.zeros((batch_size, 1, seq_len, 4), dtype=torch.float32)
            v = torch.zeros_like(k)
            for layer_idx in range(self.config.n_layer):
                kv_cache.insert_kv(layer_idx, k, v, token_mask=token_mask)
        return torch.zeros((ids.size(0), ids.size(1), self.config.vocab_size), dtype=torch.float32)


def test_generate_reuses_sampled_ids_without_forced_tokens(monkeypatch):
    first_ids = torch.tensor([[5]], dtype=torch.long)
    second_ids = torch.tensor([[6]], dtype=torch.long)
    sampled_id_queue = [first_ids, second_ids]

    def fake_sample_next_token(logits, rng, temperature=1.0, top_k=None):
        return sampled_id_queue.pop(0)

    monkeypatch.setitem(
        Engine._sample_batch_tokens.__globals__, "sample_next_token", fake_sample_next_token
    )
    engine = Engine(_TinyForwardModel(), _TinyTokenizer())
    seen_decode_ids = []

    def fake_execute_decode(ids, kv_cache, attention_mask=None, token_mask=None):
        seen_decode_ids.append(ids)
        return torch.zeros((1, 1, engine.model.config.vocab_size), dtype=torch.float32)

    engine._execute_decode = fake_execute_decode

    stream = engine.generate([2, 3], num_samples=1, max_tokens=2, temperature=0.0)

    assert next(stream) == ([5], [1])
    assert next(stream) == ([6], [1])
    assert seen_decode_ids[0] is first_ids


def test_build_attention_mask_reuses_position_buffer():
    engine = Engine(_TinyForwardModel(), _TinyTokenizer())
    lengths = torch.tensor([2, 4], dtype=torch.long)

    mask = engine._build_attention_mask(lengths, max_len=5)
    positions_ptr = engine._attention_positions.data_ptr()

    expected = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, True, False],
        ]
    )
    torch.testing.assert_close(mask, expected)

    shorter = engine._build_attention_mask(torch.tensor([1, 3], dtype=torch.long), max_len=5)

    assert engine._attention_positions.data_ptr() == positions_ptr
    torch.testing.assert_close(
        shorter,
        torch.tensor(
            [
                [True, False, False, False, False],
                [True, True, True, False, False],
            ]
        ),
    )

    grown = engine._build_attention_mask(torch.tensor([6], dtype=torch.long), max_len=6)

    assert engine._attention_positions.numel() >= 6
    torch.testing.assert_close(grown, torch.ones((1, 6), dtype=torch.bool))


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
