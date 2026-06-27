"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

import torch
from pathlib import Path
from types import SimpleNamespace
from nanochat.engine import Engine, KVCache
from nanochat.gpt import CausalSelfAttention, GPTConfig, _expand_gqa_kv_heads, apply_rotary_emb

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


def test_kv_cache_reuses_batch_index_buffer_for_padded_inserts():
    kv_cache = KVCache(
        batch_size=3,
        num_heads=1,
        seq_len=4,
        head_dim=2,
        num_layers=1,
    )
    k = torch.zeros((3, 1, 1, 2), dtype=torch.float32)
    v = torch.zeros_like(k)
    token_mask = torch.tensor([[True], [False], [True]])

    kv_cache.insert_kv(0, k, v, token_mask=token_mask)
    batch_idx_ptr = kv_cache._batch_idx.data_ptr()

    kv_cache.insert_kv(0, k, v, token_mask=token_mask)

    assert kv_cache._batch_idx.data_ptr() == batch_idx_ptr
    torch.testing.assert_close(kv_cache._batch_idx, torch.tensor([0, 1, 2]))


def test_kv_cache_dense_row_pos_insert_skips_materialized_true_mask():
    kv_cache = KVCache(
        batch_size=2,
        num_heads=1,
        seq_len=4,
        head_dim=1,
        num_layers=1,
    )
    prefill_k = torch.tensor([[[[1.0]]], [[[2.0]]]])
    prefill_v = prefill_k + 100
    token_mask = torch.tensor([[True], [False]])

    kv_cache.insert_kv(0, prefill_k, prefill_v, token_mask=token_mask)
    torch.testing.assert_close(kv_cache.row_pos, torch.tensor([1, 0]))

    decode_k = torch.tensor([[[[10.0], [11.0]]], [[[20.0], [21.0]]]])
    decode_v = decode_k + 100
    kv_cache.insert_kv(0, decode_k, decode_v)

    torch.testing.assert_close(kv_cache.row_pos, torch.tensor([3, 2]))
    torch.testing.assert_close(kv_cache.kv_cache[0, 0, 0, 0, :3, 0], torch.tensor([1.0, 10.0, 11.0]))
    torch.testing.assert_close(kv_cache.kv_cache[0, 0, 1, 0, :2, 0], torch.tensor([20.0, 21.0]))


def test_sample_batch_tokens_batches_uniform_sampling(monkeypatch):
    calls = []

    def fake_sample_next_token(logits, rng, temperature=1.0, top_k=None):
        calls.append((logits.shape[0], temperature, top_k))
        return torch.arange(10, 10 + logits.shape[0], dtype=torch.long).view(-1, 1)

    monkeypatch.setitem(
        Engine._sample_batch_tokens.__globals__, "sample_next_token", fake_sample_next_token
    )
    logits = torch.zeros((4, 8), dtype=torch.float32)
    active_mask = object()

    tokens = Engine._sample_batch_tokens(
        object(),
        logits,
        rng=None,
        temperatures=[0.7, 0.7, 0.7, 0.7],
        top_ks=[4, 4, 4, 4],
        active_mask=active_mask,
        pad_id=0,
        active_rows=[0, 2, 3],
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


def test_decode_step_helpers_reuse_token_and_active_mask_buffers():
    engine = Engine(_TinyForwardModel(), _TinyTokenizer())
    ids_buf = torch.empty((3, 1), dtype=torch.long)

    step_ids = engine._fill_ids_buffer_from_tokens(ids_buf, [4, 5, 6])
    host_ptr = engine._token_column_host

    torch.testing.assert_close(step_ids, torch.tensor([[4], [5], [6]]))

    next_step_ids = engine._fill_ids_buffer_from_tokens(ids_buf, [7, 8, 9])

    assert engine._token_column_host is host_ptr
    assert next_step_ids.data_ptr() == ids_buf.data_ptr()
    torch.testing.assert_close(next_step_ids, torch.tensor([[7], [8], [9]]))

    row_states = [
        SimpleNamespace(completed=False),
        SimpleNamespace(completed=True),
        SimpleNamespace(completed=False),
    ]
    active_mask, active_rows = engine._active_mask_for_rows(
        row_states,
        generated_counts=[0, 0, 2],
        row_max_tokens=[2, 2, 2],
        device=torch.device("cpu"),
        return_active_rows=True,
    )
    mask_ptr = engine._active_mask_device.data_ptr()

    torch.testing.assert_close(active_mask, torch.tensor([True, False, False]))
    assert active_rows == [0]

    row_states[1].completed = False
    refreshed, refreshed_rows = engine._active_mask_for_rows(
        row_states,
        generated_counts=[1, 1, 1],
        row_max_tokens=[2, 2, 2],
        device=torch.device("cpu"),
        return_active_rows=True,
    )

    assert engine._active_mask_device.data_ptr() == mask_ptr
    torch.testing.assert_close(refreshed, torch.tensor([True, True, True]))
    assert refreshed_rows == [0, 1, 2]

    tokens = engine._token_tensor_to_list(torch.tensor([10, 11], dtype=torch.long))
    sample_device_ptr = engine._sample_token_device_buffer.data_ptr()
    sample_host = engine._sample_token_host_buffer

    assert tokens == [10, 11]
    assert engine._token_tensor_to_list(torch.tensor([12, 13], dtype=torch.long)) == [12, 13]
    assert engine._sample_token_device_buffer.data_ptr() == sample_device_ptr
    assert engine._sample_token_host_buffer is sample_host


def test_generate_sampling_materializes_tokens_through_reusable_buffer():
    source = Path(__file__).resolve().parents[1] / "nanochat" / "engine.py"
    text = source.read_text(encoding="utf-8")
    sample_section = text.split("def _sample_batch_tokens", maxsplit=1)[1].split(
        "@torch.inference_mode()",
        maxsplit=1,
    )[0]
    generate_section = text.split("def generate(self, tokens", maxsplit=1)[1].split(
        "def generate_batched",
        maxsplit=1,
    )[0]

    assert "self._sample_token_device_buffer = None" in text
    assert "self._sample_token_host_buffer = None" in text
    assert "def _sample_token_buffers(self, count, device)" in text
    assert "def _token_tensor_to_list(self, token_tensor)" in text
    assert "self._token_tensor_to_list(next_ids[:, 0])" in sample_section
    assert "sampled_device[sample_idx].copy_(next_id[0, 0])" in sample_section
    assert "sampled_host.copy_(sampled_device)" in sample_section
    assert "sampled_tokens[idx] = next_id[0, 0].item()" not in sample_section
    assert "next_ids[:, 0].tolist()" not in sample_section
    assert "sampled_tokens = self._token_tensor_to_list(next_ids[:, 0])" in generate_section
    assert "next_ids[:, 0].tolist()" not in generate_section


def test_kv_cache_reuses_token_mask_row_sums():
    source = Path(__file__).resolve().parents[1] / "nanochat" / "engine.py"
    insert_section = source.read_text(encoding="utf-8").split(
        "def insert_kv", maxsplit=1,
    )[1].split(
        "# Return the full cached keys/values",
        maxsplit=1,
    )[0]

    assert "token_increments = token_mask.sum(dim=1)" in insert_section
    assert "next_row_pos = base_row_pos + token_increments" in insert_section
    assert insert_section.count("token_mask.sum(dim=1)") == 1
    assert "token_mask = torch.ones((B, T_add)" not in insert_section
    assert "if token_mask is None:\n                next_row_pos = base_row_pos + T_add" in insert_section
    assert "self.kv_cache[layer_idx, 0, batch_idx, :, positions] = k[:, :, t, :]" in insert_section
    assert "batch_idx = self._batch_index_buffer(B, k.device)" in insert_section
    assert "rows = batch_idx[active]" in insert_section
    assert "if rows.numel() == 0:" in insert_section
    assert "torch.any(active)" not in insert_section


def test_generate_batched_packs_prompt_batch_on_host_before_device_copy():
    source = Path(__file__).resolve().parents[1] / "nanochat" / "engine.py"
    generate_batched = source.read_text(encoding="utf-8").split(
        "def generate_batched", maxsplit=1,
    )[1]
    prompt_pack_section = generate_batched.split(
        "attention_mask = self._build_attention_mask", maxsplit=1,
    )[0]

    assert "lengths_host = torch.empty(" in prompt_pack_section
    assert "ids_host = torch.full(" in prompt_pack_section
    assert "ids_host.to(device=device, non_blocking=use_pinned_transfer)" in prompt_pack_section
    assert "lengths.max().item()" not in prompt_pack_section
    assert "torch.tensor(seq, dtype=torch.long, device=device)" not in prompt_pack_section


def test_attention_reuses_cu_seqlens_buffers():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=8,
        use_flash3=False,
    )
    attn = CausalSelfAttention(config, layer_idx=0)

    cu_q = attn._cu_seqlens_buffer(2, 4, torch.device("cpu"), "_cu_q_cache")
    cu_q_ptr = cu_q.data_ptr()
    cu_q_again = attn._cu_seqlens_buffer(2, 4, torch.device("cpu"), "_cu_q_cache")

    assert cu_q_again.data_ptr() == cu_q_ptr
    torch.testing.assert_close(cu_q_again, torch.tensor([0, 4, 8], dtype=torch.int32))

    cu_q_grown = attn._cu_seqlens_buffer(2, 5, torch.device("cpu"), "_cu_q_cache")

    torch.testing.assert_close(cu_q_grown, torch.tensor([0, 5, 10], dtype=torch.int32))


def test_flash3_gqa_expansion_avoids_repeat_interleave_hot_path():
    x = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).view(1, 2, 3, 4)

    expanded = _expand_gqa_kv_heads(x, 2)

    torch.testing.assert_close(expanded, x.repeat_interleave(2, dim=1))

    source = Path(__file__).resolve().parents[1] / "nanochat" / "gpt.py"
    gpt_source = source.read_text(encoding="utf-8")
    flash3_section = gpt_source.split("def _flash3_attention", maxsplit=1)[1].split(
        "def forward(self, x, cos_sin",
        maxsplit=1,
    )[0]

    assert "repeat_interleave" not in flash3_section
    assert "inspect.signature" not in flash3_section
    assert "self._flash3_accepts_clusters" in flash3_section


def test_attention_reuses_causal_mask_buffers():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=8,
        use_flash3=False,
    )
    attn = CausalSelfAttention(config, layer_idx=0)

    causal = attn._causal_mask_for(3, 3, torch.device("cpu"))
    causal_ptr = causal.data_ptr()
    causal_again = attn._causal_mask_for(3, 3, torch.device("cpu"))

    assert causal_again.data_ptr() == causal_ptr
    torch.testing.assert_close(
        causal_again,
        torch.tensor(
            [
                [True, False, False],
                [True, True, False],
                [True, True, True],
            ],
            dtype=torch.bool,
        ),
    )

    prefix = attn._prefix_causal_mask_for(2, 5, torch.device("cpu"))
    prefix_ptr = prefix.data_ptr()
    prefix_again = attn._prefix_causal_mask_for(2, 5, torch.device("cpu"))

    assert prefix_again.data_ptr() == prefix_ptr
    torch.testing.assert_close(
        prefix_again,
        torch.tensor(
            [
                [True, True, True, True, False],
                [True, True, True, True, True],
            ],
            dtype=torch.bool,
        ),
    )


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
