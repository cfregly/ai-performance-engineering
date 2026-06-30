"""
Test Engine class. Example run:

python -m pytest tests/test_engine.py -v
"""

import torch
import torch.nn.functional as F
from pathlib import Path
from types import SimpleNamespace
from nanochat.engine import Engine, KVCache, sample_next_token
from nanochat.gpt import (
    CausalSelfAttention,
    GPT,
    GPTConfig,
    _expand_gqa_kv_heads,
    _relu_square_in_place_if_safe,
    apply_rotary_emb,
)

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
    row_pos_ptr = kv_cache.row_pos.data_ptr()

    decode_k = torch.tensor([[[[10.0], [11.0]]], [[[20.0], [21.0]]]])
    decode_v = decode_k + 100
    kv_cache.insert_kv(0, decode_k, decode_v)

    assert kv_cache.row_pos.data_ptr() == row_pos_ptr
    assert kv_cache._row_position_idx is not None
    torch.testing.assert_close(kv_cache.row_pos, torch.tensor([3, 2]))
    torch.testing.assert_close(kv_cache.kv_cache[0, 0, 0, 0, :3, 0], torch.tensor([1.0, 10.0, 11.0]))
    torch.testing.assert_close(kv_cache.kv_cache[0, 0, 1, 0, :2, 0], torch.tensor([20.0, 21.0]))


def test_sample_batch_tokens_batches_uniform_sampling(monkeypatch):
    calls = []

    def fake_sample_next_token(logits, rng, temperature=1.0, top_k=None, **kwargs):
        calls.append((logits.shape[0], temperature, top_k))
        return torch.arange(10, 10 + logits.shape[0], dtype=torch.long).view(-1, 1)

    fake_self = SimpleNamespace(
        _sample_workspace=lambda *args: {},
        _token_tensor_to_list=lambda token_tensor: [int(token) for token in token_tensor.reshape(-1).tolist()],
        _sample_active_logits_buffer_for=lambda logits, count: torch.empty(
            (count, logits.size(-1)),
            dtype=logits.dtype,
            device=logits.device,
        ),
    )
    monkeypatch.setitem(
        Engine._sample_batch_tokens.__globals__, "sample_next_token", fake_sample_next_token
    )
    logits = torch.zeros((4, 8), dtype=torch.float32)
    active_mask = object()

    tokens = Engine._sample_batch_tokens(
        fake_self,
        logits,
        rng=None,
        temperatures=[0.7, 0.7, 0.7, 0.7],
        top_ks=[4, 4, 4, 4],
        active_mask=active_mask,
        pad_id=0,
        active_rows=[0, 2, 3],
        active_indices=torch.tensor([0, 2, 3], dtype=torch.long),
    )

    assert calls == [(3, 0.7, 4)]
    assert tokens == [10, 0, 11, 12]


def test_sample_batch_tokens_preserves_mixed_sampling_fallback(monkeypatch):
    calls = []

    def fake_sample_next_token(logits, rng, temperature=1.0, top_k=None, **kwargs):
        calls.append((logits.shape[0], temperature, top_k))
        return torch.tensor([[len(calls) + 20]], dtype=torch.long)

    def fake_sample_token_buffers(count, device):
        return (
            torch.empty(count, dtype=torch.long, device=device),
            torch.empty(count, dtype=torch.long, device="cpu"),
        )

    fake_self = SimpleNamespace(
        _sample_workspace=lambda *args: {},
        _sample_token_buffers=fake_sample_token_buffers,
    )
    monkeypatch.setitem(
        Engine._sample_batch_tokens.__globals__, "sample_next_token", fake_sample_next_token
    )
    logits = torch.zeros((3, 8), dtype=torch.float32)
    active_mask = torch.tensor([True, True, False])

    tokens = Engine._sample_batch_tokens(
        fake_self,
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

    def fake_sample_next_token(logits, rng, temperature=1.0, top_k=None, **kwargs):
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


def test_sample_next_token_reuses_workspace_outputs():
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.1, 0.2, 0.3],
        ],
        dtype=torch.float32,
    )
    out = torch.empty((2, 1), dtype=torch.long)
    choice = torch.empty((2, 1), dtype=torch.long)
    max_values = torch.empty((2, 1), dtype=torch.float32)
    probs = torch.empty_like(logits)
    topk_values = torch.empty((2, 2), dtype=torch.float32)
    topk_indices = torch.empty((2, 2), dtype=torch.long)
    topk_probs = torch.empty_like(topk_values)

    result = sample_next_token(
        logits,
        rng=None,
        temperature=0.0,
        out=out,
        max_values_out=max_values,
    )

    assert result is out
    torch.testing.assert_close(out, torch.tensor([[3], [0]], dtype=torch.long))

    rng = torch.Generator(device="cpu").manual_seed(1)
    result = sample_next_token(
        logits,
        rng=rng,
        temperature=1.0,
        top_k=2,
        out=out,
        choice_out=choice,
        topk_values_out=topk_values,
        topk_indices_out=topk_indices,
        topk_probs_out=topk_probs,
    )
    assert result is out
    assert out.shape == (2, 1)

    rng = torch.Generator(device="cpu").manual_seed(1)
    result = sample_next_token(
        logits,
        rng=rng,
        temperature=1.0,
        out=out,
        probs_out=probs,
    )
    assert result is out
    assert out.shape == (2, 1)


def test_gpt_generate_ids_buffer_reuses_larger_storage():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=8,
        use_flash3=False,
    )
    model = GPT(config)

    first = model._generate_ids_buffer(6, torch.device("cpu"))
    first_ptr = first.data_ptr()
    shorter = model._generate_ids_buffer(4, torch.device("cpu"))
    grown = model._generate_ids_buffer(9, torch.device("cpu"))

    assert shorter.data_ptr() == first_ptr
    assert shorter.shape == (1, 4)
    assert grown.shape == (1, 9)
    assert grown.data_ptr() != first_ptr


def test_sample_batch_tokens_reuses_sampler_workspace():
    engine = Engine(_TinyForwardModel(), _TinyTokenizer())
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.1, 0.2, 0.3],
        ],
        dtype=torch.float32,
    )

    tokens = engine._sample_batch_tokens(
        logits,
        rng=None,
        temperatures=[0.0, 0.0],
        top_ks=[None, None],
        active_mask=torch.tensor([True, True]),
        pad_id=0,
    )
    next_id_ptr = engine._sample_next_id_buffer.data_ptr()
    max_values_ptr = engine._sample_max_values_buffer.data_ptr()

    assert tokens == [3, 0]

    tokens = engine._sample_batch_tokens(
        logits,
        rng=None,
        temperatures=[0.0, 0.0],
        top_ks=[None, None],
        active_mask=torch.tensor([True, True]),
        pad_id=0,
    )

    assert tokens == [3, 0]
    assert engine._sample_next_id_buffer.data_ptr() == next_id_ptr
    assert engine._sample_max_values_buffer.data_ptr() == max_values_ptr


def test_sample_batch_tokens_reuses_sparse_active_logits_buffer():
    engine = Engine(_TinyForwardModel(), _TinyTokenizer())
    logits = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.9, 0.1, 0.2, 0.3],
            [0.1, 0.8, 0.3, 0.2],
        ],
        dtype=torch.float32,
    )

    tokens = engine._sample_batch_tokens(
        logits,
        rng=None,
        temperatures=[0.0, 0.0, 0.0],
        top_ks=[None, None, None],
        active_mask=torch.tensor([True, False, True]),
        pad_id=99,
        active_rows=[0, 2],
        active_indices=torch.tensor([0, 2], dtype=torch.long),
    )
    active_logits_ptr = engine._sample_active_logits_buffer.data_ptr()

    assert tokens == [3, 99, 1]

    tokens = engine._sample_batch_tokens(
        logits,
        rng=None,
        temperatures=[0.0, 0.0, 0.0],
        top_ks=[None, None, None],
        active_mask=torch.tensor([False, True, False]),
        pad_id=99,
        active_rows=[1],
        active_indices=torch.tensor([1], dtype=torch.long),
    )

    assert tokens == [99, 0, 99]
    assert engine._sample_active_logits_buffer.data_ptr() == active_logits_ptr


def test_build_attention_mask_reuses_position_buffer():
    engine = Engine(_TinyForwardModel(), _TinyTokenizer())
    lengths = torch.tensor([2, 4], dtype=torch.long)

    mask = engine._build_attention_mask(lengths, max_len=5)
    positions_ptr = engine._attention_positions.data_ptr()
    mask_ptr = mask.data_ptr()

    expected = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, True, False],
        ]
    )
    torch.testing.assert_close(mask, expected)

    shorter = engine._build_attention_mask(torch.tensor([1, 3], dtype=torch.long), max_len=5)

    assert engine._attention_positions.data_ptr() == positions_ptr
    assert shorter.data_ptr() == mask_ptr
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
    assert engine._attention_mask.numel() >= 6
    torch.testing.assert_close(grown, torch.ones((1, 6), dtype=torch.bool))


def test_batch_row_index_buffer_reuses_arange_storage():
    engine = Engine(_TinyForwardModel(), _TinyTokenizer())

    first = engine._batch_row_index_buffer(3, torch.device("cpu"))
    first_ptr = first.data_ptr()
    shorter = engine._batch_row_index_buffer(2, torch.device("cpu"))
    grown = engine._batch_row_index_buffer(5, torch.device("cpu"))

    torch.testing.assert_close(first, torch.tensor([0, 1, 2]))
    assert shorter.data_ptr() == first_ptr
    torch.testing.assert_close(shorter, torch.tensor([0, 1]))
    assert grown.numel() == 5
    torch.testing.assert_close(grown, torch.tensor([0, 1, 2, 3, 4]))


def test_lengths_by_batch_buffer_reuses_device_storage():
    engine = Engine(_TinyForwardModel(), _TinyTokenizer())

    first = engine._lengths_by_batch_buffer(torch.tensor([2, 4, 5], dtype=torch.long))
    first_ptr = first.data_ptr()
    first.add_(1)

    shorter = engine._lengths_by_batch_buffer(torch.tensor([1, 3], dtype=torch.long))
    grown = engine._lengths_by_batch_buffer(torch.tensor([6, 7, 8, 9], dtype=torch.long))

    assert shorter.data_ptr() == first_ptr
    torch.testing.assert_close(shorter, torch.tensor([1, 3], dtype=torch.long))
    assert grown.numel() == 4
    torch.testing.assert_close(grown, torch.tensor([6, 7, 8, 9], dtype=torch.long))


def test_full_active_mask_reuses_device_buffer():
    engine = Engine(_TinyForwardModel(), _TinyTokenizer())

    first = engine._full_active_mask(3, torch.device("cpu"))
    first_ptr = first.data_ptr()
    first[1] = False

    shorter = engine._full_active_mask(2, torch.device("cpu"))
    grown = engine._full_active_mask(5, torch.device("cpu"))

    assert shorter.data_ptr() == first_ptr
    torch.testing.assert_close(shorter, torch.tensor([True, True]))
    assert grown.numel() == 5
    torch.testing.assert_close(grown, torch.ones(5, dtype=torch.bool))


def test_prompt_pack_buffers_reuse_host_storage():
    engine = Engine(_TinyForwardModel(), _TinyTokenizer())

    lengths, ids = engine._prompt_pack_buffers(
        batch_size=3,
        max_prompt_len=5,
        pad_id=0,
        pin_memory=False,
    )
    lengths_ptr = lengths.data_ptr()
    ids_ptr = ids.data_ptr()
    ids.fill_(7)

    shorter_lengths, shorter_ids = engine._prompt_pack_buffers(
        batch_size=2,
        max_prompt_len=4,
        pad_id=1,
        pin_memory=False,
    )

    assert shorter_lengths.data_ptr() == lengths_ptr
    assert shorter_ids.data_ptr() == ids_ptr
    torch.testing.assert_close(shorter_ids, torch.ones((2, 4), dtype=torch.long))

    grown_lengths, grown_ids = engine._prompt_pack_buffers(
        batch_size=4,
        max_prompt_len=6,
        pad_id=2,
        pin_memory=False,
    )

    assert grown_lengths.numel() >= 4
    assert grown_ids.shape == (4, 6)
    torch.testing.assert_close(grown_ids, torch.full((4, 6), 2, dtype=torch.long))


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
    active_mask, active_rows, active_indices = engine._active_mask_for_rows(
        row_states,
        generated_counts=[0, 0, 2],
        row_max_tokens=[2, 2, 2],
        device=torch.device("cpu"),
        return_active_rows=True,
        return_active_indices=True,
    )
    mask_ptr = engine._active_mask_device.data_ptr()
    indices_ptr = engine._active_indices_device.data_ptr()

    torch.testing.assert_close(active_mask, torch.tensor([True, False, False]))
    assert active_rows == [0]
    torch.testing.assert_close(active_indices, torch.tensor([0]))

    row_states[1].completed = False
    refreshed, refreshed_rows, refreshed_indices = engine._active_mask_for_rows(
        row_states,
        generated_counts=[1, 1, 1],
        row_max_tokens=[2, 2, 2],
        device=torch.device("cpu"),
        return_active_rows=True,
        return_active_indices=True,
    )

    assert engine._active_mask_device.data_ptr() == mask_ptr
    assert engine._active_indices_device.data_ptr() == indices_ptr
    torch.testing.assert_close(refreshed, torch.tensor([True, True, True]))
    assert refreshed_rows == [0, 1, 2]
    torch.testing.assert_close(refreshed_indices, torch.tensor([0, 1, 2]))

    tokens = engine._token_tensor_to_list(torch.tensor([10, 11], dtype=torch.long))
    sample_host = engine._sample_token_host_buffer

    assert tokens == [10, 11]
    assert engine._sample_token_device_buffer is None
    assert engine._token_tensor_to_list(torch.tensor([12, 13], dtype=torch.long)) == [12, 13]
    assert engine._sample_token_host_buffer is sample_host

    prompt_ids = engine._single_prompt_ids([1, 2, 3], torch.device("cpu"))
    prompt_ptr = engine._prompt_ids_device.data_ptr()

    torch.testing.assert_close(prompt_ids, torch.tensor([[1, 2, 3]], dtype=torch.long))

    shorter_prompt_ids = engine._single_prompt_ids([4, 5], torch.device("cpu"))

    assert engine._prompt_ids_device.data_ptr() == prompt_ptr
    torch.testing.assert_close(shorter_prompt_ids, torch.tensor([[4, 5]], dtype=torch.long))

    first_ids_step = engine._ids_step_buffer_for(4, torch.device("cpu"))
    ids_step_ptr = engine._ids_step_buffer.data_ptr()
    second_ids_step = engine._ids_step_buffer_for(2, torch.device("cpu"))

    assert first_ids_step.shape == (4, 1)
    assert second_ids_step.shape == (2, 1)
    assert engine._ids_step_buffer.data_ptr() == ids_step_ptr


def test_generate_sampling_materializes_tokens_through_reusable_buffer():
    source = Path(__file__).resolve().parents[1] / "nanochat" / "engine.py"
    text = source.read_text(encoding="utf-8")
    sample_section = text.split("def _sample_batch_tokens", maxsplit=1)[1].split(
        "@torch.inference_mode()",
        maxsplit=1,
    )[0]
    token_list_section = text.split("def _token_tensor_to_list", maxsplit=1)[1].split(
        "def _sample_batch_tokens",
        maxsplit=1,
    )[0]
    generate_section = text.split("def generate(self, tokens", maxsplit=1)[1].split(
        "def generate_batched",
        maxsplit=1,
    )[0]

    assert "self._sample_token_device_buffer = None" in text
    assert "self._sample_token_host_buffer = None" in text
    assert "self._active_indices_device = None" in text
    assert "self._active_indices_host = None" in text
    assert "self._sample_next_id_buffer = None" in text
    assert "self._sample_probs_buffer = None" in text
    assert "self._sample_active_logits_buffer = None" in text
    assert "self._prompt_ids_device = None" in text
    assert "self._ids_step_buffer = None" in text
    assert "def _sample_host_token_buffer(self, count, source_device)" in text
    assert "def _sample_token_buffers(self, count, device)" in text
    assert "def _sample_long_buffer(self, name, shape, device)" in text
    assert "shape = tuple(int(dim) for dim in shape)" in text
    assert "or any(buffer.size(dim) < size for dim, size in enumerate(shape))" in text
    assert "return buffer[tuple(slice(0, size) for size in shape)]" in text
    assert "def _sample_like_buffer(self, name, tensor)" in text
    assert "numel = int(tensor.numel())" in text
    assert "or buffer.numel() < numel" in text
    assert "return buffer[:numel].view(shape)" in text
    assert "def _sample_active_logits_buffer_for(self, logits, row_count)" in text
    assert "def _sample_workspace(self, logits, top_k, temperature)" in text
    assert "def _single_prompt_ids(self, tokens, device)" in text
    assert "def _ids_step_buffer_for(self, batch_size, device)" in text
    assert "def _token_tensor_to_list(self, token_tensor)" in text
    assert "uniform_sampling=None" in sample_section
    assert "batch_size = logits.size(0)" in sample_section
    assert "full_active_rows = (" in sample_section
    assert "len(active_rows) == batch_size" in sample_section
    assert "host_tokens = self._sample_host_token_buffer(flat_tokens.numel(), flat_tokens.device)" in token_list_section
    assert "host_tokens.copy_(flat_tokens, non_blocking=flat_tokens.device.type == \"cuda\")" in token_list_section
    assert "return host_tokens.tolist()" in token_list_section
    assert "[int(token) for token in host_tokens.tolist()]" not in token_list_section
    assert "device_tokens.copy_(flat_tokens)" not in token_list_section
    assert "**self._sample_workspace(active_logits, first_top_k, first_temp)," in sample_section
    assert "**self._sample_workspace(row_logits, top_k, temp)," in sample_section
    assert "if uniform_sampling is None:" in sample_section
    assert "active_indices=None" in sample_section
    assert "active_indices.device.type == \"cpu\"" in sample_section
    assert "active_rows = self._token_tensor_to_list(active_indices)" in sample_section
    assert "self._token_tensor_to_list(next_ids[:, 0])" in sample_section
    assert "if full_active_rows:\n                return next_tokens" in sample_section
    assert "active_logits = self._sample_active_logits_buffer_for(logits, len(active_rows))" in sample_section
    assert "torch.index_select(logits, 0, active_indices, out=active_logits)" in sample_section
    assert "active_logits = logits.index_select(0, active_indices)" not in sample_section
    assert "sampled_tokens = [pad_id] * logits.size(0)" not in sample_section
    assert "sampled_device[sample_idx].copy_(next_id[0, 0])" in sample_section
    assert "sampled_host.copy_(sampled_device, non_blocking=sampled_device.device.type == \"cuda\")" in sample_section
    assert "sampled_tokens[idx] = next_id[0, 0].item()" not in sample_section
    assert "sampled_tokens[idx] = int(token)" not in sample_section
    assert "next_ids[:, 0].tolist()" not in sample_section
    assert generate_section.count("**self._sample_workspace(logits, top_k, temperature),") == 2
    assert "ids = self._single_prompt_ids(tokens, device)" in generate_section
    assert "ids = torch.tensor([tokens], dtype=torch.long, device=device)" not in generate_section
    assert "ids_buf = self._ids_step_buffer_for(num_samples, device)" in generate_section
    assert "torch.empty((num_samples, 1), dtype=torch.long, device=device)" not in generate_section
    assert "torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)" not in generate_section
    assert "sampled_tokens = self._token_tensor_to_list(next_ids[:, 0])" in generate_section
    assert "next_ids[:, 0].tolist()" not in generate_section
    assert "active_count = num_samples" in generate_section
    assert "if active_count == 0:" in generate_section
    assert "if all(state.completed for state in row_states):" not in generate_section


def test_long_sampling_and_generation_buffers_reuse_capacity_views():
    engine_owner = SimpleNamespace(_sample_topk_indices_buffer=None)
    large_sample = Engine._sample_long_buffer(
        engine_owner,
        "_sample_topk_indices_buffer",
        (4, 8),
        torch.device("cpu"),
    )
    sample_ptr = engine_owner._sample_topk_indices_buffer.data_ptr()
    small_sample = Engine._sample_long_buffer(
        engine_owner,
        "_sample_topk_indices_buffer",
        (2, 3),
        torch.device("cpu"),
    )

    assert large_sample.shape == (4, 8)
    assert small_sample.shape == (2, 3)
    assert engine_owner._sample_topk_indices_buffer.data_ptr() == sample_ptr

    engine_owner._sample_topk_probs_buffer = None
    large_probs = Engine._sample_like_buffer(
        engine_owner,
        "_sample_topk_probs_buffer",
        torch.empty(4, 8),
    )
    probs_ptr = engine_owner._sample_topk_probs_buffer.data_ptr()
    small_probs = Engine._sample_like_buffer(
        engine_owner,
        "_sample_topk_probs_buffer",
        torch.empty(2, 3),
    )

    assert large_probs.shape == (4, 8)
    assert small_probs.shape == (2, 3)
    assert small_probs.is_contiguous()
    assert engine_owner._sample_topk_probs_buffer.data_ptr() == probs_ptr

    gpt_owner = SimpleNamespace(_generate_topk_indices=None, _generate_topk_probs=None)
    large_generate = GPT._generate_long_buffer(
        gpt_owner,
        "_generate_topk_indices",
        (1, 8),
        torch.device("cpu"),
    )
    generate_ptr = gpt_owner._generate_topk_indices.data_ptr()
    small_generate = GPT._generate_long_buffer(
        gpt_owner,
        "_generate_topk_indices",
        (1, 3),
        torch.device("cpu"),
    )

    assert large_generate.shape == (1, 8)
    assert small_generate.shape == (1, 3)
    assert gpt_owner._generate_topk_indices.data_ptr() == generate_ptr

    large_generate_probs = GPT._generate_like_buffer(
        gpt_owner,
        "_generate_topk_probs",
        torch.empty(1, 8),
    )
    generate_probs_ptr = gpt_owner._generate_topk_probs.data_ptr()
    small_generate_probs = GPT._generate_like_buffer(
        gpt_owner,
        "_generate_topk_probs",
        torch.empty(1, 3),
    )

    assert large_generate_probs.shape == (1, 8)
    assert small_generate_probs.shape == (1, 3)
    assert small_generate_probs.is_contiguous()
    assert gpt_owner._generate_topk_probs.data_ptr() == generate_probs_ptr


def test_generate_loops_track_completion_counts_without_rescanning_rows():
    source = Path(__file__).resolve().parents[1] / "nanochat" / "engine.py"
    text = source.read_text(encoding="utf-8")
    generate_batched = text.split(
        "def generate_batched", maxsplit=1,
    )[1].split(
        "def generate_batch", maxsplit=1,
    )[0]
    generate_batch = text.split(
        "def generate_batch", maxsplit=1,
    )[1].split(
        'if __name__ == "__main__"',
        maxsplit=1,
    )[0]

    assert "active_count = sum(1 for limit in row_max_tokens if limit > 0)" in generate_batched
    assert "if active_count == 0:" in generate_batched
    assert "if all(state.completed or generated_counts[i] >= row_max_tokens[i]" not in generate_batched
    assert "remaining = num_samples" in generate_batch
    assert "if remaining == 0:" in generate_batch
    assert "if all(completed):" not in generate_batch


def test_gpt_generate_prompt_copy_reuses_ids_buffer():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=8,
        use_flash3=False,
    )
    model = GPT(config)
    ids = model._generate_ids_buffer(4, torch.device("cpu"))
    ids_ptr = model._generate_ids.data_ptr()

    model._copy_generate_prompt(ids, [1, 2, 3], torch.device("cpu"))

    torch.testing.assert_close(ids[:, :3], torch.tensor([[1, 2, 3]], dtype=torch.long))

    shorter_ids = model._generate_ids_buffer(3, torch.device("cpu"))
    model._copy_generate_prompt(shorter_ids, [4, 5], torch.device("cpu"))

    assert model._generate_ids.data_ptr() == ids_ptr
    torch.testing.assert_close(shorter_ids[:, :2], torch.tensor([[4, 5]], dtype=torch.long))


def test_kv_cache_reuses_token_mask_row_sums():
    source = Path(__file__).resolve().parents[1] / "nanochat" / "engine.py"
    text = source.read_text(encoding="utf-8")
    insert_section = text.split(
        "def insert_kv", maxsplit=1,
    )[1].split(
        "# Return the full cached keys/values",
        maxsplit=1,
    )[0]

    assert "token_increments = token_mask.sum(dim=1)" in insert_section
    assert "next_row_pos = base_row_pos + token_increments" in insert_section
    assert "if token_mask.device != k.device or token_mask.dtype != torch.bool:" in insert_section
    assert "token_mask = token_mask.to(device=k.device, dtype=torch.bool)" in insert_section
    assert insert_section.count("token_mask.sum(dim=1)") == 1
    assert "token_mask = torch.ones((B, T_add)" not in insert_section
    assert "self._row_position_idx = None" in text
    assert "def _row_position_buffer(self, batch_size, device)" in text
    assert "dense_positions = self._row_position_buffer(B, k.device)" in insert_section
    assert "dense_positions.copy_(base_row_pos)" in insert_section
    assert "dense_positions.add_(1)" in insert_section
    assert "self.row_pos.copy_(dense_positions)" in insert_section
    assert "if token_mask is None:\n                next_row_pos = base_row_pos + T_add" not in insert_section
    assert "positions = base_row_pos + t" not in insert_section
    assert "self.kv_cache[layer_idx, 0, batch_idx, :, dense_positions] = k[:, :, t, :]" in insert_section
    assert "def insert_kv(self, layer_idx, k, v, token_mask=None, max_cache_len=None)" in text
    assert "max_needed = int(max_cache_len) if max_cache_len is not None else int(dense_positions.max().item()) + T_add" in insert_section
    assert "max_needed = int(max_cache_len) if max_cache_len is not None else int(next_row_pos.max().item())" in insert_section
    assert "batch_idx = self._batch_index_buffer(B, k.device)" in insert_section
    assert "rows = batch_idx[active]" in insert_section
    assert "if rows.numel() == 0:" in insert_section
    assert "torch.any(active)" not in insert_section
    assert "def get_pos(self):\n        return self.pos" in text
    assert insert_section.count(".max().item()") == 2
    assert "self.pos = int(self.row_pos.max().item())" not in insert_section
    assert "t1_source =" not in insert_section


def test_gpt_forward_skips_matching_mask_casts():
    source = Path(__file__).resolve().parents[1] / "nanochat" / "gpt.py"
    forward_prefix = source.read_text(encoding="utf-8").split(
        "def forward(self, idx, targets=None, kv_cache=None",
        maxsplit=1,
    )[1].split(
        "# Grab the rotary embeddings",
        maxsplit=1,
    )[0]

    assert "if attention_mask.device != idx.device or attention_mask.dtype != torch.bool:" in forward_prefix
    assert "attention_mask = attention_mask.to(device=idx.device, dtype=torch.bool)" in forward_prefix
    assert "if token_mask.device != idx.device or token_mask.dtype != torch.bool:" in forward_prefix
    assert "token_mask = token_mask.to(device=idx.device, dtype=torch.bool)" in forward_prefix
    assert forward_prefix.count(".to(device=idx.device, dtype=torch.bool)") == 2
    attention_source = source.read_text(encoding="utf-8").split(
        "# Apply KV cache: insert current k,v into cache, get the full view so far",
        maxsplit=1,
    )[1].split(
        "Tq = q.size(2)",
        maxsplit=1,
    )[0]
    assert "cache_max_len = attention_mask.size(-1) if cache_token_mask is not None and attention_mask is not None else None" in attention_source
    assert "max_cache_len=cache_max_len" in attention_source


def test_generate_batched_packs_prompt_batch_on_host_before_device_copy():
    source = Path(__file__).resolve().parents[1] / "nanochat" / "engine.py"
    text = source.read_text(encoding="utf-8")
    generate_batched = text.split(
        "def generate_batched", maxsplit=1,
    )[1]
    prompt_pack_section = generate_batched.split(
        "attention_mask = self._build_attention_mask", maxsplit=1,
    )[0]

    assert "self._prompt_lengths_host = None" in text
    assert "self._prompt_ids_host = None" in text
    assert "self._lengths_by_batch = None" in text
    assert "def _prompt_pack_buffers(self, batch_size, max_prompt_len, pad_id, pin_memory)" in text
    assert "def _lengths_by_batch_buffer(self, lengths)" in text
    assert "lengths_host, ids_host = self._prompt_pack_buffers(" in prompt_pack_section
    assert "lengths_host = torch.empty(" not in prompt_pack_section
    assert "ids_host = torch.full(" not in prompt_pack_section
    assert "ids_host.to(device=device, non_blocking=use_pinned_transfer)" in prompt_pack_section
    assert "lengths.max().item()" not in prompt_pack_section
    assert "torch.tensor(seq, dtype=torch.long, device=device)" not in prompt_pack_section
    assert "self._batch_row_indices = None" in text
    assert "self._attention_mask = None" in text
    assert "def _attention_mask_buffer(self, batch_size, max_len, device)" in text
    assert "torch.lt(positions.unsqueeze(0), lengths.unsqueeze(1), out=mask)" in text
    assert "batch_rows = self._batch_row_index_buffer(batch_size, device)" in generate_batched
    assert "torch.arange(batch_size, device=device)" not in generate_batched
    assert "active_mask = self._full_active_mask(batch_size, device)" in generate_batched
    assert "torch.ones(batch_size, dtype=torch.bool, device=device)" not in generate_batched
    assert "lengths_by_batch = self._lengths_by_batch_buffer(lengths)" in generate_batched
    assert "lengths_by_batch = lengths.clone()" not in generate_batched
    assert "lengths_by_batch.add_(step_token_mask[:, 0])" in generate_batched
    assert "ids_buf = self._ids_step_buffer_for(batch_size, device)" in generate_batched
    assert "torch.empty((batch_size, 1), dtype=torch.long, device=device)" not in generate_batched
    assert "torch.tensor(token_column, dtype=torch.long, device=device).unsqueeze(1)" not in generate_batched
    assert "return_active_indices=True" in generate_batched
    assert "active_indices=active_indices" in generate_batched
    assert "active_rows=range(batch_size)" in generate_batched
    assert "active_rows=list(range(batch_size))" not in generate_batched
    assert "uniform_sampling = all(temp == first_temp and top_k == first_top_k for temp, top_k in zip(temps, top_ks, strict=True))" in generate_batched
    assert "uniform_sampling_hint = True if uniform_sampling else None" in generate_batched
    assert generate_batched.count("uniform_sampling=uniform_sampling_hint") == 2
    assert "torch.as_tensor(active_rows" not in generate_batched
    assert "current_decode_len = max_prompt_len" in generate_batched
    assert "if len(active_rows) == batch_size:" in generate_batched
    assert "current_decode_len += 1" in generate_batched
    assert "if row_len > current_decode_len:" in generate_batched
    assert "current_decode_len = row_len" in generate_batched
    assert "attn_mask = self._build_attention_mask(lengths_by_batch, current_decode_len)" in generate_batched
    assert "max(lengths_by_row)" not in generate_batched
    assert "next_lengths = lengths_by_batch +" not in generate_batched
    assert "attn_mask = self._build_attention_mask(next_lengths)" not in generate_batched


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
    q_pos_ptr = attn._mask_q_pos_cache.data_ptr()
    k_pos_ptr = attn._mask_k_pos_cache.data_ptr()
    causal_again = attn._causal_mask_for(3, 3, torch.device("cpu"))

    assert causal_again.data_ptr() == causal_ptr
    assert attn._mask_q_pos_cache.data_ptr() == q_pos_ptr
    assert attn._mask_k_pos_cache.data_ptr() == k_pos_ptr
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

    prefix_same_shape = attn._prefix_causal_mask_for(3, 3, torch.device("cpu"))
    assert attn._mask_q_pos_cache.data_ptr() == q_pos_ptr
    assert attn._mask_k_pos_cache.data_ptr() == k_pos_ptr
    torch.testing.assert_close(prefix_same_shape, causal)

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


def test_attention_reuses_padded_mask_buffer_and_skips_decode_causal_mask():
    config = GPTConfig(
        sequence_len=8,
        vocab_size=32,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=8,
        use_flash3=False,
        use_padded_attention=True,
    )
    attn = CausalSelfAttention(config, layer_idx=0)

    key_mask = torch.tensor([[[[True, True, False, False]]]])
    causal = torch.tensor(
        [
            [True, False, False, False],
            [True, True, False, False],
        ],
        dtype=torch.bool,
    )

    mask = attn._padded_attn_mask_for(key_mask, causal)
    mask_ptr = mask.data_ptr()
    mask_again = attn._padded_attn_mask_for(key_mask, causal)

    assert mask_again.data_ptr() == mask_ptr
    torch.testing.assert_close(mask, key_mask & causal)

    source = Path(__file__).resolve().parents[1] / "nanochat" / "gpt.py"
    gpt_source = source.read_text(encoding="utf-8")
    forward_section = gpt_source.split("def forward(self, x, cos_sin", maxsplit=1)[1].split(
        "# Attention: queries attend",
        maxsplit=1,
    )[1].split(
        "fa3_out = None",
        maxsplit=1,
    )[0]

    assert "self._padded_attn_mask_cache = None" in gpt_source
    assert "def _padded_attn_mask_for(self, key_mask, causal)" in gpt_source
    assert "torch.logical_and(key_mask, causal, out=attn_mask)" in gpt_source
    assert "if kv_cache is not None and Tq == 1 and Tq != Tk:" in forward_section
    assert "attn_mask = key_mask" in forward_section
    assert "attn_mask = key_mask & causal" not in forward_section


def test_apply_rotary_emb_inference_matches_reference():
    x = torch.randn(2, 3, 4, 8, dtype=torch.float32)
    cos = torch.randn(1, 3, 1, 4, dtype=torch.float32)
    sin = torch.randn(1, 3, 1, 4, dtype=torch.float32)
    x1, x2 = x[..., :4], x[..., 4:]
    expected = torch.cat([x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos], 3).to(x.dtype)

    with torch.inference_mode():
        out = torch.empty_like(x)
        actual = apply_rotary_emb(x, cos, sin, out=out)

    assert actual is out
    torch.testing.assert_close(actual, expected)


def test_attention_reuses_rotary_buffers_for_inference():
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
    q = torch.empty(1, 2, 2, 4)
    k = torch.empty(1, 2, 2, 4)

    q_buf = attn._rotary_buffer("_rotary_q_cache", q)
    k_buf = attn._rotary_buffer("_rotary_k_cache", k)

    assert attn._rotary_buffer("_rotary_q_cache", q).data_ptr() == q_buf.data_ptr()
    assert attn._rotary_buffer("_rotary_k_cache", k).data_ptr() == k_buf.data_ptr()

    q_smaller = torch.empty(1, 1, 1, 4)
    q_small_buf = attn._rotary_buffer("_rotary_q_cache", q_smaller)

    assert q_small_buf.shape == q_smaller.shape
    assert q_small_buf.is_contiguous()
    assert q_small_buf.data_ptr() == q_buf.data_ptr()
    assert attn._rotary_q_cache.numel() >= q.numel()

    source = Path(__file__).resolve().parents[1] / "nanochat" / "gpt.py"
    gpt_source = source.read_text(encoding="utf-8")
    forward_section = gpt_source.split("def forward(self, x, cos_sin", maxsplit=1)[1].split(
        "# Apply KV cache",
        maxsplit=1,
    )[0]

    assert "self._rotary_q_cache = None" in gpt_source
    assert "self._rotary_k_cache = None" in gpt_source
    assert "def _rotary_buffer(self, name, tensor)" in gpt_source
    assert "buffer.numel() < numel" in gpt_source
    assert "return buffer[:numel].view(shape)" in gpt_source
    assert "out=self._rotary_buffer(\"_rotary_q_cache\", q)" in forward_section
    assert "out=self._rotary_buffer(\"_rotary_k_cache\", k)" in forward_section


def test_apply_rotary_emb_training_path_keeps_gradients():
    x = torch.randn(2, 3, 4, 8, dtype=torch.float32, requires_grad=True)
    cos = torch.randn(1, 3, 1, 4, dtype=torch.float32)
    sin = torch.randn(1, 3, 1, 4, dtype=torch.float32)
    expected = torch.cat(
        [x[..., :4] * cos + x[..., 4:] * sin, x[..., :4] * (-sin) + x[..., 4:] * cos],
        3,
    ).to(x.dtype)

    out = apply_rotary_emb(x, cos, sin)
    out.square().mean().backward()

    torch.testing.assert_close(out, expected)
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()

    source = Path(__file__).resolve().parents[1] / "nanochat" / "gpt.py"
    rotary_section = source.read_text(encoding="utf-8").split(
        "def apply_rotary_emb",
        maxsplit=1,
    )[1].split(
        "def _expand_gqa_kv_heads",
        maxsplit=1,
    )[0]
    assert "out = torch.empty_like(x)" in rotary_section
    assert "out[..., :d] = y1" in rotary_section
    assert "out[..., d:] = y2" in rotary_section
    assert "torch.cat([y1, y2]" not in rotary_section
    assert "out.to(x.dtype)" not in rotary_section


def test_mlp_relu_square_reuses_buffer_without_grad_and_preserves_backward():
    x_fast = torch.randn(4, 8, dtype=torch.float32)
    expected_fast = F.relu(x_fast).square()

    with torch.no_grad():
        actual_fast = _relu_square_in_place_if_safe(x_fast)

    assert actual_fast.data_ptr() == x_fast.data_ptr()
    torch.testing.assert_close(actual_fast, expected_fast)

    x_ref = torch.randn(4, 8, dtype=torch.float32, requires_grad=True)
    x_test = x_ref.detach().clone().requires_grad_()

    expected = F.relu(x_ref).square()
    actual = _relu_square_in_place_if_safe(x_test)
    expected.sum().backward()
    actual.sum().backward()

    assert actual.data_ptr() != x_test.data_ptr()
    torch.testing.assert_close(actual, expected.detach())
    torch.testing.assert_close(x_test.grad, x_ref.grad)

    source = Path(__file__).resolve().parents[1] / "nanochat" / "gpt.py"
    gpt_source = source.read_text(encoding="utf-8")
    helper_source = gpt_source.split("def _relu_square_in_place_if_safe", maxsplit=1)[1].split(
        "class CausalSelfAttention",
        maxsplit=1,
    )[0]
    mlp_section = gpt_source.split("class MLP", maxsplit=1)[1].split(
        "class Block",
        maxsplit=1,
    )[0]

    assert "if torch.is_grad_enabled() and x.requires_grad:" in helper_source
    assert "F.relu(x, inplace=True)" in helper_source
    assert "x.square_()" in helper_source
    assert "x = _relu_square_in_place_if_safe(x)" in mlp_section
