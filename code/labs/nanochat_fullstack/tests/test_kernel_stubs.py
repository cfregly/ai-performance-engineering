import pytest
import torch

from nanochat.engine import Engine, KVCache
from nanochat.gpt import GPT, GPTConfig
from nanochat.kernels import stubs


CALLS = {"clustered": 0, "decode": 0}


class _Tok:
    def get_bos_token_id(self):
        return 0
    def encode_special(self, token):
        return 1


def _zero_clustered(q, k, v, attn_mask=None, causal=True, num_sm_clusters=None, enable_gqa=False):
    CALLS["clustered"] += 1
    return torch.zeros_like(q)


def _zero_decode(model, ids, kv_cache, attention_mask=None, token_mask=None):
    CALLS["decode"] += 1
    vocab = model.config.vocab_size
    return torch.zeros((ids.size(0), ids.size(1), vocab), device=ids.device, dtype=model.transformer.wte.weight.dtype)


def _cfg(**kw):
    return GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=8,
        **kw,
    )


@pytest.fixture(autouse=True)
def _reset_kernels():
    # Ensure each test sees a clean registry and no pre-registered defaults.
    stubs.clear_kernel_overrides()
    CALLS["clustered"] = 0
    CALLS["decode"] = 0
    yield
    # Restore defaults for any downstream tests
    stubs.reset_kernel_overrides()


def test_clustered_attention_stub_raises(monkeypatch):
    config = _cfg(use_clustered_attention_kernel=True, allow_kernel_stub_fallback=False)
    model = GPT(config)
    x = torch.zeros((1, 4), dtype=torch.long)
    kv_cache = None
    with torch.inference_mode(), pytest.raises(NotImplementedError):
        model(x, kv_cache=kv_cache)


def test_clustered_attention_stub_fallback():
    # Register built-in Python kernel so we don't rely on fallback behavior
    stubs.register_default_kernels()
    config = _cfg(use_clustered_attention_kernel=True, allow_kernel_stub_fallback=False)
    model = GPT(config)
    x = torch.zeros((1, 4), dtype=torch.long)
    kv_cache = None
    with torch.inference_mode():
        out = model(x, kv_cache=kv_cache)
    assert out.shape == (1, 4, config.vocab_size)


def test_clustered_attention_custom_impl_from_flag():
    config = _cfg(
        use_clustered_attention_kernel=True,
        clustered_attention_impl=f"{__name__}:_zero_clustered",
        allow_kernel_stub_fallback=False,
    )
    model = GPT(config)
    assert model.transformer.h[0].attn.clustered_attention_kernel is _zero_clustered
    x = torch.zeros((1, 2), dtype=torch.long)
    kv_cache = None
    with torch.inference_mode():
        out = model(x, kv_cache=kv_cache)
    assert CALLS["clustered"] == 1
    assert out.shape == (1, 2, config.vocab_size)


def test_gpt_reuses_position_offsets_for_padded_kv_cache():
    model = GPT(_cfg(use_padded_attention=True))

    first = model._position_offsets_for(4, torch.device("cpu"))
    first_ptr = first.data_ptr()
    second = model._position_offsets_for(3, torch.device("cpu"))

    assert second.data_ptr() == first_ptr
    torch.testing.assert_close(second, torch.arange(3, dtype=torch.long))

    grown = model._position_offsets_for(8, torch.device("cpu"))
    assert grown.numel() == 8
    assert grown.data_ptr() != first_ptr
    torch.testing.assert_close(grown, torch.arange(8, dtype=torch.long))


def test_persistent_decode_kernel_stub_raises():
    config = _cfg(use_persistent_decode_kernel=True, allow_kernel_stub_fallback=False)
    model = GPT(config)
    tok = _Tok()
    engine = Engine(model, tok, enable_batch_decode=False)
    ids = torch.zeros((1, 1), dtype=torch.long)
    kv = engine._kv_cache_params(batch_size=1, seq_len=2)
    kv_cache = KVCache(**kv)
    with pytest.raises(NotImplementedError):
        engine._decode_forward_step(ids, kv_cache)


def test_persistent_decode_kernel_fallback():
    # Register built-in Python kernel so we don't rely on fallback behavior
    stubs.register_default_kernels()
    config = _cfg(use_persistent_decode_kernel=True, allow_kernel_stub_fallback=False)
    model = GPT(config)
    tok = _Tok()
    engine = Engine(model, tok, enable_batch_decode=False)
    ids = torch.zeros((1, 1), dtype=torch.long)
    kv = engine._kv_cache_params(batch_size=1, seq_len=2)
    kv_cache = KVCache(**kv)
    out = engine._decode_forward_step(ids, kv_cache)
    assert out.shape == (1, 1, config.vocab_size)


def test_persistent_decode_custom_impl_from_flag():
    stubs.reset_kernel_overrides()
    config = _cfg(
        use_persistent_decode_kernel=True,
        persistent_decode_impl=f"{__name__}:_zero_decode",
        allow_kernel_stub_fallback=False,
    )
    model = GPT(config)
    tok = _Tok()
    engine = Engine(model, tok, enable_batch_decode=False)
    ids = torch.zeros((1, 1), dtype=torch.long)
    kv = engine._kv_cache_params(batch_size=1, seq_len=2)
    kv_cache = KVCache(**kv)
    out = engine._decode_forward_step(ids, kv_cache)
    assert engine._persistent_decode_kernel is _zero_decode
    assert CALLS["decode"] == 1
    assert out.shape == (1, 1, config.vocab_size)
