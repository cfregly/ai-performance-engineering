import os

import pytest
import torch

from nanochat.engine import Engine, KVCache
from nanochat.gpt import GPT, GPTConfig


class _StubTokenizer:
    def get_bos_token_id(self):
        return 0

    def encode_special(self, token):
        return 1


def _small_config(**overrides):
    cfg = GPTConfig(
        sequence_len=16,
        vocab_size=64,
        n_layer=1,
        n_head=2,
        n_kv_head=2,
        n_embd=8,
        **overrides,
    )
    return cfg


def test_cuda_graphs_disabled_when_persistent_decode_enabled(monkeypatch):
    # Avoid torch.compile in the test environment
    monkeypatch.setenv("NANOCHAT_DISABLE_COMPILE", "1")
    config = _small_config(enable_persistent_decode=True, use_cuda_graphs=True)
    model = GPT(config)
    tokenizer = _StubTokenizer()

    engine = Engine(model, tokenizer, enable_batch_decode=False)

    assert engine.enable_persistent_decode is True
    # Graphs remain enabled but use the default stream (no dedicated persistent stream)
    assert engine.use_cuda_graphs is True
    assert engine._persistent_stream is None


def test_cuda_graph_request_rejects_cpu_inputs(monkeypatch):
    monkeypatch.setenv("NANOCHAT_DISABLE_COMPILE", "1")
    config = _small_config(enable_persistent_decode=False, use_cuda_graphs=True)
    model = GPT(config)
    tokenizer = _StubTokenizer()
    engine = Engine(model, tokenizer, enable_batch_decode=False)

    kv_cache = KVCache(**engine._kv_cache_params(batch_size=1, seq_len=2))
    ids = torch.tensor([[1]], dtype=torch.long)
    with pytest.raises(RuntimeError, match="CUDA"):
        engine._execute_decode(ids, kv_cache)
