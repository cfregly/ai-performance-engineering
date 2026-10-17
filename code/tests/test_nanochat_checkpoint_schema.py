"""Checkpoint-boundary coverage for NanoChat model config schema migration."""

from __future__ import annotations

import logging

import pytest
import torch

import labs.nanochat_fullstack  # noqa: F401  (registers the historical nanochat package)
from nanochat import checkpoint_manager
from nanochat.gpt import GPT, GPTConfig


OFFICIAL_MODEL_CONFIG = {
    "sequence_len": 8,
    "vocab_size": 32,
    "n_layer": 1,
    "n_head": 2,
    "n_kv_head": 1,
    "n_embd": 8,
}


class _TokenizerBoundaryStub:
    """Isolate tokenizer artifact I/O after the real model has been constructed."""

    def get_vocab_size(self):
        return OFFICIAL_MODEL_CONFIG["vocab_size"]


def _write_real_checkpoint(checkpoint_dir, model_config):
    torch.manual_seed(718)
    source_config = GPTConfig(
        **OFFICIAL_MODEL_CONFIG,
        use_flash_attn_varlen=False,
    )
    source_model = GPT(source_config).eval()
    source_state = {
        key: value.detach().clone()
        for key, value in source_model.state_dict().items()
    }
    checkpoint_manager.save_checkpoint(
        str(checkpoint_dir),
        step=1,
        model_data=source_state,
        optimizer_data=None,
        meta_data={"model_config": model_config, "source": "tempfile regression"},
    )
    return source_state


def _build_from_tempfile(monkeypatch, checkpoint_dir):
    monkeypatch.setattr(
        checkpoint_manager,
        "get_tokenizer",
        lambda: _TokenizerBoundaryStub(),
    )
    return checkpoint_manager.build_model(
        str(checkpoint_dir),
        step=1,
        device=torch.device("cpu"),
        phase="eval",
    )


def _assert_state_loaded(model, source_state):
    loaded_state = model.state_dict()
    assert loaded_state.keys() == source_state.keys()
    for key, expected in source_state.items():
        torch.testing.assert_close(loaded_state[key], expected, rtol=0.0, atol=0.0)


def test_official_six_field_checkpoint_builds_without_migration(monkeypatch, tmp_path):
    assert set(OFFICIAL_MODEL_CONFIG) == {
        "sequence_len",
        "vocab_size",
        "n_layer",
        "n_head",
        "n_kv_head",
        "n_embd",
    }
    source_state = _write_real_checkpoint(tmp_path, dict(OFFICIAL_MODEL_CONFIG))

    model, tokenizer, metadata = _build_from_tempfile(monkeypatch, tmp_path)

    _assert_state_loaded(model, source_state)
    assert not model.training
    assert tokenizer.get_vocab_size() == OFFICIAL_MODEL_CONFIG["vocab_size"]
    assert model.config.use_flash_attn_varlen is True
    assert metadata["model_config"] == OFFICIAL_MODEL_CONFIG
    assert "model_config_schema_migration" not in metadata


def test_legacy_checkpoint_maps_fa2_flag_and_records_discarded_hints(
    monkeypatch,
    tmp_path,
    caplog,
):
    legacy_model_config = {
        **OFFICIAL_MODEL_CONFIG,
        "use_fp32_logits": True,
        "use_flash_sdp": True,
        "use_padded_attention": False,
        "use_flash3": False,
        "flash3_block_size": 128,
        "kv_block_size": 32,
        "kv_page_size": 1024,
        "enable_persistent_decode": True,
        "use_cuda_graphs": False,
        "use_te_weight_only": False,
        "te_weight_dtype": "fp8",
        "use_cta_clustering": True,
        "cta_cluster_size": 2,
        "cta_cluster_seq_threshold": 8,
        "use_clustered_attention_kernel": False,
        "use_persistent_decode_kernel": False,
        "clustered_attention_impl": None,
        "persistent_decode_impl": None,
        "allow_kernel_stub_fallback": False,
    }
    source_state = _write_real_checkpoint(tmp_path, legacy_model_config)
    caplog.set_level(logging.INFO, logger=checkpoint_manager.__name__)

    model, _, metadata = _build_from_tempfile(monkeypatch, tmp_path)

    _assert_state_loaded(model, source_state)
    assert model.config.use_flash_attn_varlen is False
    assert model.config.enable_persistent_decode is True
    assert model.config.kv_block_size == 32
    assert model.config.kv_page_size == 1024
    assert not hasattr(model.config, "use_flash3")
    normalized = metadata["model_config"]
    assert normalized["use_flash_attn_varlen"] is False
    for removed in (
        "use_flash3",
        "flash3_block_size",
        "use_cta_clustering",
        "cta_cluster_size",
        "cta_cluster_seq_threshold",
    ):
        assert removed not in normalized
    note = metadata["model_config_schema_migration"]
    assert "mapped use_flash3 to use_flash_attn_varlen" in note
    assert "discarded inert metadata" in note
    assert note in caplog.messages


@pytest.mark.parametrize(
    ("legacy_value", "current_value"),
    [(False, False), (False, True)],
)
def test_checkpoint_rejects_mixed_old_and_current_enable_flags(
    monkeypatch,
    tmp_path,
    legacy_value,
    current_value,
):
    mixed_config = {
        **OFFICIAL_MODEL_CONFIG,
        "use_flash3": legacy_value,
        "use_flash_attn_varlen": current_value,
    }
    _write_real_checkpoint(tmp_path, mixed_config)

    with pytest.raises(ValueError, match="contains both use_flash3 and use_flash_attn_varlen"):
        _build_from_tempfile(monkeypatch, tmp_path)


def test_checkpoint_rejects_unknown_model_config_key(monkeypatch, tmp_path):
    unknown_config = {**OFFICIAL_MODEL_CONFIG, "use_flash4": True}
    _write_real_checkpoint(tmp_path, unknown_config)

    with pytest.raises(ValueError, match="Unsupported checkpoint model_config keys: use_flash4"):
        _build_from_tempfile(monkeypatch, tmp_path)


def test_public_gpt_config_keeps_removed_checkpoint_keys_rejected():
    with pytest.raises(TypeError, match="unexpected keyword argument 'use_flash3'"):
        GPTConfig(**OFFICIAL_MODEL_CONFIG, use_flash3=False)
