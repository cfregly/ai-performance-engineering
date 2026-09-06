"""Real filesystem asset checks; these do not assert model inference validity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from labs.common import model_fetcher


def _indexed_model(directory: Path) -> None:
    directory.mkdir()
    (directory / "config.json").write_text("{}")
    (directory / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer0": "first.safetensors", "layer1": "second.safetensors"}})
    )
    (directory / "first.safetensors").write_bytes(b"asset-presence-only")
    (directory / "second.safetensors").write_bytes(b"asset-presence-only")


def test_config_alone_is_not_a_complete_model(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("{}")
    assert not model_fetcher._has_complete_weight_files(tmp_path)


@pytest.mark.parametrize("state", ["missing", "empty", "invalid-index", "external-shard"])
def test_incomplete_indexed_weights_are_rejected(tmp_path: Path, state: str) -> None:
    directory = tmp_path / "model"
    _indexed_model(directory)
    index = directory / "model.safetensors.index.json"
    if state == "missing":
        index.write_text(json.dumps({"weight_map": {"layer": "absent.safetensors"}}))
    elif state == "empty":
        (directory / "second.safetensors").write_bytes(b"")
    elif state == "invalid-index":
        index.write_text("not-json")
    else:
        (tmp_path / "external.safetensors").write_bytes(b"unrelated")
        index.write_text(json.dumps({"weight_map": {"layer": "../external.safetensors"}}))
    assert not model_fetcher._has_complete_weight_files(directory)


def test_complete_local_weights_return_an_absolute_path_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "model"
    _indexed_model(directory)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_fetcher, "snapshot_download", None)
    assert model_fetcher.ensure_gpt_oss_20b(Path("model")) == directory


def test_config_only_directory_requires_downloader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(model_fetcher, "snapshot_download", None)
    with pytest.raises(RuntimeError, match="huggingface_hub is required"):
        model_fetcher.ensure_gpt_oss_20b(tmp_path)
