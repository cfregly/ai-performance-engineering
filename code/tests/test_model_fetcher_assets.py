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


def _hugging_face_snapshot(cache_root: Path) -> Path:
    repository = cache_root / "models--openai--gpt-oss-20b"
    snapshot = repository / "snapshots" / "revision"
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    files = {
        "config.json": b"{}",
        "model.safetensors.index.json": json.dumps(
            {"weight_map": {"layer0": "first.safetensors", "layer1": "second.safetensors"}}
        ).encode(),
        "first.safetensors": b"first-asset-presence-only",
        "second.safetensors": b"second-asset-presence-only",
    }
    for position, (name, content) in enumerate(files.items()):
        blob_name = f"blob-{position}"
        (blobs / blob_name).write_bytes(content)
        (snapshot / name).symlink_to(Path("../..") / "blobs" / blob_name)
    return snapshot


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


def test_complete_hugging_face_snapshot_symlinks_are_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = _hugging_face_snapshot(tmp_path)
    monkeypatch.setattr(model_fetcher, "snapshot_download", None)
    assert model_fetcher.ensure_gpt_oss_20b(snapshot) == snapshot.resolve()


def test_hugging_face_snapshot_rejects_shard_symlink_outside_model_roots(
    tmp_path: Path,
) -> None:
    snapshot = _hugging_face_snapshot(tmp_path)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"unrelated")
    shard = snapshot / "first.safetensors"
    shard.unlink()
    shard.symlink_to(outside)
    assert not model_fetcher._has_complete_weight_files(snapshot)


@pytest.mark.parametrize("blob_state", ["broken", "empty"])
def test_hugging_face_snapshot_rejects_incomplete_blob_targets(
    tmp_path: Path, blob_state: str
) -> None:
    snapshot = _hugging_face_snapshot(tmp_path)
    blob = (snapshot / "first.safetensors").resolve()
    if blob_state == "broken":
        blob.unlink()
    else:
        blob.write_bytes(b"")
    assert not model_fetcher._has_complete_weight_files(snapshot)


@pytest.mark.parametrize("path_kind", ["absolute", "parent-traversal"])
def test_manifest_paths_cannot_address_blobs_directly(tmp_path: Path, path_kind: str) -> None:
    repository = tmp_path / "models--openai--gpt-oss-20b"
    snapshot = repository / "snapshots" / "revision"
    blobs = repository / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir()
    (snapshot / "config.json").write_text("{}")
    blob = blobs / "weight-blob"
    blob.write_bytes(b"asset-presence-only")
    manifest_path = str(blob) if path_kind == "absolute" else "../../blobs/weight-blob"
    (snapshot / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer": manifest_path}})
    )
    assert not model_fetcher._has_complete_weight_files(snapshot)


def test_manifest_shard_symlink_outside_model_roots_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "model"
    _indexed_model(directory)
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"unrelated")
    shard = directory / "first.safetensors"
    shard.unlink()
    shard.symlink_to(outside)
    assert not model_fetcher._has_complete_weight_files(directory)


def test_complete_local_weights_return_an_absolute_path_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "model"
    _indexed_model(directory)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(model_fetcher, "snapshot_download", None)
    assert model_fetcher.ensure_gpt_oss_20b(Path("model")) == directory


def test_config_only_directory_requires_downloader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "config.json").write_text("{}")
    monkeypatch.setattr(model_fetcher, "snapshot_download", None)
    with pytest.raises(RuntimeError, match="huggingface_hub is required"):
        model_fetcher.ensure_gpt_oss_20b(tmp_path)
