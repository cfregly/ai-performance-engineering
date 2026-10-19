"""Shared helpers to fetch required Hugging Face models on demand."""

from __future__ import annotations

import json
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover - huggingface_hub should be installed
    snapshot_download = None
    HF_IMPORT_ERROR = exc
else:
    HF_IMPORT_ERROR = None


def _allowed_model_roots(target_dir: Path) -> tuple[Path, ...]:
    """Return filesystem roots that may contain files for this model directory."""
    target_root = target_dir.resolve()
    roots = [target_root]
    snapshots_dir = target_root.parent
    if snapshots_dir.name != "snapshots":
        return tuple(roots)

    repository_dir = snapshots_dir.parent.resolve()
    try:
        blobs_dir = (repository_dir / "blobs").resolve(strict=True)
    except (OSError, RuntimeError):
        return tuple(roots)
    if blobs_dir.is_dir() and blobs_dir.is_relative_to(repository_dir):
        roots.append(blobs_dir)
    return tuple(roots)


def _is_allowed_model_file(path: Path, allowed_roots: tuple[Path, ...]) -> bool:
    try:
        resolved = path.resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    return resolved.is_file() and any(resolved.is_relative_to(root) for root in allowed_roots)


def _has_complete_weight_files(target_dir: Path) -> bool:
    """Check asset presence only; model loading still validates tensor contents."""
    allowed_roots = _allowed_model_roots(target_dir)
    if not _is_allowed_model_file(target_dir / "config.json", allowed_roots):
        return False
    index = target_dir / "model.safetensors.index.json"
    if not index.exists():
        weights = target_dir / "model.safetensors"
        return _is_allowed_model_file(weights, allowed_roots) and weights.stat().st_size > 0
    try:
        if not _is_allowed_model_file(index, allowed_roots):
            return False
        manifest = json.loads(index.read_text(encoding="utf-8"))
        weight_map = manifest.get("weight_map") if isinstance(manifest, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        for relative in weight_map.values():
            if not isinstance(relative, str) or not relative:
                return False
            relative_path = Path(relative)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                return False
            shard = target_dir / relative_path
            if not _is_allowed_model_file(shard, allowed_roots) or shard.stat().st_size == 0:
                return False
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return True


def ensure_gpt_oss_20b(target_dir: Path | None = None) -> Path:
    """Ensure the openai/gpt-oss-20b model is present locally.

    Downloads the full repository (including safetensors) into
    ``gpt-oss-20b/original`` at repo root by default.
    """
    if target_dir is None:
        target_dir = Path(__file__).resolve().parents[2] / "gpt-oss-20b" / "original"
    target_dir = Path(target_dir).expanduser().resolve()
    if _has_complete_weight_files(target_dir):
        return target_dir

    if snapshot_download is None:
        raise RuntimeError(
            f"huggingface_hub is required to fetch gpt-oss-20b (import error: {HF_IMPORT_ERROR})"
        )

    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="openai/gpt-oss-20b",
        local_dir=str(target_dir),
        # Fetch full repo including safetensors to avoid missing-weight errors.
        allow_patterns=None,
    )
    if not _has_complete_weight_files(target_dir):
        raise RuntimeError(
            f"Downloaded model at {target_dir} is missing config.json or complete safetensors weights"
        )
    return target_dir
