"""Shared helpers to fetch required Hugging Face models on demand."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover - huggingface_hub should be installed
    snapshot_download = None
    HF_IMPORT_ERROR = exc
else:
    HF_IMPORT_ERROR = None


def _has_complete_weight_files(target_dir: Path) -> bool:
    """Check asset presence only; model loading still validates tensor contents."""
    if not (target_dir / "config.json").is_file():
        return False
    index = target_dir / "model.safetensors.index.json"
    if not index.exists():
        weights = target_dir / "model.safetensors"
        return weights.is_file() and weights.stat().st_size > 0
    try:
        manifest = json.loads(index.read_text(encoding="utf-8"))
        weight_map = manifest.get("weight_map") if isinstance(manifest, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            return False
        for relative in weight_map.values():
            if not isinstance(relative, str) or not relative:
                return False
            shard = (target_dir / relative).resolve()
            if not shard.is_relative_to(target_dir.resolve()):
                return False
            if not shard.is_file() or shard.stat().st_size == 0:
                return False
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return True


def ensure_gpt_oss_20b(target_dir: Optional[Path] = None) -> Path:
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
            "huggingface_hub is required to fetch gpt-oss-20b "
            f"(import error: {HF_IMPORT_ERROR})"
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
