"""Verify the complete vendored source snapshot before compiling CUDA code."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path, PurePosixPath

LAB_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = LAB_DIR / "upstream"
MANIFEST_PATH = LAB_DIR / "upstream_manifest.json"
UPSTREAM_REVISION = "2dfe5e26aecfd9e5f27bf9d5837deea01acda24b"
UPSTREAM_REPOSITORY = "https://github.com/pranjalssh/fast.cu"


def verify_upstream(root: Path = UPSTREAM_DIR, manifest_path: Path = MANIFEST_PATH) -> dict:
    """Fail on missing, changed, extra, or symlinked source files.

    The explicit paths support offline integrity checks on copied source trees.
    No download, build, GPU initialization, or benchmark happens here.
    """
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported fast.cu source manifest schema")
    if manifest.get("revision") != UPSTREAM_REVISION:
        raise ValueError("Unexpected fast.cu source revision")
    if manifest.get("repository", "").removesuffix(".git") != UPSTREAM_REPOSITORY:
        raise ValueError("Unexpected fast.cu source repository")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or "LICENSE" not in files:
        raise ValueError("fast.cu manifest must include source files and LICENSE")
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"fast.cu source root must be a real directory: {root}")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in fast.cu source tree: {path.relative_to(root)}")
        if path.is_file():
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files):
        missing = sorted(set(files) - actual)
        extra = sorted(actual - set(files))
        raise ValueError(f"fast.cu source file set changed: missing={missing}, extra={extra}")
    for relative, expected in files.items():
        path = PurePosixPath(relative)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != relative:
            raise ValueError(f"Invalid fast.cu manifest path: {relative}")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"Invalid fast.cu SHA256 for {relative}")
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f"fast.cu source integrity mismatch: {relative}")
    return manifest
