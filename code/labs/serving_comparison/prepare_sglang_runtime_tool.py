#!/usr/bin/env python3
"""Create a pinned SGLang 0.5.20 overlay that initializes labeled failure counters."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SGLANG_VERSION = "0.5.20"
PACKAGE_NAME = "sglang"
COLLECTOR_RELATIVE_PATH = Path("srt/observability/metrics_collector.py")
VERSION_RELATIVE_PATH = Path("_version.py")
RECEIPT_NAME = "runtime-receipt.json"
AUTHORIZED_COLLECTOR_SHA256 = "0d85bbc763f44cec6ac7dbab3a70a5b75cd225b7d1a619994b1100fe4a16f4ff"
PATCH_IDENTITY = "sglang-0.5.20-initialize-failure-counter-labels-v1"
PACKAGE_DIGEST_ALGORITHM = "sha256(canonical-json(relative-file-manifest-v1))"

_COUNTER_DEFINITIONS = b"""        self.num_bootstrap_failed_reqs = Counter(
            name="sglang:num_bootstrap_failed_reqs_total",
            documentation="The number of bootstrap failed requests.",
            labelnames=labels.keys(),
        )
        self.num_transfer_failed_reqs = Counter(
            name="sglang:num_transfer_failed_reqs_total",
            documentation="The number of transfer failed requests.",
            labelnames=labels.keys(),
        )
"""
_COUNTER_INITIALIZATION = b"""        self.num_bootstrap_failed_reqs.labels(**self.labels)
        self.num_transfer_failed_reqs.labels(**self.labels)
"""


class PreparationError(ValueError):
    """The requested runtime overlay cannot be prepared safely."""


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as exc:
        raise PreparationError(f"cannot read package file {path}: {exc}") from exc
    return digest.hexdigest()


def _scan_package(package_dir: Path) -> list[dict[str, Any]]:
    if package_dir.is_symlink():
        raise PreparationError(f"package directory must not be a symlink: {package_dir}")
    if not package_dir.is_dir():
        raise PreparationError(f"package directory does not exist: {package_dir}")

    manifest: list[dict[str, Any]] = []
    try:
        for current_root, directory_names, file_names in os.walk(
            package_dir, topdown=True, followlinks=False
        ):
            current = Path(current_root)
            retained_directories: list[str] = []
            for name in sorted(directory_names):
                entry = current / name
                if entry.is_symlink():
                    raise PreparationError(f"package tree contains a symlink: {entry}")
                if not entry.is_dir():
                    raise PreparationError(f"package tree contains a non-directory entry: {entry}")
                if name != "__pycache__":
                    retained_directories.append(name)
            directory_names[:] = retained_directories

            for name in sorted(file_names):
                entry = current / name
                if entry.is_symlink():
                    raise PreparationError(f"package tree contains a symlink: {entry}")
                if not entry.is_file():
                    raise PreparationError(f"package tree contains a non-file entry: {entry}")
                if entry.suffix == ".pyc":
                    continue
                relative_path = entry.relative_to(package_dir).as_posix()
                manifest.append(
                    {
                        "path": relative_path,
                        "sha256": f"sha256:{_sha256_file(entry)}",
                        "size_bytes": entry.stat().st_size,
                    }
                )
    except OSError as exc:
        raise PreparationError(f"cannot scan package directory {package_dir}: {exc}") from exc

    manifest.sort(key=lambda item: item["path"])
    if not manifest:
        raise PreparationError(f"package directory contains no retained files: {package_dir}")
    return manifest


def _manifest_digest(manifest: list[dict[str, Any]]) -> str:
    encoded = json.dumps(
        manifest,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(encoded).hexdigest()}"


def _manifest_by_path(manifest: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {item["path"]: item for item in manifest}


def _path_summary(paths: list[str], limit: int = 10) -> str:
    sample = paths[:limit]
    remaining = len(paths) - len(sample)
    suffix = f", remaining={remaining}" if remaining else ""
    return f"count={len(paths)}, sample={sample}{suffix}"


def _read_version(version_path: Path) -> str:
    try:
        source = version_path.read_text(encoding="utf-8")
        module = ast.parse(source, filename=str(version_path))
    except FileNotFoundError as exc:
        raise PreparationError(f"SGLang version file is missing: {version_path}") from exc
    except (OSError, UnicodeError, SyntaxError) as exc:
        raise PreparationError(f"cannot parse SGLang version file {version_path}: {exc}") from exc

    assignments: dict[str, list[str]] = {"__version__": [], "version": []}
    for node in module.body:
        if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
            continue
        if not isinstance(node.value.value, str):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in assignments:
                assignments[target.id].append(node.value.value)

    expected = [SGLANG_VERSION]
    if assignments["__version__"] != expected or assignments["version"] != expected:
        raise PreparationError(
            f"SGLang package must declare __version__ and version exactly as {SGLANG_VERSION}"
        )
    return SGLANG_VERSION


def _patch_collector(source: bytes) -> bytes:
    source_sha256 = _sha256_bytes(source)
    if source_sha256 != AUTHORIZED_COLLECTOR_SHA256:
        raise PreparationError(
            f"SGLang metrics collector is not the authorized 0.5.20 source: sha256:{source_sha256}"
        )
    if source.count(_COUNTER_DEFINITIONS) != 1:
        raise PreparationError("authorized collector does not contain the unique patch anchor")
    if _COUNTER_INITIALIZATION in source:
        raise PreparationError("authorized collector already contains the counter initialization")
    patched = source.replace(
        _COUNTER_DEFINITIONS,
        _COUNTER_DEFINITIONS + _COUNTER_INITIALIZATION,
        1,
    )
    if b".inc(0)" in patched:
        raise PreparationError("patched collector must not initialize counters with inc(0)")
    return patched


def _copy_ignore(directory: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    root = Path(directory)
    for name in names:
        entry = root / name
        if (name == "__pycache__" and entry.is_dir()) or (
            name.endswith(".pyc") and entry.is_file()
        ):
            ignored.add(name)
    return ignored


def _relative_file_manifest(
    original: list[dict[str, Any]], output: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    original_by_path = _manifest_by_path(original)
    output_by_path = _manifest_by_path(output)
    if set(original_by_path) != set(output_by_path):
        missing = sorted(set(original_by_path) - set(output_by_path))
        added = sorted(set(output_by_path) - set(original_by_path))
        raise PreparationError(
            "output package file set differs from source, "
            f"missing({_path_summary(missing)}), added({_path_summary(added)})"
        )
    return [
        {
            "path": path,
            "original_sha256": original_by_path[path]["sha256"],
            "output_sha256": output_by_path[path]["sha256"],
            "original_size_bytes": original_by_path[path]["size_bytes"],
            "output_size_bytes": output_by_path[path]["size_bytes"],
        }
        for path in sorted(original_by_path)
    ]


def prepare_runtime_overlay(package_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Copy and patch one authorized SGLang package into a new import overlay."""
    package_argument = Path(package_dir)
    output_argument = Path(output_dir)
    if package_argument.name != PACKAGE_NAME:
        raise PreparationError(f"package directory must be named {PACKAGE_NAME}")
    if package_argument.is_symlink():
        raise PreparationError(f"package directory must not be a symlink: {package_argument}")
    try:
        source_package = package_argument.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PreparationError(f"package directory does not exist: {package_argument}") from exc
    if not source_package.is_dir():
        raise PreparationError(f"package directory is not a directory: {package_argument}")

    if os.path.lexists(output_argument):
        raise PreparationError(f"output directory already exists: {output_argument}")
    try:
        output_parent = output_argument.parent.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PreparationError(
            f"output parent directory does not exist: {output_argument.parent}"
        ) from exc
    if not output_parent.is_dir():
        raise PreparationError(f"output parent is not a directory: {output_parent}")
    resolved_output = output_parent / output_argument.name
    if resolved_output.is_relative_to(source_package):
        raise PreparationError("output directory must not be nested inside the source package")

    original_manifest = _scan_package(source_package)
    version = _read_version(source_package / VERSION_RELATIVE_PATH)
    original_by_path = _manifest_by_path(original_manifest)
    collector_key = COLLECTOR_RELATIVE_PATH.as_posix()
    collector_entry = original_by_path.get(collector_key)
    if collector_entry is None:
        raise PreparationError(f"SGLang collector is missing: {COLLECTOR_RELATIVE_PATH}")
    original_collector_sha256 = collector_entry["sha256"]
    if original_collector_sha256 != f"sha256:{AUTHORIZED_COLLECTOR_SHA256}":
        raise PreparationError(
            "SGLang metrics collector is not the authorized 0.5.20 source: "
            f"{original_collector_sha256}"
        )
    try:
        original_collector = (source_package / COLLECTOR_RELATIVE_PATH).read_bytes()
    except OSError as exc:
        raise PreparationError(f"cannot read SGLang collector: {exc}") from exc
    patched_collector = _patch_collector(original_collector)
    patched_collector_sha256 = f"sha256:{_sha256_bytes(patched_collector)}"

    resolved_output.mkdir()
    output_package = resolved_output / PACKAGE_NAME
    shutil.copytree(source_package, output_package, copy_function=shutil.copy2, ignore=_copy_ignore)
    (output_package / COLLECTOR_RELATIVE_PATH).write_bytes(patched_collector)

    source_after_copy = _scan_package(source_package)
    if source_after_copy != original_manifest:
        raise PreparationError("source package changed while the overlay was being prepared")
    output_manifest = _scan_package(output_package)
    relative_manifest = _relative_file_manifest(original_manifest, output_manifest)
    changed_files = [
        item["path"]
        for item in relative_manifest
        if item["original_sha256"] != item["output_sha256"]
        or item["original_size_bytes"] != item["output_size_bytes"]
    ]
    if changed_files != [collector_key]:
        raise PreparationError(
            "overlay did not change exactly the authorized collector: "
            f"{_path_summary(changed_files)}"
        )
    output_by_path = _manifest_by_path(output_manifest)
    if output_by_path[collector_key]["sha256"] != patched_collector_sha256:
        raise PreparationError("output collector digest does not match the deterministic patch")

    original_package_digest = _manifest_digest(original_manifest)
    output_package_digest = _manifest_digest(output_manifest)
    receipt: dict[str, Any] = {
        "schema_version": "serving-comparison.sglang-runtime-receipt.v1",
        "package_name": PACKAGE_NAME,
        "sglang_version": version,
        "patch_identity": PATCH_IDENTITY,
        "collector_relative_path": collector_key,
        "collector_original_sha256": original_collector_sha256,
        "collector_patched_sha256": patched_collector_sha256,
        "original_package_dir": str(source_package),
        "output_package_dir": PACKAGE_NAME,
        "package_digest_algorithm": PACKAGE_DIGEST_ALGORITHM,
        "original_package_digest": original_package_digest,
        "output_package_digest": output_package_digest,
        "runtime_build_id": output_package_digest,
        "changed_files": changed_files,
        "excluded_from_copy_and_digests": ["**/__pycache__/**", "**/*.pyc"],
        "relative_file_manifest": relative_manifest,
    }
    receipt_path = resolved_output / RECEIPT_NAME
    receipt_temporary_path = resolved_output / f".{RECEIPT_NAME}.tmp"
    receipt_temporary_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(receipt_temporary_path, receipt_path)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--package-dir",
        required=True,
        type=Path,
        help="Installed SGLang package directory named sglang",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="New overlay root that will contain sglang and runtime-receipt.json",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    arguments = parser.parse_args(argv)
    try:
        receipt = prepare_runtime_overlay(arguments.package_dir, arguments.output_dir)
    except (OSError, PreparationError) as exc:
        parser.error(str(exc))
    print(
        json.dumps(
            {
                "changed_files": receipt["changed_files"],
                "receipt": str(Path(arguments.output_dir) / RECEIPT_NAME),
                "runtime_build_id": receipt["runtime_build_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
