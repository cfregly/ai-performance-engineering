#!/usr/bin/env python3
"""Reconstruct the fixed public-text corpus from exact committed Git blobs."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

SOURCE_COMMIT = "03ff53d63b5ff91e6ddf556619400c6d277a7d29"
SOURCE_REPOSITORY = "https://github.com/cfregly/ai-performance-engineering"
EXTENSIONS = {
    "bazel", "bzl", "c", "cc", "cfg", "cmake", "cpp", "css", "cu", "cuh",
    "csv", "go", "h", "hpp", "html", "in", "ini", "ipynb", "java", "js",
    "json", "jsonl", "jsx", "lock", "log", "md", "nix", "patch", "proto",
    "py", "rs", "rst", "sh", "sql", "svg", "toml", "ts", "tsx", "txt",
    "xml", "yaml", "yml",
}
SOURCE_BLOB_BYTES = 32 * 1024 * 1024
PART_BYTES = 30 * 1024 * 1024
MAX_CORPUS_BYTES = 128 * 1024 * 1024
MAX_LEDGER_BYTES = 2 * 1024 * 1024
MAX_FILES = 4096
SELECTION = (
    "Distinct nonempty regular committed blobs with the fixed public text-artifact "
    "extension set, size <=32MiB, sorted by descending byte size then ascending path, "
    "first4096. No repeated Git object or SHA-256 content."
)

EXPECTED_SOURCE_BYTES = 107_929_355
EXPECTED_LEDGER = {
    "artifactName": "inputs/corpus-ledger.json",
    "sha256": "c4e456506ba8431a24dd5145622d750d9782b5a8826429062a71b874a9d12406",
    "sizeBytes": 906_568,
}
EXPECTED_CORPUS_BYTES = 112_537_600
EXPECTED_CORPUS_SHA256 = "cf533d52f44ab5c2fef3a79c6d8f4f821083a0a1f0ff961e63981ac8f31a9e4c"
EXPECTED_INDEX_BYTES = 888
EXPECTED_INDEX_SHA256 = "6a5bc63255247dff8f8398b4eb98d40e3d0dc589971968f59dbe0bbfbb899680"
EXPECTED_PARTS = [
    {
        "artifactName": "inputs/part-000.bin",
        "sha256": "ddc40c1add2581c0907c78e3cd1f617a2e07e08ddd7dbb56062aa742640532e2",
        "sizeBytes": 31_457_280,
    },
    {
        "artifactName": "inputs/part-001.bin",
        "sha256": "71ca360ba1c0a8953e758671cdd7ddae89aa933ad1cddf6ac1303c50306e5fda",
        "sizeBytes": 31_457_280,
    },
    {
        "artifactName": "inputs/part-002.bin",
        "sha256": "5c7e64b1ee1a7005504e0ec69b13cacfe4f9d4f4c37c989cdee807d03fd39a92",
        "sizeBytes": 31_457_280,
    },
    {
        "artifactName": "inputs/part-003.bin",
        "sha256": "9b1caf463fc6e7299ab31c8ab6ffc42492e5d142c6d05e1707592621fa86631b",
        "sizeBytes": 18_165_760,
    },
]


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def reference(name: str, content: bytes) -> dict:
    return {
        "artifactName": name,
        "sha256": hashlib.sha256(content).hexdigest(),
        "sizeBytes": len(content),
    }


def git(root: Path, *args: str) -> bytes:
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_OPTIONAL_LOCKS="0", GIT_NO_REPLACE_OBJECTS="1")
    return subprocess.check_output(["git", "-C", str(root), *args], env=environment)


def _extension(path: str) -> str:
    name = path.rsplit("/", 1)[-1]
    return name.rsplit(".", 1)[-1].lower() if "." in name else ""


def select_entries(lines: list[str], limit: int = MAX_FILES) -> list[dict]:
    """Select mode-100644 blobs, retaining the first path for each object ID."""
    seen, candidates = set(), []
    for line in lines:
        metadata, path = line.split("\t", 1)
        mode, kind, oid, count = metadata.split()
        if kind != "blob" or mode != "100644" or count == "-":
            continue
        size = int(count)
        if (
            not 0 < size <= SOURCE_BLOB_BYTES
            or _extension(path) not in EXTENSIONS
            or oid in seen
        ):
            continue
        seen.add(oid)
        candidates.append({"path": path, "git_blob": oid, "size_bytes": size})
    return sorted(candidates, key=lambda item: (-item["size_bytes"], item["path"]))[:limit]


def select(root: Path) -> list[dict]:
    lines = git(root, "ls-tree", "-r", "-l", "--full-tree", SOURCE_COMMIT).decode().splitlines()
    selected = select_entries(lines)
    if len(selected) != MAX_FILES:
        raise ValueError("immutable source does not supply the declared 4,096-file corpus")
    return selected


def _git_blob_id(content: bytes) -> str:
    header = b"blob " + str(len(content)).encode() + b"\0"
    return hashlib.sha1(header + content).hexdigest()


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} differs from the preregistered input authority")


def build(root: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError("output must be new")
    selected = select(root)
    objects = {}
    for entry in selected:
        content = git(root, "cat-file", "blob", entry["git_blob"])
        if _git_blob_id(content) != entry["git_blob"] or len(content) != entry["size_bytes"]:
            raise ValueError("Git blob identity mismatch")
        entry["sha256"] = hashlib.sha256(content).hexdigest()
        objects[entry["path"]] = content
    if len({entry["sha256"] for entry in selected}) != len(selected):
        raise ValueError("SHA-256 content duplicates are forbidden")
    source_bytes = sum(entry["size_bytes"] for entry in selected)
    _require_equal(source_bytes, EXPECTED_SOURCE_BYTES, "selected source bytes")

    ledger = canonical(
        {
            "schema_version": 1,
            "kind": "public_committed_artifact_corpus",
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "selection": SELECTION,
            "extensions": sorted(EXTENSIONS),
            "source_file_count": len(selected),
            "source_bytes": source_bytes,
            "files": sorted(selected, key=lambda item: item["path"]),
        }
    )
    if len(ledger) > MAX_LEDGER_BYTES:
        raise ValueError("ledger exceeds the 2 MiB finite artifact cap")
    _require_equal(reference("inputs/corpus-ledger.json", ledger), EXPECTED_LEDGER, "ledger")

    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.PAX_FORMAT) as stream:
        members = [
            ("CORPUS.json", ledger),
            *[("source/" + name, content) for name, content in sorted(objects.items())],
        ]
        for name, content in members:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ""
            stream.addfile(info, io.BytesIO(content))
    corpus = archive.getvalue()
    if len(corpus) > MAX_CORPUS_BYTES:
        raise ValueError("archive exceeds the 128 MiB finite corpus cap")
    _require_equal(len(corpus), EXPECTED_CORPUS_BYTES, "deterministic tar bytes")
    _require_equal(hashlib.sha256(corpus).hexdigest(), EXPECTED_CORPUS_SHA256, "deterministic tar hash")

    part_contents, parts = [], []
    for number, offset in enumerate(range(0, len(corpus), PART_BYTES)):
        name = f"part-{number:03d}.bin"
        content = corpus[offset:offset + PART_BYTES]
        part_contents.append((name, content))
        parts.append(reference("inputs/" + name, content))
    _require_equal(parts, EXPECTED_PARTS, "part bindings")

    index = {
        "schemaVersion": "piq.artifact-compression-chunks.v1",
        "corpusSha256": hashlib.sha256(corpus).hexdigest(),
        "corpusBytes": len(corpus),
        "sourceFileCount": len(selected),
        "ledger": reference("inputs/corpus-ledger.json", ledger),
        "chunks": parts,
    }
    index_bytes = canonical(index)
    _require_equal(len(index_bytes), EXPECTED_INDEX_BYTES, "canonical input-index bytes")
    _require_equal(hashlib.sha256(index_bytes).hexdigest(), EXPECTED_INDEX_SHA256, "input-index hash")
    receipt = {
        "source_repository": SOURCE_REPOSITORY,
        "source_commit": SOURCE_COMMIT,
        "selection": SELECTION,
        "source_file_count": len(selected),
        "source_bytes": source_bytes,
        "corpus_sha256": index["corpusSha256"],
        "corpus_bytes": len(corpus),
        "index": reference("input-index.json", index_bytes),
        "ledger": index["ledger"],
        "parts": parts,
        "compression_measured": False,
    }

    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    (inputs / "corpus-ledger.json").write_bytes(ledger)
    for name, content in part_contents:
        (inputs / name).write_bytes(content)
    (output / "input-index.json").write_bytes(index_bytes)
    (output / "INPUT-RECEIPT.json").write_bytes(canonical(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repository, args.output), indent=2))
