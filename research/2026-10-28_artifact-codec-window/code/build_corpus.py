#!/usr/bin/env python3
"""Reconstruct a deterministic corpus from distinct committed public artifacts."""
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
EXTENSIONS = {"py", "cu", "cuh", "cpp", "cc", "c", "h", "hpp", "json", "yaml", "yml",
              "toml", "md", "txt", "csv", "sql", "sh"}
PART_BYTES = 7 * 1024 * 1024
MAX_SOURCE_BYTES = 48 * 1024 * 1024
MAX_FILES = 1024
SELECTION = (
    "Distinct nonempty regular code/ blobs with listed text extensions and size <=7MiB, "
    "sorted by descending byte size then ascending path, first1024 while skipping "
    "blobs that exceed a cumulative48MiB source-byte cap. No repeated content."
)


def canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def reference(name: str, content: bytes) -> dict:
    return {"artifactName": name, "sha256": hashlib.sha256(content).hexdigest(), "sizeBytes": len(content)}


def git(root: Path, *args: str) -> bytes:
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment.update(GIT_OPTIONAL_LOCKS="0", GIT_NO_REPLACE_OBJECTS="1")
    return subprocess.check_output(["git", "-C", str(root), *args], env=environment)


def select(root: Path) -> list[dict]:
    lines = git(root, "ls-tree", "-r", "-l", "--full-tree", SOURCE_COMMIT, "code").decode().splitlines()
    seen, candidates = set(), []
    for line in lines:
        metadata, path = line.split("\t", 1)
        mode, kind, oid, count = metadata.split()
        if kind != "blob" or mode != "100644" or path.rsplit(".", 1)[-1] not in EXTENSIONS:
            continue
        size = int(count)
        if not 0 < size <= PART_BYTES or oid in seen:
            continue
        seen.add(oid)
        candidates.append({"path": path, "git_blob": oid, "size_bytes": size})
    selected, total = [], 0
    for entry in sorted(candidates, key=lambda item: (-item["size_bytes"], item["path"])):
        if len(selected) >= MAX_FILES:
            break
        if total + entry["size_bytes"] > MAX_SOURCE_BYTES:
            continue
        selected.append(entry)
        total += entry["size_bytes"]
    if len(selected) != MAX_FILES:
        raise ValueError("immutable source no longer supplies the declared 1,024-file corpus")
    return selected


def build(root: Path, output: Path) -> dict:
    if output.exists():
        raise ValueError("output must be new")
    selected = select(root)
    # Read exact committed blobs, never dirty working-tree bytes or content filters.
    objects = {}
    for entry in selected:
        content = git(root, "cat-file", "blob", entry["git_blob"])
        digest = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if digest != entry["git_blob"] or len(content) != entry["size_bytes"]:
            raise ValueError("Git blob identity mismatch")
        entry["sha256"] = hashlib.sha256(content).hexdigest()
        objects[entry["path"]] = content
    if len({entry["sha256"] for entry in selected}) != len(selected):
        raise ValueError("SHA-256 content duplicates are forbidden")
    ledger = canonical({"schema_version": 1, "kind": "public_committed_artifact_corpus",
                        "repository": SOURCE_REPOSITORY, "commit": SOURCE_COMMIT,
                        "selection": SELECTION, "extensions": sorted(EXTENSIONS),
                        "source_file_count": len(selected),
                        "source_bytes": sum(entry["size_bytes"] for entry in selected),
                        "files": sorted(selected, key=lambda item: item["path"])})
    if len(ledger) > 512 * 1024:
        raise ValueError("ledger exceeds finite artifact cap")
    archive = io.BytesIO()
    with tarfile.open(fileobj=archive, mode="w", format=tarfile.PAX_FORMAT) as stream:
        for name, content in [("CORPUS.json", ledger), *[("source/" + name, content)
                                                       for name, content in sorted(objects.items())]]:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            info.mode = 0o644
            info.mtime = info.uid = info.gid = 0
            info.uname = info.gname = ""
            stream.addfile(info, io.BytesIO(content))
    corpus = archive.getvalue()
    if len(corpus) > 64 * 1024 * 1024:
        raise ValueError("archive exceeds finite corpus cap")
    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    (inputs / "corpus-ledger.json").write_bytes(ledger)
    parts = []
    for number, offset in enumerate(range(0, len(corpus), PART_BYTES)):
        name = f"part-{number:03d}.bin"
        content = corpus[offset:offset + PART_BYTES]
        (inputs / name).write_bytes(content)
        parts.append(reference("inputs/" + name, content))
    if len({entry["sha256"] for entry in parts}) != len(parts):
        raise ValueError("duplicate archive parts are forbidden")
    index = {"schemaVersion": "piq.artifact-compression-chunks.v1",
             "corpusSha256": hashlib.sha256(corpus).hexdigest(), "corpusBytes": len(corpus),
             "sourceFileCount": len(selected), "ledger": reference("inputs/corpus-ledger.json", ledger),
             "chunks": parts}
    (output / "input-index.json").write_bytes(canonical(index))
    receipt = {"source_repository": SOURCE_REPOSITORY, "source_commit": SOURCE_COMMIT,
               "selection": SELECTION, "source_file_count": len(selected),
               "source_bytes": sum(entry["size_bytes"] for entry in selected),
               "corpus_sha256": index["corpusSha256"], "corpus_bytes": len(corpus),
               "index": reference("input-index.json", canonical(index)),
               "ledger": index["ledger"], "parts": parts, "compression_measured": False}
    (output / "INPUT-RECEIPT.json").write_bytes(canonical(receipt))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.repository, args.output), indent=2))
