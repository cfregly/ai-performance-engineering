#!/usr/bin/env python3
"""Audit the fixed corpus without invoking a codec or measuring performance."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path


AUTHORITY_PATH = Path(__file__).resolve().with_name("build_corpus.py")
AUTHORITY_SPEC = importlib.util.spec_from_file_location("public_text_corpus_authority", AUTHORITY_PATH)
authority = importlib.util.module_from_spec(AUTHORITY_SPEC)
AUTHORITY_SPEC.loader.exec_module(authority)


def _unique(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _document(path: Path, cap: int) -> tuple[bytes, dict]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("audit input must be a regular file")
    size = path.stat().st_size
    if not 0 < size <= cap:
        raise ValueError("audit input is empty or exceeds its cap")
    data = path.read_bytes()
    if len(data) != size:
        raise ValueError("audit input changed during read")
    value = json.loads(data, object_pairs_hook=_unique)
    if not isinstance(value, dict) or data != authority.canonical(value):
        raise ValueError("audit JSON must be a canonical object")
    return data, value


def _expected_index() -> dict:
    return {
        "schemaVersion": "piq.artifact-compression-chunks.v1",
        "corpusSha256": authority.EXPECTED_CORPUS_SHA256,
        "corpusBytes": authority.EXPECTED_CORPUS_BYTES,
        "sourceFileCount": authority.MAX_FILES,
        "ledger": authority.EXPECTED_LEDGER,
        "chunks": authority.EXPECTED_PARTS,
    }


def _read_part(root: Path, reference: dict) -> bytes:
    path = root / reference["artifactName"]
    if path.is_symlink() or not path.is_file():
        raise ValueError("part is absent or not a regular file")
    size = path.stat().st_size
    if size != reference["sizeBytes"] or not 0 < size <= authority.PART_BYTES:
        raise ValueError("part differs from its preregistered binding")
    data = path.read_bytes()
    if (
        len(data) != size
        or hashlib.sha256(data).hexdigest() != reference["sha256"]
    ):
        raise ValueError("part differs from its preregistered binding")
    return data


def _verify_tar(corpus: bytes, ledger_bytes: bytes, objects: dict[str, bytes]) -> int:
    expected = [("CORPUS.json", ledger_bytes)]
    expected.extend(("source/" + path, content) for path, content in sorted(objects.items()))
    with tarfile.open(fileobj=io.BytesIO(corpus), mode="r:") as stream:
        members = stream.getmembers()
        if len(members) != len(expected):
            raise ValueError("tar member count differs")
        for member, (name, content) in zip(members, expected, strict=True):
            if (
                member.name != name
                or not member.isreg()
                or member.size != len(content)
                or member.mode != 0o644
                or member.mtime != 0
                or member.uid != 0
                or member.gid != 0
                or member.uname != ""
                or member.gname != ""
            ):
                raise ValueError("tar order or member metadata differs")
            extracted = stream.extractfile(member)
            if extracted is None or extracted.read() != content:
                raise ValueError("tar member bytes differ from committed input")
    return len(expected)


def audit(repository: Path, corpus_root: Path) -> dict:
    index_bytes, index = _document(corpus_root / "input-index.json", authority.EXPECTED_INDEX_BYTES)
    if (
        index != _expected_index()
        or len(index_bytes) != authority.EXPECTED_INDEX_BYTES
        or hashlib.sha256(index_bytes).hexdigest() != authority.EXPECTED_INDEX_SHA256
    ):
        raise ValueError("canonical input index differs from the preregistered authority")

    ledger_path = corpus_root / authority.EXPECTED_LEDGER["artifactName"]
    ledger_bytes, ledger = _document(ledger_path, authority.MAX_LEDGER_BYTES)
    if authority.reference(authority.EXPECTED_LEDGER["artifactName"], ledger_bytes) != authority.EXPECTED_LEDGER:
        raise ValueError("canonical ledger differs from the preregistered authority")
    required_ledger = {
        "schema_version": 1,
        "kind": "public_committed_artifact_corpus",
        "repository": authority.SOURCE_REPOSITORY,
        "commit": authority.SOURCE_COMMIT,
        "selection": authority.SELECTION,
        "extensions": sorted(authority.EXTENSIONS),
        "source_file_count": authority.MAX_FILES,
        "source_bytes": authority.EXPECTED_SOURCE_BYTES,
    }
    if any(ledger.get(key) != value for key, value in required_ledger.items()):
        raise ValueError("ledger authority fields differ")
    files = ledger.get("files")
    if not isinstance(files, list) or len(files) != authority.MAX_FILES:
        raise ValueError("ledger file census differs")
    selected = authority.select(repository)
    if [entry["path"] for entry in files] != sorted(entry["path"] for entry in files):
        raise ValueError("ledger files are not in ascending path order")
    selected_by_path = {entry["path"]: entry for entry in selected}
    if set(selected_by_path) != {entry.get("path") for entry in files if isinstance(entry, dict)}:
        raise ValueError("ledger paths differ from the selected Git objects")

    objects, sha256_values = {}, set()
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"path", "git_blob", "size_bytes", "sha256"}:
            raise ValueError("ledger file entry is malformed")
        selected_entry = selected_by_path[entry["path"]]
        if any(entry[key] != selected_entry[key] for key in ("path", "git_blob", "size_bytes")):
            raise ValueError("ledger selection metadata differs from Git")
        content = authority.git(repository, "cat-file", "blob", entry["git_blob"])
        if (
            len(content) != entry["size_bytes"]
            or authority._git_blob_id(content) != entry["git_blob"]
            or hashlib.sha256(content).hexdigest() != entry["sha256"]
        ):
            raise ValueError("ledger object bytes differ from Git")
        if entry["sha256"] in sha256_values:
            raise ValueError("ledger contains duplicate SHA-256 content")
        sha256_values.add(entry["sha256"])
        objects[entry["path"]] = content

    parts = [_read_part(corpus_root, reference) for reference in authority.EXPECTED_PARTS]
    corpus = b"".join(parts)
    if (
        len(corpus) != authority.EXPECTED_CORPUS_BYTES
        or hashlib.sha256(corpus).hexdigest() != authority.EXPECTED_CORPUS_SHA256
    ):
        raise ValueError("aggregate tar differs from the preregistered authority")
    tar_members = _verify_tar(corpus, ledger_bytes, objects)

    receipt_bytes, receipt = _document(corpus_root / "INPUT-RECEIPT.json", 64 * 1024)
    expected_receipt = {
        "source_repository": authority.SOURCE_REPOSITORY,
        "source_commit": authority.SOURCE_COMMIT,
        "selection": authority.SELECTION,
        "source_file_count": authority.MAX_FILES,
        "source_bytes": authority.EXPECTED_SOURCE_BYTES,
        "corpus_sha256": authority.EXPECTED_CORPUS_SHA256,
        "corpus_bytes": authority.EXPECTED_CORPUS_BYTES,
        "index": authority.reference("input-index.json", index_bytes),
        "ledger": authority.EXPECTED_LEDGER,
        "parts": authority.EXPECTED_PARTS,
        "compression_measured": False,
    }
    if receipt != expected_receipt or not receipt_bytes:
        raise ValueError("input receipt differs from reconstructed bytes")

    return {
        "state": "PASS",
        "scope": "input-byte audit only, no compression experiment",
        "source_commit": authority.SOURCE_COMMIT,
        "source_files_verified": len(files),
        "source_bytes_verified": sum(entry["size_bytes"] for entry in files),
        "ledger_bytes": len(ledger_bytes),
        "ledger_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
        "index_bytes": len(index_bytes),
        "index_sha256": hashlib.sha256(index_bytes).hexdigest(),
        "parts_verified": len(parts),
        "tar_members_verified": tar_members,
        "corpus_sha256": hashlib.sha256(corpus).hexdigest(),
        "corpus_bytes": len(corpus),
        "all_git_blob_ids_and_sha256_verified": True,
        "all_distinct_content": True,
        "canonical_ledger_and_index_verified": True,
        "part_bindings_verified": True,
        "tar_member_order_metadata_and_bytes_verified": True,
        "compression_invoked": False,
    }


def render(value: dict) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("audit output must be new")
    result = audit(args.repository, args.corpus)
    args.output.write_bytes(render(result))
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
