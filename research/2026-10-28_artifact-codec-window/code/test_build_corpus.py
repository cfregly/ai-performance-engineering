"""Source-only checks for the fixed public-text input authority."""
from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = _load("build_corpus")
audit_input = _load("audit_input")


class CorpusBuilderTests(unittest.TestCase):
    def test_selection_is_case_normalized_deduplicated_and_ordered(self):
        duplicate = "a" * 40
        lines = [
            f"100644 blob {duplicate} 12\troot/first.MD",
            f"100644 blob {duplicate} 12\troot/repeated.md",
            f"100644 blob {'b' * 40} 30\tdocs/larger.JSONL",
            f"100644 blob {'c' * 40} 12\tcode/alpha.py",
            f"100644 blob {'d' * 40} 12\tcode/zeta.txt",
            f"100755 blob {'e' * 40} 99\tcode/executable.py",
            f"100644 blob {'f' * 40} 99\tcode/rejected.bin",
            f"100644 blob {'1' * 40} {builder.SOURCE_BLOB_BYTES + 1}\tdocs/too-large.md",
        ]
        selected = builder.select_entries(lines, limit=4)
        self.assertEqual(
            [(entry["size_bytes"], entry["path"]) for entry in selected],
            [(30, "docs/larger.JSONL"), (12, "code/alpha.py"),
             (12, "code/zeta.txt"), (12, "root/first.MD")],
        )
        selected_paths = {entry["path"] for entry in selected}
        self.assertNotIn("code/executable.py", selected_paths)
        self.assertNotIn("root/repeated.md", selected_paths)
        self.assertEqual(len({entry["git_blob"] for entry in selected}), len(selected))

    def test_preregistered_caps_and_counts_are_exact(self):
        self.assertEqual(builder.MAX_FILES, 4096)
        self.assertEqual(builder.SOURCE_BLOB_BYTES, 32 * 1024 * 1024)
        self.assertEqual(builder.PART_BYTES, 30 * 1024 * 1024)
        self.assertEqual(builder.MAX_LEDGER_BYTES, 2 * 1024 * 1024)
        self.assertEqual(builder.MAX_CORPUS_BYTES, 128 * 1024 * 1024)
        self.assertEqual(len(builder.EXPECTED_PARTS), 4)
        self.assertEqual(sum(part["sizeBytes"] for part in builder.EXPECTED_PARTS),
                         builder.EXPECTED_CORPUS_BYTES)
        self.assertTrue(all(part["sizeBytes"] <= builder.PART_BYTES
                            for part in builder.EXPECTED_PARTS))

    def test_expected_index_has_the_frozen_bytes_and_hash(self):
        index = {
            "schemaVersion": "piq.artifact-compression-chunks.v1",
            "corpusSha256": builder.EXPECTED_CORPUS_SHA256,
            "corpusBytes": builder.EXPECTED_CORPUS_BYTES,
            "sourceFileCount": builder.MAX_FILES,
            "ledger": builder.EXPECTED_LEDGER,
            "chunks": builder.EXPECTED_PARTS,
        }
        encoded = builder.canonical(index)
        self.assertEqual(len(encoded), builder.EXPECTED_INDEX_BYTES)
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), builder.EXPECTED_INDEX_SHA256)
        self.assertEqual(json.loads(encoded), audit_input._expected_index())

    def test_build_refuses_an_existing_output_before_reading_git(self):
        with tempfile.TemporaryDirectory(prefix="corpus-builder-existing-") as directory:
            with self.assertRaisesRegex(ValueError, "output must be new"):
                builder.build(Path(directory), Path(directory))

    def test_audit_cli_help_works_in_isolated_mode(self):
        completed = subprocess.run(
            [sys.executable, "-I", "-B", "-X", "pycache_prefix=/var/empty/piq-python-cache",
             str(HERE / "audit_input.py"), "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--repository", completed.stdout)
        self.assertIn("--corpus", completed.stdout)

    def test_tar_audit_checks_member_order_metadata_and_bytes(self):
        ledger = builder.canonical({"fixture": "source-only tar audit"})
        objects = {"docs/alpha.md": b"alpha\n", "root.TXT": b"beta\n"}
        archive = io.BytesIO()
        with tarfile.open(fileobj=archive, mode="w", format=tarfile.PAX_FORMAT) as stream:
            for name, content in [
                ("CORPUS.json", ledger),
                *[("source/" + path, data) for path, data in sorted(objects.items())],
            ]:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                info.mode = 0o644
                info.mtime = info.uid = info.gid = 0
                info.uname = info.gname = ""
                stream.addfile(info, io.BytesIO(content))
        self.assertEqual(audit_input._verify_tar(archive.getvalue(), ledger, objects), 3)


if __name__ == "__main__":
    unittest.main()
