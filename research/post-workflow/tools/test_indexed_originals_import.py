from __future__ import annotations
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools import piq_experiment_evidence as tool

class OriginalTransportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        (self.source / "binary.bin").write_bytes(b"\x00\xffnative\x00bytes")
        self.write_index()

    def write_index(self):
        raw = (self.source / "binary.bin").read_bytes()
        self.index = {"kind": "synthetic.indexed-originals.v1", "artifacts": [{
            "path": "binary.bin", "sha256": hashlib.sha256(raw).hexdigest(), "sizeBytes": len(raw)}]}
        self.save()

    def save(self):
        (self.source / "ENVELOPE.json").write_text(json.dumps(self.index))

    def test_binary_import_receipt_revalidation_and_source_unchanged(self):
        before = {p.name: p.read_bytes() for p in self.source.iterdir()}
        destination = self.root / "imported"
        tool.import_originals(self.source, destination, "ENVELOPE.json")
        self.assertEqual((destination / "binary.bin").read_bytes(), before["binary.bin"])
        self.assertEqual({p.name: p.read_bytes() for p in self.source.iterdir()}, before)
        tool.validate_originals(destination, "ENVELOPE.json")
        receipt = json.loads((destination / tool.ORIGINALS_RECEIPT).read_text())
        self.assertEqual(receipt["state"], "indexed_originals_imported")
        self.assertFalse(receipt["scientifically_verified"])
        self.assertFalse(receipt["publishable"])
        self.assertEqual(receipt["qualification"], "not_established")

    def test_existing_destination_preserved(self):
        destination = self.root / "existing"; destination.mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            tool.import_originals(self.source, destination, "ENVELOPE.json")
        self.assertEqual(list(destination.iterdir()), [])

    def test_hash_and_size_drift_refused(self):
        for field, value in [("sha256", "0" * 64), ("sizeBytes", 3)]:
            with self.subTest(field=field):
                self.write_index(); self.index["artifacts"][0][field] = value; self.save()
                with self.assertRaisesRegex(ValueError, "hash/size drift"):
                    tool.validate_originals(self.source, "ENVELOPE.json")

    def test_undeclared_file_refused(self):
        (self.source / "extra").write_bytes(b"original")
        with self.assertRaisesRegex(ValueError, "exact file set"):
            tool.validate_originals(self.source, "ENVELOPE.json")

    def test_symlink_file_and_parent_refused(self):
        actual = self.root / "actual"; actual.write_bytes(b"\x00\xffnative\x00bytes")
        (self.source / "binary.bin").unlink()
        (self.source / "binary.bin").symlink_to(actual)
        with self.assertRaises((ValueError, OSError)):
            tool.validate_originals(self.source, "ENVELOPE.json")
        link = self.root / "linked"; link.symlink_to(self.source, target_is_directory=True)
        with self.assertRaises((ValueError, OSError)):
            tool.validate_originals(link, "ENVELOPE.json")

    def test_portable_paths_duplicates_case_collisions_and_reserved_names(self):
        for name in ["../escape", "/absolute", "a//b", "a/./b", "CON", "a.", "a\\b"]:
            with self.subTest(name=name):
                self.write_index(); self.index["artifacts"][0]["path"] = name; self.save()
                with self.assertRaises(ValueError):
                    tool.validate_originals(self.source, "ENVELOPE.json")
        self.write_index(); self.index["artifacts"].append({**self.index["artifacts"][0], "path": "BINARY.BIN"}); self.save()
        with self.assertRaisesRegex(ValueError, "colliding"):
            tool.validate_originals(self.source, "ENVELOPE.json")

    def test_explicit_bounds_and_boolean_size_refused(self):
        for value in [True, -1, tool.ORIGINALS_FILE_BYTES + 1]:
            self.write_index(); self.index["artifacts"][0]["sizeBytes"] = value; self.save()
            with self.assertRaises(ValueError):
                tool.validate_originals(self.source, "ENVELOPE.json")
        self.write_index()
        with patch.object(tool, "ORIGINALS_TOTAL_BYTES", 1):
            with self.assertRaisesRegex(ValueError, "128MiB"):
                tool.validate_originals(self.source, "ENVELOPE.json")

    def test_failure_preserves_partial_stage(self):
        real = tool.validate_originals
        calls = 0
        def fail_stage(path, index):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise ValueError("synthetic copied-byte failure")
            return real(path, index)
        with patch.object(tool, "validate_originals", fail_stage):
            with self.assertRaisesRegex(ValueError, "partial evidence retained"):
                tool.import_originals(self.source, self.root / "failed", "ENVELOPE.json")
        stages = list(self.root.glob(".piq-originals-*"))
        self.assertEqual(len(stages), 1)
        self.assertTrue((stages[0] / "PARTIAL-IMPORT.json").is_file())
        self.assertTrue((stages[0] / "bundle/binary.bin").is_file())
        self.assertFalse((self.root / "failed").exists())

if __name__ == "__main__":
    unittest.main()
