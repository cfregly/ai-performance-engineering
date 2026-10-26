"""Check the exact retained capture bytes and their accepted receipt chain."""

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NATIVE_HASHES = {
    "c23f9791ae8b588a1ebb7a7737d26af6ba3f8e5c01887ea7c7f7861c92956c92",
    "5b11be1c3f8e109408d7ca59033441844d23515d48b963a9bbfd10a560692e7e",
    "28734ca6aa028737adb06ee75313b18d9a7304038ca0cbaf7139c5cffc138e03",
    "e27671e92e54d760abafc8f702e118139ee1a1102329eb763dca8f5dab883e35",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CaptureIntegrityTests(unittest.TestCase):
    def test_all_manifest_files_match_retained_bytes(self):
        for cohort in ("fresh022", "historical002", "followup"):
            root = ROOT / cohort
            manifest = json.loads((root / "manifest.json").read_text())
            for entry in manifest["files"]:
                with self.subTest(cohort=cohort, path=entry["path"]):
                    path = root / entry["path"]
                    self.assertEqual(path.stat().st_size, entry["bytes"])
                    self.assertEqual(digest(path), entry["sha256"])

    def test_four_native_reports_match_executed_export_inputs(self):
        root = ROOT / "fresh022"
        reports = list((root / "native").rglob("*.nsys-rep"))
        reports += list((root / "native").rglob("*.ncu-rep"))
        self.assertEqual(len(reports), 4)
        self.assertEqual({digest(path) for path in reports}, NATIVE_HASHES)
        receipt = (root / "exports/offline-profiler-export-result.json").read_text()
        for expected in NATIVE_HASHES:
            self.assertIn(expected, receipt)
        self.assertEqual(digest(root / "exports/offline-profiler-export-result.json"),
                         "579f69ff51fd11cdf05b7121e97dbcd0305012f7abb9ab84b2ae1e858d1592a0")

    def test_accepted_strict_and_final_receipts_are_unchanged(self):
        root = ROOT / "fresh022/qualification"
        self.assertEqual(digest(root / "STRICT-PROFILER-QUALIFICATION.json"),
                         "6c2ec3215b144b06f771e0dfa2e2b90898c0c7744c0504e57d3a9c1a399fc538")
        self.assertEqual(digest(root / "FINAL-SCIENTIFIC-QUALIFICATION.json"),
                         "0b836a908264d2592d3a79eb7ff304bf0bc9b0d9031f4ccf7627577ce9844a3a")


if __name__ == "__main__":
    unittest.main()
