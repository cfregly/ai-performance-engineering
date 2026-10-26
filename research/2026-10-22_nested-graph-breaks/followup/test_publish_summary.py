"""Non-performance checks for metadata redaction and evidence preservation."""

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import publish_summary


class PublicationTests(unittest.TestCase):
    def test_redaction_preserves_signed_values_and_all_records(self):
        source = {"profiles": [{"path": "/private/trace.json", "device": 99, "cpu_marker": {"pid": 4, "tid": 5}, "event": {"pid": 0, "tid": 7},
                    "all_five_invocations": [{"signed_delta": -3}, {"signed_delta": 7}]}],
                  "tail_pairs": [{"ratio": 2.5}], "limits": [publish_summary.PENDING_LIMIT]}
        untouched = copy.deepcopy(source)
        actual, changes = publish_summary.public_projection(source)
        self.assertEqual(source, untouched)
        self.assertEqual(actual["profiles"][0]["all_five_invocations"], source["profiles"][0]["all_five_invocations"])
        self.assertEqual(actual["tail_pairs"], source["tail_pairs"])
        self.assertEqual(actual["profiles"][0]["path"], "trace.json")
        self.assertEqual(actual["profiles"][0]["cpu_marker"]["pid"], "<cpu-pid>")
        self.assertEqual(actual["profiles"][0]["event"], source["profiles"][0]["event"])
        self.assertEqual(actual["superseded_limit"]["recorded_text"], publish_summary.PENDING_LIMIT)
        self.assertTrue(changes)

    def test_source_drift_is_rejected_without_writing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.json"
            source.write_text("{}")
            with self.assertRaisesRegex(ValueError, "source hash mismatch"):
                publish_summary.project_file(source, root / "out.json", root / "manifest.json")
            self.assertFalse((root / "out.json").exists())

    def test_retained_public_summary_hash_and_complete_coverage(self):
        root = Path(__file__).resolve().parent / "results"
        raw = (root / "summary.json").read_bytes()
        manifest = json.loads((root / "projection-manifest.json").read_text())
        summary = json.loads(raw)
        self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest["public_sha256"])
        self.assertEqual(len(summary["tail_cells"]), 8)
        self.assertEqual(len(summary["tail_pairs"]), 4)
        self.assertEqual(len(summary["memory_phase_rows"]), 20)
        self.assertEqual(len(summary["per_seed_cumulative_crossings"]), 4)
        self.assertEqual(len(summary["profiles"]), 4)
        for capture in summary["profiles"]:
            self.assertEqual(len(capture["all_five_invocations"]), 5)
            expected = [3, 4, 5] if capture["enabled"] else []
            self.assertEqual(capture["excluded_invocation_indices"], expected)
        self.assertNotIn("/opt/", raw.decode())


if __name__ == "__main__":
    unittest.main()
