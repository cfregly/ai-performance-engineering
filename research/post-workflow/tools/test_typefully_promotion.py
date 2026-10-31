import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.typefully_promotion import validate_promotion


class PromotionTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.package = Path(self.directory.name)
        self.raw = {"id": 123, "social_set_id": 456, "status": "draft", "scheduled_date": None,
                    "published_at": None, "updated_at": "2026-10-31T12:00:00Z",
                    "platforms": {"x": {"enabled": True, "settings": None}}}
        self.plan = {
            "schema_version": 1, "draft_id": 123, "social_set_id": 456,
            "auto_plug": {"state": "configured", "reply_text": "Full code and charts for my attention scratch-traffic example: https://github.com/example_author/example/tree/main/research/attention",
                          "metric": "likes", "threshold": 20,
                          "checked_at": "2026-10-31T12:01:00Z",
                          "threshold_decision": self.artifact("decision.md", b"Use 20 likes based on the retained comparable-post review."),
                          "evidence": self.artifact("ui.png", b"retained screenshot bytes")},
            "results_tracking": {"mode": "read_only", "registry_path": "/not/read/outside/registry.json",
                                 "draft_id": 123, "social_set_id": 456,
                                 "milestones_hours": [24, 168], "state": "awaiting_publication"}}

    def artifact(self, name, content):
        (self.package / name).write_bytes(content)
        return {"path": name, "sha256": hashlib.sha256(content).hexdigest()}

    def errors(self):
        self.plan["raw_readback"] = self.artifact("raw.json", json.dumps(self.raw).encode())
        (self.package / "plan.json").write_text(json.dumps(self.plan))
        errors = []
        validate_promotion(self.package, "plan.json", errors)
        return errors

    def test_valid_plan_does_not_read_external_registry(self):
        self.assertEqual(self.errors(), [])

    def test_missing_plan_and_missing_evidence_fail(self):
        errors = []
        validate_promotion(self.package, "missing.json", errors)
        self.assertTrue(any("cannot read plan" in error for error in errors))
        del self.plan["auto_plug"]["evidence"]
        self.assertTrue(any("auto_plug.evidence" in error for error in self.errors()))

    def test_invalid_threshold_and_default_copy_fail(self):
        for value in (0, -1, True, "20", 1.5):
            self.plan["auto_plug"]["threshold"] = value
            self.assertTrue(any("positive integer threshold" in error for error in self.errors()))
        self.plan["auto_plug"]["threshold"] = 20
        for copy in ("https://example.com/article", "Follow me for more useful articles https://example.com/article", "Full code and charts for my example https://github.com/owner/repo", "Full code and charts for my example https://[broken", "x" * 281):
            self.plan["auto_plug"]["reply_text"] = copy
            self.assertTrue(self.errors())

    def test_wrong_ids_nondraft_and_scheduled_fail(self):
        self.raw["id"] = 999
        self.assertTrue(any("id does not match" in error for error in self.errors()))
        self.raw["id"] = 123
        self.raw["status"] = "published"
        self.assertTrue(any("status must equal draft" in error for error in self.errors()))
        self.raw["status"] = "draft"
        self.raw["scheduled_date"] = "2026-11-01T12:00:00Z"
        self.assertTrue(any("scheduled_date must be null" in error for error in self.errors()))

    def test_not_applicable_requires_no_x_or_reply_target(self):
        self.plan["auto_plug"]["state"] = "not_applicable"
        self.assertTrue(any("not_applicable requires" in error for error in self.errors()))
        self.raw["platforms"]["x"]["settings"] = {"reply_to_url": "https://x.com/person/status/999"}
        self.assertEqual(self.errors(), [])
        self.plan["auto_plug"]["state"] = "configured"
        self.assertTrue(any("enabled original X" in error for error in self.errors()))
        self.plan["auto_plug"]["state"] = "not_applicable"
        self.raw["platforms"]["x"] = {"enabled": False}
        self.assertEqual(self.errors(), [])

    def test_missing_or_mismatched_tracking_fails(self):
        tracking = self.plan.pop("results_tracking")
        self.assertTrue(any("results_tracking" in error for error in self.errors()))
        self.plan["results_tracking"] = tracking
        tracking["draft_id"] = 999
        tracking["milestones_hours"] = [24]
        tracking["state"] = []
        self.assertTrue(any("results_tracking.draft_id" in error for error in self.errors()))
        self.assertTrue(any("milestones_hours" in error for error in self.errors()))

    def test_useful_followup_does_not_require_a_repository_cta(self):
        self.plan["auto_plug"]["reply_text"] = "Which input boundary would you check first in your own workload?"
        self.assertTrue(self.errors())
        self.plan["auto_plug"]["destination_reason"] = "A concrete reader question is the next step. No additional resource adds value."
        self.assertEqual(self.errors(), [])

    def test_artifact_hash_path_escape_symlink_and_stale_time_fail(self):
        self.plan["auto_plug"]["evidence"]["sha256"] = "0" * 64
        self.assertTrue(any("match its sha256" in error for error in self.errors()))
        self.plan["auto_plug"]["evidence"]["path"] = "../outside.png"
        self.assertTrue(any("safe package-relative" in error for error in self.errors()))
        (self.package / "link.png").symlink_to(self.package / "ui.png")
        self.plan["auto_plug"]["evidence"]["path"] = "link.png"
        self.assertTrue(any("must not use symlinks" in error for error in self.errors()))
        self.plan["auto_plug"]["checked_at"] = "2026-10-31T11:00:00Z"
        self.assertTrue(any("older than" in error for error in self.errors()))


if __name__ == "__main__":
    unittest.main()
