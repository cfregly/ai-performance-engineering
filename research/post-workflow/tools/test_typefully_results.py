"""Behavioral tests for offline Typefully publication/results tracking."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from tools.typefully_results import ingest, main, native_url, plan, snapshot, track


X = "https://x.com/example_author/status/1234567890"
LI = "https://www.linkedin.com/feed/update/urn:li:activity:1234567890/"
PUBLISHED = "2026-10-01T12:00:00Z"


class ResultsTest(unittest.TestCase):
    def setUp(self):
        self.registry = track({"owner_note": "preserve me"}, 10, 20)
        self.detail = {"id": 10, "social_set_id": 20, "status": "published",
                       "x_published_url": X, "x_post_published_at": PUBLISHED,
                       "linkedin_published_url": LI,
                       "linkedin_post_published_at": "2026-10-02T12:00:00+00:00"}

    def publications(self):
        return ingest(self.registry, self.detail, "2026-10-03T12:00:00Z")

    def observation(self, observed_at="2026-10-02T12:00:00Z", **extra):
        return {"platform": "x", "native_post_url": X, "observed_at": observed_at,
                "metrics": {"likes": 0, "impressions": 42},
                "metrics_source": "Native post analytics readback",
                "missing_reason": "Not exposed in this native readback.", **extra}

    def test_draft_planned_and_scheduled_never_admit_publication(self):
        for status in ("draft", "planned", "scheduled", "publishing", None):
            with self.subTest(status=status):
                detail = {**self.detail, "status": status}
                self.assertEqual(ingest(self.registry, detail, "2026-10-03T12:00:00Z")["publications"], [])

    def test_platforms_have_independent_publication_clocks(self):
        registry = self.publications()
        due = plan(registry, "2026-10-02T13:00:00Z")
        self.assertEqual([(item["platform"], item["actual_age_hours"]) for item in due], [("x", 25)])
        due = plan(registry, "2026-10-08T13:00:00Z")
        self.assertEqual([(item["platform"], item["milestone"]) for item in due], [("x", "7d"), ("linkedin", "24h")])

    def test_shared_timestamp_requires_explicit_platform_provenance(self):
        detail = deepcopy(self.detail)
        del detail["x_post_published_at"]
        detail["published_at"] = PUBLISHED
        registry = ingest(self.registry, detail, "2026-10-03T12:00:00Z")
        self.assertEqual([p["platform"] for p in registry["publications"]], ["linkedin"])
        envelope = {"detail": detail, "timestamp_provenance": {"x": "Native X post timestamp matches shared Typefully timestamp."}}
        registry = ingest(self.registry, envelope, "2026-10-03T12:00:00Z")
        self.assertEqual(registry["publications"][0]["timestamp_field"], "published_at")

    def test_missing_native_url_or_timestamp_does_not_admit(self):
        detail = {**self.detail, "x_published_url": None, "linkedin_post_published_at": None}
        self.assertEqual(ingest(self.registry, detail, "2026-10-03T12:00:00Z")["publications"], [])

    def test_duplicate_publication_and_snapshot_are_idempotent(self):
        registry = self.publications()
        registry["publications"][0]["manual_note"] = "keep"
        repeated = {**self.detail, "x_published_url": X.replace("x.com/example_author", "twitter.com/EXAMPLE_AUTHOR") + "?s=20"}
        registry = ingest(registry, [repeated, repeated], "2026-10-04T12:00:00Z")
        self.assertEqual(len(registry["publications"]), 2)
        registry = snapshot(registry, [self.observation(), self.observation()], "2026-10-02T12:00:00Z")
        self.assertEqual(len(registry["publications"][0]["snapshots"]), 1)
        self.assertEqual(registry["publications"][0]["manual_note"], "keep")
        self.assertEqual(registry["owner_note"], "preserve me")

    def test_overdue_first_capture_does_not_fabricate_24h_metrics(self):
        registry = self.publications()
        observation = self.observation("2026-10-10T12:00:00Z")
        registry = snapshot(registry, observation, "2026-10-10T12:00:00Z")
        captured = registry["publications"][0]["snapshots"]
        self.assertEqual(len(captured), 1)
        self.assertEqual((captured[0]["milestone"], captured[0]["actual_age_hours"], captured[0]["timing"]), ("7d", 216, "late"))
        self.assertEqual(captured[0]["missed_milestones"], ["24h"])
        self.assertFalse(any(item["platform"] == "x" for item in plan(registry, "2026-10-11T12:00:00Z")))

    def test_late_24h_snapshot_preserves_actual_age_and_due_7d(self):
        registry = snapshot(self.publications(), self.observation("2026-10-02T18:00:00Z"), "2026-10-02T18:00:00Z")
        captured = registry["publications"][0]["snapshots"][0]
        self.assertEqual((captured["milestone"], captured["actual_age_hours"]), ("24h", 30))
        self.assertEqual(next(item for item in plan(registry, "2026-10-08T12:00:00Z") if item["platform"] == "x")["milestone"], "7d")
        registry = snapshot(registry, self.observation("2026-10-08T12:00:00Z"), "2026-10-08T12:00:00Z")
        self.assertEqual([s["milestone"] for s in registry["publications"][0]["snapshots"]], ["24h", "7d"])
        self.assertFalse(any(item["platform"] == "x" for item in plan(registry, "2026-10-20T12:00:00Z")))

    def test_missing_metrics_are_null_and_zero_is_observed_zero(self):
        registry = snapshot(self.publications(), self.observation(), "2026-10-02T12:00:00Z")
        captured = registry["publications"][0]["snapshots"][0]
        self.assertEqual(captured["metrics"]["likes"], 0)
        self.assertNotIn("likes", captured["missing_reasons"])
        self.assertIsNone(captured["metrics"]["link_clicks"])
        self.assertIn("link_clicks", captured["missing_reasons"])
        with self.assertRaisesRegex(ValueError, "missing reason"):
            snapshot(self.publications(), self.observation(missing_reason=None), "2026-10-02T12:00:00Z")

    def test_invalid_native_urls_and_times_fail_closed(self):
        for bad in ("https://x.com.evil.test/example_author/status/123", "http://x.com/example_author/status/123", "https://x.com/example_author", "https://user@x.com/example_author/status/123", "https://x.com:443/example_author/status/123"):
            with self.subTest(url=bad), self.assertRaises(ValueError):
                ingest(self.registry, {**self.detail, "x_published_url": bad}, "2026-10-03T12:00:00Z")
        with self.assertRaises(ValueError):
            native_url("linkedin", "https://www.linkedin.com/in/example_author")
        for bad in ("2026-10-01T12:00:00", "not-a-date", "2026-10-04T12:00:00Z"):
            with self.subTest(timestamp=bad), self.assertRaises(ValueError):
                ingest(self.registry, {**self.detail, "x_post_published_at": bad}, "2026-10-03T12:00:00Z")
        with self.assertRaisesRegex(ValueError, "untracked"):
            ingest(self.registry, {**self.detail, "social_set_id": 99}, "2026-10-03T12:00:00Z")

    def test_conflicting_timestamp_preserves_prior_evidence(self):
        registry = self.publications()
        before = deepcopy(registry)
        with self.assertRaisesRegex(ValueError, "conflicting"):
            ingest(registry, {**self.detail, "x_post_published_at": "2026-10-01T13:00:00Z"}, "2026-10-03T12:00:00Z")
        self.assertEqual(registry, before)

    def test_future_and_prepublication_observations_fail(self):
        for observed_at in ("2026-09-30T12:00:00Z", "2026-10-04T12:00:00Z"):
            with self.subTest(observed_at=observed_at), self.assertRaises(ValueError):
                snapshot(self.publications(), self.observation(observed_at), "2026-10-03T12:00:00Z")
        with self.assertRaisesRegex(ValueError, "disagree"):
            ingest(self.registry, {**self.detail, "draft_id": 11}, "2026-10-03T12:00:00Z")

    def test_autoplug_delivery_requires_native_reply_and_evidence(self):
        observation = self.observation(auto_plug={"state": "delivered"})
        with self.assertRaisesRegex(ValueError, "exact reply_url"):
            snapshot(self.publications(), observation, "2026-10-02T12:00:00Z")
        observation["auto_plug"].update(reply_url="https://x.com/example_author/status/999999", evidence="Native reply visible in readback.")
        result = snapshot(self.publications(), observation, "2026-10-02T12:00:00Z")
        self.assertEqual(result["publications"][0]["snapshots"][0]["auto_plug"]["reply_url"], "https://x.com/example_author/status/999999")
        observation["auto_plug"]["reply_url"] = X
        with self.assertRaisesRegex(ValueError, "original post"):
            snapshot(self.publications(), observation, "2026-10-02T12:00:00Z")

    def test_plan_and_dry_run_cli_do_not_write(self):
        from contextlib import redirect_stdout
        from io import StringIO
        with tempfile.TemporaryDirectory() as directory:
            registry_path = Path(directory) / "ledger.json"
            registry_path.write_text(json.dumps(self.registry))
            original = registry_path.read_bytes()
            with redirect_stdout(StringIO()):
                main(["plan", "--registry", str(registry_path), "--now", "2026-10-03T12:00:00Z"])
                main(["track", "--registry", str(registry_path), "--draft-id", "11", "--social-set-id", "20", "--dry-run"])
            self.assertEqual(registry_path.read_bytes(), original)
            with redirect_stdout(StringIO()):
                main(["track", "--registry", str(registry_path), "--draft-id", "11", "--social-set-id", "20"])
            saved = json.loads(registry_path.read_text())
            self.assertEqual(saved["owner_note"], "preserve me")
            self.assertEqual(len(saved["tracked_drafts"]), 2)

    def test_fractional_counts_are_not_silent_estimates(self):
        with self.assertRaisesRegex(ValueError, "integer count"):
            snapshot(self.publications(), self.observation(metrics={"likes": 1.5}), "2026-10-02T12:00:00Z")

    def test_reader_response_evidence_is_retained_without_sending(self):
        responses = {"state": "observed", "items": [{
            "comment_url": "https://x.com/reader/status/999999", "summary": "Asked about startup cost.",
            "evidence": "Retained native thread screenshot."}]}
        result = snapshot(self.publications(), self.observation(reader_responses=responses), "2026-10-02T12:00:00Z")
        self.assertEqual(result["publications"][0]["snapshots"][0]["reader_responses"], responses)


if __name__ == "__main__":
    unittest.main()
