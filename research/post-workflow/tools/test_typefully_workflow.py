from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from tools.typefully_draft_guard import decide
from tools.typefully_workflow import WORKFLOW_PATH, validate_workflow


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.package = Path(self.temp.name)
        self.raw = {
            "id": "draft-123",
            "draft_id": "draft-123",
            "social_set_id": 17,
            "status": "draft",
            "publish_state": None,
            "created_at": "2026-10-26T11:00:00Z",
            "updated_at": "2026-10-26T12:00:00Z",
            "scheduled_date": None,
            "published_at": None,
            "share_url": None,
            "private_url": "https://typefully.example/draft-123",
            "x_published_url": None,
            "linkedin_published_url": None,
            "mastodon_published_url": None,
            "threads_published_url": None,
            "bluesky_published_url": None,
            "substack_published_url": None,
            "x_article_published_url": None,
            "x_post_published_at": None,
            "linkedin_post_published_at": None,
            "mastodon_post_published_at": None,
            "threads_post_published_at": None,
            "bluesky_post_published_at": None,
            "substack_post_published_at": None,
            "x_article_published_at": None,
            "draft_title": "Fixture draft",
            "tags": ["needs-review", "gpu"],
            "scratchpad_text": "Fixture notes.",
            "platforms": {
                "x": {
                    "enabled": True,
                    "posts": [{"text": "Exact X text.", "media_ids": ["media-x"]}],
                    "settings": None,
                },
                "linkedin": {
                    "enabled": True,
                    "posts": [{"text": "Exact LinkedIn text.", "media_ids": ["media-li"]}],
                    "settings": None,
                },
                "mastodon": {"enabled": False},
                "threads": {"enabled": False},
                "bluesky": {"enabled": False},
                "substack": {"enabled": False},
                "x_article": None,
            },
        }
        analytics_review = self.artifact(
            "social/typefully-analytics-review.md",
            """# Typefully analytics review

Status: completed

- Observed at: 2026-10-26T11:50:00Z
- Reviewer: Fixture analytics reviewer
- Sources checked: Retained Typefully X export and native LinkedIn availability view.
- Evidence limits: One small fixture cohort; associations are descriptive and no causal result is claimed.
- Access blocker: Native LinkedIn metrics were unavailable in this fixture.
- Decision summary: Keep the direct technical hook and record LinkedIn analytics as unavailable.
""",
        )
        people_review = self.artifact(
            "social/typefully-people-review.md",
            """# Relevant people and conversation review

Status: completed

- Observed at: 2026-10-26T11:51:00Z
- Reviewer: Fixture people reviewer
- Sources checked: Package source list, named maintainers, and directly relevant native conversations.
- Evidence limits: Fixture identities are illustrative; no live mention or relationship is established.
- Access blocker: No additional blocker affected the bounded fixture review.
- Decision summary: Use no native mention and retain no conversation target because none was verified.
""",
        )
        promotion_evidence = self.artifact(
            "social/promotion-readback.txt", "Observed after final edit and reload.\n"
        )
        self.receipt = {
            "schema_version": 1,
            "draft_id": "draft-123",
            "social_set_id": 17,
            "operation": "create",
            "prior_readback": None,
            "verified_at": "2026-10-26T12:05:00+00:00",
            "status": "draft",
            "scheduled_at": None,
            "published_at": None,
            "publication_authorized": False,
            "platforms": ["x", "linkedin"],
            "lifecycle_tag": "needs-review",
            "tags": ["needs-review", "gpu"],
            "approval": None,
            "media_ids": {"x": [["media-x"]], "linkedin": [["media-li"]]},
            "analytics": {
                "review": analytics_review,
                "x": {"status": "reviewed", "reason": ""},
                "linkedin": {"status": "unavailable", "reason": "No retained native LinkedIn view."},
            },
            "people": {
                "review": people_review,
                "mentions": {"state": "none", "reason": "No native mention selected.", "verified": []},
                "opportunities": {"state": "none", "reason": "No relevant native parent verified."},
            },
            "promotion": {
                name: {
                    "state": "off",
                    "trigger": "",
                    "reason": "Observed off in Typefully.",
                    "evidence": promotion_evidence,
                    "checked_at": "2026-10-26T12:01:00Z",
                    "dormant": True,
                }
                for name in ("auto_retweet", "auto_plug")
            },
            "reviewer": "Fixture reviewer",
        }
        self.guard_override: dict | None = None
        self.write_typefully()

    def tearDown(self) -> None:
        self.temp.cleanup()

    def write_json(self, relative: str, value: object) -> None:
        path = self.package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")

    def artifact(self, relative: str, content: str) -> dict[str, str]:
        path = self.package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def json_artifact(self, relative: str, value: object) -> dict[str, str]:
        self.write_json(relative, value)
        path = self.package / relative
        return {"path": relative, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def write_typefully(self, *, media_ids: list[str] | None = None) -> None:
        self.write_json(
            "social/typefully-receipt.json",
            {
                "draft_id": "draft-123",
                "social_set_id": 17,
                "platforms": ["x", "linkedin"],
                "media_ids": media_ids if media_ids is not None else ["media-x", "media-li"],
                "texts": {"x": "Exact X text.", "linkedin": "Exact LinkedIn text."},
            },
        )

    def guard(self) -> dict:
        payload = {
            key: copy.deepcopy(self.raw[key])
            for key in ("draft_title", "tags", "scratchpad_text", "platforms")
        }
        target = {"social_set_id": self.receipt["social_set_id"]}
        return decide(
            {
                "desired": {"target": target, "payload": payload},
                "readback_target": target,
                "list_readback": {"results": [copy.deepcopy(self.raw)]},
                "detail_readback": copy.deepcopy(self.raw),
                "prior_receipt": None,
            }
        )

    def check(self) -> list[str]:
        self.receipt["raw_readback"] = self.json_artifact("social/raw-readback.json", self.raw)
        guard = self.guard_override if self.guard_override is not None else self.guard()
        self.receipt["guard_receipt"] = self.json_artifact("social/final-guard.json", guard)
        self.write_json(WORKFLOW_PATH, self.receipt)
        errors: list[str] = []
        validate_workflow(self.package, WORKFLOW_PATH, errors)
        return errors

    def test_valid_fixture_and_cli(self) -> None:
        self.assertEqual(self.check(), [])
        result = subprocess.run(
            [sys.executable, str(Path(__file__).with_name("typefully_workflow.py")), "--package", str(self.package)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_exact_text_and_media_are_required(self) -> None:
        self.raw["platforms"]["x"]["posts"][0]["text"] = "Human edit."
        self.receipt["media_ids"]["linkedin"] = [["wrong-media"]]
        errors = "\n".join(self.check())
        self.assertIn("raw_readback x text does not match", errors)
        self.assertIn("media_ids.linkedin does not exactly match", errors)

    def test_legacy_media_union_must_match(self) -> None:
        self.write_typefully(media_ids=["media-x"])
        self.assertIn("raw_readback media IDs do not match", "\n".join(self.check()))

    def test_scheduled_published_and_extra_platform_rejected(self) -> None:
        self.raw["scheduled_date"] = "2026-10-27T12:00:00Z"
        self.raw["x_published_url"] = "https://x.com/example/status/1"
        self.raw["platforms"]["bluesky"] = {"enabled": True}
        errors = "\n".join(self.check())
        for expected in ("scheduled_date", "x_published_url", "unsupported raw_readback platform: bluesky"):
            self.assertIn(expected, errors)

    def test_publication_authorization_is_rejected(self) -> None:
        self.receipt["publication_authorized"] = True
        self.assertIn("publication_authorized must be false", "\n".join(self.check()))

    def test_guard_must_bind_current_exact_readback(self) -> None:
        self.guard_override = self.guard()
        self.raw["scratchpad_text"] = "Changed after guard."
        errors = "\n".join(self.check())
        self.assertIn("remote_fingerprint does not match raw_readback", errors)
        self.assertIn("current_fingerprint does not match raw_readback", errors)

    def test_guard_must_be_unchanged(self) -> None:
        self.guard_override = self.guard()
        self.guard_override["decision"] = "UPDATE"
        self.assertIn("final UNCHANGED exact_readback_match", "\n".join(self.check()))

    def test_missing_analytics_and_people_reviews(self) -> None:
        del self.receipt["analytics"]["review"]
        del self.receipt["people"]["review"]
        errors = "\n".join(self.check())
        self.assertIn("analytics.review", errors)
        self.assertIn("people.review", errors)

    def test_exact_templates_and_one_line_junk_are_rejected(self) -> None:
        templates = Path(__file__).resolve().parents[1] / "templates"
        self.receipt["analytics"]["review"] = self.artifact(
            "social/typefully-analytics-review.md",
            (templates / "typefully-analytics-review.md").read_text(encoding="utf-8"),
        )
        self.receipt["people"]["review"] = self.artifact(
            "social/typefully-people-review.md",
            (templates / "typefully-people-review.md").read_text(encoding="utf-8"),
        )
        errors = "\n".join(self.check())
        self.assertIn("analytics.review still matches the unfilled template", errors)
        self.assertIn("people.review still matches the unfilled template", errors)

        self.receipt["analytics"]["review"] = self.artifact(
            "social/typefully-analytics-review.md", "Status: completed — looks fine.\n"
        )
        self.receipt["people"]["review"] = self.artifact(
            "social/typefully-people-review.md", "Status: completed — no people.\n"
        )
        errors = "\n".join(self.check())
        self.assertIn("analytics.review Status must equal completed or unavailable", errors)
        self.assertIn("people.review Status must equal completed or unavailable", errors)
        self.assertIn("Decision summary must be substantive", errors)

    def test_concrete_unavailable_reviews_pass(self) -> None:
        self.receipt["analytics"]["review"] = self.artifact(
            "social/typefully-analytics-review.md",
            """# Typefully analytics review

Status: unavailable

- Observed at: 2026-10-26T11:52:00Z
- Reviewer: Fixture analytics reviewer
- Sources checked: Typefully analytics page and native X and LinkedIn analytics entry points.
- Evidence limits: No metric export was accessible, so all content choices remain editorial judgments.
- Access blocker: The authenticated analytics views returned no accessible export or metric table.
- Decision summary: Make no analytics-derived copy change and retain the explicit access limitation.
""",
        )
        self.receipt["analytics"]["x"] = {
            "status": "unavailable",
            "reason": "The analytics view exposed no accessible metric table.",
        }
        self.receipt["analytics"]["linkedin"] = {
            "status": "unavailable",
            "reason": "No native LinkedIn analytics view was accessible.",
        }
        self.receipt["people"]["review"] = self.artifact(
            "social/typefully-people-review.md",
            """# Relevant people and conversation review

Status: unavailable

- Observed at: 2026-10-26T11:53:00Z
- Reviewer: Fixture people reviewer
- Sources checked: Package sources and the available X and LinkedIn search entry points.
- Evidence limits: Profile identity and native parent targets could not be verified from retained evidence.
- Access blocker: The platform search views were unavailable for a bounded identity review.
- Decision summary: Keep mentions unresolved and prepare no reply, quote, reshare, or outreach target.
""",
        )
        self.receipt["people"]["mentions"] = {
            "state": "unresolved",
            "reason": "No native identity could be verified.",
            "verified": [],
        }
        self.receipt["people"]["opportunities"] = {
            "state": "blocked",
            "reason": "No exact native parent could be verified.",
        }
        self.assertEqual(self.check(), [])

    def test_mention_and_opportunity_states_are_explicit(self) -> None:
        self.receipt["people"]["mentions"] = {"state": "verified", "reason": "", "verified": []}
        self.receipt["people"]["opportunities"] = {"state": "blocked", "reason": ""}
        errors = "\n".join(self.check())
        self.assertIn("verified must be nonempty", errors)
        self.assertIn("opportunities.reason", errors)

    def test_unverified_and_stale_promotion(self) -> None:
        self.receipt["promotion"]["auto_retweet"]["state"] = "unverified"
        self.receipt["promotion"]["auto_plug"]["checked_at"] = "2026-10-26T11:59:59Z"
        errors = "\n".join(self.check())
        self.assertIn("auto_retweet.state", errors)
        self.assertIn("auto_plug.checked_at is stale", errors)

    def test_tag_mismatch_and_approved_evidence(self) -> None:
        self.receipt["tags"].append("approved")
        self.assertIn("exactly the selected lifecycle tag", "\n".join(self.check()))
        self.receipt["lifecycle_tag"] = "approved"
        self.receipt["tags"] = self.raw["tags"] = ["approved", "gpu"]
        self.assertIn("approval must contain", "\n".join(self.check()))
        approval = self.artifact("social/approval.md", "The author approved this exact fixture copy.\n")
        approval["reason"] = "The author explicitly approved this exact retained copy."
        self.receipt["approval"] = approval
        self.assertEqual(self.check(), [])

    def test_update_preserves_unrelated_tags(self) -> None:
        prior = copy.deepcopy(self.raw)
        prior["tags"] = ["drafting", "gpu", "owner-tag"]
        self.receipt["operation"] = "update"
        self.receipt["prior_readback"] = self.json_artifact("social/prior.json", prior)
        self.assertIn("preserve prior unrelated tags", "\n".join(self.check()))
        self.raw["tags"].append("owner-tag")
        self.receipt["tags"].append("owner-tag")
        self.assertEqual(self.check(), [])

    def test_update_requires_prior_readback(self) -> None:
        self.receipt["operation"] = "update"
        self.assertIn("prior_readback must contain", "\n".join(self.check()))

    def test_missing_timezone_and_hash_tamper(self) -> None:
        self.receipt["verified_at"] = "2026-10-26T12:05:00"
        self.receipt["people"]["review"] = copy.deepcopy(self.receipt["people"]["review"])
        self.receipt["people"]["review"]["sha256"] = "0" * 64
        errors = "\n".join(self.check())
        self.assertIn("timestamp with a timezone", errors)
        self.assertIn("people.review.sha256 does not match", errors)

    def test_unsafe_and_symlink_paths(self) -> None:
        self.receipt["people"]["review"]["path"] = "../outside"
        outside = self.package.parent / f"{self.package.name}-outside"
        outside.write_text("outside", encoding="utf-8")
        try:
            (self.package / "escape").symlink_to(outside)
            self.receipt["analytics"]["review"] = {"path": "escape", "sha256": "0" * 64}
            errors = "\n".join(self.check())
            self.assertIn("safe package-relative path", errors)
            self.assertIn("must not use a symlink", errors)
        finally:
            outside.unlink()

    def test_invalid_types_fail_without_exception(self) -> None:
        for value in (None, [], {}, {"tags": [None], "promotion": [], "analytics": None}):
            with self.subTest(value=value):
                self.write_json(WORKFLOW_PATH, value)
                errors: list[str] = []
                validate_workflow(self.package, WORKFLOW_PATH, errors)
                self.assertTrue(errors)

    def test_template_is_intentionally_incomplete(self) -> None:
        template = Path(__file__).resolve().parents[1] / "templates/typefully-workflow.json"
        self.write_json(WORKFLOW_PATH, json.loads(template.read_text(encoding="utf-8")))
        errors: list[str] = []
        validate_workflow(self.package, WORKFLOW_PATH, errors)
        self.assertTrue(errors)
        self.assertIn("state must be verified", "\n".join(errors))


if __name__ == "__main__":
    unittest.main()
