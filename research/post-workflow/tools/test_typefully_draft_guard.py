from __future__ import annotations

import copy
import json
import subprocess
import sys
import unittest
from pathlib import Path

from tools.typefully_draft_guard import decide


X_DRAFT = "synthetic-x-draft"
X_PUBLISHED = "synthetic-x-published"
LINKEDIN_DRAFT = "synthetic-linkedin-draft"

# Synthetic flat drafts:get shape. No account readback or local artifact is used.
PLAIN_DRAFT_RESPONSE = {
    "id": 123, "social_set_id": 17, "draft_id": 123,
    "status": "draft", "publish_state": None,
    "created_at": "2026-10-29T15:54:11.911Z",
    "updated_at": "2026-10-29T15:54:11.910Z",
    "scheduled_date": None, "published_at": None,
    "draft_title": "Synthetic plain draft", "tags": [],
    "preview": "Synthetic X post.",
    "share_url": None,
    "private_url": "https://typefully.example/draft-123",
    "platforms": {
        "x": {
            "enabled": True,
            "posts": [{
                "text": "Synthetic X post.",
                "media_ids": ["synthetic-media-1"],
                "quote_post_url": None, "subscribers_only": False,
                "reply_settings": None, "paid_partnership": False,
                "made_with_ai": False, "hide_link_preview": False,
            }],
            "settings": None,
        },
        "linkedin": {"enabled": False}, "mastodon": {"enabled": False},
        "threads": {"enabled": False}, "bluesky": {"enabled": False},
        "substack": {"enabled": False}, "x_article": None,
    },
    "x_published_url": None, "linkedin_published_url": None,
    "mastodon_published_url": None, "threads_published_url": None,
    "bluesky_published_url": None, "substack_published_url": None,
    "x_article_published_url": None,
    "x_post_published_at": None, "linkedin_post_published_at": None,
    "mastodon_post_published_at": None, "threads_post_published_at": None,
    "bluesky_post_published_at": None, "substack_post_published_at": None,
    "x_article_published_at": None, "scratchpad_text": None,
}


def content_from_readback(readback: dict) -> dict:
    return {
        key: copy.deepcopy(readback[key])
        for key in ("draft_title", "tags", "scratchpad_text", "platforms")
    }


def flat_envelope(readback: dict, desired_payload: dict | None = None, prior: dict | None = None) -> dict:
    target = {"social_set_id": readback["social_set_id"]}
    return {
        "desired": {"target": target, "payload": desired_payload or content_from_readback(readback)},
        "readback_target": target,
        "list_readback": {"results": [copy.deepcopy(readback)]},
        "detail_readback": copy.deepcopy(readback),
        "prior_receipt": prior,
    }


def payload(x_text: str = "Exact X text", media: str = "media-cover") -> dict:
    return {
        "draft_title": "2026-10-25-example",
        "scratchpad_text": "Private evidence note",
        "tags": ["local-draft"],
        "platforms": {
            "x": {
                "enabled": True,
                "posts": [
                    {"text": x_text, "media_ids": [media]},
                    {"text": "Exact audience reply", "media_ids": ["media-audience"]},
                ],
            },
            "linkedin": {
                "enabled": True,
                "posts": [{"text": "Exact LinkedIn text", "media_ids": [media]}],
            },
        },
    }


def envelope(
    desired_payload: dict | None = None,
    remote_payload: dict | None = None,
    *,
    target: int = 17,
    items: list[dict] | None = None,
    status: str = "draft",
    updated_at: str = "2026-10-25T20:00:00Z",
    prior: dict | None = None,
) -> dict:
    desired_payload = copy.deepcopy(desired_payload or payload())
    if items is None:
        items = [] if remote_payload is None else [
            {
                "id": "draft-1",
                "draft_title": remote_payload["draft_title"],
                "status": status,
                "updated_at": updated_at,
            }
        ]
    detail = None
    if remote_payload is not None:
        detail = {
            "id": "draft-1",
            "draft_title": remote_payload["draft_title"],
            "status": status,
            "updated_at": updated_at,
            "scheduled_at": None,
            "payload": copy.deepcopy(remote_payload),
        }
    return {
        "desired": {"target": {"social_set_id": target}, "payload": desired_payload},
        "readback_target": {"social_set_id": target},
        "list_readback": {"results": items},
        "detail_readback": detail,
        "prior_receipt": prior,
    }


class TypefullyDraftGuardTests(unittest.TestCase):
    def load_synthetic(self, kind: str) -> dict:
        readback = copy.deepcopy(PLAIN_DRAFT_RESPONSE)
        readback["scratchpad_text"] = "Synthetic notes."
        readback["platforms"]["x"]["posts"][0]["media_ids"] = []
        readback["platforms"]["x"]["settings"] = {
            "reply_to_url": "https://x.com/example/status/1"
        }
        if kind == LINKEDIN_DRAFT:
            readback["platforms"]["x"] = {"enabled": False}
            readback["platforms"]["linkedin"] = {
                "enabled": True,
                "posts": [{"text": "Synthetic LinkedIn reply.", "media_ids": []}],
                "settings": {"first_comment": "Synthetic first comment."},
            }
        elif kind == X_PUBLISHED:
            readback["status"] = "published"
            readback["published_at"] = "2026-10-29T16:00:00Z"
        elif kind != X_DRAFT:
            raise ValueError("unknown synthetic fixture")
        return readback

    def test_create_preserves_exact_payload_and_fingerprints_target_text_and_media(self) -> None:
        original = payload()
        result = decide(envelope(original))
        self.assertEqual(result["decision"], "CREATE")
        self.assertEqual(result["suggested_request"]["payload"], original)
        self.assertEqual(result["suggested_request"]["target"], {"social_set_id": 17})
        self.assertNotIn("publish_at", json.dumps(result["suggested_request"]))

        changed_media = decide(envelope(payload(media="different-media")))
        changed_text = decide(envelope(payload(x_text="Different exact X text")))
        changed_target = decide(envelope(original, target=18))
        fingerprints = {
            result["desired_fingerprint"],
            changed_media["desired_fingerprint"],
            changed_text["desired_fingerprint"],
            changed_target["desired_fingerprint"],
        }
        self.assertEqual(len(fingerprints), 4)

    def test_ambiguous_duplicate_titles_refuses(self) -> None:
        title = payload()["draft_title"]
        items = [
            {"id": "draft-1", "draft_title": title, "status": "draft"},
            {"id": "draft-2", "draft_title": title, "status": "draft"},
        ]
        result = decide(envelope(items=items))
        self.assertEqual(result["decision"], "REFUSE")
        self.assertEqual(result["reason_code"], "ambiguous_duplicate_titles")
        self.assertIsNone(result["suggested_request"])

    def test_scheduled_planned_and_published_states_refuse(self) -> None:
        for state in ("scheduled", "planned", "published"):
            with self.subTest(state=state):
                result = decide(envelope(payload(), payload(), status=state))
                self.assertEqual(result["decision"], "REFUSE")
                self.assertEqual(result["reason_code"], "non_draft_state")
                self.assertIsNone(result["suggested_request"])

        active = envelope(payload(), payload())
        active["detail_readback"]["plan_at"] = "2030-01-01T00:00:00Z"
        result = decide(active)
        self.assertEqual(result["decision"], "REFUSE")
        self.assertEqual(result["reason_code"], "planned_or_published_state")

    def test_unsafe_desired_fields_refuse_at_any_depth(self) -> None:
        cases = []
        top = payload()
        top["publish_at"] = "now"
        cases.append(top)
        platform = payload()
        platform["platforms"]["x"]["planAt"] = "next-free-slot"
        cases.append(platform)
        post = payload()
        post["platforms"]["linkedin"]["posts"][0]["share"] = True
        cases.append(post)
        status = payload()
        status["status"] = "draft"
        cases.append(status)

        for unsafe in cases:
            with self.subTest(keys=list(unsafe)):
                result = decide(envelope(unsafe))
                self.assertEqual(result["decision"], "REFUSE")
                self.assertEqual(result["reason_code"], "unsafe_desired_field")
                self.assertIsNone(result["suggested_request"])

    def test_unchanged_is_idempotent_and_emits_verified_receipt(self) -> None:
        exact = payload()
        first = decide(envelope(exact, exact))
        self.assertEqual(first["decision"], "UNCHANGED")
        self.assertIsNone(first["suggested_request"])
        self.assertEqual(first["receipt"]["remote_fingerprint"], first["current_fingerprint"])

        second = decide(envelope(exact, exact, prior=first["receipt"]))
        self.assertEqual(second["decision"], "UNCHANGED")
        self.assertEqual(second["desired_fingerprint"], first["desired_fingerprint"])
        self.assertEqual(second["receipt"], first["receipt"])

    def test_update_requires_receipt_and_preserves_human_edit(self) -> None:
        old = payload(x_text="Old guarded text")
        verified = decide(envelope(old, old))
        receipt = verified["receipt"]

        new = payload(x_text="New local text")
        update = decide(envelope(new, old, prior=receipt))
        self.assertEqual(update["decision"], "UPDATE")
        self.assertEqual(update["suggested_request"]["payload"], new)

        human = payload(x_text="Human edit after readback")
        refused = decide(
            envelope(
                new,
                human,
                prior=receipt,
                updated_at="2026-10-25T20:05:00Z",
            )
        )
        self.assertEqual(refused["decision"], "REFUSE")
        self.assertEqual(refused["reason_code"], "remote_changed_since_receipt")
        self.assertIsNone(refused["suggested_request"])

        no_receipt = decide(envelope(new, old))
        self.assertEqual(no_receipt["decision"], "REFUSE")
        self.assertEqual(no_receipt["reason_code"], "no_prior_receipt")

    def test_timestamp_change_alone_preserves_remote(self) -> None:
        old = payload(x_text="Old guarded text")
        receipt = decide(envelope(old, old))["receipt"]
        new = payload(x_text="New local text")
        changed = envelope(new, old, prior=receipt, updated_at="2026-10-25T20:05:00Z")
        result = decide(changed)
        self.assertEqual(result["decision"], "REFUSE")
        self.assertEqual(result["reason_code"], "remote_updated_since_receipt")

    def rename_envelope(self) -> dict:
        old = payload()
        receipt = decide(envelope(old, old))["receipt"]
        renamed = copy.deepcopy(old)
        renamed["draft_title"] = "2026-10-27-renamed-example"
        return envelope(renamed, old, prior=receipt)

    def test_rename_uses_receipted_id_and_preserves_exact_desired_payload(self) -> None:
        renaming = self.rename_envelope()
        result = decide(renaming)
        self.assertEqual(result["decision"], "UPDATE", result)
        self.assertEqual(result["reason_code"], "verified_remote_unchanged")
        self.assertEqual(result["suggested_request"], {
            "operation": "UPDATE",
            "target": {"social_set_id": 17},
            "draft_id": "draft-1",
            "payload": renaming["desired"]["payload"],
        })
        self.assertIsNone(result["receipt"])

    def test_rename_requires_prior_identity_in_fresh_list_and_detail(self) -> None:
        cases = []
        missing = self.rename_envelope()
        missing["list_readback"]["results"] = []
        cases.append(("missing list ID", missing, "prior_draft_missing"))
        wrong_list = self.rename_envelope()
        wrong_list["list_readback"]["results"][0]["id"] = "draft-2"
        cases.append(("different list ID", wrong_list, "prior_draft_missing"))
        wrong_detail = self.rename_envelope()
        wrong_detail["detail_readback"]["id"] = "draft-2"
        cases.append(("different detail ID", wrong_detail, "detail_id_mismatch"))
        missing_detail = self.rename_envelope()
        missing_detail["detail_readback"] = None
        cases.append(("missing detail", missing_detail, "missing_detail_readback"))
        duplicate_id = self.rename_envelope()
        duplicate_id["list_readback"]["results"].append(copy.deepcopy(duplicate_id["list_readback"]["results"][0]))
        cases.append(("duplicate ID", duplicate_id, "ambiguous_duplicate_draft_ids"))
        for name, candidate, reason in cases:
            with self.subTest(case=name):
                result = decide(candidate)
                self.assertEqual(result["decision"], "REFUSE", result)
                self.assertEqual(result["reason_code"], reason)
                self.assertIsNone(result["suggested_request"])

    def test_rename_never_selects_another_draft_by_desired_title(self) -> None:
        for prior_present in (True, False):
            with self.subTest(prior_present=prior_present):
                candidate = self.rename_envelope()
                other = copy.deepcopy(candidate["list_readback"]["results"][0])
                other.update(id="draft-2", draft_title=candidate["desired"]["payload"]["draft_title"])
                candidate["list_readback"]["results"] = (
                    candidate["list_readback"]["results"] + [other] if prior_present else [other]
                )
                # Even an exact desired detail on the other ID cannot replace receipt identity.
                candidate["detail_readback"].update(id="draft-2", payload=copy.deepcopy(candidate["desired"]["payload"]))
                result = decide(candidate)
                self.assertEqual(result["decision"], "REFUSE", result)
                self.assertEqual(result["reason_code"], "rename_title_conflict" if prior_present else "prior_draft_missing")
                self.assertIsNone(result["suggested_request"])

    def test_rename_retains_duplicate_title_protection(self) -> None:
        for duplicate_current in (True, False):
            with self.subTest(duplicate_current=duplicate_current):
                candidate = self.rename_envelope()
                title = (candidate["list_readback"]["results"][0]["draft_title"] if duplicate_current
                         else candidate["desired"]["payload"]["draft_title"])
                candidate["list_readback"]["results"].append({"id": "draft-2", "draft_title": title})
                if not duplicate_current:
                    candidate["list_readback"]["results"].append({"id": "draft-3", "draft_title": title})
                result = decide(candidate)
                self.assertEqual(result["decision"], "REFUSE", result)
                self.assertEqual(result["reason_code"], "ambiguous_duplicate_titles")

    def test_rename_preserves_receipt_target_content_and_timestamp_checks(self) -> None:
        cases = (
            ("receipt_target", "receipt_target_mismatch"),
            ("readback_target", "target_mismatch"),
            ("list_target", "readback_social_set_mismatch"),
            ("detail_target", "readback_social_set_mismatch"),
            ("fingerprint", "remote_changed_since_receipt"),
            ("human_title", "remote_changed_since_receipt"),
            ("timestamp", "remote_updated_since_receipt"),
            ("missing_timestamp", "missing_remote_update_timestamp"),
        )
        for name, reason in cases:
            with self.subTest(case=name):
                candidate = self.rename_envelope()
                if name == "receipt_target":
                    candidate["prior_receipt"]["target_fingerprint"] = "sha256:different-target"
                elif name == "readback_target":
                    candidate["readback_target"]["social_set_id"] = 99
                elif name in ("list_target", "detail_target"):
                    obj = candidate["list_readback"]["results"][0] if name == "list_target" else candidate["detail_readback"]
                    obj["social_set_id"] = 99
                elif name == "fingerprint":
                    candidate["prior_receipt"]["remote_fingerprint"] = "sha256:different-content"
                elif name == "human_title":
                    candidate["detail_readback"]["payload"]["draft_title"] = "Human rename after readback"
                else:
                    candidate["detail_readback"]["updated_at"] = None if name == "missing_timestamp" else "2026-10-27T20:00:00Z"
                result = decide(candidate)
                self.assertEqual(result["decision"], "REFUSE", result)
                self.assertEqual(result["reason_code"], reason)
                self.assertIsNone(result["suggested_request"])

    def test_rename_preserves_state_and_unknown_field_checks(self) -> None:
        for name, reason in (
            ("list_state", "non_draft_state"),
            ("detail_state", "non_draft_state"),
            ("scheduled", "planned_or_published_state"),
            ("unknown_detail", "invalid_input"),
            ("unknown_payload", "invalid_input"),
        ):
            with self.subTest(case=name):
                candidate = self.rename_envelope()
                if name == "list_state":
                    candidate["list_readback"]["results"][0]["status"] = "published"
                elif name == "detail_state":
                    candidate["detail_readback"]["status"] = "scheduled"
                elif name == "scheduled":
                    candidate["detail_readback"]["scheduled_at"] = "2030-01-01T00:00:00Z"
                elif name == "unknown_detail":
                    candidate["detail_readback"]["unknown_new_field"] = "substantive"
                else:
                    candidate["detail_readback"]["payload"]["platforms"]["x"]["posts"][0]["unknown_new_field"] = "substantive"
                result = decide(candidate)
                self.assertEqual(result["decision"], "REFUSE", result)
                self.assertEqual(result["reason_code"], reason)
                self.assertIsNone(result["suggested_request"])

    def test_rename_rejects_malformed_prior_receipts(self) -> None:
        for prior in ("draft-1", {}, {"schema_version": 99, "draft_id": "draft-1"},
                      {"schema_version": 1, "draft_id": True}, {"schema_version": 1, "draft_id": ""}):
            with self.subTest(prior=prior):
                candidate = self.rename_envelope()
                candidate["prior_receipt"] = prior
                result = decide(candidate)
                self.assertEqual(result["decision"], "REFUSE", result)
                self.assertEqual(result["reason_code"], "invalid_prior_receipt")
                self.assertIsNone(result["suggested_request"])

    def test_update_without_remote_timestamp_refuses(self) -> None:
        old = payload(x_text="Old guarded text")
        receipt = decide(envelope(old, old))["receipt"]
        new = payload(x_text="New local text")
        missing = envelope(new, old, prior=receipt, updated_at=None)
        result = decide(missing)
        self.assertEqual(result["decision"], "REFUSE")
        self.assertEqual(result["reason_code"], "missing_remote_update_timestamp")

    def test_target_mismatch_and_missing_detail_refuse(self) -> None:
        exact = payload()
        mismatch = envelope(exact)
        mismatch["readback_target"] = {"social_set_id": 99}
        self.assertEqual(decide(mismatch)["reason_code"], "target_mismatch")

        title = exact["draft_title"]
        missing = envelope(exact, items=[{"id": "draft-1", "draft_title": title, "status": "draft"}])
        self.assertEqual(decide(missing)["reason_code"], "missing_detail_readback")

    def test_synthetic_x_and_linkedin_readback_shapes_are_unchanged(self) -> None:
        for path in (X_DRAFT, LINKEDIN_DRAFT):
            with self.subTest(fixture=path):
                readback = self.load_synthetic(path)
                result = decide(flat_envelope(readback))
                self.assertEqual(result["decision"], "UNCHANGED", result)

    def test_x_only_and_linkedin_only_comment_payloads_can_be_created_without_media(self) -> None:
        for path, platform, setting in (
            (X_DRAFT, "x", "reply_to_url"),
            (LINKEDIN_DRAFT, "linkedin", "first_comment"),
        ):
            with self.subTest(fixture=path):
                readback = self.load_synthetic(path)
                request = flat_envelope(readback)
                request["list_readback"] = {"results": []}
                request["detail_readback"] = None
                result = decide(request)
                self.assertEqual(result["decision"], "CREATE", result)
                emitted = result["suggested_request"]["payload"]["platforms"][platform]
                self.assertEqual(emitted["posts"][0]["media_ids"], [])
                self.assertEqual(
                    emitted["settings"][setting],
                    readback["platforms"][platform]["settings"][setting],
                )

    def test_synthetic_published_x_is_never_mutable(self) -> None:
        readback = self.load_synthetic(X_PUBLISHED)
        result = decide(flat_envelope(readback))
        self.assertEqual(result["decision"], "REFUSE")
        self.assertEqual(result["reason_code"], "non_draft_state")

    def test_api_null_defaults_match_missing_defaults_and_optional_media(self) -> None:
        readback = self.load_synthetic(X_DRAFT)
        desired = content_from_readback(readback)
        desired["platforms"] = {
            "x": {
                "enabled": True,
                "posts": [{"text": readback["platforms"]["x"]["posts"][0]["text"], "media_ids": []}],
                "settings": {"reply_to_url": readback["platforms"]["x"]["settings"]["reply_to_url"]},
            }
        }
        result = decide(flat_envelope(readback, desired))
        self.assertEqual(result["decision"], "UNCHANGED", result)

    def test_plain_draft_null_scratchpad_matches_empty_or_omitted_and_allows_append(self) -> None:
        readback = copy.deepcopy(PLAIN_DRAFT_RESPONSE)
        expected_fingerprint = None
        for include_scratchpad in (False, True):
            with self.subTest(include_scratchpad=include_scratchpad):
                desired = content_from_readback(readback)
                if include_scratchpad:
                    desired["scratchpad_text"] = ""
                else:
                    del desired["scratchpad_text"]
                verified = decide(flat_envelope(readback, desired))
                self.assertEqual(verified["decision"], "UNCHANGED", verified)
                if expected_fingerprint is None:
                    expected_fingerprint = verified["current_fingerprint"]
                self.assertEqual(verified["current_fingerprint"], expected_fingerprint)

                desired["platforms"]["x"]["posts"].append({"text": "Append this reply."})
                appended = decide(flat_envelope(readback, desired, verified["receipt"]))
                self.assertEqual(appended["decision"], "UPDATE", appended)
                suggested = appended["suggested_request"]["payload"]
                self.assertEqual(suggested["scratchpad_text"], "")
                self.assertEqual(suggested["platforms"]["x"]["posts"][0]["text"],
                                 readback["platforms"]["x"]["posts"][0]["text"])
                self.assertEqual(suggested["platforms"]["x"]["posts"][-1]["text"], "Append this reply.")
                self.assertIsNone(readback["scratchpad_text"])

                for field in ("text", "scratchpad_text"):
                    with self.subTest(manual_edit=field):
                        human = copy.deepcopy(readback)
                        if field == "text":
                            human["platforms"]["x"]["posts"][0]["text"] += " Human edit."
                        else:
                            human["scratchpad_text"] = "Human's private notes."
                        refused = decide(flat_envelope(human, desired, verified["receipt"]))
                        self.assertEqual(refused["decision"], "REFUSE", refused)
                        self.assertEqual(refused["reason_code"], "remote_changed_since_receipt")
                        self.assertIsNone(refused["suggested_request"])

    def test_nested_remote_null_scratchpad_matches_empty_without_mutating_input(self) -> None:
        desired = payload()
        desired["scratchpad_text"] = ""
        remote = copy.deepcopy(desired)
        remote["scratchpad_text"] = None
        candidate = envelope(desired, remote)
        result = decide(candidate)
        self.assertEqual(result["decision"], "UNCHANGED", result)
        self.assertIsNone(candidate["detail_readback"]["payload"]["scratchpad_text"])

    def test_nonempty_remote_scratchpad_is_preserved_and_fingerprinted(self) -> None:
        original = payload()
        verified = decide(envelope(original, original))
        desired = copy.deepcopy(original)
        desired["platforms"]["x"]["posts"].append({"text": "New reply."})
        result = decide(envelope(desired, original, prior=verified["receipt"]))
        self.assertEqual(result["decision"], "UPDATE", result)
        self.assertEqual(result["suggested_request"]["payload"]["scratchpad_text"],
                         original["scratchpad_text"])
        human = copy.deepcopy(original)
        human["scratchpad_text"] += " Human addition."
        refused = decide(envelope(desired, human, prior=verified["receipt"]))
        self.assertEqual(refused["decision"], "REFUSE", refused)
        self.assertEqual(refused["reason_code"], "remote_changed_since_receipt")

    def test_desired_null_scratchpad_is_still_invalid(self) -> None:
        desired = payload()
        desired["scratchpad_text"] = None
        for remote in (None, desired):
            with self.subTest(existing_draft=remote is not None):
                result = decide(envelope(desired, remote))
                self.assertEqual(result["decision"], "REFUSE", result)
                self.assertEqual(result["reason_code"], "invalid_input")
                self.assertIn("desired.payload.scratchpad_text must be a string", result["message"])
                self.assertIsNone(result["suggested_request"])

    def test_changed_reply_target_is_detected_as_human_edit(self) -> None:
        readback = self.load_synthetic(X_DRAFT)
        receipt = decide(flat_envelope(readback))["receipt"]
        human = copy.deepcopy(readback)
        human["platforms"]["x"]["settings"]["reply_to_url"] = "https://x.com/example/status/999"
        human["updated_at"] = "2026-10-26T04:00:00Z"
        result = decide(flat_envelope(human, content_from_readback(readback), receipt))
        self.assertEqual(result["decision"], "REFUSE")
        self.assertEqual(result["reason_code"], "remote_changed_since_receipt")

    def test_changed_linkedin_first_comment_is_detected_as_human_edit(self) -> None:
        readback = self.load_synthetic(LINKEDIN_DRAFT)
        receipt = decide(flat_envelope(readback))["receipt"]
        human = copy.deepcopy(readback)
        human["platforms"]["linkedin"]["settings"]["first_comment"] += "\nHuman addition."
        human["updated_at"] = "2026-10-26T04:00:00Z"
        result = decide(flat_envelope(human, content_from_readback(readback), receipt))
        self.assertEqual(result["decision"], "REFUSE")
        self.assertEqual(result["reason_code"], "remote_changed_since_receipt")

    def test_changed_quote_and_disclosure_are_detected_as_human_edits(self) -> None:
        readback = self.load_synthetic(X_DRAFT)
        receipt = decide(flat_envelope(readback))["receipt"]
        for field, value in (("quote_post_url", "https://x.com/example/status/123"), ("made_with_ai", True)):
            with self.subTest(field=field):
                human = copy.deepcopy(readback)
                human["platforms"]["x"]["posts"][0][field] = value
                human["updated_at"] = "2026-10-26T04:00:00Z"
                result = decide(flat_envelope(human, content_from_readback(readback), receipt))
                self.assertEqual(result["decision"], "REFUSE")
                self.assertEqual(result["reason_code"], "remote_changed_since_receipt")

    def test_unknown_response_post_field_is_not_silently_discarded(self) -> None:
        readback = self.load_synthetic(X_DRAFT)
        readback["platforms"]["x"]["posts"][0]["unknown_new_field"] = "substantive"
        result = decide(flat_envelope(readback))
        self.assertEqual(result["decision"], "REFUSE")
        self.assertEqual(result["reason_code"], "invalid_input")

    def test_cli_help_and_refuse_exit_are_machine_readable(self) -> None:
        script = Path(__file__).with_name("typefully_draft_guard.py")
        help_result = subprocess.run(
            [sys.executable, str(script), "--help"],
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0)
        self.assertIn("No network or account mutation", help_result.stdout)

        unsafe = payload()
        unsafe["publish_at"] = "now"
        refused = subprocess.run(
            [sys.executable, str(script), "--compact"],
            input=json.dumps(envelope(unsafe)),
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(refused.returncode, 3)
        self.assertEqual(json.loads(refused.stdout)["decision"], "REFUSE")


if __name__ == "__main__":
    unittest.main()
