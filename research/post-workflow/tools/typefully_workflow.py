#!/usr/bin/env python3
"""Validate retained Typefully workflow artifacts without account mutation.

The receipt binds a fresh draft-only detail readback to the existing draft
guard, exact local platform text, exact per-post media IDs, lifecycle and mention
state, analytics/people reviews, and dormant promotion checks. Hashes make those
artifacts auditable. They do not establish remote provenance or human identity.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any

if __package__:
    from .typefully_draft_guard import GuardInputError, fingerprint, readback_fingerprint
else:
    from typefully_draft_guard import GuardInputError, fingerprint, readback_fingerprint


WORKFLOW_PATH = "social/typefully-workflow.json"
TYPEFULLY_PATH = "social/typefully-receipt.json"
LIFECYCLE = {"idea", "drafting", "needs-review", "approved"}
INERT_PLATFORMS = {"mastodon", "threads", "bluesky", "substack", "x_article"}
PLATFORMS = ("x", "linkedin")
REQUIRED_NULL_READBACK_FIELDS = (
    "publish_state",
    "scheduled_date",
    "published_at",
    "share_url",
    "x_published_url",
    "linkedin_published_url",
    "x_post_published_at",
    "linkedin_post_published_at",
)


def validate_workflow(package: Path, relative: str, errors: list[str]) -> None:
    """Append failures for one package-contained schema-1 workflow receipt."""
    package = package.resolve()

    def fail(message: str) -> None:
        errors.append(f"Typefully workflow: {message}")

    def text(value: Any) -> bool:
        return isinstance(value, str) and bool(value.strip())

    def identifier(value: Any) -> bool:
        return text(value) or (type(value) is int and value > 0)

    def path_for(value: Any, label: str) -> Path | None:
        if not text(value) or "\\" in value or "\x00" in value:
            fail(f"{label} must be a safe package-relative path")
            return None
        relative_path = Path(value)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            fail(f"{label} must be a safe package-relative path")
            return None
        candidate = package / relative_path
        current = candidate
        while current != package:
            if current.is_symlink():
                fail(f"{label} must not use a symlink")
                return None
            current = current.parent
        try:
            target = candidate.resolve()
            target.relative_to(package)
            return target
        except (OSError, RuntimeError, ValueError):
            fail(f"{label} escapes the package or cannot be resolved")
            return None

    def read_json(path: Path | None, label: str) -> dict[str, Any]:
        if path is None:
            return {}
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(value, dict):
                return value
            fail(f"{label} must be a JSON object")
        except (OSError, UnicodeError, ValueError) as exc:
            fail(f"cannot read {label}: {exc}")
        return {}

    def artifact(value: Any, label: str) -> Path | None:
        if not isinstance(value, dict):
            fail(f"{label} must contain path and sha256")
            return None
        target = path_for(value.get("path"), f"{label}.path")
        if target is None:
            return None
        try:
            content = target.read_bytes()
            if not content:
                fail(f"{label} file is empty")
            digest = value.get("sha256")
            if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
                fail(f"{label}.sha256 must be a lowercase SHA-256 digest")
            elif hashlib.sha256(content).hexdigest() != digest:
                fail(f"{label}.sha256 does not match its file")
            return target
        except OSError as exc:
            fail(f"cannot read {label}: {exc}")
            return None

    def timestamp(value: Any, label: str) -> datetime | None:
        try:
            if not isinstance(value, str) or "T" not in value:
                raise ValueError
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if parsed.utcoffset() is None:
                raise ValueError
            return parsed
        except ValueError:
            fail(f"{label} must be an ISO timestamp with a timezone")
            return None

    def review_document(
        path: Path | None,
        label: str,
        template_name: str,
        receipt_verified_at: datetime | None,
    ) -> str | None:
        if path is None:
            return None
        try:
            body = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            fail(f"cannot read {label}: {exc}")
            return None
        template = Path(__file__).resolve().parents[1] / "templates" / template_name
        try:
            if template.is_file() and path.read_bytes() == template.read_bytes():
                fail(f"{label} still matches the unfilled template")
        except OSError as exc:
            fail(f"cannot compare {label} with its template: {exc}")

        status_match = re.search(r"(?im)^Status:[ \t]*(completed|unavailable)[ \t]*$", body)
        if status_match is None:
            fail(f"{label} Status must equal completed or unavailable")
            status = None
        else:
            status = status_match.group(1).lower()

        def field(name: str) -> str:
            match = re.search(
                rf"(?im)^[ \t]*-[ \t]*{re.escape(name)}:[ \t]*(.*?)[ \t]*$",
                body,
            )
            return match.group(1).strip() if match else ""

        def require_substantive(name: str, minimum: int) -> str:
            value = field(name)
            placeholders = {"", "none", "n/a", "na", "unknown", "tbd", "not started", "not-started"}
            if len(value) < minimum or value.lower() in placeholders:
                fail(f"{label} {name} must be substantive, not a placeholder")
            return value

        observed_text = field("Observed at")
        observed_at = timestamp(observed_text, f"{label} Observed at")
        if observed_at and receipt_verified_at and observed_at > receipt_verified_at:
            fail(f"{label} Observed at is later than workflow verified_at")
        require_substantive("Reviewer", 2)
        require_substantive("Sources checked", 6)
        require_substantive("Evidence limits", 12)
        require_substantive("Decision summary", 20)
        if status == "unavailable":
            require_substantive("Access blocker", 12)
        return status

    data = read_json(path_for(relative, "receipt path"), "receipt")
    if not data:
        fail("receipt must be a nonempty JSON object")
        return
    if type(data.get("schema_version")) is not int or data["schema_version"] != 1:
        fail("schema_version must equal 1")
    for field in ("draft_id", "social_set_id"):
        if not identifier(data.get(field)):
            fail(f"{field} must be a nonempty string or positive integer")
    verified_at = timestamp(data.get("verified_at"), "verified_at")
    if data.get("status") != "draft":
        fail("status must equal 'draft'")
    for field in ("scheduled_at", "published_at"):
        if field not in data or data[field] is not None:
            fail(f"{field} must be null")
    if data.get("publication_authorized") is not False:
        fail("publication_authorized must be false")
    if data.get("platforms") != ["x", "linkedin"]:
        fail("platforms must equal ['x', 'linkedin']")

    tags = data.get("tags")
    valid_tags = isinstance(tags, list) and all(text(tag) for tag in tags)
    if not valid_tags or len(tags) != len(set(tags)):
        fail("tags must be a list of distinct nonempty strings")
        tags = []
    lifecycle = data.get("lifecycle_tag")
    if not isinstance(lifecycle, str) or lifecycle not in LIFECYCLE:
        fail("lifecycle_tag must be idea, drafting, needs-review, or approved")
    if [tag for tag in tags if tag in LIFECYCLE] != [lifecycle]:
        fail("tags must contain exactly the selected lifecycle tag")
    if lifecycle == "approved":
        approval = data.get("approval")
        artifact(approval, "approval")
        if not isinstance(approval, dict) or not text(approval.get("reason")):
            fail("approval.reason must reference Chris's explicit approval of this exact content")
    elif data.get("approval") is not None:
        fail("approval must be null unless lifecycle_tag is approved")

    raw_path = artifact(data.get("raw_readback"), "raw_readback")
    raw = read_json(raw_path, "raw draft readback")
    updated_at = timestamp(raw.get("updated_at"), "raw_readback.updated_at")
    if verified_at and updated_at and verified_at < updated_at:
        fail("verified_at is older than raw_readback.updated_at")
    for field in ("id", "draft_id", "social_set_id"):
        expected = data.get("social_set_id" if field == "social_set_id" else "draft_id")
        if field == "draft_id" and field not in raw:
            continue
        if not identifier(raw.get(field)) or str(raw[field]) != str(expected):
            fail(f"raw_readback.{field} does not match receipt")
    if raw.get("status") != "draft":
        fail("raw_readback.status must equal 'draft'")
    for field in REQUIRED_NULL_READBACK_FIELDS:
        if field not in raw or raw[field] is not None:
            fail(f"raw_readback.{field} must be present and null")
    if "scheduled_at" in raw and raw["scheduled_at"] is not None:
        fail("raw_readback.scheduled_at must be null")
    for field, value in raw.items():
        if (field.endswith("_published_at") or field.endswith("_published_url")) and value is not None:
            fail(f"raw_readback.{field} must be null")
    if raw.get("tags") != tags:
        fail("raw_readback.tags do not exactly match receipt tags")

    platforms = raw.get("platforms")
    if not isinstance(platforms, dict):
        fail("raw_readback.platforms must be an object")
        platforms = {}
    raw_media: dict[str, list[list[Any]]] = {name: [] for name in PLATFORMS}
    for name, config in platforms.items():
        if name not in PLATFORMS:
            inert = config is None or (
                isinstance(config, dict) and set(config) == {"enabled"} and config["enabled"] is False
            )
            if name not in INERT_PLATFORMS or not inert:
                fail(f"unsupported raw_readback platform: {name}")
            continue
        if not isinstance(config, dict) or config.get("enabled") is not True:
            fail(f"raw_readback.platforms.{name} must be enabled")
            continue
        posts = config.get("posts")
        if not isinstance(posts, list) or not posts:
            fail(f"raw_readback.platforms.{name}.posts must be nonempty")
            continue
        for index, post in enumerate(posts):
            label = f"raw_readback.platforms.{name}.posts[{index}]"
            if not isinstance(post, dict) or not text(post.get("text")):
                fail(f"{label}.text must be nonempty")
                continue
            media = post.get("media_ids")
            if not isinstance(media, list) or any(not identifier(item) for item in media):
                fail(f"{label}.media_ids must be a list of nonempty IDs")
                media = []
            if len({str(item) for item in media}) != len(media):
                fail(f"{label}.media_ids must not contain duplicates")
            raw_media[name].append(media)
    for name in PLATFORMS:
        if name not in platforms:
            fail(f"raw_readback.platforms must contain {name}")
        if not any(post_media for post_media in raw_media[name]):
            fail(f"raw_readback {name} must retain at least one media ID")

    declared_media = data.get("media_ids")
    if not isinstance(declared_media, dict) or set(declared_media) != set(PLATFORMS):
        fail("media_ids must contain exactly x and linkedin per-post lists")
    else:
        for name in PLATFORMS:
            value = declared_media[name]
            if not isinstance(value, list) or any(not isinstance(item, list) for item in value):
                fail(f"media_ids.{name} must be a list of per-post media-ID lists")
            elif value != raw_media[name]:
                fail(f"media_ids.{name} does not exactly match raw_readback")

    try:
        raw_fingerprint = readback_fingerprint(raw, data.get("social_set_id"))
    except (GuardInputError, TypeError, ValueError) as exc:
        raw_fingerprint = None
        fail(f"raw_readback is incompatible with typefully_draft_guard: {exc}")
    guard = read_json(artifact(data.get("guard_receipt"), "guard_receipt"), "final guard receipt")
    if guard.get("decision") != "UNCHANGED" or guard.get("reason_code") != "exact_readback_match":
        fail("guard_receipt must be the final UNCHANGED exact_readback_match decision")
    if guard.get("suggested_request") is not None:
        fail("guard_receipt.suggested_request must be null")
    if str(guard.get("draft_id")) != str(data.get("draft_id")):
        fail("guard_receipt.draft_id does not match receipt")
    guard_body = guard.get("receipt")
    if not isinstance(guard_body, dict):
        fail("guard_receipt.receipt must be an object")
        guard_body = {}
    if guard_body.get("schema_version") != 1:
        fail("guard_receipt.receipt.schema_version must equal 1")
    if str(guard_body.get("draft_id")) != str(data.get("draft_id")):
        fail("guard_receipt.receipt.draft_id does not match receipt")
    if guard_body.get("target_fingerprint") != fingerprint({"social_set_id": data.get("social_set_id")}):
        fail("guard_receipt target fingerprint does not match social_set_id")
    if raw_fingerprint is not None:
        for field in ("remote_fingerprint", "desired_fingerprint"):
            if guard_body.get(field) != raw_fingerprint:
                fail(f"guard_receipt.receipt.{field} does not match raw_readback")
        if guard.get("current_fingerprint") != raw_fingerprint:
            fail("guard_receipt.current_fingerprint does not match raw_readback")
    if guard_body.get("readback_updated_at") != raw.get("updated_at"):
        fail("guard_receipt readback timestamp does not match raw_readback.updated_at")

    legacy = read_json(path_for(TYPEFULLY_PATH, "draft receipt path"), "Typefully draft receipt")
    for field in ("draft_id", "social_set_id"):
        if str(legacy.get(field)) != str(data.get(field)):
            fail(f"{field} does not match {TYPEFULLY_PATH}")
    if legacy.get("platforms") != ["x", "linkedin"]:
        fail(f"{TYPEFULLY_PATH} must declare both X and LinkedIn")
    texts = legacy.get("texts")
    if not isinstance(texts, dict):
        fail(f"{TYPEFULLY_PATH} must contain exact texts")
        texts = {}
    for name in PLATFORMS:
        config = platforms.get(name)
        posts = config.get("posts", []) if isinstance(config, dict) else []
        if isinstance(posts, list) and all(isinstance(post, dict) and isinstance(post.get("text"), str) for post in posts):
            exact = "\n\n---\n\n".join(post["text"] for post in posts)
            if texts.get(name) != exact:
                fail(f"raw_readback {name} text does not match {TYPEFULLY_PATH}")
    flattened_media: list[Any] = []
    seen_media: set[str] = set()
    for name in PLATFORMS:
        for per_post in raw_media[name]:
            for media_id in per_post:
                key = str(media_id)
                if key not in seen_media:
                    seen_media.add(key)
                    flattened_media.append(media_id)
    if legacy.get("media_ids") != flattened_media:
        fail(f"raw_readback media IDs do not match {TYPEFULLY_PATH}")

    operation = data.get("operation")
    if operation not in ("create", "update"):
        fail("operation must equal create or update")
    if operation == "create" and data.get("prior_readback") is not None:
        fail("create operation prior_readback must be null")
    if operation == "update":
        prior = read_json(artifact(data.get("prior_readback"), "prior_readback"), "prior draft readback")
        for field in ("id", "draft_id", "social_set_id"):
            expected = data.get("social_set_id" if field == "social_set_id" else "draft_id")
            if field == "draft_id" and field not in prior:
                continue
            if not identifier(prior.get(field)) or str(prior[field]) != str(expected):
                fail(f"prior_readback.{field} does not match receipt")
        if prior.get("status") != "draft":
            fail("prior_readback.status must equal 'draft'")
        prior_tags = prior.get("tags")
        if not isinstance(prior_tags, list) or not all(text(tag) for tag in prior_tags) or len(prior_tags) != len(set(prior_tags)):
            fail("prior_readback.tags must be a list of distinct nonempty strings")
        elif any(tag not in tags for tag in prior_tags if tag not in LIFECYCLE):
            fail("update must preserve prior unrelated tags")

    analytics = data.get("analytics")
    analytics = analytics if isinstance(analytics, dict) else {}
    analytics_review_status = review_document(
        artifact(analytics.get("review"), "analytics.review"),
        "analytics.review",
        "typefully-analytics-review.md",
        verified_at,
    )
    analytics_statuses: list[Any] = []
    for name in PLATFORMS:
        entry = analytics.get(name)
        entry = entry if isinstance(entry, dict) else {}
        status = entry.get("status")
        analytics_statuses.append(status)
        if status not in ("reviewed", "unavailable", "no_history"):
            fail(f"analytics.{name}.status must be reviewed, unavailable, or no_history")
        if status in ("unavailable", "no_history") and not text(entry.get("reason")):
            fail(f"analytics.{name}.reason is required when {status}")
    if analytics_review_status == "completed" and "reviewed" not in analytics_statuses:
        fail("analytics.review completed status requires at least one reviewed platform")
    if analytics_review_status == "unavailable" and "reviewed" in analytics_statuses:
        fail("analytics.review unavailable status cannot declare a reviewed platform")

    people = data.get("people")
    people = people if isinstance(people, dict) else {}
    review_document(
        artifact(people.get("review"), "people.review"),
        "people.review",
        "typefully-people-review.md",
        verified_at,
    )
    mentions = people.get("mentions")
    mentions = mentions if isinstance(mentions, dict) else {}
    mention_state = mentions.get("state")
    if mention_state not in ("none", "verified", "unresolved", "partial"):
        fail("people.mentions.state must be none, verified, unresolved, or partial")
    verified_mentions = mentions.get("verified")
    if not isinstance(verified_mentions, list) or any(not text(item) for item in verified_mentions) or len(verified_mentions) != len(set(verified_mentions)):
        fail("people.mentions.verified must be a list of distinct native mention identifiers")
        verified_mentions = []
    if mention_state in ("verified", "partial") and not verified_mentions:
        fail(f"people.mentions.verified must be nonempty when state is {mention_state}")
    if mention_state in ("none", "unresolved") and verified_mentions:
        fail(f"people.mentions.verified must be empty when state is {mention_state}")
    if mention_state in ("none", "unresolved", "partial") and not text(mentions.get("reason")):
        fail(f"people.mentions.reason is required when state is {mention_state}")
    opportunities = people.get("opportunities")
    opportunities = opportunities if isinstance(opportunities, dict) else {}
    opportunity_state = opportunities.get("state")
    if opportunity_state not in ("none", "identified", "blocked"):
        fail("people.opportunities.state must be none, identified, or blocked")
    if opportunity_state in ("none", "blocked") and not text(opportunities.get("reason")):
        fail(f"people.opportunities.reason is required when state is {opportunity_state}")

    promotion = data.get("promotion")
    promotion = promotion if isinstance(promotion, dict) else {}
    for name in ("auto_retweet", "auto_plug"):
        entry = promotion.get(name)
        entry = entry if isinstance(entry, dict) else {}
        label = f"promotion.{name}"
        state = entry.get("state")
        if state not in ("on", "off", "not_applicable"):
            fail(f"{label}.state must be verified as on, off, or not_applicable")
        if not isinstance(entry.get("trigger"), str) or (state == "on" and not text(entry["trigger"])):
            fail(f"{label}.trigger must be a string, nonempty when on")
        if not isinstance(entry.get("reason"), str) or (state in ("off", "not_applicable") and not text(entry["reason"])):
            fail(f"{label}.reason must be a string, nonempty when off or not_applicable")
        if entry.get("dormant") is not True:
            fail(f"{label}.dormant must be true while the draft is unscheduled")
        artifact(entry.get("evidence"), f"{label}.evidence")
        checked_at = timestamp(entry.get("checked_at"), f"{label}.checked_at")
        if checked_at and updated_at and checked_at < updated_at:
            fail(f"{label}.checked_at is stale relative to raw_readback.updated_at")
        if checked_at and verified_at and checked_at > verified_at:
            fail(f"{label}.checked_at is later than verified_at")
    if not text(data.get("reviewer")):
        fail("reviewer must name the person or agent who reviewed the artifacts")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--receipt", default=WORKFLOW_PATH)
    arguments = parser.parse_args()
    failures: list[str] = []
    validate_workflow(arguments.package, arguments.receipt, failures)
    print("\n".join(failures) if failures else "PASS: Typefully workflow artifact checks")
    raise SystemExit(bool(failures))
