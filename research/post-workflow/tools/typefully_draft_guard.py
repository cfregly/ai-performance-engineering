#!/usr/bin/env python3
"""Plan a safe Typefully draft create/update from verified JSON readbacks.

This is a local, pure-standard-library guard. It makes no network calls, reads
no credentials, and performs no account mutation. Feed its suggested request
to the installed Typefully skill only after reviewing the decision. It is an
additional draft guard, not a replacement for the workspace package check,
six-audience requirement, evidence review, identity review, or live readback.

Input envelope (``--input -`` reads stdin)::

  {
    "desired": {
      "target": {"social_set_id": 123},
      "payload": {
        "draft_title": "package-slug",
        "platforms": {
          "x": {"enabled": true, "posts": [{"text": "...", "media_ids": ["m1"]}]},
          "linkedin": {"enabled": true, "posts": [{"text": "...", "media_ids": ["m1"]}]}
        }
      }
    },
    "readback_target": {"social_set_id": 123},
    "list_readback": {"results": []},
    "detail_readback": null,
    "prior_receipt": null
  }

For an existing title, provide exactly one matching list item and its fresh
detail readback. The detail may put the comparable draft body under ``payload``
or expose ``draft_title`` and ``platforms`` at top level. Save a receipt only
from an UNCHANGED result after a fresh readback. That receipt lets a later run
distinguish our last verified content from a human edit. To rename a receipted
draft, include its exact prior ID in the fresh list and detail readbacks. A
different draft with the desired title still blocks the update.

Decisions are CREATE, UPDATE, UNCHANGED, or REFUSE. CREATE/UPDATE output is a
suggestion only. Publication, scheduling, queueing, sharing, and sending fields
are rejected at any nesting depth and never appear in a suggested payload.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
PLATFORMS = ("x", "linkedin")
PAYLOAD_KEYS = {"draft_title", "platforms", "scratchpad_text", "tags"}
PLATFORM_KEYS = {"enabled", "posts", "settings"}
KNOWN_DISABLED_PLATFORMS = {"mastodon", "threads", "bluesky", "substack"}
SAFE_ACTION_LIKE_KEYS = {"share_with_followers"}
X_POST_DEFAULTS: dict[str, Any] = {
    "media_ids": [], "quote_post_url": None, "subscribers_only": False,
    "reply_settings": None, "paid_partnership": False, "made_with_ai": False,
    "hide_link_preview": False,
}
LINKEDIN_POST_DEFAULTS: dict[str, Any] = {
    "media_ids": [], "linkedin_reshare_urn": None, "hide_link_preview": False,
}
X_SETTINGS_DEFAULTS: dict[str, Any] = {
    "reply_to_url": None, "community_id": None, "share_with_followers": None,
}
LINKEDIN_SETTINGS_DEFAULTS: dict[str, Any] = {"first_comment": None}
DETAIL_RESPONSE_KEYS = {
    "id", "social_set_id", "draft_id", "status", "publish_state", "created_at", "updated_at",
    "scheduled_date", "scheduled_at", "published_at", "preview", "share_url", "private_url",
    "x_published_url", "linkedin_published_url", "mastodon_published_url",
    "threads_published_url", "bluesky_published_url", "substack_published_url",
    "x_article_published_url", "x_post_published_at", "linkedin_post_published_at",
    "mastodon_post_published_at", "threads_post_published_at", "bluesky_post_published_at",
    "substack_post_published_at", "x_article_published_at",
}
UNSAFE_KEY = re.compile(
    r"(^|_)(publish|published|schedule|scheduled|plan|planned|share|shared|sharing|"
    r"send|sent|queue|queued|auto_retweet|auto_promote|auto_plug|post_now|confirm_publish)(_|$)"
)
REMOTE_STATE_KEYS = re.compile(
    r"(^|_)(publish|published|schedule|scheduled|plan|planned|share|shared|sharing|"
    r"send|sent|queue|queued)(_|$)"
)


class GuardInputError(ValueError):
    """The input cannot establish a safe draft operation."""


class UnsafeDesiredField(GuardInputError):
    """Desired content contains an external-action field."""


def _normalized_key(key: str) -> str:
    key = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", key)
    return re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def fingerprint(value: Any) -> str:
    """Return a stable fingerprint that preserves exact JSON values and types."""
    return "sha256:" + hashlib.sha256(_json_bytes(value)).hexdigest()


def _unsafe_desired_path(value: Any, path: str = "$") -> str | None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                return f"{path}.<non-string-key>"
            norm = _normalized_key(key)
            if norm in {"status", "state"} or (norm not in SAFE_ACTION_LIKE_KEYS and UNSAFE_KEY.search(norm)):
                return f"{path}.{key}"
            found = _unsafe_desired_path(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _unsafe_desired_path(child, f"{path}[{index}]")
            if found:
                return found
    return None


def _active_remote_state(value: Any, path: str = "$") -> str | None:
    """Find a nonempty scheduling/publication/share field in a readback."""
    if isinstance(value, dict):
        for key, child in value.items():
            norm = _normalized_key(str(key))
            if norm not in SAFE_ACTION_LIKE_KEYS and REMOTE_STATE_KEYS.search(norm) and child not in (None, False, "", [], {}):
                return f"{path}.{key}"
            found = _active_remote_state(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found = _active_remote_state(child, f"{path}[{index}]")
            if found:
                return found
    return None


def _validate_target(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"social_set_id"}:
        raise GuardInputError(f"{label} must contain exactly social_set_id")
    social_set_id = value["social_set_id"]
    if isinstance(social_set_id, bool) or not isinstance(social_set_id, (str, int)):
        raise GuardInputError(f"{label}.social_set_id must be a nonempty string or integer")
    if not str(social_set_id).strip():
        raise GuardInputError(f"{label}.social_set_id must be nonempty")
    return copy.deepcopy(value)


def _normalize_settings(name: str, value: Any, label: str) -> dict[str, Any]:
    defaults = X_SETTINGS_DEFAULTS if name == "x" else LINKEDIN_SETTINGS_DEFAULTS
    if value is None:
        return copy.deepcopy(defaults)
    if not isinstance(value, dict):
        raise GuardInputError(f"{label} must be an object, null, or omitted")
    unknown = set(value) - set(defaults)
    if unknown:
        raise GuardInputError(f"{label} has unsupported substantive keys: {sorted(unknown)}")
    out = copy.deepcopy(defaults)
    out.update(value)
    for key, setting in out.items():
        if key == "share_with_followers" and setting is not None and not isinstance(setting, bool):
            raise GuardInputError(f"{label}.{key} must be boolean or null")
        if key == "community_id" and setting is not None and (
            isinstance(setting, bool) or not isinstance(setting, (str, int))
        ):
            raise GuardInputError(f"{label}.{key} must be a string, integer, or null")
        if key in {"reply_to_url", "first_comment"} and setting is not None and not isinstance(setting, str):
            raise GuardInputError(f"{label}.{key} must be a string or null")
    return out


def _normalize_post(name: str, value: Any, label: str) -> dict[str, Any]:
    defaults = X_POST_DEFAULTS if name == "x" else LINKEDIN_POST_DEFAULTS
    if not isinstance(value, dict):
        raise GuardInputError(f"{label} must be an object")
    unknown = set(value) - ({"text"} | set(defaults))
    if unknown:
        raise GuardInputError(f"{label} has unsupported substantive keys: {sorted(unknown)}")
    if not isinstance(value.get("text"), str):
        raise GuardInputError(f"{label}.text must be a string")
    out = {"text": value["text"], **copy.deepcopy(defaults)}
    out.update(value)
    media = out["media_ids"]
    if not isinstance(media, list) or any(
        isinstance(item, bool) or not isinstance(item, (str, int)) or not str(item).strip()
        for item in media
    ):
        raise GuardInputError(f"{label}.media_ids must be a list of nonempty IDs")
    for key in {"subscribers_only", "paid_partnership", "made_with_ai", "hide_link_preview"} & set(out):
        if not isinstance(out[key], bool):
            raise GuardInputError(f"{label}.{key} must be boolean")
    for key in {"quote_post_url", "linkedin_reshare_urn"} & set(out):
        if out[key] is not None and not isinstance(out[key], str):
            raise GuardInputError(f"{label}.{key} must be a string or null")
    return out


def _normalize_platform(name: str, value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GuardInputError(f"{label} must be an object")
    unknown = set(value) - PLATFORM_KEYS
    if unknown:
        raise GuardInputError(f"{label} has unsupported substantive keys: {sorted(unknown)}")
    if not isinstance(value.get("enabled"), bool):
        raise GuardInputError(f"{label}.enabled must be boolean")
    raw_posts = value.get("posts", [])
    if not isinstance(raw_posts, list):
        raise GuardInputError(f"{label}.posts must be a list")
    posts = [_normalize_post(name, post, f"{label}.posts[{i}]") for i, post in enumerate(raw_posts)]
    settings = _normalize_settings(name, value.get("settings"), f"{label}.settings")
    if value["enabled"] and not posts:
        raise GuardInputError(f"{label}.posts must be nonempty when enabled")
    if not value["enabled"] and (posts or any(v is not None for v in settings.values())):
        raise GuardInputError(f"{label} is disabled but retains substantive posts or settings")
    return {"enabled": value["enabled"], "posts": posts, "settings": settings}


def _normalize_platforms(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GuardInputError(f"{label} must be an object")
    allowed = set(PLATFORMS) | KNOWN_DISABLED_PLATFORMS | {"x_article"}
    unknown = set(value) - allowed
    if unknown:
        raise GuardInputError(f"{label} has unsupported substantive platforms: {sorted(unknown)}")
    for name in KNOWN_DISABLED_PLATFORMS:
        if name in value and value[name] != {"enabled": False}:
            raise GuardInputError(f"{label}.{name} is not the known disabled API default")
    if "x_article" in value and value["x_article"] is not None:
        raise GuardInputError(f"{label}.x_article is not the known null API default")
    out = {
        name: _normalize_platform(name, value.get(name, {"enabled": False}), f"{label}.{name}")
        for name in PLATFORMS
    }
    if not any(config["enabled"] for config in out.values()):
        raise GuardInputError(f"{label} must enable x, linkedin, or both")
    return out


def _validate_payload(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GuardInputError(f"{label} must be an object")
    unsafe = _unsafe_desired_path(value, label)
    if unsafe:
        raise UnsafeDesiredField(f"unsafe external-action field: {unsafe}")
    unknown = set(value) - PAYLOAD_KEYS
    if unknown:
        raise GuardInputError(f"{label} has unsupported keys: {sorted(unknown)}")
    if not isinstance(value.get("draft_title"), str) or not value["draft_title"].strip():
        raise GuardInputError(f"{label}.draft_title must be a nonempty string")
    if "scratchpad_text" in value and not isinstance(value["scratchpad_text"], str):
        raise GuardInputError(f"{label}.scratchpad_text must be a string")
    tags = value.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise GuardInputError(f"{label}.tags must be a list of strings")
    return {
        "draft_title": value["draft_title"], "tags": copy.deepcopy(tags),
        "scratchpad_text": value.get("scratchpad_text", ""),
        "platforms": _normalize_platforms(value.get("platforms"), f"{label}.platforms"),
    }


def _request_payload(value: dict[str, Any]) -> dict[str, Any]:
    """Remove only known response defaults while retaining substantive values."""
    out = {key: copy.deepcopy(value[key]) for key in ("draft_title", "tags", "scratchpad_text")}
    out["platforms"] = {}
    for name in PLATFORMS:
        config = value["platforms"][name]
        if not config["enabled"]:
            out["platforms"][name] = {"enabled": False}
            continue
        defaults = X_POST_DEFAULTS if name == "x" else LINKEDIN_POST_DEFAULTS
        posts = []
        for post in config["posts"]:
            clean = {"text": post["text"], "media_ids": copy.deepcopy(post["media_ids"])}
            for key, default in defaults.items():
                if key != "media_ids" and post[key] != default:
                    clean[key] = copy.deepcopy(post[key])
            posts.append(clean)
        clean_config: dict[str, Any] = {"enabled": True, "posts": posts}
        settings = {key: copy.deepcopy(v) for key, v in config["settings"].items() if v is not None}
        if settings:
            clean_config["settings"] = settings
        out["platforms"][name] = clean_config
    return out


def _list_items(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("results")
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise GuardInputError("list_readback must be a list or an object with a results list")
    return value


def _item_title(item: dict[str, Any]) -> Any:
    if "draft_title" in item:
        return item["draft_title"]
    if isinstance(item.get("payload"), dict):
        return item["payload"].get("draft_title")
    return None


def _draft_id(item: dict[str, Any]) -> Any:
    return item.get("id", item.get("draft_id"))


def _usable_draft_id(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (str, int)) and bool(str(value).strip())


def _detail_payload(detail: dict[str, Any]) -> dict[str, Any]:
    unknown = set(detail) - (DETAIL_RESPONSE_KEYS | PAYLOAD_KEYS | {"payload"})
    if unknown:
        raise GuardInputError(f"detail_readback has unsupported substantive keys: {sorted(unknown)}")
    if isinstance(detail.get("payload"), dict):
        projected = dict(detail["payload"])
        label = "detail_readback.payload"
    else:
        projected = {key: detail[key] for key in PAYLOAD_KEYS if key in detail}
        label = "detail_readback"
    # New plain drafts return null for the empty scratchpad. Normalize only
    # this response default. Explicit desired scratchpad values stay strings.
    if projected.get("scratchpad_text", "") is None:
        projected["scratchpad_text"] = ""
    return _validate_payload(projected, label)


def readback_fingerprint(detail: dict[str, Any], social_set_id: str | int) -> str:
    """Fingerprint one detail readback with the guard's exact normalization.

    Package receipts use this to bind their retained final readback to the same
    text, media, platform-setting, and target comparison used by ``decide``.
    """
    target = _validate_target({"social_set_id": social_set_id}, "readback target")
    payload = _detail_payload(detail)
    return fingerprint({"target": target, "payload": payload})


def _remote_updated_at(item: dict[str, Any], detail: dict[str, Any]) -> Any:
    return detail.get("updated_at", item.get("updated_at"))


def _refuse(reason_code: str, message: str, **facts: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": "REFUSE",
        "reason_code": reason_code,
        "message": message,
        "suggested_request": None,
        "receipt": None,
    }
    out.update(facts)
    return out


def _receipt(
    draft_id: Any,
    target: dict[str, Any],
    content_fingerprint: str,
    desired_fingerprint: str,
    updated_at: Any,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "draft_id": draft_id,
        "target_fingerprint": fingerprint(target),
        "remote_fingerprint": content_fingerprint,
        "desired_fingerprint": desired_fingerprint,
        "readback_updated_at": updated_at,
    }


def _decide(envelope: dict[str, Any]) -> dict[str, Any]:
    desired = envelope.get("desired")
    if not isinstance(desired, dict) or set(desired) != {"target", "payload"}:
        raise GuardInputError("desired must contain exactly target and payload")
    target = _validate_target(desired["target"], "desired.target")
    payload = _validate_payload(desired["payload"], "desired.payload")
    readback_target = _validate_target(envelope.get("readback_target"), "readback_target")
    if readback_target != target:
        return _refuse(
            "target_mismatch",
            "The list/detail readback target does not exactly match the desired social set.",
            desired_target=target,
            readback_target=readback_target,
        )

    title = payload["draft_title"]
    items = _list_items(envelope.get("list_readback"))
    matches = [item for item in items if _item_title(item) == title]
    desired_fp = fingerprint({"target": target, "payload": payload})
    if len(matches) > 1:
        return _refuse(
            "ambiguous_duplicate_titles",
            f"{len(matches)} drafts have the exact title {title!r}. No mutation is safe.",
            desired_fingerprint=desired_fp,
            matching_draft_ids=[_draft_id(item) for item in matches],
        )
    prior = envelope.get("prior_receipt")
    if prior is not None:
        if (
            not isinstance(prior, dict)
            or prior.get("schema_version") != SCHEMA_VERSION
            or not _usable_draft_id(prior.get("draft_id"))
        ):
            return _refuse("invalid_prior_receipt", "The prior receipt needs a supported schema version and usable draft ID.")
        prior_id = prior["draft_id"]
        identity_matches = [
            item for item in items
            if _usable_draft_id(_draft_id(item)) and str(_draft_id(item)) == str(prior_id)
        ]
        if not identity_matches:
            return _refuse(
                "prior_draft_missing",
                "The fresh list readback does not contain the draft ID named by the prior receipt.",
                draft_id=prior_id,
                desired_fingerprint=desired_fp,
            )
        if len(identity_matches) > 1:
            return _refuse(
                "ambiguous_duplicate_draft_ids",
                "The fresh list readback contains the prior draft ID more than once.",
                draft_id=prior_id,
                desired_fingerprint=desired_fp,
            )
        item = identity_matches[0]
        if matches and matches[0] is not item:
            return _refuse(
                "rename_title_conflict",
                "A different draft already has the desired title. Preserve both drafts.",
                draft_id=prior_id,
                matching_draft_ids=[_draft_id(match) for match in matches],
                desired_fingerprint=desired_fp,
            )
        current_title = _item_title(item)
        current_title_matches = [candidate for candidate in items if _item_title(candidate) == current_title]
        if current_title is not None and len(current_title_matches) > 1:
            return _refuse(
                "ambiguous_duplicate_titles",
                f"{len(current_title_matches)} drafts have the exact current title {current_title!r}. No mutation is safe.",
                desired_fingerprint=desired_fp,
                matching_draft_ids=[_draft_id(candidate) for candidate in current_title_matches],
            )
    elif not matches:
        request = {"operation": "CREATE", "target": target, "payload": _request_payload(payload)}
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "CREATE",
            "reason_code": "no_matching_title",
            "message": "No matching title exists. Review the suggested request, create with the installed skill, then read back and rerun this guard.",
            "desired_fingerprint": desired_fp,
            "current_fingerprint": None,
            "suggested_request": request,
            "receipt": None,
        }
    else:
        item = matches[0]

    draft_id = _draft_id(item)
    if not _usable_draft_id(draft_id):
        return _refuse("missing_draft_id", "The matching list item has no usable draft ID.")
    detail = envelope.get("detail_readback")
    if not isinstance(detail, dict):
        return _refuse(
            "missing_detail_readback",
            "A fresh detail readback is required before deciding about an existing draft.",
            draft_id=draft_id,
            desired_fingerprint=desired_fp,
        )
    detail_id = _draft_id(detail)
    if not _usable_draft_id(detail_id) or str(detail_id) != str(draft_id):
        return _refuse(
            "detail_id_mismatch",
            "The detail readback is not for the matching list draft.",
            draft_id=draft_id,
            detail_draft_id=detail_id,
        )
    for source, obj in (("list", item), ("detail", detail)):
        if "social_set_id" in obj and str(obj["social_set_id"]) != str(target["social_set_id"]):
            return _refuse(
                "readback_social_set_mismatch",
                f"The {source} readback belongs to a different social set.",
                draft_id=draft_id,
            )

    for source, obj in (("list", item), ("detail", detail)):
        status = obj.get("status")
        if not isinstance(status, str) or status.lower() != "draft":
            return _refuse(
                "non_draft_state",
                f"The {source} readback state is {status!r}, not an unscheduled draft.",
                draft_id=draft_id,
                remote_status=status,
            )
        active = _active_remote_state(obj)
        if active:
            return _refuse(
                "planned_or_published_state",
                f"The {source} readback has an active scheduling/publication/share field at {active}.",
                draft_id=draft_id,
                remote_status=status,
            )

    try:
        remote_payload = _detail_payload(detail)
    except UnsafeDesiredField as exc:
        return _refuse("unsafe_remote_payload", str(exc), draft_id=draft_id)
    current_fp = fingerprint({"target": readback_target, "payload": remote_payload})
    updated_at = _remote_updated_at(item, detail)
    if current_fp == desired_fp:
        return {
            "schema_version": SCHEMA_VERSION,
            "decision": "UNCHANGED",
            "reason_code": "exact_readback_match",
            "message": "The fresh detail readback exactly matches text, media, platform settings, and target. Save this receipt. No mutation is needed.",
            "draft_id": draft_id,
            "desired_fingerprint": desired_fp,
            "current_fingerprint": current_fp,
            "suggested_request": None,
            "receipt": _receipt(draft_id, target, current_fp, desired_fp, updated_at),
        }

    if not isinstance(prior, dict):
        return _refuse(
            "no_prior_receipt",
            "The existing draft differs from desired content and has no prior verified receipt. Preserve it as human-owned.",
            draft_id=draft_id,
            desired_fingerprint=desired_fp,
            current_fingerprint=current_fp,
        )
    if prior.get("schema_version") != SCHEMA_VERSION:
        return _refuse("invalid_prior_receipt", "The prior receipt schema version is missing or unsupported.", draft_id=draft_id)
    if str(prior.get("draft_id")) != str(draft_id):
        return _refuse("receipt_draft_mismatch", "The prior receipt names a different draft.", draft_id=draft_id)
    if prior.get("target_fingerprint") != fingerprint(target):
        return _refuse("receipt_target_mismatch", "The prior receipt belongs to a different target.", draft_id=draft_id)
    if updated_at in (None, ""):
        return _refuse(
            "missing_remote_update_timestamp",
            "An update needs the remote updated_at value so a human edit cannot be missed.",
            draft_id=draft_id,
        )
    if prior.get("remote_fingerprint") != current_fp:
        return _refuse(
            "remote_changed_since_receipt",
            "The fresh readback no longer matches the last verified remote fingerprint. Preserve the human edit.",
            draft_id=draft_id,
            desired_fingerprint=desired_fp,
            current_fingerprint=current_fp,
        )
    if prior.get("readback_updated_at") != updated_at:
        return _refuse(
            "remote_updated_since_receipt",
            "The draft update timestamp changed after the saved readback. Preserve it even though comparable content matches.",
            draft_id=draft_id,
            desired_fingerprint=desired_fp,
            current_fingerprint=current_fp,
        )

    request = {
        "operation": "UPDATE", "target": target, "draft_id": draft_id,
        "payload": _request_payload(payload),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": "UPDATE",
        "reason_code": "verified_remote_unchanged",
        "message": "Remote content still matches the prior verified receipt. Review the suggested request, update with the installed skill, then read back and rerun this guard.",
        "draft_id": draft_id,
        "desired_fingerprint": desired_fp,
        "current_fingerprint": current_fp,
        "suggested_request": request,
        "receipt": None,
    }


def decide(envelope: Any) -> dict[str, Any]:
    """Return a fail-closed decision without performing external work."""
    if not isinstance(envelope, dict):
        return _refuse("invalid_input", "The input envelope must be a JSON object.")
    try:
        return _decide(envelope)
    except UnsafeDesiredField as exc:
        return _refuse("unsafe_desired_field", str(exc))
    except (GuardInputError, TypeError, ValueError) as exc:
        return _refuse("invalid_input", str(exc))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Decide CREATE/UPDATE/UNCHANGED/REFUSE for one Typefully draft from local desired "
            "JSON and fresh list/detail readbacks. No network or account mutation is performed."
        ),
        epilog=(
            "Use the installed Typefully skill for an approved suggested_request, fetch fresh "
            "list/detail readbacks, rerun this guard, and save only an UNCHANGED receipt. "
            "REFUSE exits 3. Malformed JSON exits 2."
        ),
    )
    parser.add_argument("--input", default="-", metavar="FILE", help="JSON envelope path, or - for stdin (default)")
    parser.add_argument("--compact", action="store_true", help="emit one-line JSON")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        raw = sys.stdin.read() if args.input == "-" else Path(args.input).read_text(encoding="utf-8")
        envelope = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        result = _refuse("malformed_json", str(exc))
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":") if args.compact else None, indent=None if args.compact else 2))
        return 2
    result = decide(envelope)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":") if args.compact else None, indent=None if args.compact else 2))
    return 3 if result["decision"] == "REFUSE" else 0


if __name__ == "__main__":
    raise SystemExit(main())
