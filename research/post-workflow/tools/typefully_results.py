#!/usr/bin/env python3
"""Offline publication/results ledger. No network access or account mutations.

Readbacks are raw Typefully detail objects (or a list), optionally wrapped as
{detail, observed_at, source, timestamp_provenance: {x/linkedin: explanation}}.
Snapshot input is a list of {platform, native_post_url, observed_at, metrics,
metrics_source, missing_reason/missing_reasons, auto_plug?}. All times need a
timezone. Milestones are targets. actual_age_hours is the observation's age.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from urllib.parse import parse_qs, urlsplit

PLATFORMS = ("x", "linkedin")
METRICS = ("impressions", "likes", "reposts", "replies", "bookmarks", "link_clicks", "profile_clicks")
MILESTONES = (("24h", 24), ("7d", 168))


def timestamp(value):
    if not isinstance(value, str) or "T" not in value:
        raise ValueError("timestamp must be ISO 8601 with a timezone")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(timezone.utc)


def iso(value):
    return value.isoformat().replace("+00:00", "Z")


def required_text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty text")
    return value


def identifier(value):
    if isinstance(value, bool) or not re.fullmatch(r"[1-9][0-9]*", str(value)):
        raise ValueError("draft_id and social_set_id must be positive integer IDs")
    return str(value)


def native_url(platform, value):
    """Validate the exact native host/path. Canonicalize aliases/tracking only."""
    parsed = urlsplit(required_text(value, "native post URL"))
    if parsed.scheme != "https" or parsed.username or parsed.password or parsed.port:
        raise ValueError("native post URL must use HTTPS without credentials or a port")
    if platform == "x":
        valid = parsed.hostname in {"x.com", "www.x.com", "twitter.com", "www.twitter.com"}
        path = re.fullmatch(r"/[A-Za-z0-9_]{1,15}/status/[1-9][0-9]*/?", parsed.path)
        host = "x.com"
    elif platform == "linkedin":
        valid = parsed.hostname in {"linkedin.com", "www.linkedin.com"}
        path = re.fullmatch(r"/(?:feed/update/urn:li:(?:activity|share|ugcPost):[1-9][0-9]*|posts/[A-Za-z0-9_-]+_activity-[1-9][0-9]*-[A-Za-z0-9_-]+)/?", parsed.path)
        host = "www.linkedin.com"
    else:
        raise ValueError("platform must be x or linkedin")
    if not valid or not path or any(c.isspace() for c in value):
        raise ValueError(f"invalid {platform} native post URL")
    path = parsed.path.rstrip("/")
    if platform == "x":
        handle, status, post_id = path.lstrip("/").split("/")
        path = f"/{handle.lower()}/{status}/{post_id}"
    return f"https://{host}{path}"


def registry_copy(registry):
    if not isinstance(registry, dict) or registry.get("schema_version", 1) != 1:
        raise ValueError("registry must be a schema_version 1 JSON object")
    if any(not isinstance(registry.get(field, []), list) for field in ("tracked_drafts", "publications")):
        raise ValueError("tracked_drafts and publications must be lists")
    return deepcopy(registry)


def track(registry, draft_id, social_set_id, platforms=PLATFORMS):
    """Return a copy, preserving manual fields and existing publication evidence."""
    result = registry_copy(registry)
    result.setdefault("schema_version", 1)
    draft_id, social_set_id = identifier(draft_id), identifier(social_set_id)
    if not platforms or set(platforms) - set(PLATFORMS):
        raise ValueError("tracked platforms must be x and/or linkedin")
    rows = result.setdefault("tracked_drafts", [])
    row = next((r for r in rows if str(r["draft_id"]) == draft_id and str(r["social_set_id"]) == social_set_id), None)
    if row is None:
        row = {"draft_id": draft_id, "social_set_id": social_set_id}
        rows.append(row)
    row["platforms"] = list(dict.fromkeys([*row.get("platforms", []), *platforms]))
    result.setdefault("publications", [])
    return result


def ingest(registry, readbacks, now):
    """Admit only published native URLs with evidence-backed publication times."""
    result = registry_copy(registry)
    current = timestamp(now)
    for envelope in readbacks if isinstance(readbacks, list) else [readbacks]:
        detail = envelope.get("detail", envelope)
        draft_id = identifier(detail.get("draft_id", detail.get("id")))
        if detail.get("id") is not None and identifier(detail["id"]) != draft_id:
            raise ValueError("readback id and draft_id disagree")
        social_set_id = identifier(detail.get("social_set_id"))
        row = next((r for r in result.get("tracked_drafts", []) if str(r["draft_id"]) == draft_id and str(r["social_set_id"]) == social_set_id), None)
        if row is None:
            raise ValueError(f"untracked draft/social set: {draft_id}/{social_set_id}")
        observed = timestamp(envelope.get("observed_at", now))
        if observed > current:
            raise ValueError("readback observation is in the future")
        if detail.get("status") != "published":
            continue
        for platform in row.get("platforms", PLATFORMS):
            url = detail.get(f"{platform}_published_url")
            if not url:
                continue
            url = native_url(platform, url)
            field = f"{platform}_post_published_at"
            published = detail.get(field)
            provenance = field
            if not published:
                provenance = envelope.get("timestamp_provenance", {}).get(platform)
                if not provenance:
                    continue
                required_text(provenance, "shared timestamp provenance")
                field, published = "published_at", detail.get("published_at")
            if not published:
                continue
            published_time = timestamp(published)
            if published_time > observed:
                raise ValueError("publication time is later than its readback")
            key = f"{platform}:{url}"
            rows = result.setdefault("publications", [])
            existing = next((p for p in rows if p["key"] == key), None)
            if existing:
                if timestamp(existing["published_at"]) != published_time:
                    raise ValueError("conflicting publication timestamp. Reconcile evidence manually")
                continue
            rows.append({"key": key, "platform": platform, "native_post_url": url,
                         "draft_id": draft_id, "social_set_id": social_set_id,
                         "published_at": iso(published_time), "timestamp_field": field,
                         "timestamp_provenance": provenance, "publication_evidence": {
                             "observed_at": iso(observed),
                             "source": envelope.get("source", "provided Typefully draft detail"),
                             "detail_sha256": hashlib.sha256(json.dumps(detail, sort_keys=True).encode()).hexdigest()},
                         "snapshots": []})
    return result


def plan(registry, now):
    """Return due work without changing the registry or backfilling old metrics."""
    registry_copy(registry)
    current, due = timestamp(now), []
    for publication in registry.get("publications", []):
        age = (current - timestamp(publication["published_at"])).total_seconds() / 3600
        completed = set()
        for snapshot in publication.get("snapshots", []):
            completed.update([snapshot["milestone"], *snapshot.get("missed_milestones", [])])
        pending = [(name, hours) for name, hours in MILESTONES if age >= hours and name not in completed]
        if pending:
            name, hours = pending[-1]
            due.append({"key": publication["key"], "platform": publication["platform"],
                        "native_post_url": publication["native_post_url"], "milestone": name,
                        "target_age_hours": hours, "actual_age_hours": round(age, 6),
                        "missed_milestones": [item[0] for item in pending[:-1]]})
    return due


def snapshot(registry, observations, now):
    """Capture one currently due milestone per post from explicitly observed data."""
    result = registry_copy(registry)
    current = timestamp(now)
    for observation in observations if isinstance(observations, list) else [observations]:
        observed = timestamp(observation.get("observed_at"))
        if observed > current:
            raise ValueError("metrics observation is in the future")
        platform = observation.get("platform")
        url = native_url(platform, observation.get("native_post_url"))
        key = f"{platform}:{url}"
        publication = next((p for p in result.get("publications", []) if p["key"] == key), None)
        if publication is None:
            raise ValueError("metrics observation has no admitted publication")
        if observed < timestamp(publication["published_at"]):
            raise ValueError("metrics observation predates publication")
        due = next((p for p in plan(result, iso(observed)) if p["key"] == key), None)
        if due is None:
            continue
        source = required_text(observation.get("metrics_source"), "metrics_source")
        values, reasons = {}, {}
        for metric in METRICS:
            value = observation.get("metrics", {}).get(metric)
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{metric} must be a nonnegative integer count or null")
            values[metric] = value
            if value is None:
                reasons[metric] = required_text(observation.get("missing_reasons", {}).get(metric, observation.get("missing_reason")), f"missing reason for {metric}")
        auto_plug = deepcopy(observation.get("auto_plug", {"state": "unknown", "reason": "Delivery not observed."}))
        if auto_plug.get("state") not in {"unknown", "not_configured", "configured", "pending", "delivered", "failed", "not_supported"}:
            raise ValueError("invalid AutoPlug delivery state")
        if auto_plug.get("reply_url"):
            reply = native_url(platform, auto_plug["reply_url"])
            comment = parse_qs(urlsplit(auto_plug["reply_url"]).query).get("commentUrn")
            if reply == url and not (platform == "linkedin" and comment):
                raise ValueError("AutoPlug reply_url identifies the original post, not a reply")
        if auto_plug.get("state") == "delivered" and not (auto_plug.get("reply_url") and auto_plug.get("evidence")):
            raise ValueError("delivered AutoPlug requires exact reply_url and evidence")
        responses = deepcopy(observation.get("reader_responses", {
            "state": "not_collected", "items": [],
            "reason": "Existing comment monitors own reader-response preparation. No response evidence supplied."}))
        if not isinstance(responses, dict) or responses.get("state") not in {"observed", "none_observed", "not_collected", "unavailable"} or not isinstance(responses.get("items"), list):
            raise ValueError("reader_responses requires a valid state and items list")
        if responses["state"] == "observed" and not responses["items"]:
            raise ValueError("observed reader_responses requires exact comment evidence")
        for item in responses["items"]:
            native_url(platform, item.get("comment_url"))
            required_text(item.get("summary"), "reader response summary")
            required_text(item.get("evidence"), "reader response evidence")
        publication.setdefault("snapshots", []).append({
            **due, "observed_at": iso(observed),
            "timing": "on_target" if due["actual_age_hours"] == due["target_age_hours"] else "late",
            "metrics": values, "metrics_source": source, "missing_reasons": reasons,
            "auto_plug": auto_plug, "reader_responses": responses})
    return result


def atomic_save(path, value):
    path = Path(path)
    if path.suffix != ".json" or path.is_symlink():
        raise ValueError("registry must be an explicit .json file, not a symlink")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("track", "ingest", "plan", "snapshot"))
    parser.add_argument("--registry", required=True, type=Path)
    parser.add_argument("--readbacks", type=Path, help="JSON details for ingest, metrics observations for snapshot")
    parser.add_argument("--now", default=iso(datetime.now(timezone.utc)))
    parser.add_argument("--draft-id")
    parser.add_argument("--social-set-id")
    parser.add_argument("--platform", action="append", choices=PLATFORMS)
    parser.add_argument("--dry-run", action="store_true", help="print proposed registry without writing")
    args = parser.parse_args(argv)
    try:
        timestamp(args.now)
        registry = json.loads(args.registry.read_text()) if args.registry.exists() else {}
        if args.command == "track":
            result = track(registry, args.draft_id, args.social_set_id, args.platform or PLATFORMS)
        elif args.command == "plan":
            result = plan(registry, args.now)
        else:
            if not args.readbacks:
                raise ValueError("--readbacks is required for ingest and snapshot")
            readbacks = json.loads(args.readbacks.read_text())
            result = (ingest if args.command == "ingest" else snapshot)(registry, readbacks, args.now)
        if args.command != "plan" and not args.dry_run:
            atomic_save(args.registry, result)
        print(json.dumps(result, indent=2, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
