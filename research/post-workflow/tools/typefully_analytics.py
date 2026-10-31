#!/usr/bin/env python3
"""Summarize a retained Typefully X CSV offline without account changes."""
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
from pathlib import Path
import re
from zoneinfo import ZoneInfo

TIMEZONE = "America/Los_Angeles"
COUNTS = ("retweet_count", "reply_count", "like_count", "quote_count", "impression_count",
          "user_profile_clicks", "bookmark_count", "url_link_clicks", "total_engagements")
FLAGS = ("is_thread_head", "is_thread_part", "is_note_tweet")
REQUIRED = {"tweet_id", "created_at", "text", "url", "engagement_rate", *COUNTS, *FLAGS}


def timestamp(value: str, label: str) -> datetime:
    try:
        if "T" not in value:
            raise ValueError
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if result.utcoffset() is None:
            raise ValueError
        return result
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an ISO timestamp with a timezone") from exc


def number(value: str | None, label: str, *, count: bool = True) -> int | float | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = Decimal(value)
        if not parsed.is_finite() or parsed < 0 or (count and parsed != parsed.to_integral_value()):
            raise ValueError
        if count:
            return int(parsed)
        converted = float(parsed)
        if converted == float("inf"):
            raise ValueError
        return converted
    except (InvalidOperation, OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite nonnegative {'integer count' if count else 'number'} or blank") from exc


def per_thousand(numerator: int | None, impressions: int | None) -> float | None:
    return numerator * 1000 / impressions if numerator is not None and impressions else None


def summarize(csv_path: Path, start: str, end: str, observed_at: str) -> dict:
    try:
        if any(re.fullmatch(r"\d{4}-\d{2}-\d{2}", day) is None for day in (start, end)):
            raise ValueError
        first, last = date.fromisoformat(start), date.fromisoformat(end)
        if last < first:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError("start and end must be YYYY-MM-DD dates in ascending order") from exc
    observed = timestamp(observed_at, "observed_at")
    if last > observed.astimezone(ZoneInfo(TIMEZONE)).date():
        raise ValueError("end cannot be after the observed_at date in America/Los_Angeles")
    source = csv_path.read_bytes()
    reader = csv.DictReader(io.StringIO(source.decode("utf-8-sig"), newline=""))
    missing = REQUIRED - set(reader.fieldnames or [])
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
    if len(reader.fieldnames) != len(set(reader.fieldnames)):
        raise ValueError("CSV contains duplicate column names")
    posts, seen = [], set()
    for row_number, row in enumerate(reader, 2):
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV row {row_number} has an incorrect number of fields")
        tweet_id = row["tweet_id"].strip()
        if not tweet_id or tweet_id in seen:
            raise ValueError(f"CSV row {row_number} has an empty or duplicate tweet_id: {tweet_id}")
        seen.add(tweet_id)
        created = timestamp(row["created_at"], f"row {row_number} created_at")
        if created > observed:
            raise ValueError(f"CSV row {row_number} was created after observed_at")
        counts = {field: number(row[field], f"row {row_number} {field}") for field in COUNTS}
        source_rate = number(row["engagement_rate"], f"row {row_number} engagement_rate", count=False)
        flags = {}
        for field in FLAGS:
            value = row[field].strip().lower()
            if value not in ("", "true", "false"):
                raise ValueError(f"row {row_number} {field} must be true, false, or blank")
            flags[field] = None if value == "" else value == "true"
        local_date = created.astimezone(ZoneInfo(TIMEZONE)).date()
        if first <= local_date <= last:
            impressions = counts["impression_count"]
            posts.append({
                "tweet_id": tweet_id, "created_at": row["created_at"], "creation_date_local": str(local_date),
                "text": row["text"], "url": row["url"], **flags, "counts": counts,
                "rate_denominator_impressions": impressions,
                "reposts_per_1000_impressions": per_thousand(counts["retweet_count"], impressions),
                "bookmarks_per_1000_impressions": per_thousand(counts["bookmark_count"], impressions),
                "source_engagement_rate": source_rate,
            })
    missing_counts = {field: sum(post["counts"][field] is None for post in posts) for field in COUNTS}
    totals = {field: (sum(post["counts"][field] for post in posts)
                      if posts and missing_counts[field] == 0 else None) for field in COUNTS}
    ranked = sorted((post for post in posts if post["reposts_per_1000_impressions"] is not None),
                    key=lambda post: (-post["reposts_per_1000_impressions"], post["tweet_id"]))
    return {
        "schema_version": 1, "platform": "x", "observed_at": observed_at,
        "source": {"path": str(csv_path.resolve()), "sha256": hashlib.sha256(source).hexdigest(), "rows": len(seen)},
        "window": {"start": start, "end": end, "inclusive": True, "timezone": TIMEZONE,
                   "basis": "Post creation dates. Metrics are a cumulative snapshot, not engagement newly earned in this window."},
        "unit": "individual posts, including thread parts", "sample_size": len(posts),
        "aggregate": {"totals": totals, "missing_counts": missing_counts,
                      "reposts_per_1000_impressions": per_thousand(totals["retweet_count"], totals["impression_count"]),
                      "bookmarks_per_1000_impressions": per_thousand(totals["bookmark_count"], totals["impression_count"])},
        "rankable_sample_size": len(ranked),
        "ranked_by_reposts_per_1000_impressions": [{
            key: post[key] for key in ("tweet_id", "url", "reposts_per_1000_impressions", "rate_denominator_impressions")
        } for post in ranked],
        "posts": posts,
        "cautions": [
            "Post ages and thread exposure differ. Rankings are descriptive. No minimum sample or winner is assumed.",
            "Missing values are null, not zero. Aggregate totals and rates are null when required values are missing.",
            "No tags, causality, or optimal posting times are inferred.",
            "This export provides no LinkedIn analytics.",
            "Typefully UI aggregates and CSV exports may differ.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--observed-at", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.csv.resolve() == args.output.resolve():
            raise ValueError("output must differ from the source CSV")
        result = summarize(args.csv, args.start, args.end, args.observed_at)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, UnicodeError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
