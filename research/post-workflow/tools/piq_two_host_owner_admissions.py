#!/usr/bin/env python3
"""Validate the fixed source-reviewed trust anchor for owner handbacks."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

if __package__:
    from .check_piq_two_host_functional import SHA256_RE, _load_json
else:
    from check_piq_two_host_functional import SHA256_RE, _load_json


ADMISSIONS_PATH = (
    Path(__file__).resolve().parents[1] / "trust" / "piq-two-host-owner-admissions.json"
)
ADMISSIONS_KIND = "piq_two_host_owner_admissions"
UTC_RE = re.compile(
    r"^(?P<whole>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})(?:\.(?P<fraction>\d{1,9}))?Z$"
)
PORTABLE_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,254}$")
SPECIFIC_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{7,254}$")
GENERIC_IDENTIFIERS = {
    "admission",
    "authority",
    "controller",
    "custodian",
    "execution-owner",
    "execution_owner",
    "owner",
    "reviewer",
    "source",
    "system",
    "unknown",
}
ENTRY_FIELDS = {
    "admission_id",
    "admitted_at",
    "reviewer_id",
    "source",
    "owner_id",
    "authority_id",
    "producer_receipt_sha256",
    "envelope_sha256",
    "owner_handback_sha256",
    "run_id",
    "attempt_ids",
}
TUPLE_FIELDS = (
    "owner_id",
    "authority_id",
    "producer_receipt_sha256",
    "envelope_sha256",
    "owner_handback_sha256",
    "run_id",
    "attempt_ids",
)


def _utc_nanoseconds(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    match = UTC_RE.fullmatch(value)
    if match is None:
        return None
    try:
        whole = datetime.strptime(match.group("whole"), "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=timezone.utc
        )
    except ValueError:
        return None
    fraction = (match.group("fraction") or "").ljust(9, "0")
    return int(whole.timestamp()) * 1_000_000_000 + int(fraction or "0")


def _specific(value: Any) -> bool:
    return (
        isinstance(value, str)
        and SPECIFIC_IDENTIFIER_RE.fullmatch(value) is not None
        and value.casefold() not in GENERIC_IDENTIFIERS
    )


def validate_admissions(document: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(document, dict):
        return ["PIQ two-host owner admissions must be a JSON object"]
    if set(document) != {"schema_version", "kind", "entries"}:
        errors.append(
            "PIQ two-host owner admissions fields do not match schema version 1"
        )
    if (
        type(document.get("schema_version")) is not int
        or document.get("schema_version") != 1
    ):
        errors.append("PIQ two-host owner admissions schema_version must equal 1")
    if document.get("kind") != ADMISSIONS_KIND:
        errors.append("PIQ two-host owner admissions kind is unsupported")
    entries = document.get("entries")
    if not isinstance(entries, list):
        errors.append("PIQ two-host owner admissions entries must be a list")
        entries = []
    admission_ids: set[str] = set()
    admitted_tuples: set[str] = set()
    for index, value in enumerate(entries):
        label = f"PIQ two-host owner admissions entries[{index}]"
        if not isinstance(value, dict):
            errors.append(f"{label} must be a JSON object")
            continue
        if set(value) != ENTRY_FIELDS:
            errors.append(f"{label} fields do not match schema version 1")
        for field in ("admission_id", "reviewer_id", "owner_id", "authority_id"):
            if not _specific(value.get(field)):
                errors.append(
                    f"{label}.{field} must be a specific non-generic identifier"
                )
        owner_id = value.get("owner_id")
        authority_id = value.get("authority_id")
        if (
            isinstance(owner_id, str)
            and isinstance(authority_id, str)
            and owner_id.casefold() == authority_id.casefold()
        ):
            errors.append(
                f"{label} owner_id and authority_id must be distinct identities"
            )
        admission_id = value.get("admission_id")
        if isinstance(admission_id, str):
            admission_identity = admission_id.casefold()
            if admission_identity in admission_ids:
                errors.append("PIQ two-host owner admission IDs must be distinct")
            admission_ids.add(admission_identity)
        admitted_at = _utc_nanoseconds(value.get("admitted_at"))
        if admitted_at is None:
            errors.append(f"{label}.admitted_at must be a canonical UTC timestamp")
        reviewer_id = value.get("reviewer_id")
        authority_identities = {
            item.casefold()
            for item in (value.get("owner_id"), value.get("authority_id"))
            if isinstance(item, str)
        }
        if (
            isinstance(reviewer_id, str)
            and reviewer_id.casefold() in authority_identities
        ):
            errors.append(f"{label}.reviewer_id must be independent of owner authority")
        source = value.get("source")
        if not isinstance(source, dict) or set(source) != {"id", "sha256"}:
            errors.append(f"{label}.source must contain exactly id and sha256")
        else:
            if not _specific(source.get("id")):
                errors.append(
                    f"{label}.source.id must be a specific non-generic identifier"
                )
            if (
                not isinstance(source.get("sha256"), str)
                or SHA256_RE.fullmatch(source["sha256"]) is None
            ):
                errors.append(
                    f"{label}.source.sha256 must be a lowercase SHA-256 digest"
                )
        for field in (
            "producer_receipt_sha256",
            "envelope_sha256",
            "owner_handback_sha256",
        ):
            digest = value.get(field)
            if not isinstance(digest, str) or SHA256_RE.fullmatch(digest) is None:
                errors.append(f"{label}.{field} must be a lowercase SHA-256 digest")
        if (
            not isinstance(value.get("run_id"), str)
            or PORTABLE_IDENTIFIER_RE.fullmatch(value["run_id"]) is None
        ):
            errors.append(f"{label}.run_id must be a portable ASCII identifier")
        attempt_ids = value.get("attempt_ids")
        if (
            not isinstance(attempt_ids, list)
            or not attempt_ids
            or any(
                not isinstance(item, str)
                or PORTABLE_IDENTIFIER_RE.fullmatch(item) is None
                for item in attempt_ids
            )
            or len(attempt_ids)
            != len({item.casefold() for item in attempt_ids if isinstance(item, str)})
        ):
            errors.append(
                f"{label}.attempt_ids must be a casefold-distinct portable ASCII identifier list"
            )
            attempt_ids = []
        try:
            identity = json.dumps(
                {
                    field: attempt_ids if field == "attempt_ids" else value.get(field)
                    for field in TUPLE_FIELDS
                },
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError):
            errors.append(f"{label} admitted tuple must contain JSON values")
            continue
        if identity in admitted_tuples:
            errors.append("PIQ two-host owner admitted tuples must be distinct")
        admitted_tuples.add(identity)
    return errors


def load_fixed_admissions() -> tuple[Any, list[str]]:
    """Load only the repository-owned trust anchor. No override is supported."""
    errors: list[str] = []
    if ADMISSIONS_PATH.parent.is_symlink() or ADMISSIONS_PATH.is_symlink():
        return None, [
            "PIQ two-host owner admissions trust anchor must not use symlinks"
        ]
    if not ADMISSIONS_PATH.is_file():
        return None, [
            f"missing PIQ two-host owner admissions trust anchor: {ADMISSIONS_PATH}"
        ]
    document = _load_json(ADMISSIONS_PATH, "PIQ two-host owner admissions", errors)
    errors.extend(validate_admissions(document))
    return document, errors


def require_admitted_tuple(
    document: Any,
    expected: dict[str, Any],
    *,
    attested_at: Any,
) -> list[str]:
    """Require exactly one source-reviewed entry for the owner handback tuple."""
    errors = validate_admissions(document)
    if errors:
        return errors
    entries = document["entries"]
    matches = [
        entry
        for entry in entries
        if all(entry.get(field) == expected.get(field) for field in TUPLE_FIELDS)
    ]
    if len(matches) != 1:
        return [
            "PIQ two-host owner handback tuple is not admitted by the fixed source-reviewed trust anchor"
        ]
    admitted_at = _utc_nanoseconds(matches[0].get("admitted_at"))
    attested = _utc_nanoseconds(attested_at)
    if admitted_at is None or attested is None or admitted_at < attested:
        return ["PIQ two-host owner admission must occur at or after owner attestation"]
    return []
