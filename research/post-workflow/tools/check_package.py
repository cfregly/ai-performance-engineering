#!/usr/bin/env python3
"""Fail-closed artifact gate for a post package.

This checks retained files, hashes, review/readback receipts, and declared state.
It does not judge visual quality or prove that a GPU experiment was qualified.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import struct
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

if __package__:
    from .check_profile_conventions import validate_profile_conventions
    from .check_social_preview import validate_social_preview
    from .check_public_projection import validate_public_projection
else:
    from check_profile_conventions import validate_profile_conventions
    from check_social_preview import validate_social_preview
    from check_public_projection import validate_public_projection


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
CONCEPTS = ["images/a", "images/b", "images/c"]
COVER = "images/fun"
REVIEW_PATH = "images/review.json"
TYPEFULLY_PATH = "social/typefully-receipt.json"
MEASUREMENT_RECEIPT_PATH = "evidence/measurement-receipt.json"
SOURCE_REPORTED_RECEIPT_PATH = "evidence/source-reported-receipt.json"
SOCIAL_FILES = {"x": "social/x.txt", "linkedin": "social/linkedin.txt"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
REQUIRED_FILES = [
    "README.md",
    "tweets.md",
    "sources.json",
    "claims-allow.txt",
    "VERIFICATION.md",
    "images/README.md",
    "images/audiences.md",
]
PUBLIC_COPY_ROOT_FILES = ("README.md", "tweets.md")
PUBLIC_COPY_PATTERNS = (
    (re.compile(r"\bwith\s+False\b", re.I), "contextless comparison 'with False'"),
    (re.compile(r"\bwith\s+True\b", re.I), "contextless comparison 'with True'"),
    (
        re.compile(r"\bRaw\s+metrics\s+and\s+units\s+are\s+retained\.", re.I),
        "boilerplate 'Raw metrics and units are retained.'",
    ),
)


def without_fenced_code_blocks(text: str) -> str:
    """Remove Markdown fenced code so code literals are not treated as prose."""
    prose: list[str] = []
    fence_character: str | None = None
    fence_length = 0
    for line in text.splitlines(keepends=True):
        candidate = line.rstrip("\r\n")
        if fence_character is None:
            opening = re.match(r"^[ \t]*(`{3,}|~{3,})", candidate)
            if opening:
                marker = opening.group(1)
                fence_character = marker[0]
                fence_length = len(marker)
            else:
                prose.append(line)
            continue
        closing = re.match(
            rf"^[ \t]*{re.escape(fence_character)}{{{fence_length},}}[ \t]*$",
            candidate,
        )
        if closing:
            fence_character = None
            fence_length = 0
    return "".join(prose)


def validate_public_copy(package: Path, stems: list[str], errors: list[str]) -> None:
    """Lint only user-facing prose, never evidence, source, or provenance files."""
    relatives = [
        *PUBLIC_COPY_ROOT_FILES,
        *SOCIAL_FILES.values(),
        *(f"{stem}.txt" for stem in stems),
        "image.txt",
    ]
    for relative in dict.fromkeys(relatives):
        path = package / relative
        if not path.is_file():
            continue
        try:
            prose = without_fenced_code_blocks(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError) as exc:
            errors.append(f"invalid public copy {relative}: {exc}")
            continue
        for pattern, label in PUBLIC_COPY_PATTERNS:
            if pattern.search(prose):
                errors.append(f"public copy contains disallowed {label}: {relative}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_json(path: Path, errors: list[str], label: str) -> Any | None:
    if not path.is_file():
        errors.append(f"missing {label}: {path}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"invalid {label} {path}: {exc}")
        return None


def png_dimensions(path: Path) -> tuple[int, int] | None:
    """Read PNG dimensions from the signature and IHDR using only stdlib."""
    try:
        with path.open("rb") as stream:
            header = stream.read(24)
    except OSError:
        return None
    if len(header) != 24 or header[:8] != PNG_SIGNATURE:
        return None
    length, chunk = struct.unpack(">I4s", header[8:16])
    if length != 13 or chunk != b"IHDR":
        return None
    return struct.unpack(">II", header[16:24])


def safe_relative(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = Path(value)
    return not path.is_absolute() and ".." not in path.parts


def valid_https_url(value: Any) -> bool:
    if not isinstance(value, str) or not value or any(character.isspace() for character in value):
        return False
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError:
        return False
    return parsed.scheme == "https" and bool(hostname)


def has_named_author(source: dict[str, Any]) -> bool:
    for field in ("author", "authors"):
        value = source.get(field)
        if isinstance(value, str) and value.strip():
            return True
        if isinstance(value, list) and value and all(isinstance(item, str) and item.strip() for item in value):
            return True
    return False


def require_file(package: Path, relative: str, errors: list[str], *, nonempty: bool = True) -> Path:
    path = package / relative
    if not path.is_file():
        errors.append(f"missing required file: {relative}")
    elif nonempty and path.stat().st_size == 0:
        errors.append(f"required file is empty: {relative}")
    return path


def validate_hashed_file(
    package: Path,
    value: Any,
    label: str,
    errors: list[str],
    *,
    prefix: str | None = None,
) -> str | None:
    """Validate a package-contained file reference and return its relative path."""
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object with path and sha256")
        return None
    relative = value.get("path")
    if not safe_relative(relative):
        errors.append(f"{label}.path must be a safe package-relative path")
        return None
    if prefix is not None and not str(relative).startswith(prefix):
        errors.append(f"{label}.path must be under {prefix}")
    target = package / relative
    try:
        target.resolve().relative_to(package.resolve())
    except ValueError:
        errors.append(f"{label}.path escapes the package: {relative}")
        return None
    if not target.is_file():
        errors.append(f"missing {label} file: {relative}")
        return relative
    if target.stat().st_size == 0:
        errors.append(f"{label} file is empty: {relative}")
    expected = value.get("sha256")
    if not isinstance(expected, str) or not SHA256_RE.fullmatch(expected):
        errors.append(f"{label}.sha256 must be a lowercase SHA-256 hex digest")
    elif sha256(target) != expected:
        errors.append(f"{label}.sha256 does not match {relative}")
    return relative


def validate_manifest(data: Any, errors: list[str]) -> tuple[list[str], str | None, str | None]:
    if not isinstance(data, dict):
        errors.append("manifest must be a JSON object")
        return [], None, None
    if data.get("schema_version") != 1:
        errors.append("manifest.schema_version must equal 1")
    if data.get("content_state") not in {"hypothetical", "measured", "source_reported"}:
        errors.append("manifest.content_state must be 'hypothetical', 'measured', or 'source_reported'")
    if data.get("concepts") != CONCEPTS:
        errors.append(f"manifest.concepts must equal {CONCEPTS!r}")
    if data.get("cover_stem") != COVER:
        errors.append(f"manifest.cover_stem must equal {COVER!r}")
    if data.get("review_receipt") != REVIEW_PATH:
        errors.append(f"manifest.review_receipt must equal {REVIEW_PATH!r}")
    if data.get("typefully_receipt") != TYPEFULLY_PATH:
        errors.append(f"manifest.typefully_receipt must equal {TYPEFULLY_PATH!r}")
    state = data.get("content_state")
    if state == "measured" and data.get("measurement_receipt") != MEASUREMENT_RECEIPT_PATH:
        errors.append(f"measured manifest.measurement_receipt must equal {MEASUREMENT_RECEIPT_PATH!r}")
    if state == "source_reported" and data.get("source_reported_receipt") != SOURCE_REPORTED_RECEIPT_PATH:
        errors.append(f"source-reported manifest.source_reported_receipt must equal {SOURCE_REPORTED_RECEIPT_PATH!r}")
    if state == "source_reported" and "measurement_receipt" in data:
        errors.append("source-reported manifest must not declare measurement_receipt")
    if state != "source_reported" and "source_reported_receipt" in data:
        errors.append("only a source-reported manifest may declare source_reported_receipt")
    if state != "measured" and state != "source_reported" and "measurement_receipt" in data:
        errors.append("only a measured manifest may declare measurement_receipt")

    audiences = data.get("audiences")
    audience_stems: list[str] = []
    if not isinstance(audiences, list) or len(audiences) != 6:
        errors.append("manifest.audiences must contain exactly 6 entries")
    else:
        keys: set[str] = set()
        readers: set[str] = set()
        for index, item in enumerate(audiences):
            label = f"manifest.audiences[{index}]"
            if not isinstance(item, dict):
                errors.append(f"{label} must be an object with key, reader, and stem")
                continue
            missing = {"key", "reader", "stem"} - set(item)
            if missing:
                errors.append(f"{label} missing fields: {', '.join(sorted(missing))}")
            key, reader, stem = item.get("key"), item.get("reader"), item.get("stem")
            if not isinstance(key, str) or not key.strip():
                errors.append(f"{label}.key must be a nonempty string")
            elif key in keys:
                errors.append(f"duplicate audience key: {key}")
            else:
                keys.add(key)
            if not isinstance(reader, str) or not reader.strip():
                errors.append(f"{label}.reader must be a nonempty string")
            elif reader in readers:
                errors.append(f"duplicate audience reader: {reader}")
            else:
                readers.add(reader)
            if not safe_relative(stem) or not str(stem).startswith("images/aud-"):
                errors.append(f"{label}.stem must be a safe images/aud-* path without an extension")
            else:
                audience_stems.append(stem)
        if len(set(audience_stems)) != len(audience_stems):
            errors.append("audience stems must be unique")

    selected = data.get("selected_image_stem")
    allowed = [*CONCEPTS, COVER, *audience_stems]
    if selected not in allowed:
        errors.append("manifest.selected_image_stem must name one declared concept, cover, or audience stem")
        selected = None
    return audience_stems, selected, data.get("content_state")


def validate_variant(package: Path, stem: str, errors: list[str]) -> str | None:
    expected = {
        f"{stem}.png": (1600, 900),
        f"{stem}-square.png": (1080, 1080),
        f"{stem}-400.png": (400, 225),
    }
    main_hash: str | None = None
    for relative, dimensions in expected.items():
        path = require_file(package, relative, errors)
        if path.is_file() and path.stat().st_size:
            actual = png_dimensions(path)
            if actual is None:
                errors.append(f"invalid PNG signature or IHDR: {relative}")
            elif actual != dimensions:
                errors.append(f"wrong dimensions for {relative}: expected {dimensions[0]}x{dimensions[1]}, got {actual[0]}x{actual[1]}")
            if relative == f"{stem}.png":
                main_hash = sha256(path)
    require_file(package, f"{stem}.txt", errors)
    return main_hash


def validate_review(package: Path, stems: list[str], hashes: dict[str, str], errors: list[str]) -> None:
    data = load_json(package / REVIEW_PATH, errors, "independent review receipt")
    if data is None:
        return
    if not isinstance(data, dict):
        errors.append("independent review receipt must be a JSON object")
        return
    if not isinstance(data.get("reviewer"), str) or not data["reviewer"].strip():
        errors.append("review receipt reviewer must be a nonempty string")
    if data.get("independent") is not True:
        errors.append("review receipt independent must be true")
    if data.get("verdict") != "PASS":
        errors.append("review receipt verdict must be 'PASS'")
    images = data.get("images")
    if not isinstance(images, dict):
        errors.append("review receipt images must be an object covering every declared stem")
        return
    for stem in stems:
        entry = images.get(stem)
        if not isinstance(entry, dict):
            errors.append(f"review receipt missing image verdict: {stem}")
            continue
        if entry.get("verdict") != "PASS":
            errors.append(f"review verdict for {stem} must be 'PASS'")
        expected = hashes.get(stem)
        if expected and entry.get("sha256") != expected:
            errors.append(f"stale review hash for {stem}: expected {expected}")


def validate_typefully(package: Path, errors: list[str]) -> None:
    data = load_json(package / TYPEFULLY_PATH, errors, "Typefully receipt")
    if data is None:
        return
    if not isinstance(data, dict):
        errors.append("Typefully receipt must be a JSON object")
        return
    if data.get("status") != "draft":
        errors.append("Typefully receipt status must be 'draft'")
    if data.get("scheduled_at") is not None:
        errors.append("Typefully receipt scheduled_at must be null")
    if data.get("platforms") != ["x", "linkedin"]:
        errors.append("Typefully receipt platforms must equal ['x', 'linkedin']")
    if data.get("content_files") != SOCIAL_FILES:
        errors.append(f"Typefully receipt content_files must equal {SOCIAL_FILES!r}")
    if data.get("readback_verified") is not True:
        errors.append("Typefully receipt readback_verified must be true")
    draft_id = data.get("draft_id")
    if isinstance(draft_id, bool) or not isinstance(draft_id, (str, int)) or str(draft_id).strip() == "":
        errors.append("Typefully receipt draft_id must be a nonempty string or integer")
    editor_url = data.get("editor_url")
    if not isinstance(editor_url, str) or not re.match(r"^https://", editor_url):
        errors.append("Typefully receipt editor_url must be a nonempty https URL")

    shipping = package / "image.png"
    if shipping.is_file() and data.get("image_sha256") != sha256(shipping):
        errors.append("Typefully receipt image_sha256 does not match current image.png")
    texts = data.get("texts")
    if not isinstance(texts, dict):
        errors.append("Typefully receipt texts must contain exact x and linkedin draft text")
        texts = {}
    for platform, relative in SOCIAL_FILES.items():
        path = require_file(package, relative, errors)
        if path.is_file():
            exact = path.read_text(encoding="utf-8")
            if texts.get(platform) != exact:
                errors.append(f"Typefully receipt texts.{platform} does not exactly match {relative}")


def validate_measurement(package: Path, errors: list[str]) -> None:
    data = load_json(package / MEASUREMENT_RECEIPT_PATH, errors, "measurement receipt")
    if data is None:
        return
    if not isinstance(data, dict):
        errors.append("measurement receipt must be a JSON object")
        return
    if data.get("schema_version") != 1:
        errors.append("measurement receipt schema_version must equal 1")
    if data.get("state") != "measured":
        errors.append("measurement receipt state must be 'measured'")
    if not isinstance(data.get("workload_scope"), str) or not data["workload_scope"].strip():
        errors.append("measurement receipt workload_scope must be a nonempty string")
    for field in ("runtime_identifier", "hardware_identifier"):
        if not isinstance(data.get(field), str) or not data[field].strip():
            errors.append(f"measurement receipt {field} must be a nonempty string")

    source_artifacts = data.get("source_artifacts")
    if not isinstance(source_artifacts, list) or not source_artifacts:
        errors.append("measurement receipt source_artifacts must be a nonempty list")
    else:
        source_paths: set[str] = set()
        for index, artifact in enumerate(source_artifacts):
            label = f"measurement receipt source_artifacts[{index}]"
            relative = validate_hashed_file(package, artifact, label, errors)
            if relative in source_paths:
                errors.append(f"measurement receipt source_artifacts paths must be distinct: {relative}")
            elif relative:
                source_paths.add(relative)

    accepted_repeats = data.get("accepted_repeats")
    repeat_ids: set[str] = set()
    repeat_paths: set[str] = set()
    if not isinstance(accepted_repeats, list) or len(accepted_repeats) < 2:
        errors.append("measurement receipt accepted_repeats must contain at least 2 entries")
    else:
        for index, repeat in enumerate(accepted_repeats):
            label = f"measurement receipt accepted_repeats[{index}]"
            if not isinstance(repeat, dict):
                errors.append(f"{label} must be an object")
                continue
            repeat_id = repeat.get("id")
            if not isinstance(repeat_id, str) or not repeat_id.strip():
                errors.append(f"{label}.id must be a nonempty string")
            elif repeat_id in repeat_ids:
                errors.append(f"measurement receipt accepted repeat IDs must be distinct: {repeat_id}")
            else:
                repeat_ids.add(repeat_id)
            if repeat.get("accepted") is not True:
                errors.append(f"{label}.accepted must be true")
            relative = validate_hashed_file(
                package,
                repeat.get("raw_result"),
                f"{label}.raw_result",
                errors,
                prefix="evidence/",
            )
            if relative in repeat_paths:
                errors.append(f"measurement receipt accepted repeat raw-result paths must be distinct: {relative}")
            elif relative:
                repeat_paths.add(relative)

    failed_attempts = data.get("failed_attempts")
    failed_ids: set[str] = set()
    failed_paths: set[str] = set()
    if not isinstance(failed_attempts, list):
        errors.append("measurement receipt failed_attempts must be a list, which may be empty")
    else:
        for index, attempt in enumerate(failed_attempts):
            label = f"measurement receipt failed_attempts[{index}]"
            if not isinstance(attempt, dict):
                errors.append(f"{label} must be an object")
                continue
            attempt_id = attempt.get("id")
            if not isinstance(attempt_id, str) or not attempt_id.strip():
                errors.append(f"{label}.id must be a nonempty string")
            elif attempt_id in failed_ids or attempt_id in repeat_ids:
                errors.append(f"measurement receipt attempt IDs must be distinct across accepted and failed attempts: {attempt_id}")
            else:
                failed_ids.add(attempt_id)
            if attempt.get("accepted") is not False:
                errors.append(f"{label}.accepted must be false")
            if not isinstance(attempt.get("failure_reason"), str) or not attempt["failure_reason"].strip():
                errors.append(f"{label}.failure_reason must be a nonempty string")
            relative = validate_hashed_file(
                package,
                attempt.get("raw_result"),
                f"{label}.raw_result",
                errors,
                prefix="evidence/",
            )
            if relative in failed_paths or relative in repeat_paths:
                errors.append(f"measurement receipt raw-result paths must be distinct across accepted and failed attempts: {relative}")
            elif relative:
                failed_paths.add(relative)

    for field in ("guard_receipts", "profile_evidence"):
        entries = data.get(field)
        if not isinstance(entries, list) or not entries:
            errors.append(f"measurement receipt {field} must be a nonempty list")
            continue
        names: set[str] = set()
        for index, entry in enumerate(entries):
            label = f"measurement receipt {field}[{index}]"
            if not isinstance(entry, dict):
                errors.append(f"{label} must be an object")
                continue
            name = entry.get("name")
            if not isinstance(name, str) or not name.strip():
                errors.append(f"{label}.name must be a nonempty string")
            elif name in names:
                errors.append(f"measurement receipt {field} names must be distinct: {name}")
            else:
                names.add(name)
            if entry.get("accepted") is not True:
                errors.append(f"{label}.accepted must be true")
            validate_hashed_file(package, entry, label, errors, prefix="evidence/")

    correctness = data.get("correctness")
    if not isinstance(correctness, dict):
        errors.append("measurement receipt correctness must be an object")
    else:
        if correctness.get("accepted") is not True:
            errors.append("measurement receipt correctness.accepted must be true")
        validate_hashed_file(package, correctness, "measurement receipt correctness", errors, prefix="evidence/")

    expected_power = data.get("expected_power_limit_w")
    actual_power = data.get("power_limit_w")
    expected_valid = (
        not isinstance(expected_power, bool)
        and isinstance(expected_power, (int, float))
        and math.isfinite(expected_power)
        and expected_power > 0
    )
    actual_valid = (
        not isinstance(actual_power, bool)
        and isinstance(actual_power, (int, float))
        and math.isfinite(actual_power)
        and actual_power > 0
    )
    if not expected_valid:
        errors.append("measurement receipt expected_power_limit_w must be a positive finite number")
    if not actual_valid:
        errors.append("measurement receipt power_limit_w must be a positive finite number")
    elif expected_valid and actual_power != expected_power:
        errors.append("measurement receipt power_limit_w must equal declared expected_power_limit_w")

    claims = data.get("claims")
    if not isinstance(claims, list) or not claims:
        errors.append("measurement receipt claims must be a nonempty list")
    else:
        claim_ids: set[str] = set()
        for index, claim in enumerate(claims):
            label = f"measurement receipt claims[{index}]"
            if not isinstance(claim, dict):
                errors.append(f"{label} must be an object")
                continue
            claim_id = claim.get("id")
            if not isinstance(claim_id, str) or not claim_id.strip():
                errors.append(f"{label}.id must be a nonempty string")
            elif claim_id in claim_ids:
                errors.append(f"measurement receipt claim IDs must be distinct: {claim_id}")
            else:
                claim_ids.add(claim_id)
            validate_hashed_file(package, claim.get("source"), f"{label}.source", errors)
            linked = claim.get("accepted_repeat_ids")
            if not isinstance(linked, list) or len(linked) < 2 or any(not isinstance(item, str) for item in linked):
                errors.append(f"{label}.accepted_repeat_ids must contain at least 2 repeat IDs")
            elif len(set(linked)) != len(linked):
                errors.append(f"{label}.accepted_repeat_ids must be distinct")
            else:
                unknown = set(linked) - repeat_ids
                if unknown:
                    errors.append(f"{label}.accepted_repeat_ids references unknown repeats: {', '.join(sorted(unknown))}")


def validate_source_reported(package: Path, errors: list[str]) -> None:
    """Keep attributed primary-source gains separate from local measurements."""
    data = load_json(package / SOURCE_REPORTED_RECEIPT_PATH, errors, "source-reported receipt")
    if not isinstance(data, dict):
        if data is not None:
            errors.append("source-reported receipt must be an object")
        return
    if data.get("schema_version") != 1 or data.get("state") != "source_reported":
        errors.append("source-reported receipt must declare schema_version 1 and state 'source_reported'")
    if data.get("locally_measured_performance_claims") is not False:
        errors.append("source-reported receipt must explicitly exclude locally measured performance claims")
    if data.get("local_gpu_execution") != "not_run":
        errors.append("source-reported receipt requires local_gpu_execution 'not_run'. Executed packages use the measured contract")
    source_ref = data.get("sources")
    if not isinstance(source_ref, dict) or source_ref.get("path") != "sources.json":
        errors.append("source-reported receipt sources must bind sources.json")
    validate_hashed_file(package, source_ref, "source-reported receipt sources", errors)
    sources = load_json(package / "sources.json", errors, "source definitions")
    definitions = sources.get("sources") if isinstance(sources, dict) else None
    if not isinstance(definitions, list) or not definitions:
        errors.append("source definitions sources must be a nonempty list")
        definitions = []
    primary: dict[str, dict[str, Any]] = {}
    for index, item in enumerate(definitions):
        if not isinstance(item, dict):
            errors.append(f"source definitions sources[{index}] must be an object")
            continue
        source_id = item.get("id")
        if item.get("type") not in {"primary_paper", "primary_author_article", "official_results"}:
            continue
        if not isinstance(source_id, str) or not source_id.strip():
            errors.append(f"source definitions sources[{index}].id must be a nonempty string")
            continue
        if source_id in primary:
            errors.append(f"source definitions primary source IDs must be distinct: {source_id}")
            continue
        primary[source_id] = item
    results = data.get("results")
    if not isinstance(results, list) or not results:
        errors.append("source-reported receipt results must contain an attributed primary-source gain")
        results = []
    seen: set[str] = set()
    for index, result in enumerate(results):
        label = f"source-reported receipt results[{index}]"
        if not isinstance(result, dict):
            errors.append(f"{label} must be an object")
            continue
        for field in ("id", "task", "candidate", "baseline", "hardware", "metric", "quality_condition", "source_revision"):
            if not isinstance(result.get(field), str) or not result[field].strip():
                errors.append(f"{label}.{field} must be a nonempty string")
        identity = result.get("id")
        if isinstance(identity, str):
            if identity in seen:
                errors.append(f"{label}.id must be distinct")
            seen.add(identity)
        source_id = result.get("source_id")
        source = primary.get(source_id) if isinstance(source_id, str) else None
        if not isinstance(source_id, str) or not source_id.strip() or source is None:
            errors.append(f"{label}.source_id must identify a linked primary source")
        elif not valid_https_url(source.get("url")):
            errors.append(f"{label}.source_id primary source must have a valid HTTPS URL with a host")
        elif not has_named_author(source):
            errors.append(f"{label}.source_id primary source must name its author or authors")
        if result.get("attribution") != "authors":
            errors.append(f"{label}.attribution must equal 'authors'")
        gain = result.get("speedup")
        if isinstance(gain, bool) or not isinstance(gain, (int, float)) or not math.isfinite(gain) or gain <= 1:
            errors.append(f"{label}.speedup must be a finite gain greater than 1")
        validate_hashed_file(package, result.get("retained_result"), f"{label}.retained_result", errors, prefix="evidence/")
    review_ref = data.get("independent_review")
    relative = validate_hashed_file(package, review_ref, "source-reported independent review", errors, prefix="evidence/")
    if relative:
        review = load_json(package / relative, errors, "source-claim review")
        if not isinstance(review, dict) or review.get("independent") is not True or review.get("verdict") != "PASS":
            errors.append("source-claim review must independently PASS the attributed claims")
        elif not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
            errors.append("source-claim review reviewer must be a nonempty string")
        elif not (package / "sources.json").is_file() or review.get("sources_sha256") != sha256(package / "sources.json"):
            errors.append("source-claim review does not bind the current sources.json")
        elif review.get("results_sha256") != hashlib.sha256(
            json.dumps(results, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest():
            errors.append("source-claim review does not bind the current attributed results")


def validate_state(package: Path, state: str | None, selected: str | None, errors: list[str]) -> None:
    verification = require_file(package, "VERIFICATION.md", errors)
    if not verification.is_file():
        return
    text = verification.read_text(encoding="utf-8")
    match = re.search(r"(?ims)^##\s+GPU stage\s*$\n(.*?)(?=^##\s+|\Z)", text)
    if not match:
        errors.append("VERIFICATION.md must contain a nonempty '## GPU stage' section")
        gpu = ""
    else:
        gpu = match.group(1).strip()
        if not gpu:
            errors.append("VERIFICATION.md '## GPU stage' section must not be empty")
    if state == "hypothetical":
        if selected:
            sidecar = package / f"{selected}.txt"
            if sidecar.is_file() and not re.search(r"hypothetical|illustrative", sidecar.read_text(encoding="utf-8"), re.I):
                errors.append("hypothetical selected-image sidecar must say 'hypothetical' or 'illustrative'")
        if "hypothetical" not in text.lower():
            errors.append("hypothetical package VERIFICATION.md must state that status explicitly")
        nonmeasurement = re.search(
            r"no (?:gpu )?(?:workload|measurement|benchmark).{0,24}(?:launched|run|made|performed)?|not run|no measured",
            gpu,
            re.I | re.S,
        )
        if not nonmeasurement:
            errors.append("hypothetical package GPU stage must explicitly state that no GPU workload or measurement was run")
    elif state == "measured" and gpu and "measured" not in gpu.lower():
        errors.append("measured package GPU stage must explicitly describe itself as measured")
    elif state == "source_reported":
        if not re.search(r"source[-_ ]reported", text, re.I):
            errors.append("source-reported package verification must state that evidence class")
        if not re.search(r"no (?:\w+ )?GPU workload (?:was )?(?:launched|run)|not run", gpu, re.I):
            errors.append("source-reported package GPU stage must state that the local workload was not run")


def validate_package(package: Path) -> list[str]:
    package = package.resolve()
    errors: list[str] = []
    if not package.is_dir():
        return [f"package directory does not exist: {package}"]
    for relative in REQUIRED_FILES:
        require_file(package, relative, errors)
    load_json(package / "sources.json", errors, "sources JSON")

    manifest = load_json(package / "manifest.json", errors, "manifest")
    if manifest is None:
        return errors
    audience_stems, selected, state = validate_manifest(manifest, errors)
    if isinstance(manifest, dict) and manifest.get("review_status") in {"PENDING", "REVIEW_BLOCKED"}:
        errors.append("required technical review is " + manifest["review_status"] + "; private preview only")
    stems = [*CONCEPTS, COVER, *audience_stems]
    validate_public_copy(package, stems, errors)
    hashes: dict[str, str] = {}
    for stem in stems:
        value = validate_variant(package, stem, errors)
        if value:
            hashes[stem] = value
    by_hash: dict[str, list[str]] = {}
    for stem, digest in hashes.items():
        by_hash.setdefault(digest, []).append(stem)
    for duplicates in by_hash.values():
        if len(duplicates) > 1:
            errors.append(f"main variant PNGs must be distinct; identical hash: {', '.join(duplicates)}")

    if selected:
        for suffix in (".png", "-square.png", ".txt"):
            shipping = package / f"image{suffix}"
            chosen = package / f"{selected}{suffix}"
            require_file(package, f"image{suffix}", errors)
            if shipping.is_file() and chosen.is_file() and shipping.read_bytes() != chosen.read_bytes():
                errors.append(f"image{suffix} does not byte-match selected variant {selected}{suffix}")

    validate_review(package, stems, hashes, errors)
    validate_typefully(package, errors)
    if state == "measured":
        validate_measurement(package, errors)
    elif state == "source_reported":
        validate_source_reported(package, errors)
    if "claim_scope_review" in manifest:
        if manifest["claim_scope_review"] != "claim-scope-review.md":
            errors.append("manifest claim_scope_review must equal claim-scope-review.md")
        else:
            require_file(package, "claim-scope-review.md", errors)
    validate_state(package, state, selected, errors)
    errors.extend(validate_profile_conventions(package))
    errors.extend(validate_social_preview(package))
    errors.extend(validate_public_projection(package))
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python3 tools/check_package.py posts/<slug>", file=sys.stderr)
        return 2
    package = Path(argv[0])
    errors = validate_package(package)
    if errors:
        print(f"FAIL {package} ({len(errors)} problem{'s' if len(errors) != 1 else ''})", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"PASS {package}")
    print("Artifact/readback gate only: visual artistry and GPU qualification require their own review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
