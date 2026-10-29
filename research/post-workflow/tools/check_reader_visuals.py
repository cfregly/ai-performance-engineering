#!/usr/bin/env python3
"""Bind a declared reader progression and selected chart to current review evidence.

Packages opt in with manifest.reader_visuals; absence preserves historical checks.
This checks files, hashes, excerpt order, saved draft text/media, and a named
review receipt. It cannot infer whether an example explains a mechanism, whether
pixels contain a scientific chart, or whether labels are readable. Those are
explicit judgments in the retained review, with platform screenshot evidence.

Run: python3 tools/check_reader_visuals.py PACKAGE
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

if __package__:
    from .check_social_preview import _platform_posts, validate_social_preview
else:
    from check_social_preview import _platform_posts, validate_social_preview


STAGES = ("example", "mechanism", "measured_comparison", "scope")
REVIEW_CHECKS = (
    "example_before_jargon", "example_explains_mechanism",
    "measured_comparison_visible", "scope_and_conditions_visible",
    "scientific_chart_selected", "phone_readable",
)
ANALYTICAL_STAGES = ("example", "mechanism", "analytical_comparison", "scope")
ANALYTICAL_REVIEW_CHECKS = (
    "example_before_jargon", "example_explains_mechanism",
    "analytical_comparison_visible", "analytical_values_match_derivation",
    "assumptions_and_limits_visible",
    "scope_and_conditions_visible", "no_measured_performance_implication",
    "scientific_chart_selected", "phone_readable",
)
DERIVATION_REVIEW_CHECKS = (
    "independent_recalculation", "arithmetic_and_units_correct",
    "assumptions_explicit", "claimed_scope_supported", "limitations_explicit",
    "no_measured_performance_implication", "no_serving_speedup_inference",
)
COPY_PATHS = {"article": "README.md", "x": "social/x.txt", "linkedin": "social/linkedin.txt"}


def declaration_sha256(value: dict[str, Any]) -> str:
    """Canonical digest used by the reviewer to bind the manifest declaration."""
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()


def _timestamp(value: Any) -> datetime | None:
    try:
        result = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return result if result.utcoffset() is not None else None
    except (AttributeError, TypeError, ValueError):
        return None


def validate_reader_visuals(package: Path) -> list[str]:
    """Validate an opt-in manifest contract without changing files or accounts."""
    package = Path(package).resolve()
    errors: list[str] = []

    def failure(message: str) -> None:
        errors.append(f"reader visuals: {message}")

    def local_path(relative: Any, label: str) -> Path | None:
        if (not isinstance(relative, str) or not relative.strip() or "\\" in relative
                or Path(relative).is_absolute() or ".." in Path(relative).parts):
            failure(f"{label} must name a package-relative file")
            return None
        path = package / relative
        try:
            path.resolve().relative_to(package)
            if any(parent.is_symlink() for parent in (path, *path.parents) if parent != package):
                failure(f"{label} must not use symlinks")
                return None
        except (ValueError, OSError, RuntimeError):
            failure(f"{label} escapes the package")
            return None
        return path

    def retained(spec: Any, label: str) -> bytes | None:
        if not isinstance(spec, dict):
            failure(f"{label} requires path and sha256")
            return None
        path = local_path(spec.get("path"), f"{label}.path")
        if path is None:
            return None
        try:
            data = path.read_bytes()
        except OSError:
            failure(f"{label}: missing retained file {spec.get('path')}")
            return None
        digest = spec.get("sha256")
        if (not data or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or hashlib.sha256(data).hexdigest() != digest):
            failure(f"{label}: sha256 mismatch or empty file {spec.get('path')}")
            return None
        return data

    def read_json(path: Path | None, label: str) -> Any:
        if path is None:
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            failure(f"{label}: cannot read JSON ({exc})")
            return None

    manifest = read_json(local_path("manifest.json", "manifest"), "manifest")
    if not isinstance(manifest, dict):
        failure("manifest must be an object")
        return errors
    if "reader_visuals" not in manifest:
        return errors
    declaration = manifest["reader_visuals"]
    if not isinstance(declaration, dict):
        failure("manifest.reader_visuals must be an object")
        return errors
    version = declaration.get("schema_version")
    if type(version) is not int or version not in (1, 2):
        failure("schema_version must equal 1 or 2")
    analytical = type(version) is int and version == 2
    stages = ANALYTICAL_STAGES if analytical else STAGES
    review_checks = ANALYTICAL_REVIEW_CHECKS if analytical else REVIEW_CHECKS
    if analytical:
        _analytical_contract(package, manifest, declaration, retained, failure)
    else:
        state = manifest.get("content_state")
        basis_path = {"measured": manifest.get("measurement_receipt"),
                      "source_reported": manifest.get("source_reported_receipt")}.get(
                          state if isinstance(state, str) else None)
        basis = declaration.get("measurement_basis")
        retained(basis, "measurement_basis")
        if not basis_path or not isinstance(basis, dict) or basis.get("path") != basis_path:
            failure("measurement_basis must bind the manifest's measured or source-reported receipt")
    chart = declaration.get("scientific_chart")
    chart_data = retained(chart, "scientific_chart")
    chart_text = retained(declaration.get("chart_text"), "chart_text")
    if chart_data is not None and not _image_bytes(chart_data):
        failure("scientific_chart must be a retained PNG, JPEG or WebP image")
    if chart_text is not None:
        try:
            chart_text.decode("utf-8")
        except UnicodeError:
            failure("chart_text must be retained UTF-8 text")

    copy = declaration.get("copy")
    if not isinstance(copy, dict):
        failure("copy must contain article, x and linkedin")
        copy = {}
    for surface, expected_path in COPY_PATHS.items():
        spec = copy.get(surface)
        data = retained(spec, f"copy.{surface}")
        if not isinstance(spec, dict):
            continue
        if spec.get("path") != expected_path:
            failure(f"copy.{surface}.path must equal {expected_path}")
        try:
            text = data.decode("utf-8") if data is not None else None
        except UnicodeError:
            failure(f"copy.{surface} must be UTF-8 text")
            text = None
        _sequence(spec.get("sequence"), text, f"copy.{surface}", failure, stages)

    preview_spec = declaration.get("preview_receipt")
    preview_bytes = retained(preview_spec, "preview_receipt")
    preview = None
    if preview_bytes is not None:
        try:
            preview = json.loads(preview_bytes)
        except (ValueError, UnicodeError):
            failure("preview_receipt must contain JSON")
        errors.extend(validate_social_preview(package, Path(preview_spec["path"])))
    if not isinstance(preview, dict):
        failure("preview_receipt must be an object")
        preview = {}
    platforms = preview.get("platforms")
    if not isinstance(platforms, dict):
        platforms = {}
    screenshots: dict[str, Any] = {}
    for platform in ("x", "linkedin"):
        spec = platforms.get(platform)
        if not isinstance(spec, dict):
            failure(f"preview_receipt requires {platform}")
            continue
        images = spec.get("images")
        if not isinstance(images, list) or not any(
            isinstance(image, dict) and isinstance(chart, dict)
            and image.get("source_path") == chart.get("path")
            and image.get("source_sha256") == chart.get("sha256") for image in images
        ):
            failure(f"{platform}: selected scientific chart is absent from rendered attachments")
        screenshot = spec.get("screenshot")
        screenshots[platform] = screenshot
        screenshot_data = retained(screenshot, f"{platform}.screenshot")
        if screenshot_data is not None and not _image_bytes(screenshot_data):
            failure(f"{platform}.screenshot must be a retained PNG, JPEG or WebP image")
        raw = retained(spec.get("raw_readback"), f"{platform}.raw_readback")
        if raw is None:
            continue
        try:
            readback = json.loads(raw)
        except (ValueError, UnicodeError):
            failure(f"{platform}.raw_readback must contain JSON")
            continue
        draft_id = spec.get("draft_id")
        saved = list(_platform_posts(readback, str(draft_id), platform))
        copy_spec = copy.get(platform)
        if not saved:
            failure(f"{platform}: no matching saved draft text")
        for _, posts in saved:
            actual = "\n\n---\n\n".join(post.get("text", "") for post in posts
                                        if isinstance(post.get("text"), str))
            if isinstance(copy_spec, dict):
                _sequence(copy_spec.get("sequence"), actual, f"{platform} saved draft", failure, stages)
                copy_data = retained(copy_spec, f"copy.{platform}")
                if copy_data is not None:
                    try:
                        if copy_data.decode("utf-8").strip() != actual.strip():
                            failure(f"{platform}: saved draft text differs from declared current copy")
                    except UnicodeError:
                        pass  # Reported by the copy check above.

    review = read_json(local_path(declaration.get("review_receipt"), "review_receipt"), "review_receipt")
    if analytical:
        review_ref = manifest.get("reader_visuals_review")
        retained(review_ref, "reader_visuals_review")
        if not isinstance(review_ref, dict) or review_ref.get("path") != declaration.get("review_receipt"):
            failure("reader_visuals_review must hash the declared reader review receipt")
    if not isinstance(review, dict):
        failure("review_receipt must be an object")
        return errors
    if type(review.get("schema_version")) is not int or review["schema_version"] != 1:
        failure("review_receipt.schema_version must equal 1")
    if review.get("status") != "PASS":
        failure("review_receipt.status must equal PASS")
    if not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
        failure("review_receipt requires a named reviewer")
    reviewed_at = _timestamp(review.get("reviewed_at"))
    preview_at = _timestamp(preview.get("checked_at"))
    if reviewed_at is None or reviewed_at > datetime.now(timezone.utc) + timedelta(minutes=5):
        failure("review_receipt.reviewed_at requires a completed ISO time with timezone")
    elif preview_at is not None and reviewed_at < preview_at:
        failure("review_receipt predates its platform preview")
    if review.get("declaration_sha256") != declaration_sha256(declaration):
        failure("review_receipt declaration_sha256 is stale or missing")
    checks = review.get("checks")
    for check in review_checks:
        value = checks.get(check) if isinstance(checks, dict) else None
        if not _review_finding(value):
            failure(f"review_receipt.checks.{check} requires PASS and specific notes")
    platform_reviews = review.get("platforms")
    for platform in ("x", "linkedin"):
        value = platform_reviews.get(platform) if isinstance(platform_reviews, dict) else None
        if not _review_finding(value):
            failure(f"review_receipt.platforms.{platform} requires PASS and specific visual notes")
        elif value.get("screenshot") != screenshots.get(platform) or not screenshots.get(platform):
            failure(f"review_receipt.platforms.{platform} must bind the current preview screenshot")
    return errors


def _analytical_contract(package: Path, manifest: dict[str, Any], declaration: dict[str, Any],
                         retained: Any, failure: Any) -> None:
    """Check a single explicitly authorized synthetic arithmetic exception.

    No code is executed here. Hashes establish custody; named reviewers must
    judge the assumptions, arithmetic and public claims in retained evidence.
    """
    def object_from(spec: Any, label: str) -> dict[str, Any]:
        data = retained(spec, label)
        value: Any = None
        if data is not None:
            try:
                value = json.loads(data)
            except (ValueError, UnicodeError):
                pass
        if not isinstance(value, dict):
            failure(f"{label} must contain a retained JSON object")
            return {}
        return value

    def required_text(value: Any, label: str) -> None:
        if not isinstance(value, str) or not value.strip():
            failure(f"{label} requires specific nonempty text")

    def completed_time(value: Any, label: str) -> datetime | None:
        result = _timestamp(value)
        if result is None or result > datetime.now(timezone.utc) + timedelta(minutes=5):
            failure(f"{label} requires a completed ISO time with timezone")
            return None
        return result

    def version_one(value: dict[str, Any], label: str) -> None:
        if type(value.get("schema_version")) is not int or value.get("schema_version") != 1:
            failure(f"{label}.schema_version must equal 1")

    if manifest.get("content_state") != "hypothetical":
        failure("schema 2 requires content_state hypothetical")
    if manifest.get("locally_measured_performance_claims") is not False:
        failure("schema 2 requires locally_measured_performance_claims false")
    if manifest.get("local_gpu_execution") != "not_run":
        failure("schema 2 requires local_gpu_execution not_run")
    for key in ("measurement_receipt", "source_reported_receipt"):
        if key in manifest:
            failure(f"schema 2 must not declare {key}")
    if "measurement_basis" in declaration:
        failure("schema 2 must not declare measurement_basis")

    authorization_ref = manifest.get("analytical_authorization")
    authorization = object_from(authorization_ref, "analytical_authorization")
    version_one(authorization, "analytical_authorization")
    for key, expected in (("authorization_kind", "explicit_user_instruction"),
                          ("permission", "single_analytical_package_without_measured_speedup"),
                          ("package_directory", package.name)):
        if authorization.get(key) != expected:
            failure(f"analytical_authorization.{key} must equal {expected!r}")
    required_text(authorization.get("authorized_by"), "analytical_authorization.authorized_by")
    authorized_at = completed_time(authorization.get("authorized_at"), "analytical_authorization.authorized_at")
    source = authorization.get("source")
    if not isinstance(source, dict):
        source = {}
    if source.get("kind") != "user_message":
        failure("analytical_authorization.source.kind must equal user_message")
    for key in ("reference", "excerpt"):
        required_text(source.get(key), f"analytical_authorization.source.{key}")

    basis_ref = declaration.get("analytical_basis")
    basis = object_from(basis_ref, "analytical_basis")
    version_one(basis, "analytical_basis")
    if basis.get("result_kind") != "synthetic_arithmetic":
        failure("analytical_basis.result_kind must equal synthetic_arithmetic")
    if basis.get("package_directory") != package.name:
        failure("analytical_basis.package_directory must bind this one package")
    for key in ("author", "claimed_scope"):
        required_text(basis.get(key), f"analytical_basis.{key}")
    if authorization.get("claimed_scope") != basis.get("claimed_scope"):
        failure("analytical_authorization.claimed_scope must bind the analytical basis")
    for key in ("assumptions", "limitations"):
        value = basis.get(key)
        if (not isinstance(value, list) or not value
                or any(not isinstance(item, str) or not item.strip() for item in value)
                or len(set(item for item in value if isinstance(item, str))) != len(value)):
            failure(f"analytical_basis.{key} requires a nonempty list of distinct specific statements")
    code_ref, results_ref = basis.get("code"), basis.get("results")
    code = retained(code_ref, "analytical_basis.code")
    if code is not None:
        try:
            code.decode("utf-8")
        except UnicodeError:
            failure("analytical_basis.code must be UTF-8 source")
    results = object_from(results_ref, "analytical_basis.results")
    version_one(results, "analytical_basis.results")
    for key in ("result_kind", "claimed_scope", "assumptions", "limitations", "code"):
        if results.get(key) != basis.get(key):
            failure(f"analytical_basis.results.{key} must bind the analytical basis")
    values = results.get("values")
    if not isinstance(values, list) or not values:
        failure("analytical_basis.results.values requires named numeric values and units")
    else:
        names = []
        for value in values:
            if not isinstance(value, dict):
                failure("analytical_basis.results.values requires named numeric values and units")
                continue
            for key in ("name", "unit"):
                required_text(value.get(key), f"analytical_basis.results.values.{key}")
            number = value.get("value")
            if type(number) not in (int, float) or (isinstance(number, float) and not math.isfinite(number)):
                failure("analytical_basis.results.values.value must be a finite number")
            names.append(value.get("name"))
        if len(set(name for name in names if isinstance(name, str))) != len(names):
            failure("analytical_basis.results.values names must be distinct")

    independent = object_from(declaration.get("independent_review"), "independent_review")
    version_one(independent, "independent_review")
    if independent.get("status") != "PASS":
        failure("independent_review.status must equal PASS")
    required_text(independent.get("reviewer"), "independent_review.reviewer")
    reviewer, author = independent.get("reviewer"), basis.get("author")
    if (isinstance(reviewer, str) and isinstance(author, str)
            and reviewer.strip().casefold() == author.strip().casefold()):
        failure("independent_review.reviewer must differ from the derivation author")
    reviewed_at = completed_time(independent.get("reviewed_at"), "independent_review.reviewed_at")
    if authorized_at is not None and reviewed_at is not None and reviewed_at < authorized_at:
        failure("independent_review predates the explicit analytical authorization")
    if independent.get("method") != "independent_recalculation":
        failure("independent_review.method must equal independent_recalculation")
    for key, expected in (("analytical_authorization", authorization_ref), ("analytical_basis", basis_ref),
                          ("code", code_ref), ("results", results_ref)):
        if independent.get(key) != expected or not isinstance(expected, dict):
            failure(f"independent_review.{key} must bind the retained current input")
    checks = independent.get("checks")
    for check in DERIVATION_REVIEW_CHECKS:
        if not _review_finding(checks.get(check) if isinstance(checks, dict) else None):
            failure(f"independent_review.checks.{check} requires PASS and specific notes")


def _sequence(value: Any, text: str | None, label: str, failure: Any,
              stages: tuple[str, ...] = STAGES) -> None:
    if not isinstance(value, list) or [item.get("stage") if isinstance(item, dict) else None
                                       for item in value] != list(stages):
        failure(f"{label}.sequence must declare {', '.join(stages)} in order")
        return
    cursor = 0
    for item in value:
        excerpt = item.get("excerpt")
        if not isinstance(excerpt, str) or not excerpt.strip():
            failure(f"{label}.{item['stage']} requires a verbatim excerpt")
            continue
        if text is None:
            continue
        position = text.find(excerpt, cursor)
        if position < 0:
            failure(f"{label}.{item['stage']} excerpt is absent or out of order")
        else:
            cursor = position + len(excerpt)


def _review_finding(value: Any) -> bool:
    return (isinstance(value, dict) and value.get("status") == "PASS"
            and isinstance(value.get("notes"), str) and bool(value["notes"].strip()))


def _image_bytes(data: bytes) -> bool:
    return (data.startswith(b"\x89PNG\r\n\x1a\n") or data.startswith(b"\xff\xd8\xff")
            or (data.startswith(b"RIFF") and data[8:12] == b"WEBP"))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: check_reader_visuals.py PACKAGE", file=sys.stderr)
        return 2
    errors = validate_reader_visuals(Path(argv[0]))
    for error in errors:
        print(f"FAIL: {error}")
    if not errors:
        print("PASS: reader visual declaration and review evidence (when declared)")
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
