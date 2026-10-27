"""Check retained scientific-image previews against saved draft readbacks.

Packages with zymtrace/views/profile-semantics.json require a schema_version=1
social/profile-preview.json receipt. The same applies to profiles/profile-semantics.json
or manifest.json with scientific_profiles=true. Each platform records raw_readback with a
package-relative path and sha256. Its JSON must retain the matching draft id and
platform posts with media_ids (or media objects with id fields). Every attached
media ID in those posts requires a reviewed image or document-page entry.
Related drafts use additional_drafts entries with platform, draft_id,
raw_readback and images. They pass the same checks as the primary platforms.
Each platform also requires attachment_sources, keyed by every attached media
ID, with path, sha256 and media_type=image|pdf. PDFs require page_count matching
the retained PDF, and reviews must cover every page from 1 through page_count.

PDF carousel pages may repeat one media_id with distinct positive page_index
values. Each page records document_source_path and document_sha256 for the same
retained PDF, plus source_path and source_sha256 for its reviewed PNG rendering.
Document pages also record safe_content_box=[x,y,width,height] in PNG pixels,
overlay_insets={top,bottom,left,right} in rendered CSS pixels, and a visual
controls_clear review. Scientific content must stay clear of those controls.

Geometry checks catch cropping and distortion. The visual_review flags record a
human or image-capable review of the actual rendered preview. Geometry alone
cannot establish label readability. checked_at records that review, not a TTL.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import struct
import subprocess


MAX_AREA_LOSS = 0.005


def _pdf_page_count(data):
    """Read actual PDF pages with installed Poppler or pypdf. Never guess counts."""
    if len(data) > 128 * 1024 * 1024:
        raise ValueError("PDF exceeds the 128 MiB preview-check limit")
    pdfinfo = shutil.which("pdfinfo")
    if pdfinfo:
        result = subprocess.run([pdfinfo, "-"], input=data, capture_output=True, timeout=15, check=False)
        match = re.search(rb"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
        if result.returncode or not match:
            raise ValueError("pdfinfo could not parse retained PDF pages")
        count = int(match.group(1))
    else:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ValueError("PDF page validation requires an existing pdfinfo or pypdf installation") from exc
        count = len(PdfReader(io.BytesIO(data)).pages)
    if not 1 <= count <= 1000:
        raise ValueError("PDF page count must be between 1 and 1000")
    return count


def _identifier(value):
    return str(value) if isinstance(value, (str, int)) and not isinstance(value, bool) and str(value).strip() else None


def _timestamp(value):
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.utcoffset() is not None else None
    except ValueError:
        return None


def _objects(value):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _objects(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _objects(nested)


def _media_ids(post):
    result = set()
    for item in _objects(post):
        for key in ("media_ids", "mediaIds", "media"):
            media = item.get(key)
            if not isinstance(media, list):
                continue
            for entry in media:
                value = entry.get("id", entry.get("media_id")) if isinstance(entry, dict) else entry
                identifier = _identifier(value)
                if identifier is not None:
                    result.add(identifier)
        if "media_id" in item:
            identifier = _identifier(item["media_id"])
            if identifier is not None:
                result.add(identifier)
    return result


def _platform_posts(readback, draft_id, platform):
    """Select posts under the saved matching draft, never unrelated draft media."""
    for candidate in _objects(readback):
        if not any(_identifier(candidate.get(key)) == draft_id for key in ("id", "draft_id")):
            continue
        platforms = candidate.get("platforms")
        section = platforms.get(platform) if isinstance(platforms, dict) else None
        if section is None and candidate.get("platform") == platform:
            section = candidate
        if isinstance(section, dict) and isinstance(section.get("posts"), list):
            yield candidate, [post for post in section["posts"] if isinstance(post, dict)]


def validate_social_preview(package: Path) -> list[str]:
    """Return actionable errors for retained rendered scientific-image previews."""
    package = Path(package).resolve()
    receipt_path = package / "social/profile-preview.json"
    manifest_path = package / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass  # The package checker reports invalid manifests.
    scientific_profiles = isinstance(manifest, dict) and manifest.get("scientific_profiles") is True
    if not receipt_path.exists() and not scientific_profiles and not any((package / relative).exists() for relative in ("zymtrace/views/profile-semantics.json", "profiles/profile-semantics.json")):
        return []
    errors: list[str] = []
    pdf_page_counts = {}

    def read_json(path, label):
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{label}: cannot read JSON ({exc})")
            return None

    def retained(relative, digest, label):
        if not isinstance(relative, str) or not relative.strip():
            errors.append(f"{label}: requires a retained source path")
            return None
        path = (package / relative).resolve()
        if Path(relative).is_absolute() or not path.is_relative_to(package):
            errors.append(f"{label}: path must stay within the package")
            return None
        try:
            data = path.read_bytes()
        except OSError:
            errors.append(f"{label}: missing retained file {relative}")
            return None
        if not data or hashlib.sha256(data).hexdigest() != digest:
            errors.append(f"{label}: sha256 mismatch or empty retained file {relative}")
            return None
        return data

    receipt = read_json(receipt_path, "social/profile-preview.json")
    if not isinstance(receipt, dict):
        return [*errors, "social/profile-preview.json must be an object"]
    if receipt.get("schema_version") != 1 or isinstance(receipt.get("schema_version"), bool):
        errors.append("profile preview schema_version must be 1")
    if receipt.get("verified") is not True:
        errors.append("profile preview verified must be true after rendered review")
    checked_at = _timestamp(receipt.get("checked_at"))
    if checked_at is None or checked_at > datetime.now(timezone.utc) + timedelta(minutes=5):
        errors.append("profile preview checked_at must be a completed ISO 8601 time with timezone")
    platforms = receipt.get("platforms")
    if not isinstance(platforms, dict):
        return [*errors, "profile preview requires x and linkedin platforms"]
    review_specs = [(platform, platforms.get(platform), f"profile preview {platform}") for platform in ("x", "linkedin")]
    additional_drafts = receipt.get("additional_drafts", [])
    if not isinstance(additional_drafts, list):
        errors.append("profile preview additional_drafts must be a list")
    else:
        for index, spec in enumerate(additional_drafts):
            label = f"profile preview additional draft {index + 1}"
            if not isinstance(spec, dict) or spec.get("platform") not in ("x", "linkedin"):
                errors.append(f"{label}: platform must be x or linkedin")
                continue
            review_specs.append((spec["platform"], spec, label))
    for platform, spec, label in review_specs:
        if not isinstance(spec, dict):
            errors.append(f"{label}: requires platform receipt")
            continue
        draft_id = _identifier(spec.get("draft_id"))
        if draft_id is None:
            errors.append(f"{label}: requires draft_id")
        raw = spec.get("raw_readback")
        readback = None
        if not isinstance(raw, dict):
            errors.append(f"{label}: requires raw_readback path and sha256")
        else:
            raw_bytes = retained(raw.get("path"), raw.get("sha256"), f"{label} raw_readback")
            if raw_bytes is not None:
                try:
                    readback = json.loads(raw_bytes)
                except (ValueError, UnicodeError):
                    errors.append(f"{label}: raw_readback must be saved JSON")
        saved_drafts = list(_platform_posts(readback, draft_id, platform)) if draft_id else []
        if not saved_drafts:
            errors.append(f"{label}: saved readback does not contain matching draft_id and platform posts")
        posts = []
        for draft, draft_posts in saved_drafts:
            updated = _timestamp(draft.get("updated_at"))
            if updated and checked_at and updated > checked_at:
                errors.append(f"{label}: visual review is stale relative to draft updated_at")
            posts.extend(draft_posts)
        attached_media = set().union(*(_media_ids(post) for post in posts))
        sources = spec.get("attachment_sources")
        if not isinstance(sources, dict):
            errors.append(f"{label}: attachment_sources must map every attached media_id to retained source metadata")
            sources = {}
        for media_id in sorted(attached_media):
            source_label = f"{label} attachment {media_id}"
            source = sources.get(media_id)
            if not isinstance(source, dict):
                errors.append(f"{source_label}: requires attachment_sources path, sha256 and media_type")
                continue
            data = retained(source.get("path"), source.get("sha256"), source_label)
            media_type = source.get("media_type")
            if media_type == "image":
                if data is not None and not data.startswith(b"\x89PNG\r\n\x1a\n"):
                    errors.append(f"{source_label}: image attachment must have PNG signature, never PDF")
            elif media_type == "pdf":
                if type(source.get("page_count")) is not int or not 1 <= source["page_count"] <= 1000:
                    errors.append(f"{source_label}: PDF page_count must be an integer from 1 through 1000")
                if data is not None:
                    if not data.startswith(b"%PDF"):
                        errors.append(f"{source_label}: PDF attachment must have %PDF signature")
                    else:
                        digest = source.get("sha256")
                        if digest not in pdf_page_counts:
                            try:
                                pdf_page_counts[digest] = _pdf_page_count(data)
                            except Exception as exc:
                                errors.append(f"{source_label}: cannot validate actual PDF page count ({exc})")
                                pdf_page_counts[digest] = None
                        actual_count = pdf_page_counts[digest]
                        if actual_count is not None and source.get("page_count") != actual_count:
                            errors.append(f"{source_label}: declared page_count differs from actual retained PDF ({actual_count})")
            else:
                errors.append(f"{source_label}: media_type must be image or pdf")
        images = spec.get("images")
        if not isinstance(images, list) or not images:
            errors.append(f"{label}: requires nonempty reviewed images")
            continue
        seen = {}
        for index, image in enumerate(images):
            image_label = f"{label} image {index + 1}"
            if not isinstance(image, dict):
                errors.append(f"{image_label}: must be an object")
                continue
            media_id = _identifier(image.get("media_id"))
            if media_id is None:
                errors.append(f"{image_label}: requires media_id")
            page_index = image.get("page_index")
            document_identity = None
            if any(key in image for key in ("page_index", "document_source_path", "document_sha256")):
                if type(page_index) is not int or page_index < 1:
                    errors.append(f"{image_label}: document page_index must be a positive integer")
                document_identity = (image.get("document_source_path"), image.get("document_sha256"))
                document = retained(*document_identity, f"{image_label} document")
                if document is not None and not document.startswith(b"%PDF"):
                    errors.append(f"{image_label}: retained document must have %PDF magic")
            source = sources.get(media_id)
            if isinstance(source, dict):
                source_identity = (source.get("path"), source.get("sha256"))
                if source.get("media_type") == "pdf":
                    if document_identity != source_identity:
                        errors.append(f"{image_label}: PDF attachment requires matching document_source_path and document_sha256")
                elif source.get("media_type") == "image":
                    if document_identity is not None or (image.get("source_path"), image.get("source_sha256")) != source_identity:
                        errors.append(f"{image_label}: image source must match its attachment_sources path and sha256")
            previous_pages = seen.setdefault(media_id, [])
            if previous_pages:
                if document_identity is None or type(page_index) is not int or page_index < 1 or any(previous_document != document_identity or type(previous_page) is not int or previous_page < 1 for previous_document, previous_page in previous_pages):
                    errors.append(f"{image_label}: repeated media_id requires distinct positive page_index values for the same retained document")
                elif any(previous_page == page_index for _, previous_page in previous_pages):
                    errors.append(f"{image_label}: duplicate page_index for document media_id")
            previous_pages.append((document_identity, page_index))
            matching_posts = [post for post in posts if media_id and media_id in _media_ids(post)]
            if not matching_posts:
                errors.append(f"{image_label}: media_id is missing or stale in saved draft platform posts")
            if type(image.get("preview_group_size")) is not int or image["preview_group_size"] != 1:
                errors.append(f"{image_label}: detailed scientific previews require preview_group_size=1")
            if any(len(_media_ids(post)) != 1 for post in matching_posts):
                errors.append(f"{image_label}: saved post groups multiple images into one preview")
            data = retained(image.get("source_path"), image.get("source_sha256"), image_label)
            if document_identity is not None and data is not None and not data.startswith(b"\x89PNG\r\n\x1a\n"):
                errors.append(f"{image_label}: document page source must be a retained PNG rendering")
            source_size = None
            dimensions = [image.get(key) for key in ("natural_width", "natural_height", "rendered_width", "rendered_height")]
            valid_dimensions = all(type(value) in (int, float) and math.isfinite(value) and value > 0 for value in dimensions)
            if not valid_dimensions:
                errors.append(f"{image_label}: natural and rendered dimensions must be positive finite numbers")
            else:
                natural_width, natural_height, rendered_width, rendered_height = dimensions
                natural_ratio = natural_width / natural_height
                rendered_ratio = rendered_width / rendered_height
                visible_fraction = min(natural_ratio / rendered_ratio, rendered_ratio / natural_ratio)
                if data and data.startswith(b"\x89PNG\r\n\x1a\n"):
                    if len(data) < 24 or data[12:16] != b"IHDR":
                        errors.append(f"{image_label}: invalid PNG IHDR")
                    else:
                        source_width, source_height = struct.unpack(">II", data[16:24])
                        if not source_width or not source_height:
                            errors.append(f"{image_label}: PNG dimensions must be positive")
                        else:
                            source_size = (source_width, source_height)
                            source_ratio = source_width / source_height
                            if min(source_ratio / natural_ratio, natural_ratio / source_ratio) < 1 - MAX_AREA_LOSS:
                                errors.append(f"{image_label}: natural aspect ratio differs from retained PNG")
                fit = image.get("object_fit")
                if fit == "cover" and visible_fraction < 1 - MAX_AREA_LOSS:
                    errors.append(f"{image_label}: cover crops {100 * (1 - visible_fraction):.2f}% of scientific image area")
                if fit == "fill" and visible_fraction < 1 - MAX_AREA_LOSS:
                    errors.append(f"{image_label}: fill distorts scientific image aspect ratio")
            if image.get("object_fit") not in ("cover", "contain", "fill"):
                errors.append(f"{image_label}: object_fit must be cover, contain or fill")
            if not isinstance(image.get("object_position"), str) or not image["object_position"].strip():
                errors.append(f"{image_label}: requires observed object_position")
            review = image.get("visual_review")
            if not isinstance(review, dict) or any(review.get(key) is not True for key in ("complete_profile", "headings_visible", "key_labels_readable")) or not isinstance(review.get("reviewer"), str) or not review["reviewer"].strip():
                errors.append(f"{image_label}: visual_review requires complete_profile, headings_visible and key_labels_readable=true plus reviewer")
            if document_identity is not None:
                if not isinstance(review, dict) or review.get("controls_clear") is not True:
                    errors.append(f"{image_label}: document visual_review.controls_clear must be true")
                box = image.get("safe_content_box")
                insets = image.get("overlay_insets")
                valid_box = isinstance(box, list) and len(box) == 4 and all(type(value) in (int, float) and math.isfinite(value) and value >= 0 for value in box) and box[2] > 0 and box[3] > 0
                valid_insets = isinstance(insets, dict) and all(type(insets.get(side)) in (int, float) and math.isfinite(insets[side]) and insets[side] >= 0 for side in ("top", "bottom", "left", "right"))
                if not valid_box:
                    errors.append(f"{image_label}: safe_content_box requires finite nonnegative x,y and positive width,height")
                if not valid_insets:
                    errors.append(f"{image_label}: overlay_insets requires finite nonnegative top,bottom,left,right CSS pixels")
                if valid_box and source_size:
                    x, y, width, height = box
                    source_width, source_height = source_size
                    if x + width > source_width or y + height > source_height:
                        errors.append(f"{image_label}: safe_content_box must stay within the source PNG")
                    elif valid_dimensions and valid_insets:
                        scale_x = rendered_width / source_width
                        scale_y = rendered_height / source_height
                        clearances = {"left": x * scale_x, "right": rendered_width - (x + width) * scale_x, "top": y * scale_y, "bottom": rendered_height - (y + height) * scale_y}
                        if any(clearances[side] + 1e-6 < insets[side] for side in clearances):
                            errors.append(f"{image_label}: safe scientific content intersects preview overlay controls")
        for missing_media in sorted(attached_media - set(seen)):
            errors.append(f"{label}: attached media_id {missing_media} is missing rendered image review")
        for media_id in sorted(attached_media):
            source = sources.get(media_id)
            if isinstance(source, dict) and source.get("media_type") == "pdf" and type(source.get("page_count")) is int and 1 <= source["page_count"] <= 1000:
                reviewed_pages = {page for _, page in seen.get(media_id, []) if type(page) is int and page > 0}
                if reviewed_pages != set(range(1, source["page_count"] + 1)):
                    errors.append(f"{label} attachment {media_id}: rendered review must cover every PDF page 1..{source['page_count']}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    errors = validate_social_preview(args.package)
    for error in errors:
        print(error)
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
