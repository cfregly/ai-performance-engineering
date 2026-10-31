#!/usr/bin/env python3
"""Check an explicitly declared public projection without exporting any files.

Historical packages with no public_projection declaration are unchanged. The
declaration names a staged directory and maps every public file to its retained
package source. This checks that boundary, not scientific or publication quality.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit

if __package__:
    from .qualified_gpu_training import (
        build_gpu_public_summary, canonical_json, strict_load_json,
        validate_gpu_receipt,
    )
else:
    from qualified_gpu_training import (
        build_gpu_public_summary, canonical_json, strict_load_json,
        validate_gpu_receipt,
    )


PUBLIC_DIRECTORIES = {"images", "code", "captures", "evidence"}
PRIVATE_NAMES = {"_internal", "__internal"}
TEXT_SUFFIXES = {
    ".md", ".txt", ".rst", ".html", ".htm", ".json", ".jsonl", ".csv",
    ".tsv", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".py", ".sh",
    ".bash", ".zsh", ".c", ".h", ".cc", ".cpp", ".cu", ".cuh",
    ".js", ".ts", ".tsx", ".jsx", ".css", ".ipynb", ".log",
}
PRIVATE_TEXT = (
    (re.compile(r"(?<![A-Za-z0-9])_{1,2}internal(?![A-Za-z0-9])", re.I), "internal reference"),
    (re.compile(r"\btypefully\b", re.I), "Typefully reference"),
    (re.compile(
        r"(?<![\w.-])(?:VERIFICATION\.md|brief\.md|topic\.md|tweets\.md|"
        r"sources\.json|claims-allow\.txt|drafts\.json|analysis-plan\.md|"
        r"follow-up-answers\.md|profile-preview\.json)(?![\w.-])", re.I,
    ), "pipeline file reference"),
    (re.compile(
        r"^#{1,6}\s*(?:post this|cross-model review|independent review|review findings)\b|"
        r"/approve\b|\bSTATUS:\s*incomplete\b|\*\*Image options\*\*|"
        r"github\.com/[^/\s]+/research-drafts\b", re.I | re.M,
    ), "private review material"),
    (re.compile(r"(?:images|social)/review\.json\b", re.I), "private review receipt"),
)
MARKDOWN_IMAGE = re.compile(r"!\[[^\]]*\]\(\s*(?:<([^>]+)>|([^\s)]+))(?:\s+[^)]*)?\)")
HTML_IMAGE = re.compile(r"<img\b[^>]*\bsrc\s*=\s*([\"'])(.*?)\1", re.I | re.S)
REFERENCE_IMAGE = re.compile(r"!\[([^\]]*)\]\[([^\]]*)\]")
REFERENCE_DEFINITION = re.compile(r"^\s{0,3}\[([^\]]+)\]:\s*<?([^\s>]+)>?", re.M)
_PRIVATE_IDENTITY_SCAN_CHUNK_BYTES = 64 * 1024
_MAX_PRIVATE_IDENTITY_BYTES = 16 * 1024


def _duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant: {value}")


def _strict_json(path: Path) -> object:
    return json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_duplicates,
        parse_constant=_constant,
    )


def _gpu_v3_projection_errors(package: Path, manifest: dict,
                              sources: object) -> tuple[list[str], set[str]]:
    """Validate only the deterministic public summary and private redaction."""
    if (manifest.get("content_state") == "measured"
            and manifest.get("measurement_contract") != "qualified_gpu_training_v3"):
        return [
            "measured public projection uses an unsupported measurement contract. "
            "only qualified_gpu_training_v3 is active"
        ], set()
    if manifest.get("measurement_contract") != "qualified_gpu_training_v3":
        return [], set()
    errors: list[str] = []
    receipt_relative = manifest.get("measurement_receipt")
    if not _relative(receipt_relative):
        return ["GPU v3 public projection requires a safe measurement receipt path"], set()
    receipt_path = _member(package, receipt_relative, "GPU v3 measurement receipt", errors)
    if receipt_path is None or not receipt_path.is_file():
        return errors, set()
    try:
        receipt = strict_load_json(receipt_path, "GPU v3 measurement receipt")
    except ValueError as exc:
        return [f"GPU v3 public projection has invalid receipt: {exc}"], set()
    if not isinstance(receipt, dict):
        return ["GPU v3 public projection receipt must be an object"], set()
    receipt_errors = validate_gpu_receipt(package, receipt)
    if receipt_errors:
        return [
            "GPU v3 public projection requires a qualified receipt: "
            + ". ".join(receipt_errors)
        ], set()
    summary_relative = manifest.get("gpu_training_public_summary")
    if not _relative(summary_relative) or not str(summary_relative).startswith("evidence/"):
        errors.append("GPU v3 manifest.gpu_training_public_summary must be a safe evidence/ path")
        return errors, set()
    summary_path = _member(package, summary_relative, "GPU v3 public summary", errors)
    if summary_path is None or not summary_path.is_file():
        return errors, set()
    try:
        summary = strict_load_json(summary_path, "GPU public summary")
        expected = build_gpu_public_summary(package, receipt)
    except ValueError as exc:
        errors.append(f"GPU v3 public summary is invalid: {exc}")
        return errors, set()
    if summary != expected or summary_path.read_bytes() != canonical_json(expected):
        errors.append("GPU v3 public summary must equal canonical deterministic projection")
    if not isinstance(sources, dict) or summary_relative not in sources.values():
        errors.append("GPU v3 public_projection.sources must map the deterministic public summary")

    forbidden = {str(receipt_relative)}
    for field in (
        "design", "registration", "raw_census", "correctness_quality",
        "qualification_report", "baseline_review", "telemetry_review",
        "scientific_review", "piq_envelope", "checkpoint_restart",
        "telemetry_coverage", "custody_readback_cleanup", "source_import_closure",
    ):
        ref = receipt.get(field)
        if isinstance(ref, dict) and isinstance(ref.get("path"), str):
            forbidden.add(ref["path"])
    for collection in ("accepted_repeats", "profile_evidence"):
        for item in receipt.get(collection, []):
            if not isinstance(item, dict):
                continue
            ref = item.get("raw_result" if collection == "accepted_repeats" else "artifact")
            if isinstance(ref, dict) and isinstance(ref.get("path"), str):
                forbidden.add(ref["path"])
    envelope_ref = receipt.get("piq_envelope")
    envelope = None
    if isinstance(envelope_ref, dict):
        try:
            envelope = strict_load_json(package / envelope_ref["path"], "PIQ v2 envelope")
        except ValueError as exc:
            errors.append(f"GPU v3 private envelope cannot be read for redaction: {exc}")
    private_values: set[str] = set()
    if isinstance(envelope, dict):
        for attempt in envelope.get("attempts", []):
            if isinstance(attempt, dict):
                private_values.update(str(attempt.get(field)) for field in ("id", "run_id") if attempt.get(field))
        for execution in envelope.get("executions", []):
            if not isinstance(execution, dict):
                continue
            observed = execution.get("observed_resources", {})
            if isinstance(observed, dict):
                if observed.get("host"):
                    private_values.add(str(observed["host"]))
                private_values.update(str(value) for value in observed.get("device_uuids", []))
            assignment = execution.get("assignment_evidence", {})
            if isinstance(assignment, dict) and isinstance(assignment.get("path"), str):
                bundle = (package / envelope_ref["path"]).parent
                try:
                    document = strict_load_json(bundle / assignment["path"], "GPU assignment")
                except ValueError:
                    document = None
                if isinstance(document, dict):
                    for field in ("scheduler_job_id", "owner", "node", "session_id", "cgroup"):
                        if document.get(field):
                            private_values.add(str(document[field]))
        bundle_prefix = PurePosixPath(envelope_ref["path"]).parent
        for artifact in envelope.get("artifacts", []):
            if not isinstance(artifact, dict) or not isinstance(artifact.get("path"), str):
                continue
            relative = artifact["path"]
            if any(token in relative for token in (
                "attempts/", "assignments/", "rank-outputs/", "telemetry/",
                "custody/", "checkpoint", "raw-census", "cleanup", "readback",
            )):
                forbidden.add((bundle_prefix / relative).as_posix())
    if isinstance(sources, dict):
        for public, source in sources.items():
            if source in forbidden:
                errors.append(f"{public}: GPU v3 public projection maps forbidden private evidence {source}")
    summary_text = json.dumps(summary, sort_keys=True, ensure_ascii=False)
    for value in sorted(private_values):
        if value and value in summary_text:
            errors.append(f"GPU v3 public summary leaks a private execution identity: {value}")
    return errors, private_values


def _relative(value: object) -> bool:
    return (
        isinstance(value, str) and bool(value) and "\\" not in value
        and "\x00" not in value and not re.match(r"^[A-Za-z]:", value)
        and not PurePosixPath(value).is_absolute()
        and all(part not in {"", ".", ".."} for part in value.split("/"))
    )


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _member(root: Path, relative: object, label: str, errors: list[str]) -> Path | None:
    if not _relative(relative):
        errors.append(f"{label}: must be a safe package-relative path")
        return None
    assert isinstance(relative, str)
    path = root
    try:
        for part in PurePosixPath(relative).parts:
            path = path / part
            if path.is_symlink():
                errors.append(f"{label}: symlinks are not allowed")
                return None
        resolved = path.resolve()
    except (OSError, RuntimeError) as exc:
        errors.append(f"{label}: cannot resolve path: {exc}")
        return None
    if not _inside(resolved, root):
        errors.append(f"{label}: path escapes the package")
        return None
    return resolved


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _staged_private_identity_errors(
        files: dict[str, Path], private_values: set[str]) -> list[str]:
    """Scan every staged file as bytes with bounded memory and fail closed."""
    errors: list[str] = []
    needles: set[bytes] = set()
    for value in private_values:
        try:
            needle = value.encode("utf-8")
        except UnicodeError:
            errors.append("GPU v3 private execution identity is not valid UTF-8")
            continue
        if len(needle) > _MAX_PRIVATE_IDENTITY_BYTES:
            errors.append("GPU v3 private execution identity exceeds the safe scan limit")
            continue
        if needle:
            needles.add(needle)
    if not needles:
        return errors

    overlap = max(len(needle) for needle in needles) - 1
    for relative, path in sorted(files.items()):
        try:
            carry = b""
            with path.open("rb") as stream:
                while chunk := stream.read(_PRIVATE_IDENTITY_SCAN_CHUNK_BYTES):
                    window = carry + chunk
                    if any(needle in window for needle in needles):
                        errors.append(
                            f"{relative}: staged public file leaks a private execution identity"
                        )
                        break
                    carry = window[-overlap:] if overlap else b""
        except OSError as exc:
            errors.append(
                f"{relative}: cannot scan staged file for private execution identities: {exc}"
            )
    return errors


def _image_paths(text: str) -> list[str]:
    definitions = {key.casefold(): value for key, value in REFERENCE_DEFINITION.findall(text)}
    urls = [left or right for left, right in MARKDOWN_IMAGE.findall(text)]
    urls.extend(url for _, url in HTML_IMAGE.findall(text))
    urls.extend(definitions.get((label or alt).casefold(), "") for alt, label in REFERENCE_IMAGE.findall(text))
    return urls


def validate_public_projection(package: Path) -> list[str]:
    """Return boundary errors for a manifest-declared staged public directory."""
    errors: list[str] = []
    if not package.is_dir():
        return ["public projection: package directory is missing"]
    package = package.resolve()
    manifest = package / "manifest.json"
    if not manifest.exists():
        return []
    if manifest.is_symlink():
        return ["public projection: manifest.json must not be a symlink"]
    try:
        data = _strict_json(manifest)
    except (OSError, UnicodeError, ValueError) as exc:
        return [f"public projection: invalid manifest.json: {exc}"]
    if not isinstance(data, dict):
        return ["public projection: manifest.json must be an object"]
    if "public_projection" not in data:
        return []
    declaration = data["public_projection"]
    if not isinstance(declaration, dict):
        return ["public_projection must be an object"]
    directory = declaration.get("directory")
    output = _member(package, directory, "public_projection.directory", errors)
    if isinstance(directory, str) and any(part.casefold() in PRIVATE_NAMES for part in directory.split("/")):
        errors.append("public_projection.directory: internal directories are not public")
    sources = declaration.get("sources")
    if not isinstance(sources, dict) or not sources:
        errors.append("public_projection.sources must be a nonempty public-path to source-path mapping")
        return errors
    gpu_errors, private_values = _gpu_v3_projection_errors(package, data, sources)
    errors.extend(gpu_errors)
    if output is None:
        return errors
    if not output.is_dir():
        errors.append("public_projection.directory: staged directory is missing")
        return errors

    expected: set[str] = set()
    mapped: dict[str, Path] = {}
    for public, source in sources.items():
        if not _relative(public):
            errors.append(f"public projection target {public!r}: must be a safe relative path")
            continue
        expected.add(public)
        parts = PurePosixPath(public).parts
        if any(part.casefold() in PRIVATE_NAMES for part in parts):
            errors.append(f"{public}: internal paths are not public")
        if public != "README.md" and (len(parts) < 2 or parts[0] not in PUBLIC_DIRECTORIES):
            errors.append(f"{public}: public files must be README.md or under images/code/captures/evidence")
        retained = _member(package, source, f"source for {public}", errors)
        if retained is not None:
            if _inside(retained, output):
                errors.append(f"source for {public}: source must not be inside the staged output")
            elif not retained.is_file():
                errors.append(f"source for {public}: retained source file is missing")
            else:
                mapped[public] = retained

    actual: set[str] = set()
    safe_files: dict[str, Path] = {}
    for current, directories, files in os.walk(output, followlinks=False):
        for name in [*directories, *files]:
            path = Path(current) / name
            relative = path.relative_to(output).as_posix()
            parts = PurePosixPath(relative).parts
            if any(part.casefold() in PRIVATE_NAMES for part in parts):
                errors.append(f"{relative}: internal paths are not public")
            if len(parts) == 1 and (name not in PUBLIC_DIRECTORIES and name != "README.md"):
                errors.append(f"{relative}: unsupported public top-level path")
            if path.is_symlink():
                errors.append(f"{relative}: symlinks are not allowed in staged output")
                continue
            if path.is_dir():
                if relative == "README.md":
                    errors.append("README.md: must be a file")
                continue
            actual.add(relative)
            if not path.is_file():
                errors.append(f"{relative}: staged path must be a regular file")
            else:
                safe_files[relative] = path
    for relative in sorted(expected - actual):
        errors.append(f"{relative}: declared public file is missing")
    for relative in sorted(actual - expected):
        errors.append(f"{relative}: staged file is not declared in public_projection.sources")
    for relative in sorted(set(mapped) & set(safe_files)):
        try:
            if _digest(mapped[relative]) != _digest(safe_files[relative]):
                errors.append(f"{relative}: sha256 mismatch with retained source")
        except OSError as exc:
            errors.append(f"{relative}: cannot compare retained source: {exc}")

    errors.extend(_staged_private_identity_errors(safe_files, private_values))

    readme = ""
    for relative, path in sorted(safe_files.items()):
        try:
            if path.suffix.lower() not in TEXT_SUFFIXES:
                if path.stat().st_size > 2 * 1024 * 1024:
                    continue
                with path.open("rb") as stream:
                    if b"\x00" in stream.read(8192):
                        continue
            raw = path.read_bytes()
            text = raw.decode("utf-8")
        except UnicodeError:
            if relative == "README.md" or path.suffix.lower() in TEXT_SUFFIXES:
                errors.append(f"{relative}: reader text must be UTF-8")
            continue
        except OSError as exc:
            errors.append(f"{relative}: cannot read public file: {exc}")
            continue
        if relative == "README.md":
            readme = text
        for pattern, label in PRIVATE_TEXT:
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                errors.append(f"{relative}:{line}: {label}: {match.group(0)!r}")
        if relative == "README.md" and "<!--" in text:
            errors.append("README.md: HTML comments may contain private review material")
    if not readme.strip():
        errors.append("README.md: public article must be nonempty UTF-8 text")
    if "README.md" not in expected:
        errors.append("README.md: must be declared in public_projection.sources")

    shown: set[str] = set()
    for url in _image_paths(readme):
        try:
            parsed = urlsplit(html.unescape(url))
        except ValueError:
            errors.append(f"README.md: invalid image URL: {url!r}")
            continue
        if parsed.scheme or parsed.netloc:
            continue
        relative = unquote(parsed.path)
        if relative.startswith("./"):
            relative = relative[2:]
        if not _relative(relative):
            errors.append(f"README.md: image path escapes or is invalid: {url!r}")
            continue
        if relative.startswith("images/"):
            shown.add(relative)
        if relative not in actual:
            errors.append(f"README.md: shown image is missing: {relative}")
    images = {relative for relative in actual if relative.startswith("images/") and relative.lower().endswith(".png")}
    for relative in sorted(images - shown):
        errors.append(f"{relative}: PNG is not shown in README.md")
    return list(dict.fromkeys(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    args = parser.parse_args()
    errors = validate_public_projection(args.package)
    for error in errors:
        print(f"FAIL: {error}")
    print(f"public projection: {len(errors)} problem(s)" if errors else "public projection: PASS")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
