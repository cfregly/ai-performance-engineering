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
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
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
