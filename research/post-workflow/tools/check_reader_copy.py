#!/usr/bin/env python3
"""Keep private review state and file labels out of retained reader copy.

Only article/social copy, optional review galleries, public-assets prose,
image text sidecars and SVG text are checked. Evidence, manifests, receipts,
renderer notes and raw captures remain outside this gate. This checks wording,
not scientific validity, image pixels or the actual platform preview.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path
import re
import sys
import xml.etree.ElementTree as ET


PROCESS_COPY = re.compile(
    r"\billustrative routing\b|\bno new performance measurements?\b|"
    r"\bnot a performance result\b|\bno captured trace\b|"
    r"\breview[_ -](?:blocked|pending|passed|failed|status)\b|"
    r"\b(?:internal review only|for internal review)\b|\bQA (?:pending|blocked)\b|"
    r"^\s*(?:STATUS:\s*(?:incomplete|pending|blocked|ready|complete|draft|passed|failed)|"
    r"Verification:|Confidence:|Reviewers:|"
    r"Internal:\s*nothing|#{1,6}\s+Review:)",
    re.I | re.M,
)
INTERNAL_LABEL = re.compile(
    r"^\s*(?:#{1,6}\s+)?(?:option|variant)\s+[a-z]\b|"
    r"^\s*#{1,6}\s+(?:[a-z]|fun)\s*$|"
    r"^\s*(?:#{1,6}\s+)?(?:[a-z]|fun|aud-[\w-]+):\s|"
    r"(?<![\w/.-])aud-[\w-]+\b|"
    r"(?<![\w.-])(?:[a-z]|fun|aud-[\w-]+)(?:-square|-400)?\.(?:png|svg|txt)\b",
    re.I | re.M,
)
PRIVATE_PARTS = {
    "_internal", "evidence", "captures", "raw", "receipts", "verification", "superseded",
}
TEXT_SUFFIXES = {".md", ".txt", ".html", ".htm", ".svg"}


class _HTMLText(HTMLParser):
    """Read visible HTML text and image alt text, never attributes or scripts."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self.hidden.append(tag)
        if self.hidden:
            return
        if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n# ")
        elif tag in {"p", "div", "br", "li", "tr"}:
            self.parts.append("\n")
        if tag == "img":
            self.parts.append(dict(attrs).get("alt") or "")

    def handle_endtag(self, tag: str) -> None:
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
            return
        if tag in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def _without_inline_targets(text: str) -> str:
    """Keep Markdown labels while dropping balanced inline link destinations."""
    out: list[str] = []
    cursor = 0
    while cursor < len(text):
        if text[cursor:cursor + 2] != "](":
            out.append(text[cursor])
            cursor += 1
            continue
        end = cursor + 2
        depth = 1
        while end < len(text) and depth:
            if text[end] == "\\":
                end += 2
                continue
            if text[end] == "(":
                depth += 1
            elif text[end] == ")":
                depth -= 1
            end += 1
        out.append("]")
        if depth:
            # A malformed target is still visible text, rather than an escape
            # from the copy check.
            cursor += 1
        else:
            cursor = end
    return "".join(out)


def visible_markdown(text: str) -> str:
    """Discard destinations, reference definitions, comments and HTML markup."""
    text = re.sub(r"^ {0,3}\[[^]\n]+\]:[^\n]*(?:\n[ \t]+[^\n]*)*", "", text, flags=re.M)
    text = _without_inline_targets(text)
    text = re.sub(r"\] *\[[^]\n]*\]", "]", text)
    text = re.sub(r"https?://[^\s<>]+", "", text)
    parser = _HTMLText()
    parser.feed(text)
    parser.close()
    # Formatting characters must not hide visible labels such as **Option A**.
    return re.sub(r"[*`]+", "", "".join(parser.parts))


def svg_text(text: str) -> str:
    """Read SVG text elements, including nested tspan/textPath content."""
    root = ET.fromstring(text)
    return "\n".join(
        "".join(element.itertext())
        for element in root.iter()
        if element.tag.rsplit("}", 1)[-1] == "text"
    )


def check_reader_copy(text: str, where: str, *, labels: bool = False) -> list[str]:
    """Check already-visible wording, preserving useful scientific conditions."""
    checks = [(PROCESS_COPY, "internal review/process disclaimer")]
    if labels:
        checks.append((INTERNAL_LABEL, "internal option/filename label"))
    return [
        f"{where}: {reason}: {match.group(0).strip()!r}"
        for pattern, reason in checks
        for match in pattern.finditer(text)
    ]


def reader_files(package: Path) -> dict[Path, bool]:
    """Select public surfaces without recursively scanning private package data."""
    paths = {
        package / "README.md": False,
        package / "review.md": True,
        package / "social/x.txt": False,
        package / "social/linkedin.txt": False,
        package / "image.txt": True,
        package / "image.svg": True,
    }
    for directory in ("images", "public-assets"):
        for path in sorted((package / directory).rglob("*")):
            if any(part.lower() in PRIVATE_PARTS for part in path.relative_to(package).parts):
                continue
            if directory == "images" and path.suffix.lower() not in {".svg", ".txt"}:
                continue
            if directory == "public-assets" and path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            paths[path] = True
    return {path: labels for path, labels in paths.items() if path.is_file()}


def validate_reader_copy(package: Path) -> list[str]:
    errors: list[str] = []
    for path, labels in reader_files(package).items():
        relative = path.relative_to(package).as_posix()
        try:
            path.resolve().relative_to(package.resolve())
            text = path.read_text(encoding="utf-8")
            visible = svg_text(text) if path.suffix.lower() == ".svg" else visible_markdown(text)
        except (OSError, UnicodeError, ValueError, ET.ParseError) as exc:
            errors.append(f"invalid reader copy {relative}: {exc}")
            continue
        errors.extend(check_reader_copy(visible, relative, labels=labels))
    # tweets.md is a private working sheet. Its image-options section is a
    # reader gallery, but alternate drafts and coordination notes are not.
    tweets = package / "tweets.md"
    if tweets.is_file():
        try:
            text = tweets.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"invalid reader copy tweets.md image gallery: {exc}")
        else:
            for gallery in re.finditer(r"^## Image options\b(.*?)(?=^## |\Z)", text, re.S | re.M | re.I):
                errors.extend(check_reader_copy(
                    visible_markdown(gallery.group(1)), "tweets.md image gallery", labels=True,
                ))
    return errors


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: check_reader_copy.py PACKAGE", file=sys.stderr)
        return 2
    package = Path(argv[0]).resolve()
    if not package.is_dir():
        print(f"FAIL: package directory does not exist: {package}")
        return 1
    errors = validate_reader_copy(package)
    for error in errors:
        print(f"FAIL: {error}")
    if not errors:
        print("PASS: reader copy")
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
