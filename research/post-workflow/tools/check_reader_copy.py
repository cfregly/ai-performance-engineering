#!/usr/bin/env python3
"""Keep private review state and file labels out of retained reader copy.

Article/social copy, optional review galleries, public-assets prose and explicit
reader figure references are checked. Unlinked renderer/support artifacts,
evidence, manifests, receipts and raw captures remain outside this wording gate.
Scientific figure variables are distinct from gallery labels. This checks no
scientific validity, image pixels or actual platform preview.
Only exact raw Typefully organization mentions in retained social/linkedin.txt
bypass URN reference rejection.
"""

from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
import json
import posixpath
import re
import sys
from urllib.parse import unquote, urlsplit
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
    r"\b(?!(?:options|variants)\b)(?:option|variant)\s*[a-z]\b|"
    r"^\s*#{1,6}\s+(?:[a-z]|fun)\s*$|"
    r"^\s*#{1,6}\s+[a-z]:\s|"
    r"^\s*(?:#{1,6}\s+)?(?:fun|aud-[\w-]+):\s|"
    r"(?<![\w/.-])aud-[\w-]+\b|"
    r"(?<![\w.-])(?:[a-z]|fun|aud-[\w-]+)(?:-square|-400)?\.(?:png|svg|txt)\b",
    re.I | re.M,
)
GALLERY_LETTER_LABEL = re.compile(r"^\s*[a-z]:\s", re.I | re.M)
PRIVATE_PARTS = {
    "_internal", "evidence", "captures", "raw", "receipts", "verification", "superseded",
}
TEXT_SUFFIXES = {".md", ".txt", ".html", ".htm", ".svg"}
REFERENCE_SUFFIXES = TEXT_SUFFIXES | {".json"}
TYPEFULLY_LINKEDIN_ORGANIZATION_MENTION = re.compile(
    r"@\[(?P<label>[^\[\]\r\n]+)\]\(urn:li:organization:[1-9][0-9]*\)"
)


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


def check_reader_copy(text: str, where: str, *, labels: bool = False,
                      figure: bool = False) -> list[str]:
    """Check already-visible wording, preserving useful scientific conditions."""
    checks = [(PROCESS_COPY, "internal review/process disclaimer")]
    if labels:
        checks.append((INTERNAL_LABEL, "internal option/filename label"))
        # A figure's A:/B: legend or H: tensor shape is not a filename label.
        # Explicit option names, filenames and aud-* tokens still fail above.
        if not figure:
            checks.append((GALLERY_LETTER_LABEL, "internal option/filename label"))
    return [
        f"{where}: {reason}: {match.group(0).strip()!r}"
        for pattern, reason in checks
        for match in pattern.finditer(text)
    ]


def _backslash_escaped(text: str, position: int) -> bool:
    """Check whether an odd run of backslashes escapes one character."""
    start = position
    while start > 0 and text[start - 1] == "\\":
        start -= 1
    return bool((position - start) % 2)


def _without_linkedin_organization_mentions(text: str) -> tuple[str, bool]:
    """Remove exact raw mention destinations before any Markdown cleanup."""
    parts = []
    cursor = 0
    for match in TYPEFULLY_LINKEDIN_ORGANIZATION_MENTION.finditer(text):
        if (any(_backslash_escaped(text, position)
                for position in (match.start(), match.end("label")))
                or not match.group("label").strip()):
            continue
        parts.extend((text[cursor:match.start()], match.group("label")))
        cursor = match.end()
    remaining = "".join(parts) + text[cursor:]
    return remaining, re.search(r"urn:li:", remaining, re.I) is not None


def _inline_reference_targets(text: str) -> list[str]:
    """Read nested link labels without adding Python recursion depth."""
    targets = []
    pending = [text]
    while pending:
        references, labels = _inline_reference_level(pending.pop())
        targets.extend(references)
        pending.extend(labels)
    return targets


def _inline_reference_level(text: str) -> tuple[list[str], list[str]]:
    """Read balanced links at one level and queue their nested labels."""
    targets = []
    labels = []
    cursor = 0
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
            continue
        if text[cursor] != "[":
            cursor += 1
            continue
        end = cursor + 1
        depth = 1
        while end < len(text) and depth:
            if text[end] == "\\":
                end += 2
                continue
            if text[end] == "[":
                depth += 1
            elif text[end] == "]":
                depth -= 1
            end += 1
        if depth or text[end:end + 1] != "(":
            cursor += 1
            continue
        position = end + 1
        while position < len(text) and text[position].isspace():
            position += 1
        angle = text[position:position + 1] == "<"
        start = position + int(angle)
        position = start
        depth = 0
        while position < len(text):
            char = text[position]
            if char == "\\":
                position += 2
                continue
            if angle:
                if char == ">":
                    break
            else:
                if char.isspace() or (char == ")" and not depth):
                    break
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
            position += 1
        if position >= len(text) or depth or (angle and text[position] != ">"):
            cursor += 1
            continue
        target = text[start:position]
        after = position + int(angle)
        # Preserve the previous optional-title handling. Destination balancing
        # ends before this tail, so a title cannot truncate a real file suffix.
        tail = re.match(r"(?:\s+[^)]*)?\)", text[after:])
        if target and tail:
            token_end = after + tail.end()
            # Markdown escapes punctuation, not arbitrary letters or spaces.
            target = re.sub(r"\\([!-/:-@\[-`{-~])", r"\1", target)
            targets.append(target)
            label = text[cursor + 1:end - 1]
            if "[" in label:
                labels.append(label)
            cursor = token_end
        else:
            cursor += 1
    return targets, labels


def _reference_targets(text: str, *, svg: bool = False,
                       machine: bool = False) -> list[str]:
    """Read actual local link/image targets, not code or bare filenames."""
    if svg:
        root = ET.fromstring(text)
        return [value for element in root.iter()
                for key, value in element.attrib.items()
                if key.rsplit("}", 1)[-1] == "href"]
    if machine:
        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, list):
                for item in value:
                    yield from strings(item)
            elif isinstance(value, dict):
                for item in value.values():
                    yield from strings(item)
        # Raw metadata path strings do not make a reader figure. Actual
        # Markdown/HTML image or link markup inside a record still does.
        return [target for value in strings(json.loads(text))
                for target in _reference_targets(value)]
    text = re.sub(r"(?ms)^\s*(`{3,}|~{3,})[^\n]*\n.*?^\s*\1\s*$", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    text = re.sub(r"<(script|style)\b[^>]*>.*?</\1\s*>", "", text, flags=re.I | re.S)
    text = re.sub(r"(`+)(.*?)\1", "", text, flags=re.S)
    definitions = {key.casefold(): value for key, value in re.findall(
        r"^ {0,3}\[([^]\n]+)\]:\s*<?([^\s>]+)>?", text, re.M)}
    text = re.sub(r"^ {0,3}\[[^]\n]+\]:[^\n]*", "", text, flags=re.M)
    targets = _inline_reference_targets(text)
    for alt, key in re.findall(r"!?\[([^]\n]*)\]\[([^]\n]*)\]", text):
        if (key or alt).casefold() in definitions:
            targets.append(definitions[(key or alt).casefold()])
    for key in re.findall(r"!?\[([^]\n]+)\](?![\[(])", text):
        if key.casefold() in definitions:
            targets.append(definitions[key.casefold()])

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag in {"img", "a", "object", "image"}:
                targets.extend(value for key, value in attrs
                               if key in {"src", "href", "data"} and value)
    parser = Links()
    parser.feed(text)
    parser.close()
    return targets


def reader_files(package: Path, errors: list[str] | None = None) -> dict[Path, bool]:
    """Use explicit public roles and reader references, not an images glob.

    SVG/text siblings describe an explicitly selected or linked figure. Public
    projection aliases resolve links from their public path, not the retained
    source directory. Private records and unlinked support sheets stay private.
    """
    package = package.resolve()
    issues: list[str] = []
    paths: dict[Path, bool] = {}
    pending: list[tuple[Path, str, bool]] = []
    seen: set[tuple[Path, str, bool]] = set()
    mapped: dict[str, Path] = {}

    def member(relative: object, label: str, *, strict: bool = True) -> Path | None:
        if (not isinstance(relative, str) or not relative or "\\" in relative
                or "\x00" in relative or re.match(r"^[A-Za-z]:", relative)
                or PurePosixPath(relative).is_absolute()
                or (strict and any(part in {"", ".", ".."} for part in relative.split("/")))):
            issues.append(f"invalid reader path {label}: must stay within the package")
            return None
        parts: list[str] = []
        for part in relative.split("/"):
            if part in {"", "."}:
                continue
            if part == "..":
                if not parts:
                    issues.append(f"invalid reader path {label}: escapes the package")
                    return None
                parts.pop()
            else:
                parts.append(part)
                if package.joinpath(*parts).is_symlink():
                    issues.append(f"invalid reader path {label}: symlinks are not allowed")
                    return None
        return package.joinpath(*parts)

    def add(path: Path | None, logical: str, labels: bool | None,
            public: bool = False, *, required: bool = False) -> None:
        if path is None:
            return
        if not path.is_file():
            if required:
                issues.append(f"invalid reader path {logical}: declared reader file is missing")
            return
        if labels is not None:
            paths[path] = paths.get(path, False) or labels
        pending.append((path, logical, public))

    def family(path: Path | None, logical: str, public: bool = False) -> None:
        if path is None:
            return
        stem = PurePosixPath(logical).with_suffix("")
        for variant in ("", "-square"):
            for suffix in (".svg", ".txt"):
                key = str(stem) + variant + suffix
                target = mapped.get(key) if public else None
                if target is None:
                    target = member(str(path.relative_to(package).with_suffix("")) + variant + suffix, key)
                add(target, key, True, public)

    for name, labels in (("README.md", False), ("review.md", True),
                         ("social/x.txt", False), ("social/linkedin.txt", False),
                         ("image.txt", True), ("image.svg", True)):
        add(member(name, name), name, labels)
    # These guides supply references without turning their coordination prose
    # into a reader gallery. Public-assets prose keeps its existing gate role.
    for name in ("images/README.md", "images/audiences.md", "code/README.md"):
        add(member(name, name), name, None)
    public_assets = member("public-assets", "public-assets")
    for path in sorted(public_assets.rglob("*")) if public_assets is not None else ():
        relative = path.relative_to(package)
        if not any(part.lower() in PRIVATE_PARTS for part in relative.parts) and path.suffix.lower() in TEXT_SUFFIXES - {".svg"}:
            add(member(relative.as_posix(), relative.as_posix()), relative.as_posix(), True)
    manifest_path = member("manifest.json", "manifest.json")
    manifest = {}
    if manifest_path is not None and manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(manifest, dict):
                raise ValueError("must be an object")
        except (OSError, UnicodeError, ValueError) as exc:
            issues.append(f"invalid reader selection manifest.json: {exc}")
            manifest = {}
    declaration = manifest.get("public_projection")
    if declaration is not None:
        if not isinstance(declaration, dict) or not isinstance(declaration.get("sources"), dict) or not declaration["sources"]:
            issues.append("invalid reader selection public_projection.sources: requires a nonempty mapping")
        else:
            member(declaration.get("directory"), "public_projection.directory")
            for logical, source in declaration["sources"].items():
                if member(logical, "public target") is not None:
                    retained = member(source, f"source for {logical}")
                    if retained is not None:
                        mapped[logical] = retained
            for logical, retained in mapped.items():
                suffix = PurePosixPath(logical).suffix.lower()
                private = any(part.lower() in PRIVATE_PARTS for part in PurePosixPath(logical).parts)
                if suffix in {".svg", ".txt"}:
                    # Explicit publication overrides historical folder names.
                    add(retained, logical, True, True, required=True)
                elif logical == "README.md" or suffix in {".md", ".html", ".htm"}:
                    labels = None if private else logical != "README.md" and logical.startswith("public-assets/")
                    add(retained, logical, labels, True, required=True)
                elif suffix == ".json":
                    add(retained, logical, None, True, required=True)
    stems = [manifest[key] for key in ("cover_stem", "selected_image_stem") if key in manifest]
    concepts = manifest.get("concepts", [])
    audiences = manifest.get("audiences", [])
    if not isinstance(concepts, list) or not isinstance(audiences, list):
        issues.append("invalid reader selection: concepts and audiences must be lists")
    else:
        stems.extend(concepts)
        for audience in audiences:
            if not isinstance(audience, dict) or "stem" not in audience:
                issues.append("invalid reader selection: audience requires a stem")
            else:
                stems.append(audience["stem"])
    for stem in stems:
        family(member(stem, "selected image stem"), str(stem))
    reader_visuals = manifest.get("reader_visuals", {})
    if isinstance(reader_visuals, dict):
        for key in ("scientific_chart", "chart_text"):
            if key in reader_visuals:
                value = reader_visuals[key]
                path = member(value.get("path") if isinstance(value, dict) else None, f"reader_visuals.{key}")
                if path is not None:
                    if path.suffix.lower() in TEXT_SUFFIXES:
                        add(path, str(path.relative_to(package)), True, required=True)
                    family(path, str(path.relative_to(package)))
    tweets = member("tweets.md", "tweets.md")
    if tweets is not None and tweets.is_file():
        try:
            text = tweets.read_text(encoding="utf-8")
            if re.search(r"^## Image options\b", text, re.M | re.I):
                add(tweets, "tweets.md", None)
        except (OSError, UnicodeError) as exc:
            issues.append(f"invalid reader copy tweets.md image gallery: {exc}")
    while pending:
        path, logical, public = pending.pop()
        identity = (path, logical, public)
        if identity in seen:
            continue
        seen.add(identity)
        if not public and path.relative_to(package).parts[0] == "public-assets" and path in mapped.values():
            # The mapped public-path context is already queued. A retained
            # alias directory is not the base of its published relative links.
            continue
        try:
            text = path.read_text(encoding="utf-8")
            if path.name == "tweets.md":
                text = "\n".join(g.group(1) for g in re.finditer(r"^## Image options\b(.*?)(?=^## |\Z)", text, re.S | re.M | re.I))
            if (not public and logical == "social/linkedin.txt"
                    and path == package / "social/linkedin.txt"):
                text, stray_urn = _without_linkedin_organization_mentions(text)
                if stray_urn:
                    issues.append(f"invalid reader reference {logical}: "
                                  "LinkedIn URN outside an exact native organization mention")
            targets = _reference_targets(text, svg=path.suffix.lower() == ".svg",
                                         machine=path.suffix.lower() == ".json")
        except (OSError, UnicodeError, ValueError, ET.ParseError) as exc:
            issues.append(f"invalid reader references {logical}: {exc}")
            continue
        for target in targets:
            try:
                split = urlsplit(target)
            except ValueError as exc:
                issues.append(f"invalid reader reference {logical}: {exc}")
                continue
            # Check the scheme before the authority. file://host/path.svg is
            # not a safe external web link merely because it has a netloc.
            if split.scheme and split.scheme not in {"http", "https", "data", "mailto"}:
                issues.append(f"invalid reader reference {logical}: unsupported local scheme")
                continue
            if split.scheme in {"http", "https", "data", "mailto"} or split.netloc or not split.path:
                continue
            reference = unquote(split.path)
            suffix = PurePosixPath(reference).suffix.lower()
            if suffix not in REFERENCE_SUFFIXES | {".png", ".jpg", ".jpeg", ".webp"}:
                continue
            relative = str(PurePosixPath(logical).parent / reference)
            checked = member(relative, f"reference from {logical}", strict=False)
            normalized = posixpath.normpath(relative)
            if checked is None:
                continue
            retained = mapped.get(normalized, checked) if public else checked
            if suffix in REFERENCE_SUFFIXES:
                # Linked methods prose is not an option gallery. Explicit
                # figure links override historical/private directory names.
                private = any(part.lower() in PRIVATE_PARTS for part in PurePosixPath(normalized).parts)
                labels = True if suffix in {".svg", ".txt"} else None if private or suffix == ".json" else suffix != ".md"
                add(retained, normalized, labels, public, required=True)
            if suffix in {".svg", ".png", ".jpg", ".jpeg", ".webp"}:
                family(retained, normalized, public)
    if errors is not None:
        errors.extend(issues)
    elif issues:
        raise ValueError("; ".join(issues))
    return paths


def validate_reader_copy(package: Path) -> list[str]:
    package = package.resolve()
    errors: list[str] = []
    for path, labels in reader_files(package, errors).items():
        relative = path.relative_to(package).as_posix()
        try:
            path.resolve().relative_to(package.resolve())
            text = path.read_text(encoding="utf-8")
            visible = svg_text(text) if path.suffix.lower() == ".svg" else visible_markdown(text)
        except (OSError, UnicodeError, ValueError, ET.ParseError) as exc:
            errors.append(f"invalid reader copy {relative}: {exc}")
            continue
        figure = path.suffix.lower() in {".svg", ".txt"}
        errors.extend(check_reader_copy(visible, relative, labels=labels, figure=figure))
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
