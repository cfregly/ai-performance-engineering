#!/usr/bin/env python3
"""Extract the AI Systems Performance Engineering book PDF into per-chapter markdown.

The PDF is a professionally typeset (Antenna House) build with a reliable text
layer. This script reconstructs clean markdown using font / size / position
heuristics that were calibrated against the actual PDF:

- Body text:     MinionPro-Regular ~10.5pt at x0~=72
- Headings:      MyriadPro-SemiboldCond  (>=22 chapter title, 17-22 ``##``, 13-17 ``###``)
                 "CHAPTER N" eyebrow at ~16.8pt is folded into the H1 title
- Code listings: UbuntuMono ~8.5pt at x0~=89 (indentation preserved)
- Captions:      "Figure/Table/Example N-N." in MyriadPro-Cond ~9pt
- Callouts:      O'Reilly TIP/NOTE icons (raster xref 106 / 108, ~42x56 / ~38x50)
                 with serif body text indented to x0~=137
- Footer band:   running head + page number below y~=600 (stripped)

Cross-page handling: the footer/header band is dropped, and a paragraph that is
split by a page break is rejoined (no spurious paragraph break) unless the prior
page ended on terminal punctuation. Line-break hyphenation is normalized using
the PDF's own encoding: U+2010 (and U+00AD/U+2011) are soft hyphens (removed);
U+002D is a hard compound hyphen (kept).

Usage:
    python core/scripts/extract_book_from_pdf.py --pdf /path/to/book.pdf --out book
    python core/scripts/extract_book_from_pdf.py --pdf ... --out book --chapters 1 7 10
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

# --- calibrated constants -------------------------------------------------
BODY_X = 72.0
CODE_X = 89.0
LIST_X_LO, LIST_X_HI = 84.0, 98.0
CALLOUT_X = 125.0
FOOTER_Y = 600.0          # strip spans whose top is below this
HEADER_Y = 44.0           # strip spans whose bottom is above this (none observed, safety)
MONO_FLAG = 8             # PyMuPDF span flag bit3 = monospaced
ITALIC_FLAG = 2
TIP_SIZE = (40.0, 44.0, 54.0, 58.0)   # w_lo, w_hi, h_lo, h_hi  (xref 106)
NOTE_SIZE = (36.0, 41.0, 47.0, 52.0)  # (xref 108)
SOFT_HYPHENS = ("\u2010", "\u00ad", "\u2011")

CAPTION_RE = re.compile(r"^(Figure|Table|Example)\s+\d+[-\u2013.]\d+\.")
EYEBROW_RE = re.compile(r"^CHAPTER\s+\d+$")
BULLET_RE = re.compile(r"^\s*([\u2022\u2023\u25aa\u2043\u00b7\u2219\u2013\u2014\u2010]|[-*])\s+")
NUMLIST_RE = re.compile(r"^\s*(\d{1,2})[.)]\s+")


@dataclass
class Line:
    text: str          # markdown text with inline code backticked
    raw: str           # plain text (no markup) for hyphen/caption logic
    x0: float
    y0: float
    y1: float
    size: float
    font: str
    mono_frac: float
    page: int


def span_text_with_inline_code(spans: list[dict]) -> tuple[str, str, float, float, str]:
    """Return (markdown_text, raw_text, max_size, mono_fraction, dominant_font)."""
    md_parts: list[str] = []
    raw_parts: list[str] = []
    mono_chars = 0
    total_chars = 0
    sizes: dict[str, float] = {}
    in_code = False
    for s in spans:
        t = s["text"]
        if not t:
            continue
        is_mono = bool(s["flags"] & MONO_FLAG)
        stripped = t.strip()
        if stripped:
            total_chars += len(stripped)
            if is_mono:
                mono_chars += len(stripped)
            sizes[s["font"]] = max(sizes.get(s["font"], 0.0), s["size"])
        raw_parts.append(t)
        # inline code: wrap contiguous mono runs in backticks (only for prose lines)
        if is_mono and stripped:
            if not in_code:
                md_parts.append("`")
                in_code = True
            md_parts.append(t)
        else:
            if in_code:
                md_parts.append("`")
                in_code = False
            md_parts.append(t)
    if in_code:
        md_parts.append("`")
    md = "".join(md_parts)
    raw = "".join(raw_parts)
    max_size = max(sizes.values()) if sizes else 0.0
    dominant = max(sizes, key=sizes.get) if sizes else ""
    mono_frac = (mono_chars / total_chars) if total_chars else 0.0
    return md, raw, max_size, mono_frac, dominant


def content_lines(page: fitz.Page) -> list[Line]:
    out: list[Line] = []
    d = page.get_text("dict")
    pno = page.number
    for b in d["blocks"]:
        if b.get("type", 0) != 0:
            continue  # image block
        for l in b.get("lines", []):
            spans = [s for s in l.get("spans", []) if s.get("text")]
            if not spans:
                continue
            y0 = l["bbox"][1]
            y1 = l["bbox"][3]
            if y0 >= FOOTER_Y or y1 <= HEADER_Y:
                continue  # footer / running head band
            md, raw, size, mono_frac, font = span_text_with_inline_code(spans)
            if not raw.strip():
                continue
            # drop standalone page-number / separator lines that survived the band
            if re.fullmatch(r"[\d\s|]+", raw.strip()):
                continue
            out.append(Line(md.rstrip(), raw.rstrip(), round(l["bbox"][0], 1),
                            round(y0, 1), round(y1, 1), round(size, 1), font, mono_frac, pno))
    out.sort(key=lambda ln: (ln.y0, ln.x0))
    return out


def callout_icons(page: fitz.Page) -> list[tuple[float, float, str]]:
    icons: list[tuple[float, float, str]] = []
    for im in page.get_image_info(xrefs=True):
        bb = fitz.Rect(im["bbox"])
        w, h = bb.width, bb.height
        if TIP_SIZE[0] <= w <= TIP_SIZE[1] and TIP_SIZE[2] <= h <= TIP_SIZE[3]:
            icons.append((bb.y0, bb.y1, "Tip"))
        elif NOTE_SIZE[0] <= w <= NOTE_SIZE[1] and NOTE_SIZE[2] <= h <= NOTE_SIZE[3]:
            icons.append((bb.y0, bb.y1, "Note"))
    return icons


def is_heading(ln: Line) -> bool:
    return "Myriad" in ln.font and "Semibold" in ln.font and ln.size >= 13.0


def heading_level(size: float) -> int:
    if size >= 22.0:
        return 1
    if size >= 17.0:
        return 2
    return 3


def is_code(ln: Line) -> bool:
    # Real code listings are set in a monospaced font (UbuntuMono, 8.5pt in
    # listings and 10pt for standalone constant-width terms), so the line's
    # DOMINANT (largest) font is monospaced. A prose line that merely contains a
    # long inline-code token (e.g. ``cudaFuncAttributePreferredSharedMemoryCarveout
    # to select the memory ...``) keeps the serif body font (MinionPro ~10.5pt) as
    # its dominant font, so it can have a high mono fraction while NOT being code.
    # Gate on the dominant font, not on mono_frac alone, so body prose is never
    # fenced as code and real constant-width code at 10pt is never dropped.
    if "Mono" in ln.font and ln.mono_frac >= 0.6:
        return True
    return ln.size <= 9.2 and ln.x0 >= 84.0 and ln.mono_frac > 0.0


def is_italic_serif(ln: Line) -> bool:
    return "Minion" in ln.font and "It" in ln.font


def is_caption_start(ln: Line) -> bool:
    # Figure/Table/Example captions are set in italic serif (MinionPro-It).
    return bool(CAPTION_RE.match(ln.raw)) and is_italic_serif(ln)


def join_prose(lines: list[str]) -> str:
    out = ""
    for i, raw in enumerate(lines):
        ln = raw.rstrip()
        if not ln:
            continue
        if not out:
            out = ln
            continue
        if out.endswith(SOFT_HYPHENS):
            out = out[:-1] + ln.lstrip()
        elif out.endswith("-"):
            out = out + ln.lstrip()  # hard compound hyphen wrapped at line end
        elif out.endswith("`") and ln.lstrip().startswith("`"):
            # An inline-code identifier wrapped across a line break (e.g.
            # "...pipeline: `pro`" + "`ducer_acquire()`..."). The two backtick
            # spans are one identifier with no source space, so merge them by
            # dropping the adjacent boundary backticks. Genuine space-separated
            # inline pairs (e.g. `const` `__restrict__`) live WITHIN one source
            # line and never reach this cross-line join.
            out = out[:-1] + ln.lstrip()[1:]
        else:
            out = out + " " + ln.lstrip()
    out = re.sub(r" {2,}", " ", out).strip()
    # Normalize hyphenation that survived line-join: U+00AD is a zero-width soft
    # hyphen (always drop); U+2010/U+2011 used MID-WORD are typographic compound
    # hyphens (e.g. "remote\u2010NUMA", "round\u2010robin") -> normalize to ASCII
    # "-". A trailing U+2010/U+2011 is a dangling cross-block break; leave it so it
    # stays visible rather than silently corrupting the word.
    out = out.replace("\u00ad", "")
    out = re.sub(r"[\u2010\u2011](?=\S)", "-", out)
    return out


def guess_lang(code: str) -> str:
    low = code
    if any(k in low for k in ("#include", "__global__", "__device__", "cudaMalloc",
                              "cudaMemcpy", "<<<", "nvcuda", "__shared__", "cooperative_groups")):
        return "cpp"
    if low.lstrip().startswith("#!/bin/bash") or re.search(r"(?m)^\s*(sudo |apt |export |nvidia-smi|numactl|docker |kubectl |nsys |ncu )", low):
        return "bash"
    if re.search(r"(?m)^\s*(import |from \S+ import|def |class |@|print\()", low):
        return "python"
    if re.search(r"(?m)^\s*(apiVersion:|kind:|metadata:|spec:)", low):
        return "yaml"
    return ""


@dataclass
class Stats:
    pages: int = 0
    headings: int = 0
    code_blocks: int = 0
    callouts: int = 0
    captions: int = 0
    tables: int = 0
    paragraphs: int = 0
    words: int = 0


def render_chapter(doc: fitz.Document, start0: int, end0: int, title: str,
                   chap_num: int | None) -> tuple[str, Stats]:
    md: list[str] = []
    stats = Stats(pages=end0 - start0)

    # element buffers
    para: list[str] = []
    code: list[str] = []
    callout: list[str] = []
    callout_kind = "Note"
    listing: list[str] = []  # current list block (already rendered "- ..." items)
    cur_list_item: list[str] = []
    cap: list[str] = []
    head: list[str] = []
    head_lvl = 2
    table_lines: list[Line] = []
    in_table = False
    pending_table = False
    title_parts: list[str] = []
    have_title = False

    prev_kind = None
    prev_raw_end = ""  # last raw text emitted, for cross-page merge decisions
    prev_page = start0
    prev_y0 = -1.0

    def flush_para():
        nonlocal para
        if para:
            text = join_prose(para)
            if text:
                md.append(text)
                md.append("")
                stats.paragraphs += 1
                stats.words += len(text.split())
            para = []

    def flush_code():
        nonlocal code
        if code:
            while code and not code[0].strip():
                code.pop(0)
            while code and not code[-1].strip():
                code.pop()
        if code:
            lang = guess_lang("\n".join(code))
            md.append(f"```{lang}")
            md.extend(code)
            md.append("```")
            md.append("")
            stats.code_blocks += 1
            code = []
        else:
            code = []

    def flush_callout():
        nonlocal callout, callout_kind
        if callout:
            text = join_prose(callout)
            if text:
                md.append(f"> **{callout_kind}**")
                md.append(">")
                for seg in text.split("\n"):
                    md.append(f"> {seg}")
                md.append("")
                stats.callouts += 1
            callout = []

    def flush_list_item():
        nonlocal cur_list_item
        if cur_list_item:
            listing.append("- " + join_prose(cur_list_item))
            cur_list_item = []

    def flush_list():
        nonlocal listing
        flush_list_item()
        if listing:
            md.extend(listing)
            md.append("")
            listing = []

    def flush_caption():
        nonlocal cap
        if cap:
            text = join_prose(cap)
            if text:
                md.append(f"*{text}*")
                md.append("")
                stats.captions += 1
            cap = []

    def emit_table():
        nonlocal table_lines
        if not table_lines:
            return
        rows: list[list[Line]] = []
        cur: list[Line] = []
        ref: float | None = None
        for ln in sorted(table_lines, key=lambda l: (l.y0, l.x0)):
            if ref is None or abs(ln.y0 - ref) <= 4.0:
                cur.append(ln)
                ref = ln.y0 if ref is None else ref
            else:
                rows.append(cur)
                cur = [ln]
                ref = ln.y0
        if cur:
            rows.append(cur)
        grid = [[c.text.strip() for c in sorted(r, key=lambda l: l.x0)] for r in rows]
        ncol = max((len(r) for r in grid), default=0)
        if ncol >= 2 and len(grid) >= 2:
            for r in grid:
                r += [""] * (ncol - len(r))
            md.append("| " + " | ".join(c.replace("|", "\\|") for c in grid[0]) + " |")
            md.append("| " + " | ".join(["---"] * ncol) + " |")
            for r in grid[1:]:
                md.append("| " + " | ".join(c.replace("|", "\\|") for c in r) + " |")
            md.append("")
            stats.tables += 1
        else:
            for r in grid:
                md.append(join_prose(r) if len(r) == 1 else "  ".join(r))
            md.append("")
        table_lines = []

    def flush_head():
        nonlocal head
        if head:
            text = join_prose(head)
            if text:
                md.append(f"{'#' * head_lvl} {text}")
                md.append("")
                stats.headings += 1
            head = []

    def flush_all_but(keep: str):
        flush_head()
        if keep != "para":
            flush_para()
        if keep != "code":
            flush_code()
        if keep != "callout":
            flush_callout()
        if keep != "list":
            flush_list()
        if keep != "cap":
            flush_caption()

    # title line first
    for pno in range(start0, end0):
        page = doc[pno]
        icons = callout_icons(page)
        for ln in content_lines(page):
            # --- title assembly on the opening page ---
            if not have_title:
                if EYEBROW_RE.match(ln.raw.strip()):
                    continue  # skip "CHAPTER N" eyebrow
                if is_heading(ln) and ln.size >= 22.0:
                    title_parts.append(ln.raw.strip())
                    continue
                # first non-title element closes the title
                if title_parts or chap_num is not None:
                    t = join_prose(title_parts) if title_parts else title
                    if chap_num is not None:
                        md.append(f"# Chapter {chap_num}: {t}")
                    else:
                        md.append(f"# {t}")
                    md.append("")
                    have_title = True
                else:
                    md.append(f"# {title}")
                    md.append("")
                    have_title = True

            # --- captions (italic serif "Figure/Table/Example N-N." + continuations) ---
            if is_caption_start(ln):
                flush_all_but("cap")
                if cap:
                    flush_caption()
                cap = [ln.raw.strip()]
                pending_table = ln.raw.strip().startswith("Table")
                prev_kind = "cap"
                prev_raw_end = ln.raw.strip()
                continue
            if (prev_kind == "cap" and cap and is_italic_serif(ln)
                    and ln.x0 < 100 and not is_heading(ln) and not is_code(ln)):
                cap.append(ln.raw.strip())
                prev_raw_end = ln.raw.strip()
                continue
            if prev_kind == "cap" and pending_table:
                flush_caption()              # the caption is done; a table body follows
                in_table = True
                pending_table = False
                table_lines = []
                prev_kind = "table"

            # --- table body (started by a "Table N-N." caption) ---
            if in_table:
                end_table = (is_heading(ln) or is_code(ln) or is_caption_start(ln)
                             or any(iy0 - 8 <= ln.y0 <= iy1 + 6 for iy0, iy1, _ in icons)
                             or (ln.x0 <= 74.0 and len(ln.raw) > 70))
                if not end_table:
                    table_lines.append(ln)
                    prev_kind = "table"
                    prev_raw_end = ln.raw
                    continue
                emit_table()
                in_table = False  # fall through and classify this line normally

            # --- callout must be anchored by a TIP/NOTE icon (else it is a table cell) ---
            co_kind = None
            if ln.x0 >= CALLOUT_X and ln.mono_frac < 0.6:
                for iy0, iy1, ik in icons:
                    if iy0 - 8 <= ln.y0 <= iy1 + 18:
                        co_kind = ik
                        break

            kind = None
            if is_heading(ln):
                kind = "heading"
            elif co_kind is not None:
                kind = "callout"
            elif (prev_kind == "callout" and ln.x0 >= CALLOUT_X
                  and ln.mono_frac < 0.6):
                kind = "callout"  # wrapped continuation of an icon-anchored callout
            elif is_code(ln):
                kind = "code"
            elif (LIST_X_LO <= ln.x0 <= LIST_X_HI and ln.mono_frac < 0.6
                  and ln.size >= 9.6):
                kind = "list"
            else:
                kind = "body"

            if kind == "heading":
                lvl = heading_level(ln.size)
                if prev_kind == "heading" and lvl == head_lvl:
                    head.append(ln.raw.strip())  # continuation of a wrapped heading
                else:
                    flush_all_but("")
                    head = [ln.raw.strip()]
                    head_lvl = lvl
                prev_kind = "heading"
                prev_raw_end = ln.raw.strip()
                continue

            if kind == "callout":
                if prev_kind != "callout":
                    flush_all_but("callout")
                    callout_kind = co_kind or "Note"
                callout.append(ln.raw)
                prev_kind = "callout"
                prev_raw_end = ln.raw
                continue

            if kind == "code":
                if prev_kind != "code":
                    flush_all_but("code")
                code.append(ln.raw.rstrip())
                prev_kind = "code"
                prev_raw_end = ln.raw
                continue

            if kind == "list":
                if prev_kind != "list":
                    flush_all_but("list")
                m = BULLET_RE.match(ln.text) or NUMLIST_RE.match(ln.text)
                stripped = ln.text
                if m:
                    flush_list_item()
                    stripped = ln.text[m.end():]
                    cur_list_item = [stripped]
                else:
                    if cur_list_item:
                        cur_list_item.append(ln.text)
                    else:
                        cur_list_item = [stripped]
                prev_kind = "list"
                prev_raw_end = ln.raw
                continue

            # body
            if prev_kind == "body":
                # same-page paragraph break by vertical gap handled here
                if para and ln.page == prev_page and (ln.y0 - prev_y0) > 17.0:
                    flush_para()
                elif para and ln.page != prev_page:
                    # cross-page: new paragraph only if prior ended a sentence
                    if prev_raw_end.rstrip().endswith((".", "!", "?", ":", "”", ")")):
                        flush_para()
            else:
                flush_all_but("para")
            para.append(ln.text)
            prev_kind = "body"
            prev_raw_end = ln.raw
            prev_y0 = ln.y0
            prev_page = ln.page
            continue

        # remember last y for next page's gap calc
        prev_y0 = -1.0  # reset; cross-page handled via page comparison
    emit_table()
    flush_all_but("")
    # collapse >2 blank lines
    text = "\n".join(md)
    text = re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"
    return text, stats


def derive_chapters(doc: fitz.Document):
    """Return list of (slug, display_title, chap_num_or_None, start0, end0)."""
    toc = [(lvl, t, p) for lvl, t, p in doc.get_toc(simple=True) if lvl == 1]
    entries = []
    for i, (lvl, title, page) in enumerate(toc):
        start0 = page - 1
        end0 = (toc[i + 1][2] - 1) if i + 1 < len(toc) else doc.page_count
        m = re.match(r"Chapter\s+(\d+)\.\s*(.+)", title)
        if m:
            num = int(m.group(1))
            entries.append((f"ch{num:02d}", m.group(2).strip(), num, start0, end0))
        elif title.strip().lower() == "preface":
            entries.append(("preface", "Preface", None, start0, end0))
        elif title.strip().startswith("Appendix"):
            entries.append(("appendix", title.replace("Appendix.", "").strip(), None, start0, end0))
    return entries


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="output book/ dir")
    ap.add_argument("--chapters", nargs="*", type=int, help="only these chapter numbers")
    ap.add_argument("--front-matter", action="store_true", help="also emit preface/appendix")
    args = ap.parse_args()

    doc = fitz.open(args.pdf)
    args.out.mkdir(parents=True, exist_ok=True)
    chapters = derive_chapters(doc)

    print(f"{'slug':10} {'pages':>5} {'hd':>4} {'code':>5} {'note':>5} {'cap':>4} {'tbl':>4} {'para':>5} {'words':>7}")
    for slug, title, num, start0, end0 in chapters:
        if num is None and not args.front_matter:
            continue
        if args.chapters and (num not in args.chapters):
            continue
        text, st = render_chapter(doc, start0, end0, title, num)
        path = args.out / f"{slug}.md"
        path.write_text(text, encoding="utf-8")
        print(f"{slug:10} {st.pages:>5} {st.headings:>4} {st.code_blocks:>5} "
              f"{st.callouts:>5} {st.captions:>4} {st.tables:>4} {st.paragraphs:>5} {st.words:>7}  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
