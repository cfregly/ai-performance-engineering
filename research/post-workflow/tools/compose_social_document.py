#!/usr/bin/env python3
"""Compose unchanged square Gamma cards into a viewer-safe rectangular PDF.

Use the bundled Python runtime; Pillow, ReportLab and pypdf must already exist.
The only outputs are the requested new PDF and its sibling .composition.json.
Neither an existing PDF nor an existing receipt is overwritten. No raster is
resized, cropped, annotated or recolored: each source's decoded RGB bytes are
embedded losslessly and checked in the completed PDF before anything is saved.

Example:
  python3 tools/compose_social_document.py --package /absolute/package \
    --output images/attention-resource-guide.pdf --scale .78 --top-clearance .12

The image retains its width-based scale and equal side/bottom bands. Extra page
height above it clears the platform's title bar without shrinking the artwork.
The default extra height comes from the shared visual tokens. An explicit
--top-clearance 0 retains the legacy square-page composition.

The receipt identifies the canonical embedded PNG, not a full-page rendering.
It records that PNG's exact placement inside the PDF page. An uploaded platform
preview still needs actual geometry, controls and readability inspection.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
from typing import Any

try:
    from PIL import Image
    from pypdf import PdfReader
    from pypdf.generic import ContentStream
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
except ImportError as exc:
    raise SystemExit(
        f"Required PDF dependency is unavailable: {exc}. Use the existing bundled "
        "Python runtime. This tool does not install dependencies."
    ) from exc


PAGE_POINTS = 540.0
DEFAULT_SCALE = 0.78
STYLE_TOKENS = Path(__file__).resolve().parents[1] / "style/visual-tokens.json"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _member(package: Path, relative: Any, label: str) -> Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative
            or "\0" in relative or Path(relative).is_absolute()
            or any(part in {"", ".", ".."} for part in relative.split("/"))):
        raise ValueError(f"{label} must be a safe package-relative path")
    path = package
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink():
            raise ValueError(f"{label} must not use symlinks: {relative}")
    if not path.resolve().is_relative_to(package):
        raise ValueError(f"{label} escapes the package: {relative}")
    return path


@dataclass(frozen=True)
class Source:
    asset_index: int
    title: str
    reference: dict[str, Any]
    path: Path
    png: bytes
    rgb: bytes
    side: int


def _sources(package: Path, asset_index: str = "images/asset-index.json") -> tuple[bytes, list[Source]]:
    index_path = _member(package, asset_index, "asset index")
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    assets = index.get("assets") if isinstance(index, dict) else None
    if not isinstance(assets, list) or not 1 <= len(assets) <= 1000:
        raise ValueError("asset index requires between 1 and 1000 assets")
    result = []
    for position, asset in enumerate(assets, 1):
        if not isinstance(asset, dict) or type(asset.get("index")) is not int or asset["index"] != position:
            raise ValueError("assets must retain contiguous one-based index order")
        selected = asset.get("selected")
        if not isinstance(selected, dict):
            raise ValueError(f"asset {position} requires a selected PNG reference")
        path = _member(package, selected.get("path"), f"asset {position} selected PNG")
        expected = selected.get("sha256")
        if not isinstance(expected, str) or not SHA256.fullmatch(expected):
            raise ValueError(f"asset {position} requires a lowercase SHA-256 digest")
        png = path.read_bytes()
        if not png.startswith(b"\x89PNG\r\n\x1a\n") or _sha(png) != expected:
            raise ValueError(f"asset {position} selected PNG hash/signature mismatch: {selected['path']}")
        with Image.open(io.BytesIO(png)) as image:
            image.load()
            if image.mode != "RGB":
                raise ValueError(f"asset {position} must be RGB to preserve its exact decoded pixels")
            width, height = image.size
            if width != height or width < 1:
                raise ValueError(f"asset {position} selected PNG must be square")
            if (type(selected.get("width")) is not int or type(selected.get("height")) is not int
                    or (selected["width"], selected["height"]) != (width, height)):
                raise ValueError(f"asset {position} selected PNG dimensions do not match the index")
            rgb = image.tobytes()
        result.append(Source(position, str(asset.get("title", "")),
                             {key: selected[key] for key in ("path", "sha256", "width", "height")},
                             path, png, rgb, width))
    return index_bytes, result


def _multiply(left: tuple[float, ...], right: list[float]) -> tuple[float, ...]:
    a, b, c, d, e, f = left
    g, h, i, j, k, l = right
    return (a*g+c*h, b*g+d*h, a*i+c*j, b*i+d*j, a*k+c*l+e, b*k+d*l+f)


def _verify_pdf(data: bytes, sources: list[Source], scale: float, top_clearance: float = 0.0) -> None:
    """Check actual pages, decoded embedded pixels, and effective image CTMs."""
    reader = PdfReader(io.BytesIO(data), strict=True)
    if len(reader.pages) != len(sources):
        raise ValueError("composed PDF page count does not match the asset index")
    side = PAGE_POINTS * scale
    margin = (PAGE_POINTS - side) / 2
    page_height = PAGE_POINTS + PAGE_POINTS * top_clearance
    expected_page = (0.0, 0.0, PAGE_POINTS, page_height)
    expected_ctm = (side, 0.0, 0.0, side, margin, margin)
    for number, (page, source) in enumerate(zip(reader.pages, sources), 1):
        if (page.rotation != 0
                or any(abs(float(actual)-expected) > 1e-6
                       for box in (page.mediabox, page.cropbox)
                       for actual, expected in zip(box, expected_page))):
            raise ValueError(f"PDF page {number} has an unexpected page box")
        xobjects = page["/Resources"]["/XObject"].get_object()
        images = [(str(name), obj.get_object()) for name, obj in xobjects.items()
                  if obj.get_object().get("/Subtype") == "/Image"]
        if len(images) != 1:
            raise ValueError(f"PDF page {number} must contain exactly one image")
        name, embedded = images[0]
        if (embedded.get("/ColorSpace") != "/DeviceRGB"
                or embedded.get("/BitsPerComponent") != 8
                or (embedded.get("/Width"), embedded.get("/Height")) != (source.side, source.side)
                or "/SMask" in embedded or "/Mask" in embedded
                or embedded.get_data() != source.rgb):
            raise ValueError(f"PDF page {number} does not preserve the source RGB pixels")
        ctm = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
        stack = []
        placements = []
        for operands, operator in ContentStream(page.get_contents(), reader).operations:
            if operator == b"q":
                stack.append(ctm)
            elif operator == b"Q":
                if not stack:
                    raise ValueError(f"PDF page {number} has an unbalanced graphics state")
                ctm = stack.pop()
            elif operator == b"cm":
                ctm = _multiply(ctm, [float(value) for value in operands])
            elif operator == b"Do":
                placements.append((str(operands[0]), ctm))
        if (stack or len(placements) != 1 or placements[0][0] != name
                or any(abs(actual-expected) > 1e-6
                       for actual, expected in zip(placements[0][1], expected_ctm))):
            raise ValueError(f"PDF page {number} image placement does not match the requested scale and clearance")


def compose(package: Path | str, output: str, scale: float = DEFAULT_SCALE,
            top_clearance: float | None = None, asset_index: str = "images/asset-index.json") -> dict[str, Any]:
    package = Path(package)
    if not package.is_absolute() or not package.is_dir():
        raise ValueError("--package must name an existing absolute package directory")
    package = package.resolve()
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale < 1:
        raise ValueError("--scale must be finite and strictly between 0 and 1")
    clearance_source = "explicit_argument"
    style_reference = None
    if top_clearance is None:
        try:
            token_bytes = STYLE_TOKENS.read_bytes()
            top_clearance = json.loads(token_bytes)["spacing"]["pdf_extra_top_clearance_ratio"]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"shared PDF top-clearance token is unavailable: {exc}") from exc
        clearance_source = "style/visual-tokens.json:spacing.pdf_extra_top_clearance_ratio"
        style_reference = {"path": "style/visual-tokens.json", "sha256": _sha(token_bytes)}
    if (type(top_clearance) not in (int, float) or not math.isfinite(top_clearance)
            or top_clearance < 0 or not math.isfinite(PAGE_POINTS + PAGE_POINTS * top_clearance)):
        raise ValueError("--top-clearance must be a finite nonnegative page-width ratio")
    destination = _member(package, output, "output")
    if Path(output).parts[0] != "images" or destination.suffix.lower() != ".pdf":
        raise ValueError("--output must be a new PDF under images/")
    receipt_path = destination.with_suffix(".composition.json")
    receipt_relative = receipt_path.relative_to(package).as_posix()
    _member(package, receipt_relative, "composition receipt")
    if destination.exists() or receipt_path.exists():
        raise ValueError("output PDF and composition receipt must be new; existing files are never overwritten")

    # All inputs are validated before creating output directories or files.
    index_bytes, sources = _sources(package, asset_index)
    side = PAGE_POINTS * scale
    margin = (PAGE_POINTS-side) / 2
    extra_top = PAGE_POINTS * top_clearance
    page_height = PAGE_POINTS + extra_top
    stream = io.BytesIO()
    pdf = canvas.Canvas(stream, pagesize=(PAGE_POINTS, page_height), pageCompression=1, invariant=1)
    pdf.setTitle(destination.stem.replace("-", " "))
    for source in sources:
        pdf.setFillColorRGB(1, 1, 1)
        pdf.rect(0, 0, PAGE_POINTS, page_height, fill=1, stroke=0)
        pdf.drawImage(ImageReader(io.BytesIO(source.png)), margin, margin, width=side, height=side)
        pdf.showPage()
    pdf.save()
    data = stream.getvalue()
    _verify_pdf(data, sources, scale, top_clearance)
    if _member(package, asset_index, "asset index").read_bytes() != index_bytes:
        raise ValueError("asset index changed while composing the PDF")
    for source in sources:
        if source.path.read_bytes() != source.png:
            raise ValueError(f"source PNG changed while composing: {source.reference['path']}")

    receipt = {
        "schema_version": 1,
        "kind": "lossless_square_image_pdf_composition",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "document": {"path": output, "sha256": _sha(data), "page_count": len(sources)},
        "asset_index": {"path": asset_index, "sha256": _sha(index_bytes)},
        "composition_receipt": receipt_relative,
        "page_size_points": {"width": PAGE_POINTS, "height": page_height},
        "image_scale": scale,
        "extra_top_clearance_ratio": top_clearance,
        "extra_top_clearance_points": extra_top,
        "top_clearance_source": clearance_source,
        "shared_visual_tokens": style_reference,
        "outer_band_points": {"top": margin + extra_top, "bottom": margin,
                              "left": margin, "right": margin},
        "background_rgb": [1, 1, 1],
        "pages": [{
            "page_index": source.asset_index,
            "asset_index": source.asset_index,
            "title": source.title,
            "source_png": source.reference,
            "decoded_rgb_sha256": _sha(source.rgb),
            "page_box_points": [0, 0, PAGE_POINTS, page_height],
            "image_box_points": [margin, margin, side, side],
            "image_transform": [side, 0, 0, side, margin, margin],
            "embedded_rgb_pixels_verified": True,
        } for source in sources],
        "validation": {
            "actual_pdf_page_count": len(sources),
            "all_embedded_rgb_pixels_identical": True,
            "all_image_transforms_centered": top_clearance == 0,
            "all_image_transforms_horizontally_centered": True,
            "all_image_placements_verified": True,
            "crop_or_raster_resampling": "none",
            "preview_source_note": "source_png is the canonical image actually embedded inside the page, not a full-page raster. Use image_box_points to map its actual preview geometry.",
            "limits": "This verifies composition and source-pixel identity only. Actual uploaded pages, overlay controls and phone readability still require platform inspection.",
        },
    }
    receipt_bytes = (json.dumps(receipt, indent=2, ensure_ascii=False)+"\n").encode("utf-8")
    destination.parent.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for path, content in ((destination, data), (receipt_path, receipt_bytes)):
            with path.open("xb") as target:
                created.append(path)
                target.write(content)
    except OSError:
        for path in reversed(created):
            path.unlink(missing_ok=True)
        raise
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--asset-index", default="images/asset-index.json",
                        help="package-relative selected-card index; the full gallery stays intact")
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--top-clearance", type=float, default=None,
                        help="extra page height above the image as a page-width ratio; "
                             "defaults to the shared visual token, or use 0 for a square page")
    args = parser.parse_args(argv)
    try:
        receipt = compose(args.package, args.output, args.scale, args.top_clearance, args.asset_index)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"Document composition failed: {exc}\n")
    print(json.dumps({"document": receipt["document"], "receipt": receipt["composition_receipt"],
                      "scale": receipt["image_scale"],
                      "top_clearance": receipt["extra_top_clearance_ratio"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
