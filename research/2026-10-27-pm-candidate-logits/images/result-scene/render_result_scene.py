#!/usr/bin/env python3
"""Render the selected result scene from retained art and timing evidence.

Requires existing Pillow, librsvg's rsvg-convert and Fontconfig executables.
No generation service, network request or installation is used. Run from any
working directory. Outputs are new result-scene assets in the package's images
directory. Pass --overwrite only to rebuild these owned outputs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET

from PIL import Image, ImageFont


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
IMAGES = PACKAGE / "images"
TOKENS_PATH = HERE / "visual-tokens.json"
FONT = IMAGES / "fonts/Inter.ttf"
ART = HERE / "matrix-cut-art.png"
SUMMARY = PACKAGE / "evidence/summary.json"
OUTPUT = IMAGES / "result-scene"
W, H = 1200, 1500


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path: Path) -> str:
    return path.relative_to(PACKAGE).as_posix()


def render(overwrite: bool = False):
    tokens = json.loads(TOKENS_PATH.read_text())
    assert tokens["typography"]["font_family"] == "Inter"
    palette = tokens["colors"]
    roles = tokens["typography"]["roles_px"]
    margin = tokens["spacing"]["safe_margin_px"]
    # At a 312 px image width, 48 source pixels become 12.48 CSS pixels.
    body = roles["body"] + tokens["spacing"]["scale_px"][0] // 2
    assert body == 48 and margin == 84
    data = json.loads(SUMMARY.read_text())
    cell = next(c for c in data["cells"] if c["size"] == 64 and c["mode"] == "changing")
    baseline = cell["baseline_median_wall_ms"]
    candidate = cell["candidate_median_wall_ms"]
    ratio = cell["speedup"]
    assert data["model_measured"] is False and data["serving_measured"] is False
    assert round(baseline, 3) == 5.399 and round(candidate, 3) == 0.421
    assert round(ratio, 2) == 12.84
    inputs = {relative(p): sha(p) for p in [TOKENS_PATH, FONT, ART, SUMMARY]}
    outputs = [OUTPUT.with_suffix(s) for s in [".svg", ".png", ".txt"]]
    outputs += [IMAGES / f"result-scene-{width}px.png" for width in [400, 312]]
    receipt_path = HERE / "receipt.json"
    if not overwrite:
        existing = [relative(p) for p in outputs + [receipt_path] if p.exists()]
        if existing:
            raise SystemExit("Owned outputs already exist. Use --overwrite to rebuild: " + ", ".join(existing))
    items = []
    text_items = []

    def rect(x, y, width, height, fill):
        items.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}"/>')

    def line(x1, y1, x2, y2, width=2):
        items.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{palette["ink"]}" stroke-width="{width}"/>')

    def text(x, y, value, size=body, weight=400, fill=None, anchor="start"):
        assert ";" not in value
        face = ImageFont.truetype(str(FONT), size)
        face.set_variation_by_axes([14, weight])
        width = face.getlength(value)
        left = x if anchor == "start" else x - width if anchor == "end" else x - width / 2
        assert left >= margin, (value, left)
        assert left + width <= W - margin + 1, (value, left + width)
        assert y <= H - margin, (value, y)
        text_items.append({"text": value, "font_size": size, "font_weight": weight, "baseline": [x, y], "width": round(width, 3), "size_at_312px_image_width": size * 312 / W})
        items.append(f'<text x="{x}" y="{y}" font-family="Inter" font-size="{size}" font-weight="{weight}" fill="{fill or palette["ink"]}" text-anchor="{anchor}">{html.escape(value)}</text>')

    def arrow(d):
        items.append(f'<path d="{d}" fill="none" stroke="{palette["ink"]}" stroke-width="3" marker-end="url(#arrow)"/>')

    rect(0, 0, W, H, palette["background"])
    text(margin, 133, "Select allowed rows", roles["heading"], 700)
    text(margin, 211, "before scoring", roles["heading"], 700)
    # Crop only the conceptual scene's outer matrix and shadow. The selection
    # action stays visible. This image does not encode measured dimensions.
    items.append(f'<image x="84" y="165" width="1032" height="{1032 * 941 / 1672}" href="result-scene/matrix-cut-art.png" clip-path="url(#scene-clip)"/>')
    text(100, 330, "Full weights", body, 600, palette["blue"])
    arrow("M 390 342 L 555 362")
    text(100, 480, "Allowed rows", body, 600)
    arrow("M 420 491 L 505 520")
    text(margin, 709, "Conceptual view · rows not to scale", body)
    text(margin, 779, f"{ratio:.2f}× faster · output head only", body, 700)
    text(margin, 837, "Median per warm 16-step block", body)
    text(margin, 905, "Full + filter", body, 600, palette["blue"])
    text(W - margin, 905, f"{baseline:.3f} ms", body, 600, anchor="end")
    rect(margin, 922, (W - margin * 2) * baseline / 6, 36, palette["blue"])
    text(margin, 1015, "Allowed rows only", body, 600)
    text(W - margin, 1015, f"{candidate:.3f} ms", body, 600, anchor="end")
    rect(margin, 1032, (W - margin * 2) * candidate / 6, 36, palette["orange"])
    axis_y = 1100
    line(margin, axis_y, W - margin, axis_y)
    for value in [0, 2, 4, 6]:
        x = margin + value / 6 * (W - margin * 2)
        line(x, axis_y, x, axis_y + 9)
        text(x, axis_y + 54, f"{value} ms" if value == 6 else str(value), body, anchor="start" if value == 0 else "end" if value == 6 else "middle")
    for i, value in enumerate([
        "Synthetic strict FP32 · B200",
        "64 allowed tokens shared by batch",
        "New set each step · weight gather included",
        "Whole-model gains unmeasured",
    ]):
        text(margin, 1220 + i * 56, value, body, 500 if i == 3 else 400)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1500" viewBox="0 0 1200 1500">
<title>Select allowed rows before scoring</title>
<desc>Conceptual row selection above a zero-based comparison of synthetic strict-FP32 B200 output-head timings. Matrix dimensions and measured row count are not encoded by the illustration.</desc>
<defs>
<style>@font-face {{ font-family: Inter; src: url("fonts/Inter.ttf") format("truetype"); font-weight: 100 900; }}</style>
<clipPath id="scene-clip"><rect x="84" y="245" width="1032" height="410"/></clipPath>
<marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="{palette["ink"]}"/></marker>
</defs>
{''.join(items)}
</svg>'''
    svg_path = OUTPUT.with_suffix(".svg")
    svg_path.write_text(svg)
    ET.parse(svg_path)
    OUTPUT.with_suffix(".txt").write_text("\n".join(item["text"] for item in text_items) + "\n")
    # Fontconfig is process-local. Its temporary absolute paths are neither
    # shipped nor written into the public renderer or provenance receipt.
    with tempfile.TemporaryDirectory(prefix="result-scene-font-") as folder:
        temp = Path(folder)
        config = temp / "fonts.conf"
        config.write_text(f'<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><dir>{html.escape(str(FONT.parent))}</dir><cachedir>{html.escape(str(temp / "cache"))}</cachedir></fontconfig>')
        env = {**os.environ, "FONTCONFIG_FILE": str(config)}
        matched = subprocess.check_output(["fc-match", "-f", "%{file}\n%{family}\n", "Inter"], env=env, text=True).splitlines()
        assert Path(matched[0]).resolve() == FONT.resolve(), "Inter source file did not resolve"
        subprocess.run(["rsvg-convert", "--output", str(OUTPUT.with_suffix(".png")), str(svg_path)], env=env, check=True)
    with Image.open(OUTPUT.with_suffix(".png")) as image:
        assert image.size == (W, H)
        for width in [400, 312]:
            image.resize((width, round(H * width / W)), Image.Resampling.LANCZOS).save(IMAGES / f"result-scene-{width}px.png")
    assert inputs == {relative(p): sha(p) for p in [TOKENS_PATH, FONT, ART, SUMMARY]}
    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state": "rendered local asset, independent review and platform preview pending",
        "family": "scene_evidence_split",
        "dimensions_px": [W, H],
        "renderer": relative(Path(__file__).resolve()),
        "renderer_sha256": sha(Path(__file__).resolve()),
        "illustration": {"source": relative(ART), "retained_original": "images/creative-v2/matrix-cut-art.png", "provider": "retained built-in image_gen illustration", "gamma_generated": False, "new_generation": False, "source_file_unchanged": True, "conceptual_only": True, "display_crop": "Outer matrix and shadow cropped to preserve room for readable text", "svg_relative_href": "result-scene/matrix-cut-art.png"},
        "style_authority": {"name": "Whiteboard", "shared_source": "style/visual-tokens.json", "packaged_exact_copy": relative(TOKENS_PATH), "shared_source_sha256": inputs[relative(TOKENS_PATH)], "font_shared_source": "style/fonts/Inter.ttf", "package_font": relative(FONT), "shared_and_package_font_sha256": inputs[relative(FONT)]},
        "inputs_sha256": inputs,
        "unchanged_inputs_after_render": True,
        "source_cell": cell,
        "displayed": {"baseline_ms": f"{baseline:.3f}", "candidate_ms": f"{candidate:.3f}", "speedup": f"{ratio:.2f}×", "unrounded_speedup": ratio, "timing_unit": "median milliseconds per warm block of 16 projection/filter/argmax steps"},
        "bar_geometry": {"axis_minimum_ms": 0, "axis_maximum_ms": 6, "linear": True, "source_values_used_unrounded": [baseline, candidate], "flat_2d": True},
        "text": text_items,
        "phone_size_calculation": {"image_width_css_px": 312, "minimum_source_text_px": min(t["font_size"] for t in text_items), "minimum_text_css_px": min(t["size_at_312px_image_width"] for t in text_items), "limit": "Calculated image scaling only. Actual platform preview remains required."},
        "svg_resource_resolution": {"base": "images/result-scene.svg file location", "image_href": "result-scene/matrix-cut-art.png", "font_css_href": "fonts/Inter.ttf", "renderer_font_resolution": "Exact package Inter.ttf verified through process-local Fontconfig"},
        "outputs_sha256": {relative(p): sha(p) for p in outputs},
        "omissions": ["No model or serving measurement", "No actual platform upload or preview", "No new provider job", "No publication or scheduling"],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": relative(OUTPUT.with_suffix('.png')), "phone_text_css_px_at_312px_image_width": receipt["phone_size_calculation"]["minimum_text_css_px"], "input_hashes_unchanged": True}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true", help="Rebuild only the owned result-scene outputs")
    render(parser.parse_args().overwrite)
