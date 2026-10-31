#!/usr/bin/env python3
"""Render the selected result scene from retained art and timing evidence.

Requires existing Pillow, librsvg's rsvg-convert and Fontconfig executables.
No generation service, network request or installation is used. Run from any
working directory. --check regenerates in temporary storage and compares the
canonical SVG, PNG and TXT byte-for-byte without changing package files.
Pass --overwrite to rebuild those three files, the 400/312 px PNG previews and
images/result-scene/receipt.json. PNG byte equality requires a matching renderer
and font stack; see the public README for the versions used.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import io
import math
import json
import os
from pathlib import Path
import shutil
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


def require(condition: bool, message: str) -> None:
    """Keep data and layout guards active under python -O."""
    if not condition:
        raise ValueError(message)


def render(overwrite: bool = False, check: bool = False):
    require(not (overwrite and check), "--check and --overwrite are mutually exclusive")
    tokens = json.loads(TOKENS_PATH.read_text())
    require(tokens["typography"]["font_family"] == "Inter", "The retained font family must be Inter")
    palette = tokens["colors"]
    roles = tokens["typography"]["roles_px"]
    margin = tokens["spacing"]["safe_margin_px"]
    # At a 312 px image width, 48 source pixels become 12.48 CSS pixels.
    body = roles["body"] + tokens["spacing"]["scale_px"][0] // 2
    require(body == 48 and margin == 84, "The retained layout requires 48 px body text and 84 px margins")
    data = json.loads(SUMMARY.read_text())
    cells = [c for c in data["cells"] if c["size"] == 64 and c["mode"] == "changing"]
    require(len(cells) == 1, "Expected exactly one changing-set, 64-token cell")
    cell = cells[0]
    baseline = cell["baseline_median_wall_ms"]
    candidate = cell["candidate_median_wall_ms"]
    ratio = cell["speedup"]
    require(data["model_measured"] is False and data["serving_measured"] is False,
            "The retained result must not claim model or serving measurements")
    require(all(isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(value) and value > 0
                for value in [baseline, candidate, ratio]),
            "Timing values and speedup must be positive finite numbers")
    require(round(baseline, 3) == 5.399 and round(candidate, 3) == 0.421,
            "Timings do not match the retained result displayed by this scene")
    require(round(ratio, 2) == 12.84, "Speedup does not match the retained result")
    inputs = {relative(p): sha(p) for p in [TOKENS_PATH, FONT, ART, SUMMARY]}
    outputs = [OUTPUT.with_suffix(s) for s in [".svg", ".png", ".txt"]]
    outputs += [IMAGES / f"result-scene-{width}px.png" for width in [400, 312]]
    receipt_path = HERE / "receipt.json"
    if not overwrite and not check:
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
        require(";" not in value, "Scene text must not contain semicolons")
        face = ImageFont.truetype(str(FONT), size)
        face.set_variation_by_axes([14, weight])
        width = face.getlength(value)
        left = x if anchor == "start" else x - width if anchor == "end" else x - width / 2
        require(left >= margin, f"Text extends beyond the left margin: {value!r} at {left}")
        require(left + width <= W - margin + 1, f"Text extends beyond the right margin: {value!r} at {left + width}")
        require(y <= H - margin, f"Text extends beyond the bottom margin: {value!r} at {y}")
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
    ET.fromstring(svg)
    rendered = {
        OUTPUT.with_suffix(".svg"): svg.encode("utf-8"),
        OUTPUT.with_suffix(".txt"): ("\n".join(item["text"] for item in text_items) + "\n").encode("utf-8"),
    }
    # All rendering, resource resolution and Fontconfig caches stay in temporary
    # storage. Relative SVG resource paths match the final images/ layout.
    with tempfile.TemporaryDirectory(prefix="result-scene-render-") as folder:
        temp = Path(folder)
        temp_images = temp / "images"
        (temp_images / "result-scene").mkdir(parents=True)
        (temp_images / "fonts").mkdir()
        temp_font = temp_images / "fonts/Inter.ttf"
        shutil.copyfile(FONT, temp_font)
        shutil.copyfile(ART, temp_images / "result-scene/matrix-cut-art.png")
        temp_svg = temp_images / "result-scene.svg"
        temp_svg.write_bytes(rendered[OUTPUT.with_suffix(".svg")])
        temp_png = temp_images / "result-scene.png"
        config = temp / "fonts.conf"
        config.write_text(f'<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><dir>{html.escape(str(temp_font.parent))}</dir><cachedir>{html.escape(str(temp / "cache"))}</cachedir></fontconfig>')
        env = {**os.environ, "FONTCONFIG_FILE": str(config)}
        matched = subprocess.check_output(["fc-match", "-f", "%{file}\n%{family}\n", "Inter"], env=env, text=True).splitlines()
        require(bool(matched) and Path(matched[0]).resolve() == temp_font.resolve(),
                "Inter source file did not resolve")
        subprocess.run(["rsvg-convert", "--output", str(temp_png), str(temp_svg)], env=env, check=True)
        rendered[OUTPUT.with_suffix(".png")] = temp_png.read_bytes()
    with Image.open(io.BytesIO(rendered[OUTPUT.with_suffix(".png")])) as image:
        require(image.size == (W, H), f"Unexpected rendered dimensions: {image.size}")
        for width in [400, 312]:
            preview = io.BytesIO()
            image.resize((width, round(H * width / W)), Image.Resampling.LANCZOS).save(preview, format="PNG")
            rendered[IMAGES / f"result-scene-{width}px.png"] = preview.getvalue()
    require(inputs == {relative(p): sha(p) for p in [TOKENS_PATH, FONT, ART, SUMMARY]},
            "Retained inputs changed during rendering")
    if check:
        canonical = [OUTPUT.with_suffix(s) for s in [".svg", ".png", ".txt"]]
        different = [relative(p) for p in canonical if not p.is_file() or p.read_bytes() != rendered[p]]
        require(not different, "Regeneration differs for canonical outputs: " + ", ".join(different)
                + ". PNG byte equality depends on the renderer and font stack; compare the documented versions.")
        print(json.dumps({"state": "canonical outputs match byte-for-byte", "checked": [relative(p) for p in canonical], "package_files_written": False}))
        return
    for path in outputs:
        path.write_bytes(rendered[path])
    receipt = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "state": "rendered from retained inputs. Review and platform preview evidence are retained separately",
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--overwrite", action="store_true", help="Rebuild canonical SVG/PNG/TXT, 400/312 px previews and receipt.json")
    mode.add_argument("--check", action="store_true", help="Compare canonical SVG/PNG/TXT byte-for-byte using temporary storage; write no package files")
    args = parser.parse_args()
    render(overwrite=args.overwrite, check=args.check)
