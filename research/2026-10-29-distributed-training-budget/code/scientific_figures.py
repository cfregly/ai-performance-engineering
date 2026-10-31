#!/usr/bin/env python3
"""Render only the selected placement and ordered-floor figures.

Uses the retained workbook, shared visual tokens, Inter and the retained barrier
illustration. PNG rendering needs an existing rsvg-convert executable. No network
or GPU is used. Run with --check for exact SVG, PNG and text regeneration.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

import visual_elements

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images/scientific"
TOKENS = OUT / "visual-tokens.json"
FONT = ROOT / "images/fonts/Inter.ttf"
SOURCE = ROOT / "evidence/reader-summary.json"
NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", NS)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def topology(source, colors):
    """Preserve the retained topology geometry and make its identity explicit."""
    element = ET.fromstring(visual_elements.topology_b(source).validate())
    root = ET.Element(f"{{{NS}}}svg", width="1200", height="1200", viewBox="0 0 1200 1200", role="img")
    ET.SubElement(root, f"{{{NS}}}title").text = "Placement B: pair replicas within each rack"
    ET.SubElement(root, f"{{{NS}}}desc").text = element.find(f"{{{NS}}}desc").text
    ET.SubElement(root, f"{{{NS}}}rect", width="1200", height="1200", fill=colors["background"])
    heading = ET.SubElement(root, f"{{{NS}}}text", x="600", y="100", **{"font-family": "Inter", "font-size": "54", "font-weight": "600", "text-anchor": "middle", "fill": colors["ink"]})
    heading.text = "Pair replicas within each rack"
    root.append(copy.deepcopy(element.find(f"{{{NS}}}defs")))
    group = ET.SubElement(root, f"{{{NS}}}g", transform="translate(150 140) scale(0.9)")
    for child in element:
        tag = child.tag.rsplit("}", 1)[-1]
        if tag in {"title", "desc", "defs"} or (tag == "rect" and child.get("width") == "1000"):
            continue
        group.append(copy.deepcopy(child))
    labels = [e.text for e in root.iter(f"{{{NS}}}text")]
    assert "Replica 0" in labels and "Replica 1" in labels
    assert not any(s in {"R0", "R1", "14"} for s in labels)
    return (ET.tostring(root, encoding="utf-8") + b"\n", {"minimum_effective_font_px": 46.8, "geometry": "Unchanged workbook-derived topology, uniformly scaled by 0.9"})


def floors(source, colors):
    """Preserve exact zero-based, additive phase geometry and retained artwork."""
    bg, ink, blue, orange = (colors[k] for k in ("background", "ink", "blue", "orange"))
    parts = [f'<svg xmlns="{NS}" width="1200" height="1200" viewBox="0 0 1200 1200" role="img">',
             '<title>Two ordered lower bounds</title>',
             '<desc>Hypothetical full DP all-reduce before norm, then clipping and FP32 Muon. The busiest rank updates its own experts and the gathered attention matrices. Blue and orange show ordered phase lower bounds under assumed service rates, not runtimes.</desc>',
             f'<rect width="1200" height="1200" fill="{bg}"/>',
             f'<g font-family="Inter" fill="{ink}">']
    labels = []

    def text(x, y, value, size=48, anchor="middle", color=ink, weight=400):
        labels.append(value)
        assert size >= 48
        parts.append(f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{escape(value)}</text>')

    def rect(x, y, width, height, color):
        parts.append(f'<rect x="{x:g}" y="{y:g}" width="{width:.9f}" height="{height:g}" fill="{color}"/>')

    def line(x1, y1, x2, y2, dashed=False):
        dash = ' stroke-dasharray="12 10"' if dashed else ''
        parts.append(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{ink}" stroke-width="3"{dash}/>')

    text(600, 112, "Two ordered lower bounds", 58, weight=600)
    text(600, 175, "Hypothetical plan · ms per update")
    for x, label, color in ((220, "Before clip", blue), (710, "After clip", orange)):
        rect(x, 212, 32, 32, color)
        text(x + 48, 248, label, anchor="start")
    scene = base64.b64encode((OUT / "ordered-barrier-scene.png").read_bytes()).decode()
    parts.append(f'<image x="90" y="270" width="1020" height="193.263157895" href="data:image/png;base64,{scene}"/>')
    x0, scale, budget = 150, 880 / 300, 150
    text(600, 510, "Illustrative 150 ms budget")
    line(x0 + budget * scale, 520, x0 + budget * scale, 780, True)
    geometry = []
    for y, name in ((540, "A"), (675, "B")):
        row = source["traffic_and_screens"][name]["floors_ms"]
        pre, post = row["pre_clipping_phase"], row["post_clipping_phase"]
        total = pre + post
        assert abs(total - row["necessary_step_floor"]) < 1e-6
        rect(x0, y, pre * scale, 85, blue)
        rect(x0 + pre * scale, y, post * scale, 85, orange)
        text(108, y + 61, name, 58, weight=600)
        text(x0 + pre * scale / 2, y + 61, f"{pre:.2f}", 58, color=bg)
        text(x0 + (pre + post / 2) * scale, y + 61, f"{post:.2f}", 58, color=ink)
        text(x0 + total * scale + 18, y + 61, f"{total:.2f}", 58, "start", weight=600)
        geometry.append({"placement": name, "pre_ms": pre, "post_ms": post, "total_ms": total, "px_per_ms": scale, "pre_width": pre * scale, "post_width": post * scale})
    line(x0, 530, x0, 790)
    line(x0, 790, 1030, 790)
    for value in (0, 150, 300):
        text(x0 + value * scale, 850, str(value))
    caps = source["assumed_capabilities"]
    for y, value in zip((915, 965, 1015, 1065, 1115), (
        "Full DP all-reduce before norm",
        "FP32 Muon · one attention owner",
        f"Assumed {caps['Muon_FP32_TF_s_GPU']} TFLOP/s per GPU",
        f"Rack {caps['rack_uplink_GB_s_each_direction']} / node {caps['node_NIC_GB_s_each_direction']} GB/s per direction",
        "Balanced routing · not runtimes",
    )):
        text(600, y, value)
    parts.append("</g></svg>\n")
    data = "\n".join(parts).encode()
    ET.fromstring(data)
    assert "150 ms target" not in labels and "16" not in labels
    return data, {"minimum_effective_font_px": 48, "geometry": geometry}


def png(data, font_config):
    env = dict(os.environ, FONTCONFIG_FILE=str(font_config), LC_ALL="C")
    return subprocess.run(["rsvg-convert", "--width", "2400", "--height", "2400"], input=data, capture_output=True, check=True, env=env).stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    tokens = json.loads(TOKENS.read_text())
    shared = ROOT.parents[1] / "style/visual-tokens.json"
    if shared.exists():
        assert shared.read_bytes() == TOKENS.read_bytes(), "Packaged visual tokens differ from shared authority"
    assert tokens["typography"]["font_family"] == "Inter"
    source = json.loads(SOURCE.read_text())
    receipt = {"source_sha256": digest(SOURCE.read_bytes()), "tokens_sha256": digest(TOKENS.read_bytes()), "font_sha256": digest(FONT.read_bytes()), "renderer": subprocess.check_output(["rsvg-convert", "--version"], text=True).splitlines()[0], "figures": []}
    with tempfile.TemporaryDirectory(prefix="selected-scientific-") as temp:
        config = Path(temp) / "fonts.conf"
        config.write_text(f'<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><dir>{escape(str(FONT.parent))}</dir><cachedir>{escape(temp)}</cachedir></fontconfig>')
        for name, builder in (("14-topology-b", topology), ("16-ordered-floors", floors)):
            svg, details = builder(source, tokens["colors"])
            labels = [e.text for e in ET.fromstring(svg).iter(f"{{{NS}}}text") if e.text]
            outputs = {"svg": svg, "png": png(svg, config), "txt": ("\n".join(labels) + "\n").encode()}
            for extension, data in outputs.items():
                path = OUT / f"{name}.{extension}"
                if args.check:
                    assert path.read_bytes() == data, f"Regeneration differs: {path}"
                else:
                    path.write_bytes(data)
            minimum = details["minimum_effective_font_px"] * 312 / 1200
            assert minimum >= tokens["typography"]["minimum_phone_px"]["material_condition"]
            receipt["figures"].append({"name": name, "outputs": {extension: digest(data) for extension, data in outputs.items()}, "minimum_phone_font_px_at_312_width": minimum, **details})
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"{'Verified' if args.check else 'Wrote'} two selected SVG/PNG figures; material labels are at least 12 px at 312 px width.")


if __name__ == "__main__":
    main()
