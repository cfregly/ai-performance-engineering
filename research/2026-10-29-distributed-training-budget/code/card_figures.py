#!/usr/bin/env python3
"""Generate four self-contained scientific SVG cards from reader-summary.json.

Run with --check to verify exact retained bytes. No packages or network needed.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/reader-summary.json"
OUT = ROOT / "images/card-elements"
BG, INK, BLUE, ORANGE = "#FBFBF8", "#202326", "#3478BA", "#E87823"


class Card:
    def __init__(self, title, description):
        self.parts = [
            '<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="1000" viewBox="0 0 1000 1000" role="img" aria-labelledby="title desc">',
            f'<title id="title">{html.escape(title)}</title>',
            f'<desc id="desc">{html.escape(description)}</desc>',
            '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="9" refY="5" orient="auto" markerUnits="userSpaceOnUse"><path d="M1 1L9 5L1 9" fill="none" stroke="#202326" stroke-width="2"/></marker></defs>',
            f'<rect width="1000" height="1000" fill="{BG}"/>',
            f'<g font-family="Inter" font-weight="400" fill="{INK}">',
        ]
        self.bounds = []
        self.fonts = []
        self.geometry = []

    def bound(self, x, y, w, h, kind):
        self.bounds.append((x, y, w, h, kind))

    def text(self, x, y, text, size=36, anchor="start", color=INK):
        width = sum(.36 if c in " ilI1.,:;|" else .68 if c in "MW@" else .59 for c in text) * size
        left = x if anchor == "start" else x - width / 2 if anchor == "middle" else x - width
        self.bound(left, y - size, width, size * 1.25, "text")
        self.fonts.append(size)
        self.parts.append(f'<text x="{x:g}" y="{y:g}" font-size="{size:g}" text-anchor="{anchor}" fill="{color}">{html.escape(text)}</text>')

    def rect(self, x, y, w, h, fill=BG, stroke=INK, sw=2):
        self.bound(x, y, w, h, "rect")
        self.parts.append(f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" fill="{fill}" stroke="{stroke}" stroke-width="{sw:g}"/>')

    def line(self, x1, y1, x2, y2, color=INK, arrow=False, dashed=False):
        self.bound(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1), "line")
        extra = ' marker-end="url(#arrow)"' if arrow else ''
        extra += ' stroke-dasharray="10 8"' if dashed else ''
        self.parts.append(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{color}" stroke-width="2"{extra}/>')

    def box(self, x, y, w, h, lines, size=40):
        self.rect(x, y, w, h)
        gap = size * 1.12
        first = y + h/2 + size*.33 - (len(lines)-1)*gap/2
        for i, text in enumerate(lines):
            self.text(x+w/2, first+i*gap, text, size, "middle")

    def validate(self):
        for x, y, w, h, kind in self.bounds:
            if min(x, y) < 80 or x+w > 920 or y+h > 920:
                raise ValueError(f"{kind} outside 80 px inset: {(x,y,w,h)}")
        assert min(self.fonts) >= 36
        data = ("\n".join(self.parts+["</g></svg>"])+"\n").encode()
        ET.fromstring(data)
        return data


def stage_work(source):
    work = source["compute"]["stage_training_TF_both_DP"]
    extra = source["compute"]["stage3_extra_percent_full_head"]
    c = Card("The full head adds work", "Hypothetical forward/backward dense GEMM work for both data replicas. Stage 3 has the complete output head. Work is not elapsed time.")
    c.text(500, 140, "The full head adds work", 54, "middle")
    c.text(500, 198, "Hypothetical F/B GEMM · both replicas", 36, "middle")
    c.text(80, 270, "TFLOP / update", 36)
    x0, y0, scale = 140, 720, 8.4
    for value in (0, 20, 40):
        y = y0-value*scale
        c.line(x0, y, 900, y)
        c.text(130, y+12, str(value), 36, "end")
    c.line(x0, 300, x0, y0)
    for stage, value in enumerate(work):
        x = 180+stage*180
        height = value*scale
        color = ORANGE if stage == 3 else BLUE
        c.rect(x, y0-height, 120, height, color, color, 0)
        c.text(x+60, y0-height-25, f"{value:.2f}", 48, "middle")
        c.text(x+60, 780, f"S{stage}", 42, "middle")
        c.geometry.append({"stage": stage, "value_TF": value, "origin_y": y0, "height": height, "px_per_TF": scale})
    c.text(500, 850, f"Full head: +{extra:.2f}% work", 48, "middle")
    c.text(500, 904, "Work count, not elapsed time.", 36, "middle")
    return c


def norm_schedules(source):
    assert "declared choice" in source["optimizer_and_routing"]["barrier"]
    assert "RS ->" in source["alternative_DP_schedule"]["order"]
    c = Card("Same norm. Different waiting.", "The declared full DP all-reduce plan and a reduce-scatter-first alternative preserve total payload and norm mathematics. Possible allgather/optimizer overlap is optimistic and must respect the full dependency graph.")
    c.text(500, 140, "Same norm.", 60, "middle")
    c.text(500, 206, "Different waiting.", 60, "middle")
    for x, label in ((90, "Declared plan"), (540, "Alternative")):
        c.text(x+185, 270, label, 42, "middle")
        for y1, y2 in ((375, 410), (510, 545), (615, 650)):
            c.line(x+185, y1, x+185, y2, arrow=True)
    c.box(90, 290, 370, 85, ["Full all-reduce"])
    c.box(90, 410, 370, 100, ["Unique norm"])
    c.box(90, 545, 370, 70, ["Clip"])
    c.box(90, 650, 370, 140, ["Optimizer"])
    c.box(540, 290, 370, 85, ["Reduce-scatter"])
    c.box(540, 410, 370, 100, ["Disjoint terms", "one global norm"], 36)
    c.box(540, 545, 370, 70, ["Clip"])
    c.box(540, 650, 370, 140, ["Allgather +", "optimizer", "possible overlap"], 36)
    c.text(500, 830, "Same bytes and norm math.", 36, "middle")
    c.text(500, 870, "Optimistic floors need a full", 36, "middle")
    c.text(500, 910, "dependency graph before speed claims.", 36, "middle")
    return c


def tier_traffic(source):
    traffic = source["traffic_and_screens"]
    a = traffic["A"]["aggregate_sent_bytes"]
    b = traffic["B"]["aggregate_sent_bytes"]
    assert a["inter_rack"]+a["intra_rack"] == b["inter_rack"]+b["intra_rack"]
    c = Card("Placement moves rack traffic", "Logical directed payload, both directions per update. Cross-rack payload moves within the racks; total off-node payload is unchanged.")
    c.text(500, 140, "Placement moves rack traffic", 50, "middle")
    c.text(500, 198, "Hypothetical cross-rack GB / update", 36, "middle")
    c.text(500, 250, "Both directions", 36, "middle")
    x0, scale = 195, 700/30
    for value in (0, 10, 20, 30):
        x = x0+value*scale
        c.line(x, 350 if value == 0 else 690, x, 690 if value == 0 else 704)
        c.text(x, 740, str(value), 36, "middle")
    c.line(x0, 690, 895, 690)
    for y, label, values, color in ((360,"A",a,BLUE),(570,"B",b,ORANGE)):
        value = values["inter_rack"]/1e9
        width = value*scale
        c.rect(x0, y, width, 85, color, color, 0)
        c.text(110, y+61, label, 52)
        c.text(x0, y-25, f"{value:.3f} GB", 56)
        c.geometry.append({"placement": label, "cross_rack_GB": value, "origin_x": x0, "width": width, "px_per_GB": scale})
    c.text(500, 830, f"Within rack: {a['intra_rack']/1e9:.3f} → {b['intra_rack']/1e9:.3f} GB", 36, "middle")
    c.text(500, 890, "Total off-node payload unchanged", 36, "middle")
    return c


def ordered_floors(source):
    traffic = source["traffic_and_screens"]
    caps = source["assumed_capabilities"]
    c = Card("Two ordered lower bounds", "The declared full DP all-reduce-before-norm plan has optimistic ordered resource floors, not runtime estimates. Blue is before clipping and orange is after clipping. A 150 ms target is shown on a zero-based millisecond axis.")
    c.text(500, 140, "Two ordered lower bounds", 52, "middle")
    c.text(500, 198, "Hypothetical plan · ms per update", 36, "middle")
    for x, color, label in ((115, BLUE, "Before clip"), (540, ORANGE, "After clip")):
        c.rect(x, 244, 28, 28, color, color, 0)
        c.text(x+45, 273, label, 36)
    x0, scale, target = 140, 700/300, 150
    for value in (0, 150, 300):
        x = x0+value*scale
        c.line(x, 600 if value == 300 else 350, x, 612 if value == 300 else 600, dashed=value==target)
        c.text(x, 645, str(value), 36, "middle")
    c.line(x0, 600, 840, 600)
    c.text(x0+target*scale, 330, "150 ms target", 36, "middle")
    for y, label in ((365,"A"),(510,"B")):
        values = traffic[label]["floors_ms"]
        pre, post = values["pre_clipping_phase"], values["post_clipping_phase"]
        total = pre+post
        assert abs(total-values["necessary_step_floor"]) < 1e-6
        c.rect(x0, y, pre*scale, 70, BLUE, BLUE, 0)
        c.rect(x0+pre*scale, y, post*scale, 70, ORANGE, ORANGE, 0)
        c.text(95, y+51, label, 52)
        c.text(x0+pre*scale/2, y+51, f"{pre:.2f}", 48, "middle", BG)
        c.text(x0+(pre+post/2)*scale, y+51, f"{post:.2f}", 48, "middle", BG)
        c.text(x0+total*scale+18, y+51, f"{total:.2f}", 52)
        c.geometry.append({"placement": label, "pre_ms": pre, "post_ms": post, "total_ms": total, "origin_x": x0, "pre_width": pre*scale, "post_width": post*scale, "px_per_ms": scale})
    c.text(500, 703, "Declared full DP all-reduce before norm", 36, "middle")
    c.text(500, 743, "FP32 Muon, one attention owner", 36, "middle")
    c.text(500, 783, f"Assumed {caps['Muon_FP32_TF_s_GPU']} TFLOP/s/GPU,", 36, "middle")
    c.text(500, 823, f"rack {caps['rack_uplink_GB_s_each_direction']} GB/s / node {caps['node_NIC_GB_s_each_direction']} GB/s", 36, "middle")
    c.text(500, 863, "Each direction. Balanced routing.", 36, "middle")
    c.text(500, 903, "Lower bounds, not runtimes.", 36, "middle")
    return c


FIGURES = {"04-stage-work": stage_work, "12-norm-schedules": norm_schedules,
           "15-tier-traffic": tier_traffic, "16-ordered-floors": ordered_floors}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    source_bytes = SOURCE.read_bytes()
    source = json.loads(source_bytes)
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {"source": str(SOURCE), "source_sha256": hashlib.sha256(source_bytes).hexdigest(), "cards": []}
    for name, builder in FIGURES.items():
        card = builder(source)
        data = card.validate()
        path = OUT / (name+".svg")
        if args.check:
            assert path.read_bytes() == data, f"Regeneration differs: {path}"
        else:
            path.write_bytes(data)
        receipt["cards"].append({"file": str(path), "sha256": hashlib.sha256(data).hexdigest(), "viewbox": [0,0,1000,1000], "minimum_font_px": min(card.fonts), "safe_inset_px": 80, "xml_parse": "pass", "bounds": "pass", "geometry": card.geometry})
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt,indent=2)+"\n")
    print(f"{'Verified' if args.check else 'Wrote'} four cards: XML, 80 px bounds and minimum 36 px labels pass.")


if __name__ == "__main__":
    main()
