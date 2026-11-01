#!/usr/bin/env python3
"""Build six native SVG diagram elements from the frozen reader summary.

Run from any directory. This writes only images/elements/*.svg and summary.json.
No chart renders, account actions, packages, or network access are required.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence" / "reader-summary.json"
README = ROOT / "code" / "README.md"
OUT = ROOT / "images" / "elements"
W, H = 1600, 900
BG, INK, BLUE, ORANGE = "#FBFBF8", "#202326", "#3478BA", "#E87823"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def at_pointer(source: dict, pointer: str) -> object:
    value = source
    for part in pointer.lstrip("/").split("/"):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


class SVG:
    """Small explicit SVG vocabulary, with bounds recorded before serialization."""

    def __init__(self, title: str, description: str, width=W, height=H):
        self.width, self.height = width, height
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f'<title id="title">{html.escape(title)}</title>',
            f'<desc id="desc">{html.escape(description)}</desc>',
            '<defs>',
        ]
        for name, color in (("ink", INK), ("blue", BLUE), ("orange", ORANGE)):
            self.parts.append(
                f'<marker id="arrow-{name}" markerWidth="12" markerHeight="10" refX="11" refY="5" orient="auto-start-reverse" markerUnits="userSpaceOnUse">'
                f'<path d="M 1 1 L 11 5 L 1 9" fill="none" stroke="{color}" stroke-width="3"/></marker>'
            )
        self.parts.extend(["</defs>", f'<rect x="0" y="0" width="{width}" height="{height}" fill="{BG}"/>'])
        self.bounds = []
        self.font_sizes = []

    def bound(self, x: float, y: float, width: float, height: float, kind: str):
        self.bounds.append((x, y, width, height, kind))

    def rect(self, x, y, width, height, stroke=INK, fill=BG, opacity=None, sw=3):
        self.bound(x - sw / 2, y - sw / 2, width + sw, height + sw, "rect")
        extra = f' fill-opacity="{opacity}"' if opacity is not None else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{fill}"{extra} stroke="{stroke}" stroke-width="{sw}"/>')

    def line(self, points, color=INK, arrow=False, both=False, sw=3):
        xs, ys = zip(*points)
        self.bound(min(xs) - 6, min(ys) - 6, max(xs) - min(xs) + 12, max(ys) - min(ys) + 12, "line")
        key = {INK: "ink", BLUE: "blue", ORANGE: "orange"}[color]
        markers = f' marker-end="url(#arrow-{key})"' if arrow or both else ""
        if both:
            markers += f' marker-start="url(#arrow-{key})"'
        points_text = " ".join(f"{x},{y}" for x, y in points)
        self.parts.append(f'<polyline points="{points_text}" fill="none" stroke="{color}" stroke-width="{sw}" stroke-linejoin="miter"{markers}/>')

    def text(self, x, y, text, size=34, color=INK, anchor="start", weight=400):
        self.font_sizes.append(size)
        # Conservative Inter-width estimate catches clipped labels without a font renderer.
        width = sum(0.36 if c in " ilI1.,:;|" else 0.68 if c in "MW@" else 0.59 for c in text) * size
        left = x if anchor == "start" else x - width / 2 if anchor == "middle" else x - width
        self.bound(left, y - size, width, size * 1.25, "text")
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Inter" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{html.escape(text)}</text>')

    def lines(self, x, y, lines, size=34, color=INK, anchor="middle", gap=None, weight=400):
        for i, text in enumerate(lines):
            self.text(x, y + i * (gap or size * 1.3), text, size, color, anchor, weight)

    def gpu_row(self, x, y):
        for rank in range(8):
            self.rect(x + rank * 58, y, 34, 34, BLUE, BLUE, 0.12, 2)

    def gpu_two_rows(self, x, y):
        for rank in range(8):
            self.rect(x + (rank % 4) * 54, y + (rank // 4) * 48, 32, 32, BLUE, BLUE, 0.12, 2)

    def validate(self):
        for x, y, width, height, kind in self.bounds:
            if min(x, y) < 20 or x + width > self.width - 20 or y + height > self.height - 20:
                raise ValueError(f"{kind} outside safe inset: {(x, y, width, height)}")
        if min(self.font_sizes) < 30:
            raise ValueError("A label is smaller than 30 px")
        data = "\n".join(self.parts + ["</svg>"]) + "\n"
        root = ET.fromstring(data)
        if root.attrib["viewBox"] != f"0 0 {self.width} {self.height}":
            raise ValueError("Unexpected viewBox")
        return data.encode()


def topology_a(source):
    s = SVG("Data replicas in separate racks", "Eight nodes, four in each rack. One pipeline stage occupies each eight-GPU node. Every corresponding-stage data-parallel pair crosses racks.", 1000, 1000)
    racks = source["placement_nodes_by_rack"]["A"]
    for rack, (x, nodes) in enumerate(zip((30, 590), racks)):
        s.rect(x, 40, 380, 830)
        s.text(x + 190, 105, f"Rack {rack}", 60, anchor="middle", weight=600)
        s.text(x + 190, 170, f"R{rack}", 60, anchor="middle", weight=600)
        for i, node in enumerate(nodes):
            y = 210 + i * 180
            replica, stage = divmod(node, source["model"]["PP"])
            s.rect(x + 30, y, 320, 110)
            s.text(x + 190, y + 78, f"P{stage}", 64, anchor="middle", weight=600)
            if i < 3:
                s.line([(x + 190, y + 110), (x + 190, y + 180)], BLUE, arrow=True, sw=4)
    for stage in range(source["model"]["PP"]):
        y = 265 + stage * 180
        s.line([(380, y), (620, y)], ORANGE, both=True, sw=4)
        s.text(500, y - 22, "DP", 48, ORANGE, "middle", 600)
    s.line([(50, 935), (110, 935)], BLUE, arrow=True, sw=4)
    s.text(130, 951, "PP", 48, BLUE)
    s.line([(300, 935), (360, 935)], ORANGE, both=True, sw=4)
    s.text(380, 951, "DP", 48, ORANGE)
    s.text(575, 951, "8 GPUs/node", 48)
    return s


def topology_b(source):
    s = SVG("Corresponding stages share a rack", "Rack 0 has stages 0 and 1 of both replicas; rack 1 has stages 2 and 3 of both replicas. Data-parallel pairs stay inside a rack. The stage 1 to stage 2 pipeline boundary crosses racks for both replicas.", 1000, 1100)
    rack_nodes = source["placement_nodes_by_rack"]["B"]
    positions = {}
    for rack, (x, nodes) in enumerate(zip((30, 530), rack_nodes)):
        s.rect(x, 40, 440, 920)
        s.text(x + 220, 120, f"Rack {rack}", 60, anchor="middle", weight=600)
        for replica in range(source["model"]["DP"]):
            s.text(x + 220, 220 + replica * 460, f"Replica {replica}", 52, anchor="middle", weight=600)
        for node in nodes:
            replica, stage = divmod(node, source["model"]["PP"])
            nx, ny = x + 20 + (stage % 2) * 240, 280 + replica * 460
            positions[node] = (nx, ny)
            s.rect(nx, ny, 160, 140)
            s.text(nx + 80, ny + 92, f"P{stage}", 64, anchor="middle", weight=600)
        for stage in (rack * 2, rack * 2 + 1):
            top_x, top_y = positions[stage]
            s.line([(top_x + 80, top_y + 140), (top_x + 80, top_y + 460)], ORANGE, both=True, sw=4)
    for replica in range(source["model"]["DP"]):
        for stage in range(source["model"]["PP"] - 1):
            ax, ay = positions[replica * 4 + stage]
            bx, by = positions[replica * 4 + stage + 1]
            s.line([(ax + 160, ay + 70), (bx, by + 70)], BLUE, arrow=True, sw=4)
    s.line([(50, 1020), (110, 1020)], BLUE, arrow=True, sw=4)
    s.text(130, 1036, "PP", 54, BLUE)
    s.line([(300, 1020), (360, 1020)], ORANGE, both=True, sw=4)
    s.text(380, 1036, "DP", 54, ORANGE)
    s.text(575, 1036, "8 GPUs/node", 54)
    return s


def rank_groups(source):
    s = SVG("Eight GPUs, two execution views", "A single physical two-by-four grid contains ranks 0 to 7. Tensor-parallel groups are rows of four, context-parallel groups are vertical pairs, and expert parallelism uses all eight ranks.", 1000, 900)
    s.rect(30, 130, 940, 600, INK)
    s.text(500, 85, "EP8 · ranks 0–7", 60, anchor="middle", weight=600)
    for row in range(2):
        y = 215 + row * 300
        s.rect(170, y - 30, 785, 190, BLUE, BLUE, 0.06)
        s.text(100, y + 90, "TP4", 48, BLUE, "middle", 600)
        for col in range(4):
            rank = source["groups_each_node"]["attention_TP4_groups"][row][col]
            x = 190 + col * 200
            s.rect(x, y, 150, 130, BLUE)
            s.text(x + 75, y + 88, str(rank), 64, anchor="middle", weight=600)
    for pair in source["groups_each_node"]["attention_CP2_groups"]:
        col = pair[0]
        x = 265 + col * 200
        s.line([(x, 345), (x, 515)], ORANGE, both=True, sw=4)
        s.rect(x - 52, 399, 104, 66, ORANGE, BG, sw=0)
        s.text(x, 451, "CP2", 48, ORANGE, "middle", 600)
    s.text(500, 800, "Attention: TP4 × CP2", 48, anchor="middle")
    s.text(500, 860, "MoE: EP8 · same 8 GPUs", 42, anchor="middle")
    return s


def pipeline_1f1b(source):
    unit = source["unit_1F1B"]
    s = SVG("Twenty-two unit slots of one-forward one-backward scheduling", "One schedule continues from slots 1 to 11 in the upper panel to slots 12 to 22 below. Four pipeline stages follow the exact frozen schedule, with equal forward/backward unit service and zero handoff time. Blue cells are forward microbatches, orange cells backward, and literally blank cells idle. Slots are not milliseconds.", 1000, 1100)
    rows = [row.split() for row in unit["rows"]]
    if unit["slots"] != 22 or len(rows) != 4 or any(len(row) != 22 for row in rows):
        raise ValueError("The wrapped layout requires the frozen four-stage, 22-slot schedule")
    x0, pitch, row_pitch = 140, 75, 75
    for start, heading_y, numbers_y, y0 in ((0, 75, 130, 150), (11, 540, 595, 615)):
        heading = "Slots 1–11" if start == 0 else "Slots 12–22 · continued"
        s.text(500, heading_y, heading, 48, anchor="middle", weight=600)
        for col in range(11):
            s.text(x0 + col * pitch + 35, numbers_y, str(start + col + 1), 42, anchor="middle")
        for stage, row in enumerate(rows):
            y = y0 + stage * row_pitch
            s.text(80, y + 47, f"P{stage}", 48, anchor="middle", weight=600)
            for col, token in enumerate(row[start:start + 11]):
                x = x0 + col * pitch
                color = BLUE if token.startswith("F") else ORANGE if token.startswith("B") else INK
                if token != "..":
                    s.rect(x, y, 70, 62, color, color, 0.13, 2)
                    s.text(x + 35, y + 45, token, 42, anchor="middle", weight=600)
    s.rect(100, 950, 46, 46, BLUE, BLUE, 0.13, 2)
    s.text(165, 990, "F forward", 42)
    s.rect(430, 950, 46, 46, ORANGE, ORANGE, 0.13, 2)
    s.text(495, 990, "B backward", 42)
    s.text(780, 990, "Blank: idle", 37)
    s.text(500, 1065, "Equal unit service · not milliseconds", 42, anchor="middle")
    return s


def muon_whole_matrix(source):
    s = SVG("Gather before the whole-matrix Muon update", "Four attention-state shards contain FP32 gradient, master weights and momentum. Gather a whole two-dimensional matrix on stage rank 0, run five finite FP32 Newton-Schulz iterations, then scatter and publish weights and optimizer state. This approximate transform is not exact SVD. Expert matrices already reside whole on their local owners.", 1000, 1100)
    s.text(500, 85, "Whole-matrix Muon", 60, anchor="middle", weight=600)
    s.text(500, 160, "4 attention-state shards", 52, anchor="middle")
    s.text(500, 218, "FP32 gradient / master / momentum", 48, anchor="middle")
    for shard in range(4):
        x = 70 + shard * 220
        s.rect(x, 250, 200, 70, BLUE, BLUE, 0.09)
        s.text(x + 100, 302, str(shard), 48, anchor="middle")
        s.line([(x + 100, 320), (x + 100, 347)], BLUE)
    s.line([(170, 347), (830, 347)], BLUE)
    s.line([(500, 347), (500, 385)], BLUE, arrow=True, sw=4)
    s.rect(50, 385, 900, 175, BLUE, BLUE, 0.06)
    s.rect(80, 410, 150, 125, BLUE, BLUE, 0.08)
    for row in range(1, 4):
        s.line([(80, 410 + row * 31.25), (230, 410 + row * 31.25)], BLUE, sw=2)
    for col in range(1, 4):
        s.line([(80 + col * 37.5, 410), (80 + col * 37.5, 535)], BLUE, sw=2)
    s.lines(590, 450, ["Gather whole matrix", "on stage rank 0"], 52, gap=70, weight=600)
    s.line([(500, 560), (500, 610)], INK, arrow=True, sw=4)
    s.rect(50, 610, 900, 210)
    s.text(500, 674, "5 finite Newton–Schulz", 52, anchor="middle", weight=600)
    s.text(500, 736, "FP32 iterations", 48, anchor="middle")
    s.text(500, 795, "Approximate, not exact SVD", 48, anchor="middle")
    s.line([(500, 820), (500, 880)], ORANGE, arrow=True, sw=4)
    s.rect(50, 880, 900, 130, ORANGE, ORANGE, 0.06)
    s.text(500, 934, "Scatter + publish", 52, ORANGE, "middle", 600)
    s.text(500, 992, "weights AND optimizer state", 48, anchor="middle")
    s.text(500, 1062, "Experts: whole on local owners", 48, anchor="middle")
    return s


def gradient_barrier(source):
    s = SVG("The global clipping dependency chain", "Completed gradient buckets contribute to local tensor/context-parallel and data-parallel sums. Every required sum must finish across the whole step before one all-parameter global L2 norm. Each unique Muon and AdamW gradient contributes once, excluding duplicate TP/CP/DP copies. One scalar clips every gradient before the two optimizer paths. Both updated weights and optimizer state must be published before the next step.", 1000, 1100)
    s.text(500, 78, "One global clipping barrier", 56, anchor="middle", weight=600)
    # The single chain enforces the global barrier before the optimizer fork.
    for start, end in ((190, 225), (355, 390), (520, 555), (935, 970)):
        s.line([(500, start), (500, end)], INK, arrow=True, sw=4)
    for center, color in ((255, BLUE), (745, ORANGE)):
        s.line([(500, 635), (500, 657), (center, 657), (center, 680)], color, arrow=True, sw=4)
        s.line([(center, 765), (center, 790), (500, 790)], color, sw=4)
    s.line([(500, 790), (500, 815)], INK, arrow=True, sw=4)
    s.rect(80, 110, 840, 80)
    s.text(500, 165, "Completed gradient buckets", 48, anchor="middle")
    s.rect(80, 225, 840, 130)
    s.lines(500, 278, ["All local TP/CP + DP", "sums finish"], 52, gap=57, weight=600)
    s.rect(80, 390, 840, 130)
    s.text(500, 445, "One all-parameter L2 norm", 52, anchor="middle", weight=600)
    s.text(500, 502, "Count each gradient once", 48, anchor="middle")
    s.rect(80, 555, 840, 80)
    s.text(500, 610, "Clip all with one scalar", 52, anchor="middle", weight=600)
    s.rect(60, 680, 390, 85, BLUE, BLUE, 0.06)
    s.text(255, 739, "Muon", 52, BLUE, "middle", 600)
    s.rect(550, 680, 390, 85, ORANGE, ORANGE, 0.06)
    s.text(745, 739, "AdamW", 52, ORANGE, "middle", 600)
    s.rect(80, 815, 840, 120)
    s.text(500, 868, "Both updates published", 52, anchor="middle", weight=600)
    s.text(500, 918, "weights + optimizer state", 48, anchor="middle")
    s.rect(80, 970, 840, 80)
    s.text(500, 1025, "Next step", 52, anchor="middle", weight=600)
    return s


FIGURES = [
    ("topology-a", topology_a, "Keep each data replica within one rack; show every DP pair crossing racks.", ["/model/DP", "/model/PP", "/model/nodes", "/model/GPUs", "/model/GPUs_per_node", "/placement_nodes_by_rack/A", "/placement_nodes_by_rack/node_identity"]),
    ("topology-b", topology_b, "Put both replicas' stages 0/1 in rack 0 and 2/3 in rack 1; DP pairs stay in-rack, and the P1-to-P2 boundary crosses racks.", ["/model/DP", "/model/PP", "/model/GPUs", "/placement_nodes_by_rack/B", "/placement_nodes_by_rack/node_identity"]),
    ("rank-groups", rank_groups, "Show TP4 rows, CP2 vertical pairs and EP8 on one physical eight-rank grid.", ["/model/GPUs_per_node", "/groups_each_node/attention_TP4_groups", "/groups_each_node/attention_CP2_groups", "/groups_each_node/EP8_same_ranks"]),
    ("pipeline-1f1b", pipeline_1f1b, "Project one exact 22-slot four-stage schedule as two stacked 11-slot continuation panels, with operation labels and unit-service limits.", ["/unit_1F1B/rows", "/unit_1F1B/slots", "/unit_1F1B/assumed_F_and_B_service", "/unit_1F1B/zero_transfer_latency", "/unit_1F1B/limit"]),
    ("muon-whole-matrix", muon_whole_matrix, "Show four attention-state shards gathering before a finite whole-matrix Newton-Schulz update, followed by scatter/publication.", ["/groups_each_node/attention_TP4_groups", "/optimizer_and_routing/Muon", "/optimizer_and_routing/owner", "/declared_capacity/states"]),
    ("gradient-barrier", gradient_barrier, "Preserve the all-required-reductions, unique-global-norm, common-clipping, two-optimizer and publication dependencies.", ["/optimizer_and_routing/barrier", "/optimizer_and_routing/Muon", "/optimizer_and_routing/AdamW"]),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Validate that retained SVGs exactly match regenerated bytes.")
    args = parser.parse_args()
    source = json.loads(SOURCE.read_text())
    if args.check:
        if not OUT.is_dir():
            raise ValueError("Retained element directory is missing: images/elements")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source": {"path": "evidence/reader-summary.json", "sha256": digest(SOURCE.read_bytes())},
        "text_authority": {"path": "code/README.md", "sha256": digest(README.read_bytes())},
        "source_field_hash_format": "sha256 of UTF-8 JSON, sorted keys, compact separators, ensure_ascii=False",
        "default_viewbox": [0, 0, W, H],
        "style": {"background": BG, "ink": INK, "blue": BLUE, "orange": ORANGE, "font": "Inter", "minimum_label_px": 30, "timeline_operation_label_px": 42},
        "figures": [],
    }
    for name, builder, purpose, pointers in FIGURES:
        diagram = builder(source)
        data = diagram.validate()
        target = OUT / f"{name}.svg"
        if args.check:
            if not target.is_file():
                raise ValueError(f"Retained element is missing: {target.relative_to(ROOT).as_posix()}")
            if target.read_bytes() != data:
                raise ValueError(f"Retained element differs: {target.relative_to(ROOT).as_posix()}")
        else:
            target.write_bytes(data)
        manifest["figures"].append({
            "file": target.relative_to(ROOT).as_posix(),
            "viewbox": [0, 0, diagram.width, diagram.height],
            "purpose": purpose,
            "sha256": digest(data),
            "source_fields": [{"pointer": p, "sha256": digest(canonical(at_pointer(source, p)))} for p in pointers],
            "checks": {"xml_parse": "pass", "geometry_safe_inset_px": 20, "bounds": "pass", "minimum_font_px": min(diagram.font_sizes)},
        })
    manifest_bytes = (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode()
    manifest_path = OUT / "summary.json"
    if args.check:
        if not manifest_path.is_file():
            raise ValueError("Retained provenance summary is missing")
        if manifest_path.read_bytes() != manifest_bytes:
            raise ValueError("Retained provenance summary differs")
    else:
        manifest_path.write_bytes(manifest_bytes)
    print(f"{'Verified' if args.check else 'Wrote'} {len(FIGURES)} SVG elements; XML and 20 px safe-inset bounds pass.")


if __name__ == "__main__":
    main()
