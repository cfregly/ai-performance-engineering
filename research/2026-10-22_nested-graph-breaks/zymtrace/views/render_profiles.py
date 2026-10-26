#!/usr/bin/env python3
"""Render retained CUDA event stacks with event-record widths."""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
from collections import Counter

HERE = Path(__file__).resolve().parent
CAPTURE = HERE.parent / "captures" / "torch2141-b200-20261026"
FOCUS = ("ngb_nested_break", "CUDAGraph", "CUDAWarmupNode", "cuLaunchKernel", "cuGraphLaunch")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text(x, y, value, size=18, color="#172438", weight="normal"):
    return f'<text x="{x}" y="{y}" fill="{color}" font-family="DejaVu Sans, sans-serif" font-size="{size}" font-weight="{weight}">{html.escape(str(value))}</text>'


def trie(stacks):
    root = {"n": "all matching CUDA records", "count": 0, "children": {}}
    for stack in stacks:
        node = root
        node["count"] += 1
        for name, kind in stack:
            node = node["children"].setdefault((name, kind), {"n": name, "k": kind, "count": 0, "children": {}})
            node["count"] += 1
    return root


def draw(node, x, y, width, row, output):
    color = {"cuda": "#ffcc73", "cuda-memory": "#ffe6b8", "native": "#a3daf1", "cpython": "#bfd0f4"}.get(node.get("k"), "#dce5ed")
    if "ngb_" in node["n"]:
        color = "#c2e6ce"
    output.append(f'<rect x="{x:.3f}" y="{y}" width="{width:.3f}" height="{row - 1}" fill="{color}" stroke="#ffffff" stroke-width="0.65"><title>{html.escape(node["n"])} | {node["count"]} records</title></rect>')
    label = node["n"]
    capacity = int((width - 9) / 6.9)
    if capacity >= 5:
        if len(label) > capacity:
            label = label[:capacity - 1] + "…"
        output.append(text(round(x + 4, 3), y + row - 6, label, 11))
    cursor = x
    for child in node["children"].values():
        child_width = width * child["count"] / node["count"]
        draw(child, cursor, y + row, child_width, row, output)
        cursor += child_width


def save_svg(path, body, width, height):
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#f8fafc"/>' + "".join(body) + "</svg>\n")


def main():
    provenance = {"renderer": "render_profiles.py", "width_unit": "one exported CUDA event row", "inputs": {}, "outputs": {}, "method": "Filter by audited source PID, GPU UUID, cuda kind, stream presence, and positive finite c. Build a prefix tree from actual ordered frame IDs. Width is record count. Main images retain mechanism frames and all CUDA terminal frames. Full-stack SVGs retain every frame.", "limits": ["c is used only for the existing export validity check. Its units are not inferred or plotted.", "These records span setup, warmup, timing, and correctness work. Counts are not launch totals, durations, CPU overhead, or a speedup comparison.", "Raw t timestamps have second precision. No timeline is reconstructed.", "The enabled raw export includes prior-worker records. Only its audited PID is included in these derived views.", "Narrow labels are truncated visually. Exact full labels and source frame IDs are retained in the stack JSON and SVG tooltips."]}
    for arm, word in (("disabled", "without"), ("enabled", "with")):
        source = CAPTURE / arm / "zymtrace-export" / "cuda-events.raw.json"
        audit_path = source.with_name("audit.json")
        raw_flame = source.with_name("cuda-flamegraph.raw.json")
        doc = json.loads(source.read_text())
        audit = json.loads(audit_path.read_text())
        strings, frames = doc["strings"], doc["frames"]
        pid = audit["source_pid"]
        gpu = audit["expected_gpu_uuid"].removeprefix("GPU-").lower()
        events = [e for e in doc["events"] if e["pid"] == pid and strings[e["k"]] == "cuda" and strings[e["u"]].removeprefix("GPU-").lower() == gpu and e.get("sid") is not None and isinstance(e.get("c"), (int, float)) and not isinstance(e["c"], bool) and math.isfinite(e["c"]) and e["c"] > 0]
        assert len(events) == audit["matching_cuda_rows"]
        def decode(i):
            f = frames[i]
            kind = strings[f["k"]]
            name = strings[f["n"]]
            if kind == "native" and name == "cuda" and "e" in f:
                name = Path(strings[f["e"]]).name
            return name, kind
        full = [[decode(i) for i in e["f"]] for e in events]
        focused = [[pair for pair in stack if pair[1].startswith("cuda") or any(token in pair[0] for token in FOCUS)] for stack in full]
        assert all(focused)
        prefix = f"zymtrace-{word}-nested-graph-breaks"
        raw_indices = {id(event): i for i, event in enumerate(doc["events"])}
        exact = {"arm": arm, "pid": pid, "records": [{"raw_event_index": raw_indices[id(e)], "frame_ids": e["f"], "frames": [{"name": n, "kind": k} for n, k in stack]} for e, stack in zip(events, full)]}
        (HERE / f"{prefix}.stacks.json").write_text(json.dumps(exact, indent=2) + "\n")
        width, height = 1800, 1120
        body = [text(55, 59, "ZYMTRACE / RETAINED CUDA EVENT STACKS", 20, "#526276", "bold"), text(55, 110, f"{word.capitalize()} nested graph-break support", 39, weight="bold"), text(55, 151, "PyTorch 2.14.1 · NVIDIA B200 · nested_break_depth8 · 26 October 2026", 21), text(55, 192, f"Worker PID {pid}  |  {len(events)} matching records  |  Width counts records, not elapsed time", 22, weight="bold"), text(55, 230, "Mechanism-focused flamegraph · every matching record included · other stack frames omitted", 19, "#526276")]
        draw(trie(focused), 55, 265, 1690, 25, body)
        body.extend([text(55, 790, "Exact function names found in this worker's stacks", 24, weight="bold"), text(55, 822, "Record counts below are stack membership. One record can contain several functions.", 18, "#526276")])
        counts = Counter()
        for stack in full:
            counts.update(set(n for n, k in stack))
        names = ("cuLaunchKernel", "cuGraphLaunch", "CUDAGraph.replay", "CUDAGraphTreeManager.run")
        for j, name in enumerate(names):
            x, y = 55 + (j % 2) * 845, 867 + (j // 2) * 48
            body.append(text(x, y, f"{name}   {counts[name]} records", 23))
        overlap = len(doc["events"]) - len(events)
        body.extend([text(55, 986, f"PID filter: {overlap} other-worker records excluded from {len(doc['events'])} raw exported records.", 20), text(55, 1018, "Includes setup, warmup, timing, and correctness work. Counts do not establish performance.", 20), text(55, 1052, "Full ordered stacks, source hashes, raw exports, and rendering method accompany this image.", 19, "#526276"), text(55, 1085, "Derived chart from retained Zymtrace JSON. No timestamp interpolation or duration conversion.", 18, "#526276")])
        svg = HERE / f"{prefix}.svg"
        save_svg(svg, body, width, height)
        subprocess.run(["rsvg-convert", "-o", str(HERE / f"{prefix}.png"), str(svg)], check=True)
        subprocess.run(["rsvg-convert", "-w", "400", "-o", str(HERE / f"{prefix}.preview.png"), str(svg)], check=True)
        # Preserve a zoomable full-stack view without the mechanism projection.
        full_height = 190 + 21 * (max(map(len, full)) + 1)
        full_body = [text(55, 55, f"{word.capitalize()} nested graph-break support — full exported frame stacks", 28, weight="bold"), text(55, 96, f"PID {pid} · {len(events)} records · widths count records, not time · raw native library labels preserved", 20)]
        draw(trie(full), 55, 140, 2690, 21, full_body)
        save_svg(HERE / f"{prefix}.full-stacks.svg", full_body, 2800, full_height)
        inputs = {}
        for path in (source, audit_path, raw_flame):
            inputs[str(path.relative_to(HERE.parent))] = sha(path)
        provenance["inputs"][arm] = {"sha256": inputs, "pid": pid, "accepted_records": len(events), "excluded_records": overlap, "maximum_full_stack_depth": max(map(len, full)), "maximum_projected_depth": max(map(len, focused))}
    for path in sorted(HERE.glob("zymtrace-*")):
        provenance["outputs"][path.name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    provenance["renderer_sha256"] = sha(Path(__file__))
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
