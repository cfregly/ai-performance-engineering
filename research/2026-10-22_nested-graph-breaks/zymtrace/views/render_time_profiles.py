#!/usr/bin/env python3
"""Render worker-filtered Zymtrace GPU-time flamegraphs from retained exports."""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
CAPTURE = HERE.parent / "captures/torch2141-b200-20261026"
OUTPUT = HERE / "time-weighted"
FOCUS = ("ngb_nested_break", "CUDAGraph", "CUDAWarmupNode", "cuLaunchKernel", "cuGraphLaunch")
COLORS = {"cuda": "#f4ae49", "cuda-memory": "#ffe0a6", "native": "#a2d9ea", "cpython": "#becff0"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_events(doc, pid, gpu):
    """Retain one worker, one GPU, and positive finite CUDA-time weights."""
    def string(value):
        return doc["strings"][value]
    selected = []
    for index, event in enumerate(doc["events"]):
        if (event["pid"] != pid or string(event["k"]) != "cuda"
                or string(event["u"]).removeprefix("GPU-").lower() != gpu.removeprefix("GPU-").lower()
                or event.get("sid") is None):
            continue
        value = event.get("c")
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Worker-scoped CUDA time must be positive and finite")
        selected.append((index, event))
    if not selected:
        raise ValueError("Worker filter selected no CUDA activity")
    return selected


def tree(records):
    """Accumulate exported GPU-time weights through each ordered launch stack."""
    root = {"name": "All captured GPU work", "kind": "root", "weight": 0.0, "children": {}}
    for frames, weight in records:
        node = root
        node["weight"] += weight
        for name, kind in frames:
            node = node["children"].setdefault((name, kind), {
                "name": name, "kind": kind, "weight": 0.0, "children": {}})
            node["weight"] += weight
    return root


def label(x, y, value, size=20, bold=False):
    return (f'<text x="{x}" y="{y}" fill="#172438" font-family="DejaVu Sans,sans-serif" '
            f'font-size="{size}" font-weight="{"bold" if bold else "normal"}">{html.escape(value)}</text>')


def bars(node, x, y, width, row, total, out):
    color = "#c2e6ce" if "ngb_" in node["name"] else COLORS.get(node["kind"], "#dce5ed")
    tooltip = f'{node["name"]}: {100 * node["weight"] / total:.6f}% of captured GPU time'
    out.append(f'<rect x="{x:.6f}" y="{y}" width="{width:.6f}" height="{row - 1}" fill="{color}" stroke="white" stroke-width="0.6"><title>{html.escape(tooltip)}</title></rect>')
    capacity = int((width - 8) / 6.8)
    if capacity >= 6:
        name = node["name"] if len(node["name"]) <= capacity else node["name"][:capacity - 1] + "…"
        out.append(label(round(x + 4, 3), y + row - 5, name, 11))
    cursor = x
    for child in sorted(node["children"].values(), key=lambda n: (n["name"], n["kind"])):
        child_width = width * child["weight"] / node["weight"]
        bars(child, cursor, y + row, child_width, row, total, out)
        cursor += child_width


def svg(path, body, width, height):
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#f8fafc"/>' + "".join(body) + "</svg>\n")


def read_authority():
    path = HERE / "units/authority.json"
    authority = json.loads(path.read_text())
    if (authority.get("verified") is not True or authority.get("event_field") != "c"
            or authority.get("unit") != "fraction" or authority.get("quantity") != "gpu_time"
            or authority.get("normalization") != "per_worker_sum" or not authority.get("evidence")):
        raise ValueError("Verified GPU-time quantity and fraction normalization are required before rendering")
    return path, authority


def main():
    authority_path, authority = read_authority()
    OUTPUT.mkdir(exist_ok=True)
    old_provenance = HERE / "record-count-provenance.json"
    provenance_path = HERE / "provenance.json"
    if not old_provenance.exists():
        old_provenance.write_bytes(provenance_path.read_bytes())
    provenance = {
        "renderer": "render_time_profiles.py", "renderer_sha256": digest(Path(__file__)),
        "width_unit": "Percent of captured GPU time, independently normalized per worker",
        "method": "Filter the original CUDA event export by worker PID, GPU UUID and stream. Sum the API's event-time field c through each ordered stack, then divide by the worker sum. Every bar uses GPU time.",
        "inputs": {}, "outputs": {},
        "limits": ["Includes setup, warmup, timing and correctness work.",
                   "These separate captures are not steady-state latency measurements.",
                   "GPU time attributed to a host stack is not CPU execution time.",
                   "Summed GPU activity can overlap and is not elapsed wall time.",
                   "Widths describe exported attributed GPU time. No additional sampling extrapolation is applied."]}
    semantics = {
        "schema_version": 1, "weights": {"quantity": "gpu_time_share", "unit": "fraction", "field": "c", "normalization": "per_worker_sum"},
        "unit_authority": {"verified": True, "schema_field": "JsonEvent.c", "schema_value": "Time spent in this event.", "absolute_export_unit": "undocumented",
                           "source_url": authority["source_url"], "source_path": str(authority_path.relative_to(PACKAGE)), "sha256": digest(authority_path)},
        "scope": {"phases": ["setup", "warmup", "timing", "correctness"], "normalized_width": True,
                  "disclosure": "Each worker fills its own 100% GPU-time width. Percentages divide exported time weights by their worker sum. No absolute time conversion is applied."}, "arms": {}}
    for arm, word in (("disabled", "without"), ("enabled", "with")):
        source = CAPTURE / arm / "zymtrace-export/cuda-events.raw.json"
        audit_path = source.with_name("audit.json")
        native_path = source.with_name("cuda-flamegraph.raw.json")
        doc, audit = json.loads(source.read_text()), json.loads(audit_path.read_text())
        pid, gpu = audit["source_pid"], audit["expected_gpu_uuid"]
        selected = select_events(doc, pid, gpu)
        assert len(selected) == audit["matching_cuda_rows"]
        all_weight = math.fsum(e["c"] for e in doc["events"])
        native_root = json.loads(native_path.read_text())["fg"]["nodes"][0]["t"]
        if not math.isclose(all_weight, native_root, rel_tol=2e-6, abs_tol=1e-9):
            raise ValueError("Native GPU-time root does not match raw exported time")
        def decode(index):
            f = doc["frames"][index]
            name, kind = doc["strings"][f["n"]], doc["strings"][f["k"]]
            if kind == "native" and name == "cuda" and "e" in f:
                name = Path(doc["strings"][f["e"]]).name
            return name, kind
        full = [([decode(i) for i in e["f"]], e["c"]) for _, e in selected]
        focused = [([(n, k) for n, k in frames if k.startswith("cuda") or any(t in n for t in FOCUS)], weight) for frames, weight in full]
        assert all(frames for frames, _ in focused)
        total = math.fsum(weight for _, weight in full)
        prefix = OUTPUT / f"zymtrace-{word}-nested-graph-breaks"
        sidecar = {"pid": pid, "gpu_uuid": gpu, "unit": "fraction of captured GPU time", "records": [
            {"raw_event_index": i, "exported_time_weight": e["c"], "gpu_time_fraction": e["c"] / total, "frame_ids": e["f"],
             "frames": [{"name": n, "kind": k} for n, k in frames]}
            for (i, e), (frames, _) in zip(selected, full)]}
        Path(str(prefix) + ".stacks.json").write_text(json.dumps(sidecar, indent=2) + "\n")
        body = [label(55, 57, "ZYMTRACE / GPU-TIME FLAMEGRAPH", 22, True),
                label(55, 111, f"{word.capitalize()} nested graph-break support", 39, True),
                label(55, 151, "NVIDIA B200 · PyTorch 2.14.1 · depth-8 compiler microbenchmark", 22),
                label(55, 196, "Width = percentage of captured GPU time", 29, True),
                label(55, 231, "Filtered launch stacks · each worker normalized to its own 100% · time-weighted bars", 20)]
        bars(tree(focused), 55, 270, 1690, 25, total, body)
        body.extend([label(55, 815, "GPU work attributed to the code that launched it", 26, True),
                     label(55, 851, "Python/native frame width includes descendant GPU work. It does not measure CPU execution.", 21),
                     label(55, 892, f"Worker PID {pid} · {len(selected)} CUDA records · {len(doc['events']) - len(selected)} other-worker records excluded", 22),
                     label(55, 936, "Includes setup, warmup, timing and correctness. This is not a steady-state latency comparison.", 21),
                     label(55, 979, "Summed GPU activity can overlap. It is not elapsed wall time or a CPU/GPU causal percentage.", 20),
                     label(55, 1022, "Derived from retained Zymtrace exports. Full weighted stacks, unit evidence and code accompany this view.", 19),
                     label(55, 1065, "Bar tooltips retain inclusive GPU time and exact function names. Horizontal order is not chronology.", 19)])
        image_svg = Path(str(prefix) + ".svg")
        svg(image_svg, body, 1800, 1120)
        subprocess.run(["rsvg-convert", "-o", str(prefix) + ".png", str(image_svg)], check=True)
        subprocess.run(["rsvg-convert", "-w", "400", "-o", str(prefix) + ".preview.png", str(image_svg)], check=True)
        full_body = [label(55, 55, f"{word.capitalize()} nested graph-break support — full GPU-time stacks", 28, True),
                     label(55, 98, f"PID {pid} · width = percentage of captured GPU time · normalized within this worker", 21)]
        bars(tree(full), 55, 140, 2690, 21, total, full_body)
        svg(Path(str(prefix) + ".full-stacks.svg"), full_body, 2800, 190 + 21 * (max(len(frames) for frames, _ in full) + 1))
        metadata = {"pid": pid, "gpu_uuid": gpu, "accepted_records": len(selected),
                    "excluded_records": len(doc["events"]) - len(selected), "sum_exported_time_weights": total,
                    "sum_gpu_time_fractions": math.fsum(e["c"] / total for _, e in selected),
                    "native_unfiltered_root_weight": native_root,
                    "sha256": {str(p.relative_to(PACKAGE)): digest(p) for p in (source, audit_path, native_path)}}
        provenance["inputs"][arm] = metadata
        semantics["arms"][arm] = {"expected_pid": pid, "gpu_uuid": gpu,
            "events": {"path": str(source.relative_to(PACKAGE)), "sha256": digest(source)},
            "accepted_records": len(selected), "excluded_records": metadata["excluded_records"],
            "sum_exported_time_weights": total, "sum_gpu_time_fractions": metadata["sum_gpu_time_fractions"],
            "images": [str(Path(str(prefix) + ".png").relative_to(PACKAGE))]}
    for path in sorted(OUTPUT.iterdir()):
        if path.is_file():
            provenance["outputs"][str(path.relative_to(PACKAGE))] = {"sha256": digest(path), "bytes": path.stat().st_size}
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    (HERE / "profile-semantics.json").write_text(json.dumps(semantics, indent=2) + "\n")


if __name__ == "__main__":
    main()
