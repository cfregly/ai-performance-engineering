#!/usr/bin/env python3
"""Render the three selected scientific figures from retained inputs.

Requires existing Pillow, rsvg-convert and Fontconfig. No network or GPU is used.
Run with --check for exact SVG, PNG and text regeneration without output writes.
All essential text is at least 52 px on a 1200 px canvas, or 12.87 px at 297 px.
Editable SVGs require their retained PNG artwork beside them. Only PNG conversion
embeds those exact assets in memory, so stored SVGs stay portable and readable.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "images/scientific"
SOURCE = ROOT / "evidence/reader-summary.json"
TOKENS = OUT / "visual-tokens.json"
FONT = ROOT / "images/fonts/Inter.ttf"
NS = "http://www.w3.org/2000/svg"
SIZE, SAFE, MINIMUM = 1200, 84, 52
LINE_TEXT_CLEARANCE = 10


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def segment_box_clearance(line, box):
    """Return the shortest text-box distance from the visible line stroke."""
    x1, y1, x2, y2 = (line[key] for key in ('x1', 'y1', 'x2', 'y2'))
    dx, dy = x2 - x1, y2 - y1
    enter, leave = 0.0, 1.0
    for p, q in ((-dx, x1 - box['left']), (dx, box['right'] - x1),
                 (-dy, y1 - box['top']), (dy, box['bottom'] - y1)):
        if p == 0:
            if q < 0:
                break
        elif p < 0:
            enter = max(enter, q / p)
        else:
            leave = min(leave, q / p)
    else:
        if enter <= leave:
            return -line['stroke_width'] / 2

    def point_segment_distance(x, y):
        length_squared = dx * dx + dy * dy
        t = max(0, min(1, ((x - x1) * dx + (y - y1) * dy) / length_squared)) if length_squared else 0
        return math.hypot(x - x1 - t * dx, y - y1 - t * dy)

    def point_box_distance(x, y):
        return math.hypot(max(box['left'] - x, 0, x - box['right']),
                          max(box['top'] - y, 0, y - box['bottom']))

    distance = min(point_box_distance(x1, y1), point_box_distance(x2, y2),
                   *(point_segment_distance(x, y)
                     for x in (box['left'], box['right'])
                     for y in (box['top'], box['bottom'])))
    return distance - line['stroke_width'] / 2


def contrast(foreground, background):
    def luminance(color):
        channels = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        linear = [c / 12.92 if c <= .04045 else ((c + .055) / 1.055) ** 2.4 for c in channels]
        return sum(c * weight for c, weight in zip(linear, (.2126, .7152, .0722)))
    low, high = sorted((luminance(foreground), luminance(background)))
    return (high + .05) / (low + .05)


class Figure:
    def __init__(self, title, description, colors):
        self.colors = colors
        self.parts = [f'<svg xmlns="{NS}" width="1200" height="1200" viewBox="0 0 1200 1200" role="img" aria-labelledby="title description">',
                      f'<title id="title">{html.escape(title)}</title>',
                      f'<desc id="description">{html.escape(description)}</desc>',
                      f'<rect width="1200" height="1200" fill="{colors["background"]}"/>',
                      '<defs>']
        for key in ("ink", "blue", "orange"):
            self.parts.append(f'<marker id="{key}" markerWidth="20" markerHeight="18" refX="18" refY="9" orient="auto-start-reverse" markerUnits="userSpaceOnUse"><path d="M 2 2 L 18 9 L 2 16" fill="none" stroke="{colors[key]}" stroke-width="4"/></marker>')
        self.parts.append('</defs><g font-family="Inter">')
        self.labels, self.text_boxes, self.geometry, self.lines = [], [], [], []

    def text(self, x, y, value, size=52, anchor="middle", color="ink", weight=400, surface="background"):
        require(size >= MINIMUM, f"Essential label below {MINIMUM} px: {value}")
        font = ImageFont.truetype(str(FONT), size)
        font.set_variation_by_axes([14, weight])
        width = font.getlength(value)
        left = x if anchor == "start" else x - width if anchor == "end" else x - width / 2
        box = font.getbbox(value, anchor="ls")
        top, bottom = y + box[1], y + box[3]
        require(left >= SAFE and left + width <= SIZE - SAFE, f"Text overflows horizontal safe margin: {value}")
        require(top >= SAFE and bottom <= SIZE - SAFE, f"Text overflows vertical safe margin: {value}")
        ratio = contrast(self.colors.get(color, color), self.colors[surface])
        require(ratio >= 4.5, f'Text contrast below 4.5:1: {value}')
        self.labels.append(value)
        self.text_boxes.append({"text": value, "left": left, "top": top, "right": left + width, "bottom": bottom, "font_size": size, "contrast_ratio": ratio})
        self.parts.append(f'<text x="{x:g}" y="{y:g}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{self.colors.get(color, color)}">{html.escape(value)}</text>')

    def rect(self, x, y, width, height, color="background", stroke=None, sw=3):
        require(all(math.isfinite(v) for v in (x, y, width, height)), "Nonfinite rectangle")
        require(width >= 0 and height >= 0, "Negative rectangle")
        require(x >= SAFE and y >= SAFE and x + width <= SIZE - SAFE and y + height <= SIZE - SAFE, "Rectangle outside safe margins")
        extra = f' stroke="{self.colors[stroke]}" stroke-width="{sw}"' if stroke else ''
        self.parts.append(f'<rect x="{x:.9f}" y="{y:g}" width="{width:.9f}" height="{height:g}" fill="{self.colors[color]}"{extra}/>')

    def line(self, x1, y1, x2, y2, color="ink", sw=3, arrow=False, both=False, dashed=False):
        require(all(math.isfinite(v) for v in (x1, y1, x2, y2, sw)) and sw > 0, "Invalid line")
        require(min(x1, x2, y1, y2) >= SAFE and max(x1, x2, y1, y2) <= SIZE - SAFE, "Line outside safe margins")
        extra = ' stroke-dasharray="12 10"' if dashed else ''
        if arrow or both:
            extra += f' marker-end="url(#{color})"'
        if both:
            extra += f' marker-start="url(#{color})"'
        self.lines.append({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'stroke_width': sw})
        self.parts.append(f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" stroke="{self.colors[color]}" stroke-width="{sw}"{extra}/>')

    def image(self, name, x, y, width, height, crop=None):
        require(name in {'rack-traffic-scene.png', 'ordered-barrier-scene.png'}, f'Unsupported retained artwork: {name}')
        if crop:
            self.parts.append(f'<svg x="{x}" y="{y}" width="{width}" height="{height}" viewBox="{crop}" overflow="hidden"><image width="1900" height="360" href="{name}"/></svg>')
        else:
            self.parts.append(f'<image x="{x}" y="{y}" width="{width}" height="{height}" href="{name}"/>')

    def finish(self):
        require(bool(self.labels), "Figure has no labels")
        for i, first in enumerate(self.text_boxes):
            for second in self.text_boxes[i + 1:]:
                overlap_x = min(first['right'], second['right']) - max(first['left'], second['left'])
                overlap_y = min(first['bottom'], second['bottom']) - max(first['top'], second['top'])
                require(not (overlap_x > 0 and overlap_y > 0), f"Text collision: {first['text']} / {second['text']}")
        for line in self.lines:
            for box in self.text_boxes:
                clearance = segment_box_clearance(line, box)
                require(clearance >= LINE_TEXT_CLEARANCE,
                        f"Line/text clearance below {LINE_TEXT_CLEARANCE} px: {box['text']} ({clearance:.2f} px)")
        data = ("\n".join(self.parts + ['</g></svg>']) + "\n").encode()
        ET.fromstring(data)
        return data


def validate_source(source):
    require(source['model']['GPUs'] == 64, 'Selected figures require the retained 64-GPU example')
    require(source['placement_nodes_by_rack']['B'] == [[0, 1, 4, 5], [2, 3, 6, 7]], 'Unexpected placement B')
    traffic = source['traffic_and_screens']
    a, b = (traffic[k]['aggregate_sent_bytes'] for k in ('A', 'B'))
    require(a['inter_rack'] + a['intra_rack'] == b['inter_rack'] + b['intra_rack'], 'Off-node payload is not conserved')
    for name in ('A', 'B'):
        row = traffic[name]['floors_ms']
        for key in ('pre_clipping_phase', 'post_clipping_phase', 'necessary_step_floor'):
            require(math.isfinite(row[key]) and row[key] >= 0, f'Invalid {name} {key}')
        require(abs(row['pre_clipping_phase'] + row['post_clipping_phase'] - row['necessary_step_floor']) < 1e-6, f'Ordered phase sum differs for {name}')


def traffic_figure(source, colors):
    figure = Figure('Placement moves rack traffic', 'Hypothetical logical cross-rack payload, both directions combined per update. B keeps within-rack loops and a thin symbolic cross-rack pipeline connection. Artwork and connector widths explain placement, not quantitative proportions. Within-rack payload is between nodes. Both placements have the same total off-node payload.', colors)
    figure.text(600, 136, 'Placement moves rack traffic', 64, weight=600)
    figure.text(600, 204, 'Same 64 GPUs · hypothetical update')
    figure.text(382, 268, 'A · before', 60, weight=600)
    figure.text(840, 268, 'B · after', 60, weight=600)
    figure.image('rack-traffic-scene.png', 180, 283, 840, 357.623762376)
    # One symbolic pipeline connection crosses B's rack boundary. Its fixed
    # thin stroke is an illustration convention, never a payload scale.
    figure.line(766, 466, 868, 466, 'blue', 3, arrow=True)
    figure.geometry.append({'B_illustration_cross_rack_pipeline_connection': True,
                            'artwork_connector_stroke_px': 3,
                            'artwork_connector_width_encodes_quantity': False,
                            'within_rack_loops_retained': True})
    figure.text(600, 692, 'Cross-rack GB · both directions')
    x0, scale = 160, 900 / 30
    for y, name, color in ((720, 'A', 'blue'), (830, 'B', 'orange')):
        value = source['traffic_and_screens'][name]['aggregate_sent_bytes']['inter_rack'] / 1e9
        figure.rect(x0, y, value * scale, 72, color)
        figure.text(110, y + 55, name, 60, weight=600)
        figure.text(188, y + 55, f'{value:.3f} GB', 56, 'start', '#FFFFFF' if name == 'A' else 'ink', 600, 'blue' if name == 'A' else 'background')
        figure.geometry.append({'placement': name, 'cross_rack_GB': value, 'origin_x': x0, 'width': value * scale, 'px_per_GB': scale, 'zero_baseline': True})
    figure.line(x0, 710, x0, 925)
    figure.line(x0, 925, 1060, 925)
    for value in (0, 10, 20, 30):
        figure.text(x0 + value * scale, 988, str(value))
    a, b = (source['traffic_and_screens'][k]['aggregate_sent_bytes']['intra_rack'] / 1e9 for k in ('A', 'B'))
    figure.text(600, 1052, f'Within racks: {a:.3f} → {b:.3f} GB')
    figure.text(600, 1100, 'Total off-node payload unchanged')
    return figure


def floor_figure(source, colors):
    figure = Figure('Two ordered lower bounds', 'The illustration shows a clipping barrier. It is separate from the dashed illustrative 150 ms budget on the chart below. Blue is before clipping and orange after clipping. The busiest rank updates local experts and gathered attention matrices. All values are lower bounds under assumed rates, not runtimes.', colors)
    figure.text(600, 136, 'Two ordered lower bounds', 64, weight=600)
    figure.text(600, 204, 'Hypothetical plan · ms per update')
    figure.text(108, 266, 'A: original · B: moved', anchor='start')
    require(figure.text_boxes[-1]['right'] <= 640, 'Placement key overlaps the barrier illustration')
    for y, label, color in ((326, 'Pre-clip', 'blue'), (416, 'Post-clip', 'orange')):
        figure.rect(108, y - 37, 32, 32, color)
        figure.text(160, y, label, anchor='start')
    figure.image('ordered-barrier-scene.png', 650, 245, 440, 176, '500 0 900 360')
    figure.text(898, 477, 'Clipping barrier')
    x0, scale, budget = 160, 880 / 300, 150
    figure.text(600, 542, 'Illustrative 150 ms budget')
    for y, name in ((565, 'A'), (670, 'B')):
        row = source['traffic_and_screens'][name]['floors_ms']
        pre, post = row['pre_clipping_phase'], row['post_clipping_phase']
        total = pre + post
        figure.rect(x0, y, pre * scale, 74, 'blue')
        figure.rect(x0 + pre * scale, y, post * scale, 74, 'orange')
        figure.text(110, y + 56, name, 60, weight=600)
        figure.text(x0 + pre * scale / 2, y + 56, f'{pre:.2f}', 56, color='#FFFFFF', surface='blue')
        post_label_x = x0 + (pre + post / 2) * scale - (18 if name == 'B' else 0)
        figure.text(post_label_x, y + 56, f'{post:.2f}', 56, surface='orange')
        figure.text(x0 + total * scale + 18, y + 56, f'{total:.2f}', 60, 'start', weight=600)
        figure.geometry.append({'placement': name, 'pre_ms': pre, 'post_ms': post, 'total_ms': total, 'origin_x': x0, 'pre_width': pre * scale, 'post_width': post * scale, 'px_per_ms': scale, 'zero_baseline': True})
    # Emit the budget after the bars so its trajectory stays visible across them.
    figure.line(x0 + budget * scale, 570, x0 + budget * scale, 755, dashed=True, sw=4)
    figure.line(x0, 558, x0, 755)
    figure.line(x0, 755, 1040, 755)
    for value in (0, 150, 300):
        figure.text(x0 + value * scale, 815, str(value))
    caps = source['assumed_capabilities']
    for y, label in zip((880, 935, 990, 1045, 1100), (
        'Full DP all-reduce before norm',
        'FP32 Muon · one attention owner',
        f"Assumed {caps['Muon_FP32_TF_s_GPU']} TFLOP/s per GPU",
        f"Rack {caps['rack_uplink_GB_s_each_direction']} / node {caps['node_NIC_GB_s_each_direction']} GB/s per direction",
        'Balanced routing · not runtimes',
    )):
        figure.text(600, y, label)
    figure.geometry.append({'budget_ms': budget, 'budget_x': x0 + budget * scale, 'budget_emitted_after_bars': True, 'B_post_clip_label_offset_px': -18, 'illustration_barrier_x_approx': 903, 'illustration_is_not_a_time_axis': True})
    return figure


def topology_figure(source, colors):
    figure = Figure('Pair stages inside racks', 'Placement B has stages 0 and 1 of both replicas in rack 0 and stages 2 and 3 in rack 1. The two thick blue P1 to P2 pipeline hops cross racks. Thin blue hops and orange data-parallel pairs stay inside a rack.', colors)
    figure.text(600, 136, 'Pair stages inside racks', 64, weight=600)
    figure.text(600, 204, 'B after · Replica 0 top · Replica 1 bottom')
    centers = {0: 170, 1: 470, 2: 730, 3: 1030}
    for rack, (left, title_x) in enumerate(((84, 319), (646, 881))):
        figure.rect(left, 250, 470, 730, stroke='ink')
        figure.text(title_x, 322, f'Rack {rack}', 60, weight=600)
    for stage, x in centers.items():
        figure.line(x, 562, x, 820, 'orange', 4, both=True)
        for replica, y in ((0, 442), (1, 820)):
            figure.rect(x - 70, y, 140, 120, stroke='ink')
            figure.text(x, y + 81, f'P{stage}', 64, weight=600)
            figure.geometry.append({'replica': replica, 'stage': stage, 'rack': stage // 2, 'node': replica * 4 + stage})
    for y in (502, 880):
        for stage in (0, 1, 2):
            cross = stage == 1
            figure.line(centers[stage] + 70, y, centers[stage + 1] - 70, y, 'blue', 12 if cross else 3, arrow=True)
    figure.text(600, 1045, 'P1 → P2 crosses racks', weight=600)
    figure.text(600, 1100, 'DP inside racks · 8 GPUs per node')
    figure.geometry.append({'cross_rack_PP_hops': [[1, 2], [5, 6]], 'cross_rack_stroke_px': 12, 'intrarack_stroke_px': 3, 'replica_labels_above_DP_span': True, 'replica_key_baseline_y': 204, 'DP_span_top_y': 562})
    return figure


BUILDERS = {'14-topology-b': topology_figure, '15-tier-traffic': traffic_figure, '16-ordered-floors': floor_figure}


def retained_font_environment(config):
    env = dict(os.environ, FONTCONFIG_FILE=str(config), LC_ALL='C')
    matched = subprocess.run(['fc-match', '-f', '%{family}|%{file}', 'Inter'], capture_output=True, text=True, env=env)
    require(matched.returncode == 0, f'fc-match failed: {matched.stderr}')
    family, separator, filename = matched.stdout.partition('|')
    require(separator and family == 'Inter', f'Inter resolution failed: {matched.stdout}')
    require(Path(filename).resolve() == FONT.resolve(), f'Unexpected Inter file resolved: {filename}')
    require(digest(Path(filename).read_bytes()) == digest(FONT.read_bytes()), 'Resolved Inter bytes differ')
    return env


def render_png(svg, env):
    # Preserve source geometry byte for byte while resolving only admitted assets.
    resolved = svg.decode()
    for image in ET.fromstring(svg).iter(f'{{{NS}}}image'):
        name = image.get('href')
        require(name in {'rack-traffic-scene.png', 'ordered-barrier-scene.png'}, f'Unsupported retained artwork: {name}')
        asset = OUT / name
        require(asset.is_file(), f'Required retained artwork is missing: {name}')
        data = asset.read_bytes()
        require(data.startswith(b'\x89PNG\r\n\x1a\n'), f'Retained artwork is not PNG: {name}')
        embedded = base64.b64encode(data).decode()
        resolved = resolved.replace(f'href="{name}"', f'href="data:image/png;base64,{embedded}"')
    result = subprocess.run(['rsvg-convert', '--width', '2400', '--height', '2400'], input=resolved.encode(), capture_output=True, env=env)
    require(result.returncode == 0, f'rsvg-convert failed: {result.stderr.decode(errors="replace")}')
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    for required in (SOURCE, TOKENS, FONT, OUT / 'ordered-barrier-scene.png', OUT / 'rack-traffic-scene.png'):
        require(required.is_file(), f'Required retained input is missing: {required.name}')
    source = json.loads(SOURCE.read_text())
    tokens = json.loads(TOKENS.read_text())
    require(tokens['typography']['font_family'] == 'Inter', 'Inter is required')
    validate_source(source)
    receipt = {'source_sha256': digest(SOURCE.read_bytes()), 'tokens_sha256': digest(TOKENS.read_bytes()), 'font_sha256': digest(FONT.read_bytes()), 'artwork_sha256': {name: digest((OUT / name).read_bytes()) for name in ('rack-traffic-scene.png', 'ordered-barrier-scene.png')}, 'stored_svg_artwork': 'relative hrefs resolved only in memory for PNG conversion', 'rsvg_version': subprocess.check_output(['rsvg-convert', '--version'], text=True).splitlines()[0], 'figures': []}
    outputs = []
    with tempfile.TemporaryDirectory(prefix='scientific-figures-') as temp:
        config = Path(temp) / 'fonts.conf'
        config.write_text(f'<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><dir>{html.escape(str(FONT.parent))}</dir><cachedir>{html.escape(temp)}</cachedir></fontconfig>')
        env = retained_font_environment(config)
        receipt['font_resolution'] = {'family': 'Inter', 'retained_file': 'images/fonts/Inter.ttf', 'exact_path_and_bytes': True}
        for name, builder in BUILDERS.items():
            figure = builder(source, tokens['colors'])
            svg = figure.finish()
            # Check semantic outputs before attempting rasterization or PNG
            # equality, so renderer-stack differences cannot hide TXT drift.
            generated = {'svg': svg, 'txt': ('\n'.join(figure.labels) + '\n').encode()}
            if args.check:
                for extension, data in generated.items():
                    path = OUT / f'{name}.{extension}'
                    require(path.is_file(), f'Retained output is missing: {path.name}')
                    require(path.read_bytes() == data, f'Regeneration differs: {path.name}.')
            generated['png'] = render_png(svg, env)
            for extension, data in generated.items():
                path = OUT / f'{name}.{extension}'
                if args.check and extension == 'png':
                    require(path.is_file(), f'Retained output is missing: {path.name}')
                    hint = f" PNG byte equality depends on the rendering stack. Tested with rsvg-convert 2.62.3 and Pillow 11.3.0; current {receipt['rsvg_version']}. SVG/TXT already match."
                    require(path.read_bytes() == data, f'Regeneration differs: {path.name}.' + hint)
                outputs.append((path, data))
            receipt['figures'].append({'name': name, 'hashes': {key: digest(data) for key, data in generated.items()}, 'viewbox': [0, 0, 1200, 1200], 'essential_font_min_px': min(box['font_size'] for box in figure.text_boxes), 'minimum_at_297_px': min(box['font_size'] for box in figure.text_boxes) * 297 / SIZE, 'minimum_at_312_px': min(box['font_size'] for box in figure.text_boxes) * 312 / SIZE, 'minimum_text_contrast_ratio': min(box['contrast_ratio'] for box in figure.text_boxes), 'text_safe_margins': SAFE, 'text_collision_check': 'pass', 'line_text_clearance_min_px': LINE_TEXT_CLEARANCE, 'line_text_clearance_observed_px': min(segment_box_clearance(line, box) for line in figure.lines for box in figure.text_boxes), 'line_text_clearance_check': 'pass', 'geometry': figure.geometry})
    # Validate every figure before writing any output.
    if not args.check:
        for path, data in outputs:
            path.write_bytes(data)
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
    print(f"{'Verified' if args.check else 'Wrote'} three selected SVG/PNG figures. All text >=52 px, >=12.87 px at 297 px width. Exact values, safe margins, text collisions and >=10 px line/text clearance pass.")


if __name__ == '__main__':
    main()
