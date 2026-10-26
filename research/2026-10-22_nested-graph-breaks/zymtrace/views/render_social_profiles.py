#!/usr/bin/env python3
"""Render uncropped social cards from the retained worker GPU-time stacks."""
from __future__ import annotations

import ctypes as C
import ctypes.util
import html
import json
import math
from pathlib import Path
import shutil
import subprocess

import render_time_profiles as source

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'social-layout'
KEYS = ('CUDAGraphNode.run', 'CUDAGraph.replay', 'cuLaunchKernel', 'cuGraphLaunch')


class Rectangle(C.Structure):
    _fields_ = [(key, C.c_int) for key in ('x', 'y', 'width', 'height')]


class TextMetrics:
    """Measure the same DejaVu Sans font used by librsvg through Pango."""
    def __init__(self):
        def library(name):
            path = C.util.find_library(name)
            if not path:
                path = '/opt/homebrew/lib/lib' + name + '.dylib'
            return C.CDLL(path)
        self.pango = library('pango-1.0')
        cairo = library('pangocairo-1.0')
        self.gobject = library('gobject-2.0')
        def api(lib, name, result, *args):
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = result, args
            return fn
        self.unref = api(self.gobject, 'g_object_unref', None, C.c_void_p)
        fontmap = api(cairo, 'pango_cairo_font_map_get_default', C.c_void_p)()
        self.context = api(self.pango, 'pango_font_map_create_context', C.c_void_p, C.c_void_p)(fontmap)
        self.layout_new = api(self.pango, 'pango_layout_new', C.c_void_p, C.c_void_p)
        self.description_new = api(self.pango, 'pango_font_description_from_string', C.c_void_p, C.c_char_p)
        self.description_size = api(self.pango, 'pango_font_description_set_absolute_size', None, C.c_void_p, C.c_double)
        self.description_free = api(self.pango, 'pango_font_description_free', None, C.c_void_p)
        self.layout_font = api(self.pango, 'pango_layout_set_font_description', None, C.c_void_p, C.c_void_p)
        self.layout_text = api(self.pango, 'pango_layout_set_text', None, C.c_void_p, C.c_char_p, C.c_int)
        self.extents = api(self.pango, 'pango_layout_get_pixel_extents', None, C.c_void_p, C.POINTER(Rectangle), C.POINTER(Rectangle))
        self.baseline = api(self.pango, 'pango_layout_get_baseline', C.c_int, C.c_void_p)
        self.cache = {}

    def measure(self, value, size, bold=False):
        key = value, size, bold
        if key not in self.cache:
            layout = self.layout_new(self.context)
            desc = self.description_new(('DejaVu Sans Bold' if bold else 'DejaVu Sans').encode())
            self.description_size(desc, size * 1024)
            self.layout_font(layout, desc)
            self.layout_text(layout, value.encode(), -1)
            ink, logical = Rectangle(), Rectangle()
            self.extents(layout, C.byref(ink), C.byref(logical))
            self.cache[key] = (ink.x, ink.y - self.baseline(layout) / 1024, ink.width, ink.height)
            self.description_free(desc)
            self.unref(layout)
        return self.cache[key]


class Canvas:
    def __init__(self, name, width, height, metrics):
        self.name, self.width, self.height, self.metrics = name, width, height, metrics
        self.body = [f'<rect width="{width}" height="{height}" fill="#f8fafc"/>']
        self.text_boxes, self.bar_boxes, self.profiles = [], [], []

    def text(self, x, y, value, size=32, bold=False, color='#172438'):
        dx, dy, w, h = self.metrics.measure(value, size, bold)
        box = {'text': value, 'size_px': size, 'bbox': [x + dx, y + dy, w, h]}
        self.text_boxes.append(box)
        self.body.append(f'<text x="{x}" y="{y}" fill="{color}" font-family="DejaVu Sans,sans-serif" font-size="{size}" font-weight="{"bold" if bold else "normal"}">{html.escape(value)}</text>')

    def bar(self, node, x, y, width, row, total):
        self.bar_boxes.append([x, y, width, row - 1])
        color = '#b9ddc9' if 'ngb_' in node['name'] else source.COLORS.get(node['kind'], '#dce5ed')
        tip = f"{node['name']}: {100 * node['weight'] / total:.6f}% of captured GPU time"
        self.body.append(f'<rect x="{x:.9f}" y="{y}" width="{width:.9f}" height="{row - 1}" fill="{color}" stroke="white" stroke-width="0.6"><title>{html.escape(tip)}</title></rect>')
        name = node['name']
        while name and self.metrics.measure(name, 22)[2] > width - 12:
            name = name[:-2] + '…' if not name.endswith('…') else name[:-2] + '…'
            if len(name) < 4:
                name = ''
        if name:
            self.text(x + 6, y + row - 7, name, 22)
        cursor = x
        for child in sorted(node['children'].values(), key=lambda n: (n['name'], n['kind'])):
            child_width = width * child['weight'] / node['weight']
            self.bar(child, cursor, y + row, child_width, row, total)
            cursor += child_width

    def profile(self, arm, y, row):
        height = row * (1 + max(len(frames) for frames, _ in arm['focused']))
        self.profiles.append({'arm': arm['word'], 'bounds': [60, y, 1280, height], 'accepted_records': len(arm['doc']['records']), 'focused_frames': sum(len(f) for f, _ in arm['focused']), 'weight_fraction': math.fsum(r['gpu_time_fraction'] for r in arm['doc']['records'])})
        self.bar(source.tree(arm['focused']), 60, y, 1280, row, arm['total'])

    def legend(self, arm, y, spacing=57, size=44, count=4):
        self.text(60, y, 'Key functions · inclusive GPU time', 32, True)
        for index, key in enumerate(KEYS[:count]):
            value = 100 * math.fsum(record['gpu_time_fraction'] for record in arm['doc']['records'] if any(frame['name'] == key for frame in record['frames']))
            self.text(60, y + (index + 1) * spacing, key, size)
            percent = f'{value:.2f}%'
            width = self.metrics.measure(percent, size, True)[2]
            self.text(1338 - width, y + (index + 1) * spacing, percent, size, True)

    def save(self):
        for box in [t['bbox'] for t in self.text_boxes] + self.bar_boxes + [p['bounds'] for p in self.profiles]:
            x, y, w, h = box
            if min(x, y, w, h) < -1e-6 or x + w > self.width + 1e-6 or y + h > self.height + 1e-6:
                raise ValueError(f'Off-canvas geometry in {self.name}: {box}')
        path = OUTPUT / (self.name + '.svg')
        path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" height="{self.height}" viewBox="0 0 {self.width} {self.height}">' + ''.join(self.body) + '</svg>\n')
        subprocess.run(['rsvg-convert', '-o', str(path.with_suffix('.png')), str(path)], check=True)
        subprocess.run(['rsvg-convert', '-w', '520', '-o', str(OUTPUT / (self.name + '.preview-520.png')), str(path)], check=True)
        subprocess.run(['rsvg-convert', '-w', '400', '-o', str(OUTPUT / (self.name + '.preview-400.png')), str(path)], check=True)
        return {'canvas': [self.width, self.height], 'aspect_ratio': f'{self.width}:{self.height}', 'text_boxes': self.text_boxes, 'bar_boxes': self.bar_boxes, 'profiles': self.profiles, 'outputs': {p.name: {'sha256': source.digest(p), 'bytes': p.stat().st_size} for p in (path, path.with_suffix('.png'), OUTPUT / (self.name + '.preview-520.png'), OUTPUT / (self.name + '.preview-400.png'))}}


def load_arm(word):
    path = HERE / 'time-weighted' / f'zymtrace-{word}-nested-graph-breaks.stacks.json'
    doc = json.loads(path.read_text())
    total = math.fsum(r['exported_time_weight'] for r in doc['records'])
    if not total or any(not math.isclose(r['gpu_time_fraction'], r['exported_time_weight'] / total, rel_tol=1e-12) for r in doc['records']):
        raise ValueError('Sidecar fractions do not match per-worker exported GPU-time weights')
    focused = [([(f['name'], f['kind']) for f in r['frames'] if f['kind'].startswith('cuda') or any(token in f['name'] for token in source.FOCUS)], r['exported_time_weight']) for r in doc['records']]
    if any(not frames for frames, _ in focused):
        raise ValueError('Mechanism filter would discard an accepted row')
    return {'word': word, 'path': path, 'doc': doc, 'total': total, 'focused': focused}


def main():
    source.read_authority()
    OUTPUT.mkdir(exist_ok=True)
    metrics = TextMetrics()
    arms = [load_arm(word) for word in ('without', 'with')]
    manifest = {'schema_version': 1, 'renderer': Path(__file__).name, 'renderer_sha256': source.digest(Path(__file__)), 'unit': 'fraction of captured GPU time', 'normalization': 'c / sum(c), separately for each worker', 'phases': ['setup', 'warmup', 'timing', 'correctness'], 'scope': 'Mechanism-focused stacks retain every accepted event and every frame selected by render_time_profiles.FOCUS. Full unfiltered stacks remain in time-weighted/*.full-stacks.svg.', 'text_bbox_method': 'Pango ink extents for DejaVu Sans at absolute pixel size', 'inputs': {}, 'cards': {}}
    for arm in arms:
        name = f'zymtrace-{arm["word"]}-nested-graph-breaks-square'
        card = Canvas(name, 1400, 1400, metrics)
        card.text(60, 63, 'ZYMTRACE / CAPTURED GPU TIME', 32, True)
        card.text(60, 143, f'{arm["word"].capitalize()} nested', 56, True)
        card.text(60, 209, 'graph-break support', 56, True)
        card.text(60, 259, 'B200 · PyTorch 2.14.1 · depth-8 microbenchmark', 32)
        card.text(60, 309, 'Bar width = % of this worker’s captured GPU time', 32, True)
        card.profile(arm, 340, 30)
        card.legend(arm, 966, 57, 44, 4)
        card.text(60, 1238, 'Each arm normalized separately. Inclusive shares overlap.', 28)
        card.text(60, 1283, 'Includes setup, warmup, timing and correctness work.', 28)
        card.text(60, 1328, 'GPU time attributed to host frames is not CPU execution time.', 28)
        card.text(60, 1373, 'Not steady-state latency. Full function stacks linked with the post.', 28)
        manifest['cards'][name] = card.save()
        shutil.copyfile(arm['path'], OUTPUT / (name + '.stacks.json'))
        manifest['inputs'][arm['word']] = {'path': str(arm['path'].relative_to(source.PACKAGE)), 'sha256': source.digest(arm['path']), 'pid': arm['doc']['pid'], 'gpu_uuid': arm['doc']['gpu_uuid'], 'accepted_records': len(arm['doc']['records'])}
    card = Canvas('zymtrace-nested-graph-breaks-comparison-portrait', 1400, 1750, metrics)
    card.text(60, 63, 'ZYMTRACE / CAPTURED GPU TIME', 32, True)
    card.text(60, 141, 'Where GPU work is launched', 56, True)
    card.text(60, 192, 'B200 · PyTorch 2.14.1 · depth-8 microbenchmark', 32)
    card.text(60, 241, 'Bar width = % captured GPU time · each arm normalized separately', 30)
    card.text(60, 313, 'Without nested graph-break support', 44, True)
    card.profile(arms[0], 341, 26)
    card.legend(arms[0], 880, 46, 44, 3)
    card.text(60, 1110, 'With nested graph-break support', 44, True)
    card.profile(arms[1], 1138, 26)
    card.legend(arms[1], 1490, 45, 44, 3)
    card.text(60, 1670, 'Whole capture: setup, warmup, timing and correctness. Not steady latency.', 28)
    card.text(60, 1706, 'Host frames inherit GPU time, not CPU time. Inclusive function shares overlap.', 28)
    card.text(60, 1742, 'All focused bars shown. Full unfiltered function stacks linked with the post.', 28)
    manifest['cards'][card.name] = card.save()
    refs = {'unit': 'fraction of captured GPU time', 'normalization': 'independent per worker', 'sources': manifest['inputs']}
    (OUTPUT / (card.name + '.stacks.json')).write_text(json.dumps(refs, indent=2) + '\n')
    (OUTPUT / 'layout-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
