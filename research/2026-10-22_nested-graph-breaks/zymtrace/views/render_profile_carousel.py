#!/usr/bin/env python3
"""Fit retained screenshots on square pages and assemble a four-page PDF."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import struct
import subprocess
import tempfile

from pypdf import PdfReader, PdfWriter

import render_social_profiles as social

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'social-layout'
PACKAGE = HERE.parents[1]


def png_dimensions(path):
    data = path.read_bytes()
    if data[:8] != b'\x89PNG\r\n\x1a\n':
        raise ValueError('Expected a retained PNG')
    return struct.unpack('>II', data[16:24])


def screenshot_page(name, source, title, subtitle, labels, footer, metrics):
    canvas = social.Canvas(name, 1400, 1400, metrics)
    canvas.text(60, 63, 'RETAINED NVIDIA PROFILE', 32, True)
    canvas.text(60, 142, title, 56, True)
    canvas.text(60, 196, subtitle, 32)
    canvas.text(60, 268, labels[0], 44, True)
    canvas.text(60, 324, labels[1], 44, True)
    width, height = png_dimensions(source)
    fitted_width = 1320
    fitted_height = height * fitted_width / width
    image_box = [40, 364, fitted_width, fitted_height]
    if image_box[1] + fitted_height > 1100:
        raise ValueError('Screenshot exceeds its reserved area')
    encoded = base64.b64encode(source.read_bytes()).decode('ascii')
    canvas.body.append(f'<image x="40" y="364" width="{fitted_width}" height="{fitted_height}" preserveAspectRatio="xMidYMid meet" href="data:image/png;base64,{encoded}"/>')
    canvas.text(60, 1173, footer[0], 44, True)
    canvas.text(60, 1230, footer[1], 32)
    canvas.text(60, 1278, footer[2], 32)
    canvas.text(60, 1336, 'Original capture, report and code linked with the post.', 30)
    result = canvas.save()
    result['source'] = {'path': str(source.relative_to(PACKAGE)), 'sha256': social.source.digest(source), 'dimensions_px': [width, height]}
    result['image_bounds'] = image_box
    result['image_method'] = 'Unchanged retained PNG bytes embedded as a data URI. Uniform aspect-preserving fit, with no crop or report redraw.'
    return result


def main():
    OUTPUT.mkdir(exist_ok=True)
    metrics = social.TextMetrics()
    inputs = [
        ('nsys-profile-square', PACKAGE / 'profiles/views/nsys-first-post-warmup-invocation.png',
         'Nsys: first post-warmup call', 'B200 · PyTorch 2.14.1 · graph-level trace',
         ('Top: without nested graph-break support', 'Bottom: with nested graph-break support'),
         ('One synchronized invocation per arm.', 'Retained timeline fitted in full. Shared time axis in µs.', 'Graph-level trace, separate from acceptance timing.')),
        ('ncu-profile-square', PACKAGE / 'profiles/views/ncu-single-kernel-metrics.png',
         'NCU: one kernel record per arm', 'B200 · retained Nsight Compute metrics',
         ('False column: without nested graph-break support', 'True column: with nested graph-break support'),
         ('Single-kernel profiler replay.', 'Retained report values and units remain unchanged.', 'Separate from unprofiled acceptance timing.')),
    ]
    manifest = {'schema_version': 1, 'renderer': Path(__file__).name, 'renderer_sha256': social.source.digest(Path(__file__)), 'pages': []}
    for name, source, title, subtitle, labels, footer in inputs:
        manifest['pages'].append({'name': name, **screenshot_page(name, source, title, subtitle, labels, footer, metrics)})
    pages = [OUTPUT / (item['name'] + '.svg') for item in manifest['pages']]
    for word in ('without', 'with'):
        path = OUTPUT / f'zymtrace-{word}-nested-graph-breaks-square.svg'
        if not path.is_file():
            raise FileNotFoundError('Render the social profile cards first')
        pages.append(path)
        manifest['pages'].append({'name': path.stem, 'canvas': [1400, 1400], 'source': {'path': str(path.relative_to(PACKAGE)), 'sha256': social.source.digest(path)}, 'image_method': 'Complete existing time-weighted square SVG page, unchanged.'})
    writer = PdfWriter()
    with tempfile.TemporaryDirectory(prefix='profile-carousel-') as temporary:
        for index, path in enumerate(pages):
            pdf = Path(temporary) / f'{index + 1}.pdf'
            subprocess.run(['rsvg-convert', '-f', 'pdf', '-o', str(pdf), str(path)], check=True)
            reader = PdfReader(pdf)
            if len(reader.pages) != 1:
                raise ValueError('Each profile must produce one page')
            writer.add_page(reader.pages[0])
        writer.add_metadata({'/Title': 'Nested graph-break support: retained profiles', '/Subject': 'Nsys invocation, NCU kernel replay, and per-worker Zymtrace GPU-time stacks'})
        output = OUTPUT / 'linkedin-profiles.pdf'
        with output.open('wb') as file:
            writer.write(file)
    final = PdfReader(output)
    if len(final.pages) != 4 or any(abs(float(p.mediabox.width) - float(p.mediabox.height)) > 0.01 for p in final.pages):
        raise ValueError('Expected exactly four square PDF pages')
    manifest['pdf'] = {'name': output.name, 'sha256': social.source.digest(output), 'bytes': output.stat().st_size, 'page_count': len(final.pages), 'page_sizes_pt': [[float(p.mediabox.width), float(p.mediabox.height)] for p in final.pages]}
    (OUTPUT / 'carousel-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['pdf']))


if __name__ == '__main__':
    main()
