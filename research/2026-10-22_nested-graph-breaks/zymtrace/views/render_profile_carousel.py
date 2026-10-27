#!/usr/bin/env python3
"""Fit retained screenshots on square pages and assemble a four-page PDF."""
from __future__ import annotations

import base64
import json
from pathlib import Path
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET

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
    source_manifest = json.loads((OUTPUT / 'carousel-manifest.json').read_text())
    layout_manifest = json.loads((OUTPUT / 'layout-manifest.json').read_text())
    source_names = ['nsys-profile-square', 'ncu-profile-square',
                    'zymtrace-without-nested-graph-breaks-square',
                    'zymtrace-with-nested-graph-breaks-square']
    previous = {item.get('source_square_name', item['name']): item for item in source_manifest['pages']}
    manifest = {'schema_version': 2, 'renderer': Path(__file__).name,
                'renderer_sha256': social.source.digest(Path(__file__)),
                'canvas_px': [1400, 1750], 'content_translation_px': [0, 175],
                'content_scale': 1, 'reserved_overlay_px': 120,
                'method': 'Each unchanged 1400px square SVG is nested at y=175 on a 1400x1750 portrait page. Blank top and bottom buffers protect the content from viewer controls.',
                'pages': []}
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    pages = []
    for index, source_name in enumerate(source_names, 1):
        source_path = OUTPUT / (source_name + '.svg')
        original = ET.fromstring(source_path.read_text())
        if original.attrib.get('viewBox') != '0 0 1400 1400':
            raise ValueError('Expected the existing full square source without rescaling')
        original.set('x', '0')
        original.set('y', '175')
        wrapper = ET.Element('{http://www.w3.org/2000/svg}svg', {'width': '1400', 'height': '1750', 'viewBox': '0 0 1400 1750'})
        ET.SubElement(wrapper, '{http://www.w3.org/2000/svg}rect', {'width': '1400', 'height': '1750', 'fill': '#f8fafc'})
        wrapper.append(original)
        name = f'linkedin-page-{index:02d}'
        path = OUTPUT / (name + '.svg')
        path.write_text(ET.tostring(wrapper, encoding='unicode') + '\n')
        outputs = [path]
        for width in (1400, 400, 520):
            png = OUTPUT / (name + ('.png' if width == 1400 else f'.preview-{width}.png'))
            subprocess.run(['rsvg-convert', '-w', str(width), '-o', str(png), str(path)], check=True)
            outputs.append(png)
        source_data = layout_manifest['cards'].get(source_name, previous[source_name])
        text_boxes = []
        for box in source_data.get('text_boxes', []):
            x, y, w, h = box['bbox']
            y -= source_data.get('content_bounds', [0, 0, 0, 0])[1]
            shifted = [x, y + 175, w, h]
            if y + 175 < 120 or y + 175 + h > 1750 - 120:
                raise ValueError('Text overlaps the reserved viewer control area')
            text_boxes.append({**box, 'bbox': shifted})
        manifest['pages'].append({'name': name, 'canvas': [1400, 1750],
            'source_square_name': source_name, 'source': {'path': str(source_path.relative_to(PACKAGE)), 'sha256': social.source.digest(source_path)},
            'source_artifact': previous[source_name].get('source_artifact', previous[source_name].get('source')),
            'source_canvas': [1400, 1400], 'content_bounds': [0, 175, 1400, 1400],
            'text_boxes': text_boxes,
            'image_method': 'Complete existing SVG nested at y=175. All source content retained at its original size, with no crop or content rescaling.',
            'outputs': {item.name: {'sha256': social.source.digest(item), 'bytes': item.stat().st_size} for item in outputs}})
        pages.append(path)
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
    if not output.read_bytes().startswith(b'%PDF-'):
        raise ValueError('Missing PDF file signature')
    final = PdfReader(output)
    if len(final.pages) != 4 or any(abs(float(p.mediabox.height) / float(p.mediabox.width) - 1.25) > 1e-6 for p in final.pages):
        raise ValueError('Expected exactly four 4:5 portrait PDF pages')
    manifest['pdf'] = {'name': output.name, 'sha256': social.source.digest(output), 'bytes': output.stat().st_size, 'magic': '%PDF-',
        'page_count': len(final.pages), 'page_sizes_pt': [[float(p.mediabox.width), float(p.mediabox.height)] for p in final.pages]}
    (OUTPUT / 'carousel-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest['pdf']))


if __name__ == '__main__':
    main()
