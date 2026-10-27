#!/usr/bin/env python3
"""Fit complete square profile cards to Typefully's observed X image frame."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

import render_time_profiles as source

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / 'social-layout'
PAIRS = (
    ('nsys-profile-square', 'nsys-profile-x'),
    ('ncu-profile-square', 'ncu-profile-x'),
    ('zymtrace-without-nested-graph-breaks-square', 'zymtrace-without-nested-graph-breaks-x'),
    ('zymtrace-with-nested-graph-breaks-square', 'zymtrace-with-nested-graph-breaks-x'),
)


def wrap(source_path, target_path):
    """Translate the unchanged 1400px source by 14px on a 1428px canvas."""
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    original = ET.fromstring(source_path.read_text())
    if original.attrib.get('viewBox') != '0 0 1400 1400':
        raise ValueError('Expected an existing 1400px square profile card')
    original.set('x', '14')
    original.set('y', '0')
    wrapper = ET.Element('{http://www.w3.org/2000/svg}svg', {'width': '1428', 'height': '1400', 'viewBox': '0 0 1428 1400'})
    ET.SubElement(wrapper, '{http://www.w3.org/2000/svg}rect', {'width': '1428', 'height': '1400', 'fill': '#f8fafc'})
    wrapper.append(original)
    target_path.write_text(ET.tostring(wrapper, encoding='unicode') + '\n')
    nested = ET.fromstring(target_path.read_text()).find('{http://www.w3.org/2000/svg}svg')
    assert nested.attrib.pop('x') == '14'
    assert nested.attrib.pop('y') == '0'
    assert ET.tostring(nested) == ET.tostring(ET.fromstring(source_path.read_text()))


def main():
    manifest = {'schema_version': 1, 'renderer': Path(__file__).name, 'renderer_sha256': source.digest(Path(__file__)),
                'canvas': [1428, 1400], 'source_canvas': [1400, 1400],
                'content_offset': [14, 0], 'content_scale': 1,
                'observed_display_frame': [520, 510],
                'expected_object_fit_cover_loss_fraction': 1 - (520 / 510) / (1428 / 1400),
                'method': 'Unchanged square SVG nested at x=14, y=0 with 14px side margins. No source rescaling or cropping.',
                'cards': {}}
    if manifest['expected_object_fit_cover_loss_fraction'] > 0.005:
        raise ValueError('X layout exceeds the permitted crop fraction')
    for source_name, name in PAIRS:
        source_path = OUTPUT / (source_name + '.svg')
        path = OUTPUT / (name + '.svg')
        wrap(source_path, path)
        outputs = [path]
        for width in (1428, 400, 520):
            png = OUTPUT / (name + ('.png' if width == 1428 else f'.preview-{width}.png'))
            subprocess.run(['rsvg-convert', '-w', str(width), '-o', str(png), str(path)], check=True)
            outputs.append(png)
        sidecar = OUTPUT / (source_name + '.stacks.json')
        if sidecar.exists():
            copied = OUTPUT / (name + '.stacks.json')
            shutil.copyfile(sidecar, copied)
            assert copied.read_bytes() == sidecar.read_bytes()
            outputs.append(copied)
        manifest['cards'][name] = {'source': {'path': str(source_path.relative_to(source.PACKAGE)), 'sha256': source.digest(source_path)},
            'content_bounds': [14, 0, 1400, 1400],
            'outputs': {p.name: {'sha256': source.digest(p), 'bytes': p.stat().st_size} for p in outputs}}
    (OUTPUT / 'x-layout-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'outputs': list(manifest['cards']), 'crop_fraction': manifest['expected_object_fit_cover_loss_fraction']}))


if __name__ == '__main__':
    main()
