#!/usr/bin/env python3
"""Reproduce article-only badge removal and the blank-slot legend correction.

Existing designed PNGs remain the source: this does not reconstruct their data
marks. Stored SVGs retain relative source-image links and editable annotations.
Requires existing Pillow, rsvg-convert and Fontconfig. No scientific calculator,
network, model, GPU or account operation is invoked. Default is read-only check.
"""
from __future__ import annotations
import argparse
import base64
import hashlib
import html
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

try:
    from PIL import Image, ImageChops, ImageDraw, __version__ as PILLOW_VERSION
except ImportError as exc:
    raise SystemExit('Article rendering requires an existing Pillow environment. The analytical workbook itself needs only the Python standard library.') from exc

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'images/article/render-inputs.json'
BG = (251, 251, 248)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def source_image(root, entry):
    path = root / entry['source']
    require(path.resolve().is_relative_to(root.resolve()), 'Source must stay within the package')
    data = path.read_bytes()
    require(sha(data) == entry['source_sha256'], 'Article source identity differs: ' + entry['source'])
    image = Image.open(io.BytesIO(data)).convert('RGB')
    require(list(image.size) == entry['dimensions'], 'Article source dimensions differ')
    return data, image


def svg_source(entry):
    width, height = entry['dimensions']
    target = Path(entry['target'])
    require(not target.is_absolute() and '..' not in target.parts, 'Output must stay within the package')
    href = os.path.relpath(entry['source'], target.parent).replace(os.sep, '/')
    title = html.escape(entry['description'])
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img"><title>{title}</title>',
           f'<image href="{html.escape(href, quote=True)}" x="0" y="0" width="{width}" height="{height}"/>']
    for box in entry['erase_regions']:
        x1, y1, x2, y2 = box
        require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height, 'Invalid annotation edit region')
        badge = ('/article/' in entry['target'] and x1 >= .885*width and y1 >= .94*height
                 and x2 <= .96*width and y2 <= .98*height)
        factor, offset = (1, 0) if (width, height) == (1080, 1080) else (5/6, 350)
        legend_zone = (int(785*factor+offset), int(919*factor), int(1025*factor+offset+.999), int(978*factor+.999))
        legend = bool(entry.get('blank_idle_legend') and (width, height) in ((1080, 1080), (1600, 900)) and tuple(box) == legend_zone)
        require(badge or legend, 'Only the isolated deck-number or exact non-data legend region may be erased')
        out.append(f'<rect x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" fill="#FBFBF8"/>')
    if entry.get('blank_idle_legend'):
        label = entry['blank_idle_legend']
        out.append(f'<text x="{label["x"]}" y="{label["baseline_y"]}" font-family="Inter" font-weight="400" font-size="{label["font_size"]}" fill="#202326">Blank: idle</text>')
    return ('\n'.join(out) + '\n</svg>\n').encode()


def verify_preservation(source, rendered, entry):
    require(rendered.size == source.size, 'Output dimensions changed')
    difference = ImageChops.difference(source, rendered.convert('RGB'))
    allowed = Image.new('L', source.size)
    draw = ImageDraw.Draw(allowed)
    for x1, y1, x2, y2 in entry['erase_regions']:
        draw.rectangle((x1, y1, x2-1, y2-1), fill=255)
    # The replacement legend is contained wholly inside its declared erase box.
    channels = difference.split()
    changed = ImageChops.lighter(ImageChops.lighter(channels[0], channels[1]), channels[2])
    outside = ImageChops.multiply(changed, ImageChops.invert(allowed))
    require(outside.getbbox() is None, 'A pixel outside the declared non-data annotation region changed')
    require(difference.getbbox() is not None, 'Declared reader correction made no change')
    return {'changed_pixel_bbox': difference.getbbox(), 'outside_annotation_pixels_identical': True,
            'source_and_output_dimensions': list(source.size), 'scientific_diagram_pixels_outside_annotation_identical': True}


def required_tool(name):
    tool = shutil.which(name)
    require(tool is not None, 'Article rendering requires existing tool: ' + name)
    return tool


def render(entry, root, env):
    source_bytes, original = source_image(root, entry)
    svg = svg_source(entry)
    href = os.path.relpath(entry['source'], Path(entry['target']).parent).replace(os.sep, '/')
    resolved = svg.replace(('href="' + html.escape(href, quote=True) + '"').encode(),
                           ('href="data:image/png;base64,' + base64.b64encode(source_bytes).decode() + '"').encode(), 1)
    result = subprocess.run([required_tool('rsvg-convert')], input=resolved, capture_output=True, env=env)
    require(result.returncode == 0, 'Article SVG conversion failed: ' + result.stderr.decode(errors='replace'))
    output = Image.open(io.BytesIO(result.stdout)).convert('RGB')
    proof = verify_preservation(original, output, entry)
    return svg, result.stdout, proof


def compare_retained_receipt(retained, current):
    """Exact source/output proof is required; environment strings are context."""
    environment_fields = {'Pillow', 'rsvg_version'}
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'))
    expected = {key: value for key, value in retained.items() if key not in environment_fields}
    actual = {key: value for key, value in current.items() if key not in environment_fields}
    require(canonical(expected) == canonical(actual), 'Retained article source/output proof differs')
    return {key: {'retained': retained.get(key), 'current': current.get(key)}
            for key in sorted(environment_fields) if retained.get(key) != current.get(key)}


def retained_output(path, data, output):
    hint = (' Exact PNG bytes depend on the retained source/font and rendering stack. '
            'Compare the editable SVG, source PNG, Inter, rsvg/Pango and Pillow versions before deliberately regenerating.') if path.suffix == '.png' else ''
    require(path.is_file() and path.read_bytes() == data,
            'Article retained output differs: ' + str(path.relative_to(output)) + hint)


def run(root, output, write, *, explicit_output=False):
    declaration = json.loads((root / 'images/article/render-inputs.json').read_text())
    font = root / declaration['font']['path']
    require(sha(font.read_bytes()) == declaration['font']['sha256'], 'Retained Inter identity changed')
    required_tool('rsvg-convert')
    required_tool('fc-match')
    if write:
        output.mkdir(parents=True, exist_ok=True)
    else:
        require(output.is_dir(), 'Retained output directory is missing')
    # A default check never creates a cache/work directory in the package.
    # An explicit output root may own its own temporary font environment.
    with tempfile.TemporaryDirectory(prefix='.article-render-', dir=output if explicit_output else None) as temporary:
        temporary = Path(temporary)
        config = temporary / 'fonts.conf'
        config.write_text('<fontconfig><dir>' + html.escape(str(font.parent)) + '</dir><cachedir>' + html.escape(str(temporary/'cache')) + '</cachedir></fontconfig>')
        env = dict(os.environ, FONTCONFIG_FILE=str(config), PANGOCAIRO_BACKEND='fc')
        match = subprocess.run([required_tool('fc-match'), '-f', '%{file}\n%{style}\n', 'Inter:style=Regular'], capture_output=True, text=True, env=env)
        require(match.returncode == 0 and match.stdout.splitlines() and Path(match.stdout.splitlines()[0]).resolve() == font.resolve(), 'Inter Regular did not resolve to the retained packaged font')
        require(len(match.stdout.splitlines()) > 1 and 'Regular' in match.stdout.splitlines()[1], 'Requested regular Inter face did not resolve')
        rows, rendered = [], {}
        for entry in declaration['outputs']:
            svg, png, proof = render(entry, root, env)
            for suffix, data in (('.svg', svg), ('.png', png)):
                path = output / Path(entry['target']).with_suffix(suffix)
                if write:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(data)
                else:
                    retained_output(path, data, output)
            rendered[entry['target']] = png
            rows.append({'target': entry['target'], 'source': entry['source'], 'source_sha256': entry['source_sha256'],
                         'png_sha256': sha(png), 'svg_sha256': sha(svg), 'edits': entry['erase_regions'], **proof})
        derived = []
        for entry in declaration.get('derived_outputs', []):
            target = Path(entry['target'])
            require(not target.is_absolute() and '..' not in target.parts and target.suffix == '.png', 'Derived output must stay within the package')
            require(entry['source'] in rendered and entry['dimensions'] == [400, 400], 'Phone output must derive from a declared corrected square')
            source = rendered[entry['source']]
            image = Image.open(io.BytesIO(source)).convert('RGB')
            require(image.size == (1080, 1080), 'Phone source must be the corrected square')
            stream = io.BytesIO()
            image.resize((400, 400), Image.Resampling.LANCZOS).save(stream, format='PNG')
            png = stream.getvalue()
            path = output / target
            if write:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(png)
            else:
                retained_output(path, png, output)
            derived.append({'target': entry['target'], 'source': entry['source'], 'source_sha256': sha(source),
                            'png_sha256': sha(png), 'dimensions': [400, 400], 'resampling': 'LANCZOS'})
        receipt = {'kind': 'PRESENTATION_ANNOTATION_REPAIR_ONLY', 'manifest_sha256': sha((root/'images/article/render-inputs.json').read_bytes()),
                   'renderer_sha256': sha(Path(__file__).read_bytes()),
                   'font_sha256': declaration['font']['sha256'], 'Pillow': PILLOW_VERSION,
                   'rsvg_version': subprocess.check_output([required_tool('rsvg-convert'), '--version'], text=True).splitlines()[0],
                   'scientific_calculations_run': 0, 'new_mathematical_PASS': False, 'rows': rows, 'derived_outputs': derived}
        path = output / 'images/article/render-receipt.json'
        data = (json.dumps(receipt, indent=2) + '\n').encode()
        if write:
            path.write_bytes(data)
        else:
            require(path.is_file(), 'Retained article presentation receipt is missing')
            environment_differences = compare_retained_receipt(json.loads(path.read_text()), receipt)
            receipt = dict(receipt, environment_differences=environment_differences)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--output-root', type=Path)
    args = parser.parse_args()
    require(not (args.write and args.check), 'Choose write or check')
    result = run(ROOT, args.output_root or ROOT, args.write, explicit_output=args.output_root is not None)
    print(json.dumps({'status': 'WROTE' if args.write else 'CHECKED', 'figures': len(result['rows']),
                      'phone_derivatives': len(result['derived_outputs']),
                      'environment_differences': result.get('environment_differences', {}), 'scientific_calculations_run': 0}))


if __name__ == '__main__':
    main()
