#!/usr/bin/env python3
"""Reflow the current evidence chart into landscape and square cover variants.

Uses the mapped gamma-evidence renderer, tokens and upright stamp, plus retained
raw/summary data and Inter. No benchmark, generation service or network call.
Use --check for byte comparison without replacing the cover files.
"""
from pathlib import Path
import argparse
import hashlib
import html
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from PIL import Image, ImageFont

HERE = Path(__file__).resolve().parent
GAMMA_RENDERER_SHA256 = 'a9cd439ce9f8fee9b8a5377dfa79d75a4c8d38ef330499e511c3fb04293e6236'
STAMP_SHA256 = 'f456e86072e455924bb45aca4500b27cf02d4210297c55d24179097700a946ec'
STAMP_HREF = 'gamma-evidence/fregly-take-stamp.png'

def require(value, message):
    if not value:
        raise ValueError(message)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def source(package):
    path = package / 'images/gamma-evidence/render_evidence.py'
    require(path.resolve() == path and not path.is_symlink() and sha(path.read_bytes()) == GAMMA_RENDERER_SHA256,
            'Mapped canonical evidence renderer identity differs')
    spec = importlib.util.spec_from_file_location('canonical_candidate_evidence', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(module.STAMP_SHA256 == STAMP_SHA256, 'Canonical upright stamp identity differs')
    return module

def landscape(package, canonical, basis):
    width, height = 1600, 900
    x0, x1, maximum = 72, 1528, 6
    stamp = (1364, 10, 1528, 174)
    palette = canonical.PALETTE
    tokens = json.loads((package / 'images/gamma-evidence/gamma-visual-tokens.json').read_text())
    font_path = package / 'images/fonts/Inter.ttf'
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="900" viewBox="0 0 1600 900">',
             '<title>Skip scores you\'ll discard</title>',
             '<desc>The same retained output-head evidence, reflowed for a landscape cover. Allowed-set production and transfer remain excluded from both timed paths.</desc>']
    boxes, texts, marks = [], [], []
    def rect(x, y, w, h, color, extra=''):
        parts.append(f'<rect x="{x}" y="{y}" width="{w:.13f}" height="{h}" fill="{color}" {extra}/>')
    def line(x, y, a, b, extra=''):
        parts.append(f'<line x1="{x:.13f}" y1="{y}" x2="{a:.13f}" y2="{b}" stroke="{palette["ink"]}" stroke-width="2" {extra}/>')
    def text(x, y, value, size=48, weight=400, color=None, anchor='start', background=None):
        color = color or palette['ink']
        background = background or palette['background']
        require(';' not in value and '—' not in value, 'Use plain labels')
        require(size * 400 / width >= tokens['typography']['minimum_phone_px']['material_condition'],
                'Text below400px minimum')
        require(weight in tokens['typography']['allowed_weights'], 'Unsupported Inter weight')
        face = ImageFont.truetype(str(font_path), size)
        face.set_variation_by_axes([14, weight])
        bb = face.getbbox(value, anchor='ls')
        length = face.getlength(value)
        shift = length if anchor == 'end' else length / 2 if anchor == 'middle' else 0
        bounds = (x-shift+bb[0], y+bb[1], x-shift+bb[2], y+bb[3])
        require(bounds[0] >= 48 and bounds[2] <= width-48 and bounds[1] >= 20 and bounds[3] <= height-16,
                'Text outside safe cover canvas: ' + value)
        for old, label in boxes:
            overlap = min(bounds[2], old[2])-max(bounds[0], old[0]) > 1 and min(bounds[3], old[3])-max(bounds[1], old[1]) > 1
            require(not overlap, 'Text overlap: ' + value + ' / ' + label)
        require(canonical.contrast(color, background) >= tokens['palette_policy']['ordinary_text_contrast_target'],
                'Text contrast below declared target')
        boxes.append((bounds, value))
        texts.append({'text': value, 'size': size, 'weight': weight, 'bounds': bounds, 'css_px_at400': size*400/width})
        parts.append(f'<text x="{x}" y="{y}" font-family="Inter" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{html.escape(value)}</text>')
    rect(0, 0, width, height, palette['background'])
    rect(0, 0, width, 184, palette['header'])
    text(72, 89, basis['text'][0]['text'], 64, 600, palette['header_ink'], background=palette['header'])
    text(72, 160, basis['text'][1]['text'], 52, 600, palette['header_accent'], background=palette['header'])
    parts.append(f'<image x="1364" y="10" width="164" height="164" href="{STAMP_HREF}" preserveAspectRatio="xMidYMid meet"/>')
    text(72, 246, 'Synthetic strict FP32 · B200', 48, 500)
    for source_mark, y, label in zip(basis['marks'], (341, 471), ('Full scoring + filter', 'Allowed rows first')):
        value = source_mark['value_ms']
        series = source_mark['series']
        color = source_mark['color']
        text(72, y-36, label, 52, 500)
        text(1528, y-36, f'{value:.3f} ms', 52, 600, anchor='end')
        bar_width = (x1-x0)*value/maximum
        rect(x0, y, bar_width, 30, color, f'data-science="median" data-series="{series}"')
        a, b = [x0+(x1-x0)*v/maximum for v in (source_mark['observed_min_ms'], source_mark['observed_max_ms'])]
        line(a, y+15, b, y+15, f'data-science="range" data-series="{series}"')
        line(a, y+5, a, y+25)
        line(b, y+5, b, y+25)
        marks.append({**source_mark, 'bar_x': x0, 'bar_y': y, 'bar_width': bar_width, 'bar_height': 30,
                      'scale_px_per_ms': (x1-x0)/maximum, 'whisker_x': [a, b]})
    line(x0, 520, x1, 520)
    for tick in (0, 2, 4, 6):
        x = x0+(x1-x0)*tick/maximum
        line(x, 520, x, 532)
        text(x, 574, str(tick), anchor='start' if tick == 0 else 'end' if tick == maximum else 'middle')
    text(72, 636, 'Median ms / 16-step warm block', 48, 500)
    text(1528, 636, '3-pair min–max whiskers', anchor='end')
    rect(48, 676, 1504, 208, palette['catch'])
    for y, value, weight in [(716, 'Allowed set ready BEFORE scoring · weight gathering timed', 600),
                              (766, '64 allowed tokens shared by batch · new set each step', 400),
                              (816, 'Both paths exclude set creation and transfer', 400),
                              (866, 'Whole-model gains unmeasured', 500)]:
        text(72, y, value, 48, weight, background=palette['catch'])
    for bounds, value in boxes:
        require(not(min(bounds[2], stamp[2])-max(bounds[0], stamp[0]) > 0 and min(bounds[3], stamp[3])-max(bounds[1], stamp[1]) > 0),
                'Upright stamp overlaps text: ' + value)
    parts.append('</svg>')
    svg = '\n'.join(parts) + '\n'
    ET.fromstring(svg)
    return svg, '\n'.join(t['text'] for t in texts)+'\n', {'dimensions': [width, height], 'marks': marks,
        'axis': {'minimum_ms': 0, 'maximum_ms': maximum, 'start_x': x0, 'end_x': x1, 'linear': True},
        'text': texts, 'minimum_text_css_px_at400': min(t['css_px_at400'] for t in texts),
        'stamp': {'sha256': STAMP_SHA256, 'href': STAMP_HREF, 'box': stamp, 'rotation_degrees': 0, 'mirror': False, 'skew_degrees': 0},
        'text_bar_clearance': canonical.check_text_bar_clearance(texts, marks), 'layout': 'Full-width reflow of chart and conditions. No centered square image.'}

def render(package):
    canonical = source(package)
    square, square_text, basis = canonical.build(package)
    square = square.replace('width="1200" height="1200"', 'width="1080" height="1080"', 1)
    square = square.replace('href="fregly-take-stamp.png"', f'href="{STAMP_HREF}"', 1)
    wide, wide_text, wide_basis = landscape(package, canonical, basis)
    outputs = {'fun.svg': wide.encode(), 'fun-square.svg': square.encode(),
               'fun.txt': (wide_text+'\n'+canonical.retained_methods(package)).encode(), 'fun-square.txt': square_text.encode()}
    converter, matcher = shutil.which('rsvg-convert'), shutil.which('fc-match')
    require(converter and matcher, 'Existing rsvg-convert and fc-match are required')
    stamp = package / 'images/gamma-evidence/fregly-take-stamp.png'
    require(sha(stamp.read_bytes()) == STAMP_SHA256, 'Current upright stamp bytes differ')
    font = package / 'images/fonts/Inter.ttf'
    previews = {}
    with tempfile.TemporaryDirectory(prefix='data-covers-') as directory:
        tmp = Path(directory)
        (tmp / 'gamma-evidence').mkdir()
        shutil.copyfile(stamp, tmp / STAMP_HREF)
        config = tmp / 'fonts.conf'
        config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><dir>'+html.escape(str(font.parent))+'</dir><cachedir>'+html.escape(str(tmp/'cache'))+'</cachedir></fontconfig>')
        env = dict(os.environ, PANGOCAIRO_BACKEND='fc', FONTCONFIG_FILE=str(config))
        matched = subprocess.check_output([matcher, '-f', '%{file}\n%{family}\n%{style}\n', 'Inter:style=Regular'], env=env, text=True).splitlines()
        require(Path(matched[0]).resolve() == font.resolve() and matched[1:] == ['Inter', 'Regular'], 'Exact regular Inter file did not resolve')
        for stem, dimensions in [('fun', (1600, 900)), ('fun-square', (1080, 1080))]:
            src, png = tmp/(stem+'.svg'), tmp/(stem+'.png')
            src.write_bytes(outputs[stem+'.svg'])
            subprocess.run([converter, str(src), '-o', str(png)], env=env, check=True)
            with Image.open(png) as decoded:
                require(decoded.size == dimensions, 'Cover dimensions differ')
                image = decoded.convert('RGB')
                buffer = io.BytesIO()
                image.save(buffer, format='PNG')
                outputs[stem+'.png'] = buffer.getvalue()
                preview = io.BytesIO()
                image.resize((400, round(dimensions[1]*400/dimensions[0])), Image.Resampling.LANCZOS).save(preview, format='PNG')
                previews[stem+'-400px.png'] = preview.getvalue()
        version = subprocess.check_output([converter, '--version'], text=True).strip()
    require(sha((package/'images/gamma-evidence/render_evidence.py').read_bytes()) == GAMMA_RENDERER_SHA256 and sha(stamp.read_bytes()) == STAMP_SHA256,
            'Canonical chart or stamp source drifted')
    metadata = {'landscape': wide_basis, 'square': {'dimensions': [1080, 1080], 'viewBox': [0, 0, 1200, 1200],
                 'canonical_numeric_geometry_preserved': True, 'source': 'images/gamma-evidence/comparison.svg',
                 'minimum_text_css_px_at400': min(t['size']*400/1200 for t in basis['text']), 'marks': basis['marks']},
                'canonical_renderer_sha256': GAMMA_RENDERER_SHA256, 'source_pins': canonical.SOURCE_PINS,
                'tokens_sha256': sha((package/'images/gamma-evidence/gamma-visual-tokens.json').read_bytes()),
                'stamp_sha256': STAMP_SHA256, 'font_resolution': matched, 'renderer_version': version,
                'PANGOCAIRO_BACKEND': 'fc', 'PNG_mode': 'RGB', 'new_measurements': 0,
                'scope': 'Compatibility covers from the same canonical evidence page. Exact retained median/range geometry, timing conditions and current upright stamp. No new lead concept, benchmark, generation or account action.'}
    return outputs, previews, metadata

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=HERE.parent)
    parser.add_argument('--output-root', type=Path, default=HERE)
    parser.add_argument('--receipt', type=Path)
    parser.add_argument('--preview-root', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    package, out = args.package_root.resolve(), args.output_root.resolve()
    outputs, previews, metadata = render(package)
    if args.check:
        require(all((out/name).is_file() and (out/name).read_bytes() == data for name, data in outputs.items()), 'Canonical compatibility cover bytes differ')
    else:
        require(not any((out/name).exists() for name in outputs), 'Use a new output directory or --check')
        out.mkdir(parents=True, exist_ok=True)
        for name, data in outputs.items():
            (out/name).write_bytes(data)
    if args.preview_root:
        args.preview_root.mkdir(parents=True, exist_ok=True)
        for name, data in previews.items():
            (args.preview_root/name).write_bytes(data)
    metadata['outputs_sha256'] = {name: sha(data) for name, data in outputs.items()}
    metadata['renderer_sha256'] = sha(Path(__file__).read_bytes())
    if args.receipt:
        args.receipt.write_text(json.dumps(metadata, indent=2)+'\n')
    print(json.dumps({'state': 'BYTE_EXACT_CHECK_PASS' if args.check else 'PROPOSAL_COVERS_RENDERED',
                      'written': not args.check, 'outputs': metadata['outputs_sha256'],
                      'minimum_text_css_px_at400': metadata['landscape']['minimum_text_css_px_at400']}))

if __name__ == '__main__':
    main()
