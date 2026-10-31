#!/usr/bin/env python3
"""Deterministic attention figures. Existing art is clipped, never regenerated.

Only Python's standard library is imported. Rendering uses existing rsvg-convert
and ImageMagick executables. No install, network, model, GPU or runtime call.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from html import escape

ROOT = Path(__file__).resolve().parents[1]
INPUTS = ROOT / 'images' / 'attention-scenes'
RESULTS = ROOT / 'evidence' / 'derivation' / 'RESULTS.json'
try:
    TOKENS = json.loads((INPUTS / 'visual-tokens.json').read_text())
except (OSError, ValueError) as exc:
    raise ValueError(f'Shared visual tokens unavailable: {exc}') from exc
C = TOKENS['colors']
SIZE = 1200
MIN_FONT = 52
WIDTHS = (297, 312)
MIB = 1048576
STEMS = ('03-scratch-traffic', '04-target-rejection', '06-reduction-comparison')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(args: list[str], **kwargs) -> bytes:
    return subprocess.run(args, check=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, **kwargs).stdout


def font_env(cache: Path) -> dict[str, str]:
    config = cache / 'fonts.conf'
    config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd">'
                      f'<fontconfig><dir>{escape(str(INPUTS))}</dir>'
                      f'<cachedir>{escape(str(cache / "font-cache"))}</cachedir>'
                      '</fontconfig>')
    env = os.environ.copy()
    env['FONTCONFIG_FILE'] = str(config)
    # Pin locale and timezone; raster metadata is stripped below.
    env['LC_ALL'] = 'C'
    env['TZ'] = 'UTC'
    return env


class Slide:
    def __init__(self, name: str, title: str, asset_prefix: str = '../attention-scenes'):
        self.name, self.title = name, title
        self.asset_prefix = asset_prefix
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" '
                      f'xmlns:xlink="http://www.w3.org/1999/xlink" width="{SIZE}" '
                      f'height="{SIZE}" viewBox="0 0 {SIZE} {SIZE}">',
                      f'<title>{escape(title)}</title>',
                      '<style>@font-face {font-family: Inter; '
                      f'src: url("{escape(asset_prefix)}/Inter.ttf") format("truetype"); '
                      'font-weight: 100 900; font-style: normal;}</style>',
                      f'<rect width="1200" height="1200" fill="{C["background"]}"/>']
        self.labels: list[dict] = []

    def text(self, value: str, x: float, y: float, size: int = 52,
             weight: int = 400, color: str = 'ink', anchor: str = 'start'):
        require(size >= MIN_FONT, f'Essential text below {MIN_FONT}px: {value}')
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Inter" '
                          f'font-size="{size}" font-weight="{weight}" '
                          f'text-anchor="{anchor}" fill="{C[color]}">{escape(value)}</text>')
        self.labels.append(dict(text=value, x=x, y=y, font_px=size,
                                weight=weight, anchor=anchor))

    def rect(self, x, y, w, h, color, stroke=None):
        self.parts.append(f'<rect x="{x:.6f}" y="{y:.6f}" width="{w:.6f}" '
                          f'height="{h:.6f}" fill="{C[color]}" '
                          + (f'stroke="{C[stroke]}" stroke-width="4" ' if stroke else '') + '/>')

    def line(self, x1, y1, x2, y2, color='ink', width=3, dash=''):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                          f'stroke="{C[color]}" stroke-width="{width}" '
                          + (f'stroke-dasharray="{dash}" ' if dash else '') + '/>')

    def art(self, source: str, crop: tuple[int, int, int, int],
            destination: tuple[int, int, int, int]):
        x, y, w, h = destination
        self.parts.append(f'<svg x="{x}" y="{y}" width="{w}" height="{h}" '
                          f'viewBox="{" ".join(map(str,crop))}" overflow="hidden" '
                          'preserveAspectRatio="xMidYMid meet">'
                          f'<image width="2400" height="2400" '
                          f'xlink:href="{escape(self.asset_prefix)}/{escape(source)}"/></svg>')

    def arrow(self, x1, y1, x2, y2, color='blue'):
        self.line(x1, y1, x2, y2, color, 5)
        self.line(x2-16, y2-11, x2, y2, color, 5)
        self.line(x2-16, y2+11, x2, y2, color, 5)

    def result(self):
        return ('\n'.join(self.parts) + '\n</svg>\n').encode()


def validate_inputs() -> dict:
    provenance = json.loads((INPUTS / 'provenance.json').read_text())
    for row in provenance['inputs']:
        path = ROOT / row['input']
        require(path.resolve().is_relative_to(ROOT), 'Input must remain inside package root')
        require(path.is_file(), f'Required retained input missing: {row["input"]}')
        require(sha(path.read_bytes()) == row['sha256'],
                f'Input digest mismatch: {row["input"]}')
    require(TOKENS['typography']['font_family'] == 'Inter', 'Inter token required')
    require(TOKENS['typography']['reference_canvas_px'] == SIZE, 'Canvas token changed')
    d = json.loads(RESULTS.read_text())
    v = d['derivation']
    require(d['specification']['sequence_length'] == 4096, '4K sequence required')
    require(d['specification']['kv_heads'] == 1, 'MQA example required')
    require(d['target_screen']['illustrative_target_ms'] == '0.200', 'Target changed')
    require(v['total_partial_bytes'] == 1056*MIB, 'Produced partial bytes changed')
    require(v['not_retained_bytes_at_phase_boundary'] == 928*MIB, 'Unretained bytes changed')
    require(v['partial_hbm_traffic_floor_bytes'] == 1856*MIB, 'Traffic floor changed')
    require(d['phase_contract']['maximum_on_chip_retention_bytes_between_phases'] == 128*MIB,
            'On-chip retention cap changed')
    require(v['hypothetical_bandwidth_floors_ms']['8'] == '0.243269632', 'Time floor changed')
    require(d['phase_contract']['all_producer_partials_complete_before_separate_reducer'],
            'Complete phase boundary required')
    require(not d['phase_contract']['compression_allowed'] and
            not d['phase_contract']['recomputation_allowed'], 'Representation contract changed')
    require(d['measurements']['gpu_runs'] == 0, 'Analytical example expected')
    return d


def slides(data: dict, asset_prefix: str = '../attention-scenes') -> list[Slide]:
    # Each chart uses a shared zero origin and unrounded source values.
    a = Slide(STEMS[0], 'Scratch becomes traffic', asset_prefix)
    a.text(a.title, 84, 112, 64, 700)
    a.text('Attention backward · one sequence,', 84, 182)
    a.text('one layer · one device · FP32 partials', 84, 246)
    a.art('warehouse-source.png', (0, 624, 2400, 738), (84, 278, 1032, 226))
    rows = [(1056, '1,056 MiB produced', 'blue'),
            (128, '≤128 MiB retained on chip', 'blue'),
            (1856, '≥1,856 MiB out + back', 'orange')]
    for y, (value, label, color) in zip((557, 669, 781), rows):
        a.text(label, 84, y, 52, 600)
        a.rect(84, y+22, 1032*value/2048, 27, color)
    a.line(84, 837, 1116, 837)
    for value, label, anchor in ((0,'0','start'), (1024,'1,024','middle'), (2048,'2,048 MiB','end')):
        x = 84+1032*value/2048
        a.line(x, 837, x, 849)
        a.text(label, x, 905, anchor=anchor)
    a.text('All production, then reduction.', 84, 977)
    a.text('≥928 MiB out + ≥928 MiB reread.', 84, 1041)
    a.text('No compression or recomputation.', 84, 1105)

    b = Slide(STEMS[1], 'Transfers alone miss the target', asset_prefix)
    b.text(b.title, 84, 112, 60, 700)
    b.text('Produce all, then reduce · analytical', 84, 182)
    b.text('Assumed 8 decimal TB/s, shared', 84, 246)
    b.art('factory-source.png', (190, 750, 2016, 480), (84, 288, 1032, 246))
    floor = float(data['derivation']['hypothetical_bandwidth_floors_ms']['8'])
    for y, value, label, color in ((586,.2,'Target 0.200 ms','blue'),
                                  (732,floor,'Transfer floor ≥0.243 ms','orange')):
        b.text(label, 84, y, 52, 600)
        b.rect(84, y+22, 1032*value/.25, 36, color)
    b.line(84, 816, 1116, 816)
    for value, label, anchor in ((0,'0','start'),(.125,'0.125','middle'),(.25,'0.250 ms','end')):
        x=84+1032*value/.25
        b.line(x,816,x,828)
        b.text(label,x,884,anchor=anchor)
    b.text('1,856 MiB ÷ assumed 8 TB/s.',84,977)
    b.text('Transfers alone exceed this target.',84,1041)
    b.text('Compute and other traffic excluded.',84,1105)

    c = Slide(STEMS[2], 'Compare the complete path', asset_prefix)
    c.text(c.title,84,112,64,700)
    # The mechanism gets the space previously occupied by the decorative strip.
    # Node and edge metadata keep the qualitative topology independently checkable.
    def node(identifier, kind, draw):
        c.parts.append(f'<g data-node-id="{identifier}" data-node-kind="{kind}">')
        draw()
        c.parts.append('</g>')

    def edge(source, target, x1, y1, x2, y2, color='blue'):
        c.parts.append(f'<g data-flow-source="{source}" data-flow-target="{target}">')
        # Rotate the arrowhead with its edge, including diagonal convergence.
        dx,dy=x2-x1,y2-y1
        length=(dx*dx+dy*dy)**0.5
        ux,uy=dx/length,dy/length
        c.line(x1,y1,x2,y2,color,5)
        for side in (-1,1):
            c.line(x2-12*ux+side*8*uy,y2-12*uy-side*8*ux,x2,y2,color,5)
        c.parts.append('</g>')

    rows = [(250,'Store all partials','Simple ownership.','More scratch + traffic.','all'),
            (520,'Combine smaller batches','Less live scratch.','More coordination.','bounded'),
            (790,'Accumulate directly','Less staging.','Check synchronization.','direct')]
    for y,title,first,second,kind in rows:
        c.text(title,84,y,52,700)
        c.text(first,84,y+66)
        c.text(second,84,y+130)
        # Exact topology only. Widths are not capacity, traffic or timing values.
        center=y+82
        for index,offset in enumerate((-60,0,60),1):
            yy=center+offset
            producer=f'{kind}-producer-{index}'
            node(producer,'partial',lambda yy=yy:
                 c.parts.append(f'<circle cx="802" cy="{yy}" r="14" '
                                f'fill="{C["background"]}" stroke="{C["blue"]}" '
                                'stroke-width="4"/>'))
            if kind=='all':
                buffer=f'all-buffer-{index}'
                edge(producer,buffer,824,yy,866,yy)
                node(buffer,'stored-partial',lambda yy=yy:
                     c.rect(874,yy-19,44,38,'background','orange'))
                edge(buffer,'all-reducer',926,yy,984,center+offset/5,'ink')
            elif kind=='bounded':
                edge(producer,'bounded-buffer',824,yy,908,center+offset/5)
            else:
                edge(producer,'direct-output',824,yy,1066,center+offset/5)
        if kind=='all':
            node('all-reducer','reducer',lambda:
                 c.rect(992,center-29,46,58,'background','ink'))
            c.text('Σ',1015,center+18,52,400,anchor='middle')
            edge('all-reducer','all-output',1046,center,1066,center,'ink')
        elif kind=='bounded':
            node('bounded-buffer','batch-buffer',lambda:
                 c.rect(916,center-23,46,46,'background','orange'))
            edge('bounded-buffer','bounded-reducer',970,center,984,center,'ink')
            node('bounded-reducer','reducer',lambda:
                 c.rect(992,center-29,46,58,'background','ink'))
            c.text('Σ',1015,center+18,52,400,anchor='middle')
            edge('bounded-reducer','bounded-output',1046,center,1066,center,'ink')
        else:
            node('direct-synchronization','synchronization',lambda:
                 c.line(1015,center-72,1015,center+72,'orange',7))
        node(f'{kind}-output','output',lambda:
             c.rect(1074,center-29,42,58,'background','blue'))
    c.text('Each design needs correctness checks.',84,1041)
    c.text('Sum order can change rounding.',84,1105)
    return [a,b,c]


def validate_svg(slide: Slide, svg: bytes, env: dict) -> dict:
    root=ET.fromstring(svg)
    ns={'s':'http://www.w3.org/2000/svg'}
    texts=root.findall('.//s:text',ns)
    require(len(texts)==len(slide.labels), 'Label inventory mismatch')
    for node,row in zip(texts,slide.labels):
        size=float(node.attrib['font-size'])
        require(node.attrib.get('font-family')=='Inter','Unexpected font family')
        require(size>=MIN_FONT,'Font below 52px')
        require(size*297/SIZE >= 12.87,'Essential label below 12.87px at actual 297px')
        # Independent glyph-width check uses the pinned Inter file directly.
        width=int(run(['magick','-background','none','-font',str(INPUTS/'Inter.ttf'),
                       '-pointsize',str(row['font_px']),f'label:{row["text"]}',
                       '-format','%w','info:'],env=env).decode())
        x=row['x']; anchor=row['anchor']
        left=x if anchor=='start' else x-width if anchor=='end' else x-width/2
        right=left+width
        require(left>=80 and right<=1120,
                f'Text outside 80px safe margin: {row["text"]} ({left}..{right})')
        require(row['y']-row['font_px']>=48 and row['y']+14<=1120,
                f'Text outside vertical safe area: {row["text"]}')
    return {'text_elements':len(texts),'minimum_font_px':min(r['font_px'] for r in slide.labels),
            'minimum_at_297_px':min(r['font_px']*297/SIZE for r in slide.labels),
            'minimum_at_312_px':min(r['font_px']*312/SIZE for r in slide.labels),
            'safe_margin_checked_px':80,'labels':slide.labels}


def raster_source(svg: bytes, asset_prefix: str) -> bytes:
    """Resolve pinned relative dependencies for raster input only.

    Stored SVGs remain small, readable and editable. The PNG rasterizer gets
    the exact pinned asset bytes in memory, with no external fetching.
    """
    result=svg
    for name,mime in (('warehouse-source.png','image/png'),
                      ('factory-source.png','image/png'),('Inter.ttf','font/ttf')):
        reference=f'{escape(asset_prefix)}/{name}'.encode()
        embedded=f'data:{mime};base64,'.encode()+base64.b64encode((INPUTS/name).read_bytes())
        result=result.replace(reference,embedded)
    require(b'../attention-scenes/' not in result,'Unresolved raster asset dependency')
    return result


def render(output: Path, data: dict, asset_context: Path | None = None) -> dict:
    output.mkdir(parents=True,exist_ok=True)
    asset_prefix=Path(os.path.relpath(INPUTS,asset_context or output)).as_posix()
    records={}
    with tempfile.TemporaryDirectory(prefix='attention-fonts-') as temp:
        env=font_env(Path(temp))
        matched=run(['fc-match','-f','%{family}|%{file}','Inter'],env=env).decode()
        require(matched.split('|')[0]=='Inter',f'Inter resolution failed: {matched}')
        require(str(INPUTS/'Inter.ttf') in matched,'Unexpected Inter file resolved')
        for slide in slides(data,asset_prefix):
            svg=slide.result()
            require(b'data:' not in svg and b'base64' not in svg,'Stored SVG must use relative assets')
            checks=validate_svg(slide,svg,env)
            (output/(slide.name+'.svg')).write_bytes(svg)
            rendered=run(['rsvg-convert','--format','png'],input=raster_source(svg,asset_prefix),env=env)
            # Normalize the existing renderer's metadata without touching visual pixels.
            png=run(['magick','png:-','-strip','-define','png:exclude-chunks=date,time','png:-'],
                    input=rendered,env=env)
            require(struct.unpack('>II',png[16:24])==(SIZE,SIZE),'Incorrect raster dimensions')
            (output/(slide.name+'.png')).write_bytes(png)
            (output/(slide.name+'.txt')).write_text('\n'.join(r['text'] for r in slide.labels)+'\n')
            for width in WIDTHS:
                preview=run(['magick','png:-','-filter','Lanczos','-resize',f'{width}x{width}',
                             '-strip','-define','png:exclude-chunks=date,time','png:-'],input=png,env=env)
                (output/f'{slide.name}-{width}.png').write_bytes(preview)
            records[slide.name]={'svg_sha256':sha(svg),'png_sha256':sha(png),**checks}
    return records


def verify_current(output: Path, expected: Path, second: Path) -> list[dict]:
    """Check the current selected destination, without overwriting stale files."""
    verified=[]
    for candidate in sorted(expected.iterdir()):
        if candidate.suffix not in ('.svg','.png','.txt'):
            continue
        name=candidate.name
        require(candidate.read_bytes()==(second/name).read_bytes(),
                f'Non-deterministic output: {name}')
        current=output/name
        require(current.is_file(), f'Canonical output missing: {current}')
        require(current.read_bytes()==candidate.read_bytes(),
                f'Canonical output is stale or modified: {current}')
        verified.append({'path':str(current),'sha256':sha(current.read_bytes())})
    require(len(verified)==len(STEMS)*5,'Output inventory incomplete')
    return verified


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'images'/'scientific',
                        help='Canonical output directory (default: package images/scientific)')
    parser.add_argument('--check',action='store_true',
                        help='Read-only check of current canonical files against two fresh byte-exact renders')
    parser.add_argument('--receipt',type=Path,help='Optional explicit path for the JSON receipt')
    args=parser.parse_args()
    output=args.output_dir.resolve()
    for binary in ('rsvg-convert','magick','fc-match'):
        require(shutil.which(binary) is not None,f'Required existing binary unavailable: {binary}')
    data=validate_inputs()
    verified=[]
    if args.check:
        with tempfile.TemporaryDirectory(prefix='attention-repro-') as temp:
            first,second=Path(temp)/'first',Path(temp)/'second'
            records=render(first,data,output)
            render(second,data,output)
            verified=verify_current(output,first,second)
    else:
        records=render(output,data)
    receipt={'status':'PASS','canvas':[SIZE,SIZE],
             'package_root':str(ROOT),'canonical_output_dir':str(output),
             'mode':'read-only-check' if args.check else 'render',
             'phone_widths_px':list(WIDTHS),'all_essential_text_minimum_px':MIN_FONT,
             'byte_exact_second_render':bool(args.check),
             'current_canonical_files_verified':verified,'records':records,
             'dependencies':{'python':'standard library only',
                 'rsvg':run(['rsvg-convert','--version']).decode().splitlines()[0],
                 'imagemagick':run(['magick','--version']).decode().splitlines()[0],
                 'font':'images/attention-scenes/Inter.ttf, resolved with isolated Fontconfig configuration'},
             'limitations':['Local preview only. Actual platform upload, preview and review binding remain root-owned.',
                            'Retained illustrative art is reused, not new art. Diagrams are qualitative topology.',
                            'No model runtime, attention kernel, GPU experiment or measured speedup.']}
    if args.receipt:
        args.receipt.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    main()
