#!/usr/bin/env python3
"""Render plain Inter figures from retained measurements. No benchmark execution.

Private layout: python3 images/render_illustrations.py
Public layout uses the same evidence/ files, vendored tokens and Inter font.
Use --output-root for a separate proposal/export directory.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import argparse
import base64
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from PIL import Image, ImageFont

HERE = Path(__file__).resolve().parent
PINS = {
    'evidence/summary.json': 'f695fc48f67c11b2d2e0b38c9de6ce385a5877b9e49c4703bfd01b5b9f49d380',
    'evidence/raw.jsonl': 'eca4fcb7e280baebb2dbc223fb9f55584144e797dcae419957e66fd91f7e4a4f',
    'images/fonts/Inter.ttf': '29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031',
}
SUMMARY = PAIRS = TOKENS = FONT = None
GEOMETRY = []
BOUNDS = []


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def font(size):
    result = ImageFont.truetype(str(FONT), int(size))
    if hasattr(result, 'get_variation_axes'):
        axes = result.get_variation_axes()
        values = [400 if axis['name'] == b'Weight' else axis['default'] for axis in axes]
        result.set_variation_by_axes(values)
    return result


def wrap(text, size, width):
    lines = []
    for paragraph in str(text).split('\n'):
        line = ''
        for word in paragraph.split():
            candidate = (line + ' ' + word).strip()
            if line and font(size).getlength(candidate) > width:
                lines.append(line)
                line = word
            else:
                line = candidate
        lines.append(line)
    return lines


class Board:
    def __init__(self, stem, title, subtitle, square):
        self.stem, self.square = stem, square
        self.w, self.h = (1080, 1080) if square else (1600, 900)
        self.margin = 48 if square else 64
        self.body = 36 if square else 48
        self.copy, self.boxes, self.chart = [], [], []
        self.bg, self.ink, self.blue, self.orange = [TOKENS['colors'][key] for key in ('background', 'ink', 'blue', 'orange')]
        self.parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" viewBox="0 0 {self.w} {self.h}">',
                      f'<rect width="100%" height="100%" fill="{self.bg}"/>',
                      '<defs><marker id="arrow" viewBox="0 0 12 12" refX="10" refY="6" markerWidth="5" markerHeight="5" orient="auto"><path d="M2 2 L10 6 L2 10" fill="none" stroke="context-stroke" stroke-width="2"/></marker></defs>']
        size = 60 if square else 64
        titles = wrap(title, size, self.w - self.margin * 2)
        self.text(self.margin, 82, titles, size, gap=1.12)
        last = 82 + (len(titles)-1)*size*1.12
        subtitles = wrap(subtitle, self.body, self.w-self.margin*2)
        self.text(self.margin, last+70, subtitles, self.body)
        self.top = last+70+(len(subtitles)-1)*self.body*1.2+35

    def text(self, x, y, lines, size=None, color=None, anchor='start', gap=1.2):
        size = size or self.body
        color = color or self.ink
        lines = [lines] if isinstance(lines, (str, int, float)) else lines
        for i, line in enumerate(lines):
            line = str(line)
            require(';' not in line and 'legal' not in line.lower(), 'Unapproved reader label: '+line)
            yy = y+i*size*gap
            f = font(size)
            left, top, right, bottom = f.getbbox(line, anchor='ls')
            shift = f.getlength(line) * (0.5 if anchor == 'middle' else 1 if anchor == 'end' else 0)
            box = (x-shift+left, yy+top, x-shift+right, yy+bottom)
            require(box[0] >= 20 and box[2] <= self.w-20 and box[1] >= 15 and box[3] <= self.h-15,
                    f'Text outside canvas {self.stem}: {line} {box}')
            for old, label in self.boxes:
                overlap = min(box[2],old[2])-max(box[0],old[0]) > 1 and min(box[3],old[3])-max(box[1],old[1]) > 1
                require(not overlap, f'Text overlap {self.stem}: {line!r} with {label!r}')
            self.boxes.append((box,line))
            self.copy.append(line)
            self.parts.append(f'<text x="{x}" y="{yy}" font-family="Inter" font-size="{size}" font-weight="400" font-style="normal" fill="{color}" text-anchor="{anchor}">{escape(line)}</text>')

    def line(self, x1,y1,x2,y2,color=None,width=3,arrow=False):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color or self.ink}" stroke-width="{width}"'+(' marker-end="url(#arrow)"' if arrow else '')+'/>')

    def rect(self,x,y,w,h,color=None,fill='none',width=3):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="0" fill="{fill}" stroke="{color or self.ink}" stroke-width="{width}"/>')

    def circle(self,x,y,r,color=None):
        self.parts.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="{color or self.ink}" stroke-width="3"/>')

    def matrix(self,x,y,w,h,color,rows=4,cols=5):
        self.rect(x,y,w,h,color)
        for row in range(1,rows):self.line(x,y+h*row/rows,x+w,y+h*row/rows,color,1.5)
        for col in range(1,cols):self.line(x+w*col/cols,y,x+w*col/cols,y+h,color,1.5)

    def foot(self, lines):
        size=self.body
        material=[]
        for line in lines:material += wrap(line,size,self.w-self.margin*2)
        y=self.h-30-(len(material)-1)*size*1.15
        self.text(self.margin,y,material,size,gap=1.15)

    def bar(self,cell,series,x0,y,width,height=14):
        field='baseline_median_wall_ms' if series=='baseline' else 'candidate_median_wall_ms'
        rawfield='baseline_wall_ms' if series=='baseline' else 'candidate_wall_ms'
        value=cell[field]
        rows=[row for row in PAIRS if row['size']==cell['size'] and row['mode']==cell['mode']]
        values=[row[rawfield] for row in rows]
        lo,hi=min(values),max(values)
        color=self.blue if series=='baseline' else self.orange
        w=width*value/6
        self.parts.append(f'<rect data-science="bar" data-series="{series}" data-mode="{cell["mode"]}" data-size="{cell["size"]}" x="{x0}" y="{y}" width="{w:.13f}" height="{height}" fill="{color}"/>')
        yy=y+height/2
        self.line(x0+width*lo/6,yy,x0+width*hi/6,yy,self.ink,2)
        self.line(x0+width*lo/6,yy-7,x0+width*lo/6,yy+7,self.ink,2)
        self.line(x0+width*hi/6,yy-7,x0+width*hi/6,yy+7,self.ink,2)
        self.chart.append({'series':series,'size':cell['size'],'mode':cell['mode'],'median_ms':value,
                           'minimum_ms':lo,'maximum_ms':hi,'axis_max_ms':6,'x0':x0,'scale_px_per_ms':width/6,
                           'bar_width':w,'whisker_x0':x0+width*lo/6,'whisker_x1':x0+width*hi/6,'color':color})
        return x0+w

    def finish(self):
        return '\n'.join(self.parts+['</svg>'])+'\n'


def mechanism(square):
    b=Board('a','Select allowed rows before scoring','Compare the same allowed-token outputs',square)
    if square:
        for y,color,title in ((340,b.blue,'Full scoring + filter'),(650,b.orange,'Allowed rows first')):
            b.text(55,y,title,44,color)
            b.matrix(70,y+(40 if color==b.blue else 71),170,132 if color==b.blue else 66,color,8 if color==b.blue else 2,5)
            b.line(265,y+104,335,y+104,color,3,True)
            b.rect(365,y+55,345,100,color)
            b.text(538,y+115,'Score all rows' if color==b.blue else 'Score rows',40,color,'middle')
            b.line(735,y+104,810,y+104,color,3,True)
            b.text(900,y+94,['Filter +','choose'] if color==b.blue else ['Choose','token'],36,color,'middle')
        b.foot(['One shared allowed set across batch 8','Strict FP32 output head · TF32 off'])
    else:
        for y,color,title,rows in ((280,b.blue,'Full scoring + filter',6),(555,b.orange,'Allowed rows first',2)):
            b.text(70,y,title,48,color)
            b.matrix(80,y+(15 if color==b.blue else 50),220,140 if color==b.blue else 70,color,8 if color==b.blue else 2,6)
            b.line(335,y+85,460,y+85,color,3,True)
            b.rect(500,y+35,470,100,color)
            b.text(735,y+100,'Score all rows' if color==b.blue else 'Score selected rows',48,color,'middle')
            b.line(1005,y+85,1120,y+85,color,3,True)
            b.text(1320,y+70,['Filter +','choose'] if color==b.blue else ['Choose','token'],48,color,'middle')
        b.foot(['Shared allowed set across batch 8 · strict FP32 output head','TF32 off · both paths return allowed-token scores'])
    return b


def check_text_bar_clearance(texts, marks, minimum=20):
    """Require a visible gutter between actual text bounds and emitted timing bars."""
    gaps=[]
    for (left,top,right,bottom), label in texts:
        for mark in marks:
            x,y,width,height=(float(mark.get(key)) for key in ('x','y','width','height'))
            horizontal=max(left-(x+width), x-right)
            vertical=max(top-(y+height), y-bottom)
            clearance=max(horizontal,vertical)
            require(clearance >= minimum, 'Text too close to timing bar: '+label)
            gaps.append(clearance)
    return {'minimum_required_px':minimum,'minimum_observed_px':min(gaps),'checked_pairs':len(gaps)}


def check_paragraph_gaps(board):
    """Keep the result, comparison sections and evidence footer visually distinct."""
    pairs=[]
    if board.stem=='aud-performance-engineer':
        ratio=next(label for _,label in board.boxes if label.endswith('ratio of retained medians'))
        pairs=[(ratio,'Full scoring + filter',32)]
    elif board.stem=='aud-kernel-engineer':
        pairs=[('8 × 2048  @  2048 × 131072','64 allowed outputs',32),
               ('8 × 2048  @  2048 × 64','FP32 weights used: 1 GiB → 512 KiB',64)]
    checked=[]
    for first,second,minimum in pairs:
        a=[box for box,label in board.boxes if label==first]
        b=[box for box,label in board.boxes if label==second]
        require(len(a)==len(b)==1,'Paragraph labels must be unique')
        actual=b[0][1]-a[0][3]
        require(actual>=minimum,'Paragraph separation too small: '+first+' / '+second)
        checked.append({'first':first,'second':second,'minimum_px':minimum,'actual_px':actual})
    return checked


def comparison(square):
    b=Board('b','Six warm-block comparisons','Synthetic strict-FP32 output-head stage',square)
    legend_size = 36 if square else 48
    swatch, gap, between = (22, 14, 44) if square else (28, 18, 64)
    labels = [('baseline', 'Full scoring + filter', b.blue), ('candidate', 'Allowed rows first', b.orange)]
    total = sum(swatch + gap + font(legend_size).getlength(label) for _, label, _ in labels) + between
    left = (b.w - total) / 2
    baseline = 270 if square else 220
    for series, label, color in labels:
        b.parts.append(f'<rect data-role="legend-swatch" data-series="{series}" x="{left}" y="{baseline-legend_size*.62}" width="{swatch}" height="{swatch}" fill="{color}"/>')
        b.text(left + swatch + gap, baseline, label, legend_size, color)
        left += swatch + gap + font(legend_size).getlength(label) + between
    if square:
        x0,width,top,pitch=320,440,310,88
        for tick in (0,3,6):
            x=x0+width*tick/6;b.line(x,310,x,top+pitch*5+58,'#D3D6D9',1)
            b.text(x,854,str(tick),36,anchor='middle')
        for n,cell in enumerate(SUMMARY['cells']):
            y=top+n*pitch
            b.text(55,y+29,[str(cell['size']),cell['mode']],36,gap=1.0)
            for series,offset in (('baseline',0),('candidate',44)):
                end=b.bar(cell,series,x0,y+offset,width,12)
                value=cell['baseline_median_wall_ms' if series=='baseline' else 'candidate_median_wall_ms']
                b.text(x0+width+32,y+offset+20,f'{value:.3f}',36,b.blue if series=='baseline' else b.orange)
        b.text(540,900,'Milliseconds per warm 16-step block',36,anchor='middle')
        b.foot(['Median bars · min–max whiskers · 3 pairs in one process',
                'Batch 8 · hidden 2048 · vocabulary 131072 · TF32 off',
                'Changing includes gather. Static setup is separate.'])
    else:
        for panel,mode in enumerate(('static','changing')):
            left=70+panel*790;x0=left+135;width=400
            b.text(left,292,mode.capitalize()+' IDs',48)
            for tick in (0,3,6):
                x=x0+width*tick/6;b.line(x,305,x,604,'#D3D6D9',1)
                b.text(x,652,str(tick),48,anchor='middle')
            for n,cell in enumerate(c for c in SUMMARY['cells'] if c['mode']==mode):
                y=330+n*96
                b.text(left,y+34,str(cell['size']),48)
                for series,offset in (('baseline',0),('candidate',48)):
                    end=b.bar(cell,series,x0,y+offset,width,14)
                    value=cell['baseline_median_wall_ms' if series=='baseline' else 'candidate_median_wall_ms']
                    b.text(x0+width+32,y+offset+25,f'{value:.3f}',48,b.blue if series=='baseline' else b.orange)
        b.text(800,700,'Milliseconds per warm 16-step block',48,anchor='middle')
        b.foot(['Median bars · min–max whiskers · 3 pairs in one process',
                'Batch 8 · hidden 2048 · vocabulary 131072 · TF32 off',
                'Changing includes gather. Static setup is separate.'])
    return b


def inspection(square):
    b=Board('c','Know which tokens are allowed before scoring?','Find the point where selection can move earlier',square)
    labels=[['Identify allowed','tokens'],['Select their','weights'],['Measure gathering','+ scoring']]
    if square:
        for n,lines in enumerate(labels):
            y=385+n*185;b.circle(115,y,42,b.orange)
            b.text(115,y+15,n+1,42,anchor='middle');b.text(220,y-8,lines,42)
            if n<2:b.line(115,y+57,115,y+120,b.orange,3,True)
        b.foot(['Include choosing the allowed token','Changing sets: gather every step','Static sets: report one-time setup','Compare the same allowed outputs'])
    else:
        for n,lines in enumerate(labels):
            x=290+n*510;b.circle(x,355,52,b.orange)
            b.text(x,373,n+1,54,anchor='middle');b.text(x,490,lines,48,anchor='middle')
            if n<2:b.line(x+88,355,x+420,355,b.orange,3,True)
        b.foot(['Include choosing the allowed token','Changing sets: gather every step','Static sets: report one-time setup','Compare the same allowed outputs'])
    return b


def curious(square):
    b=Board('aud-curious','Why score tokens you cannot emit?','Example: only YES or NO is allowed',square)
    if square:
        b.text(55,335,'Score everything',44,b.blue)
        for n,word in enumerate(('YES','road','apple','NO','blue','river')):
            x=60+n%3*330;y=380+n//3*105;b.rect(x,y,290,78,b.blue)
            b.text(x+145,y+52,word,40,b.blue,'middle')
            if word not in ('YES','NO'):b.line(x+20,y+15,x+270,y+65,b.ink,2)
        b.line(540,610,540,680,b.orange,3,True)
        b.text(540,745,'Select first',48,b.orange,'middle')
        for x,word in ((225,'YES'),(595,'NO')):
            b.rect(x,790,260,88,b.orange);b.text(x+130,850,word,44,b.orange,'middle')
        b.foot(['Allowed choices must be known before scoring','Toy words explain the idea, not the test inputs'])
    else:
        b.text(70,290,'Score everything',48,b.blue)
        for n,word in enumerate(('YES','road','apple','NO','blue','river')):
            x=75+n%3*220;y=350+n//3*125;b.rect(x,y,190,90,b.blue)
            b.text(x+95,y+60,word,48,b.blue,'middle')
            if word not in ('YES','NO'):b.line(x+15,y+15,x+175,y+75,b.ink,2)
        b.line(790,465,930,465,b.orange,3,True)
        b.text(1220,290,'Select first',48,b.orange,'middle')
        for y,word in ((350,'YES'),(475,'NO')):
            b.rect(1040,y,360,90,b.orange);b.text(1220,y+60,word,48,b.orange,'middle')
        b.foot(['Allowed choices must be known before scoring','Toy words explain the idea, not the test inputs'])
    return b


def ml(square):
    b=Board('aud-ml-engineer','Select weights, then score','One allowed set shared across the batch',square)
    if square:
        b.text(540,350,'H @ W[ids].T',60,b.orange,'middle')
        for x,w,label in ((65,220,['H','8 × 2048']),(440,270,['W[ids].T','2048 × 64']),(860,145,['Scores','8 × 64'])):
            b.matrix(x,430,w,110,b.orange if x>65 else b.ink)
            b.text(x+w/2,610,label,36,anchor='middle')
        b.text(365,505,'×',54,anchor='middle');b.line(745,485,820,485,b.orange,3,True)
        b.text(540,790,['Allowed argmax → token ID','Preserve ID order and tie rules'],38,anchor='middle')
        b.foot(['64 allowed IDs · strict FP32 · TF32 off','Check score tolerance and exact token agreement'])
    else:
        b.text(800,290,'H @ W[ids].T',64,b.orange,'middle')
        for x,w,label in ((80,260,'H: 8 × 2048'),(565,360,'W[ids].T: 2048 × 64'),(1210,250,'Scores: 8 × 64')):
            b.matrix(x,380,w,130,b.orange if x>80 else b.ink)
            b.text(x+w/2,590,label,48,anchor='middle')
        b.text(455,475,'×',64,anchor='middle');b.line(1000,445,1120,445,b.orange,3,True)
        b.foot(['Allowed argmax → token ID · preserve ID order and tie rules','64 allowed IDs · strict FP32 · TF32 off','Check score tolerance and exact token agreement'])
    return b


def detail(square,stem='creative-v2/matrix-cut'):
    b=Board(stem,'Compare the complete output-head stage','64 changing allowed IDs · gathering included',square)
    c=next(c for c in SUMMARY['cells'] if c['size']==64 and c['mode']=='changing')
    ratio=c['speedup'];ratios=[r['baseline_wall_ms']/r['candidate_wall_ms'] for r in PAIRS if r['size']==64 and r['mode']=='changing']
    x0,width=(100,700) if square else (100,1080)
    ys=(420,610) if square else (330,495)
    for y,series,label in zip(ys,('baseline','candidate'),('Full scoring + filter','Allowed rows first')):
        color=b.blue if series=='baseline' else b.orange
        b.text(x0,y-28,label,40 if square else 48,color)
        end=b.bar(c,series,x0,y,width,24)
        value=c['baseline_median_wall_ms' if series=='baseline' else 'candidate_median_wall_ms']
        b.text(end+18,y+24,f'{value:.3f} ms',40 if square else 48,color)
    axis_y=705 if square else 560
    for tick in (0,2,4,6):
        x=x0+width*tick/6;b.line(x,axis_y-12,x,axis_y, b.ink,2);b.text(x,axis_y+50,str(tick),36 if square else 48,anchor='middle')
    b.text(b.w/2,axis_y+108,'Milliseconds per warm 16-step block',36 if square else 48,anchor='middle')
    b.text(b.w/2,875 if square else 740,f'{ratio:.2f}× median ratio · {min(ratios):.2f}–{max(ratios):.2f}× paired range',36 if square else 48,anchor='middle')
    b.foot(['Median bars · min–max whiskers · 3 pairs in one process',
            'Synthetic FP32 · batch 8 · TF32 off' if square else 'Synthetic FP32 · B8 · H2048 · V131072 · TF32 off'])
    return b


def performance(square):
    b=Board('aud-performance-engineer','Include gathering in timing','64 changing allowed IDs · warm 16-step blocks',square)
    c=next(c for c in SUMMARY['cells'] if c['size']==64 and c['mode']=='changing')
    ratios=[r['baseline_wall_ms']/r['candidate_wall_ms'] for r in PAIRS if r['size']==64 and r['mode']=='changing']
    b.text(b.w/2,300 if square else 232,f"{c['speedup']:.2f}× ratio of retained medians",36 if square else 48,b.orange,'middle')
    x0,width=(100,700) if square else (100,1080)
    for y,series,label in zip((400,555) if square else (330,455),('baseline','candidate'),('Full scoring + filter','Allowed rows first')):
        color=b.blue if series=='baseline' else b.orange
        b.text(x0,y-28,label,40 if square else 48,color)
        end=b.bar(c,series,x0,y,width,24)
        value=c['baseline_median_wall_ms' if series=='baseline' else 'candidate_median_wall_ms']
        b.text(end+18,y+24,f'{value:.3f} ms',40 if square else 48,color)
    axis=650 if square else 520
    for tick in (0,2,4,6):
        x=x0+width*tick/6;b.line(x,axis-12,x,axis,b.ink,2);b.text(x,axis+50,str(tick),36 if square else 48,anchor='middle')
    b.text(b.w/2,axis+108,'Milliseconds per warm 16-step block',36 if square else 48,anchor='middle')
    setup = {row['size']: row['row_setup_wall_ms'] for row in SUMMARY['cells'] if row['mode']=='static'}
    b.foot([f'Static setup: 64 IDs {setup[64]:.3f} ms (unexplained outlier)',
            f'Static 16/256 IDs: {setup[16]:.3f}/{setup[256]:.3f} ms · setup excluded',
            'FP32 · B8 · H2048 · V131072 · TF32 off',
            f'3 pairs, one process · paired ratios: {min(ratios):.2f}–{max(ratios):.2f}×'])

    return b


def kernel(square):
    b=Board('aud-kernel-engineer','Reduce the output dimension','Same inputs, fewer allowed weight rows',square)
    if square:
        for y,title,col,w,shape in ((345,'Full output projection',b.blue,700,'8 × 2048  @  2048 × 131072'),(650,'64 allowed outputs',b.orange,250,'8 × 2048  @  2048 × 64')):
            b.text(55,y,title,42,col);b.matrix(80,y+45,w,100,col);b.text(80,y+207,shape,38)
        b.foot(['FP32 weights used: 1 GiB → 512 KiB','Full weight matrix stays resident','Derived sizes · traffic unmeasured'])
    else:
        for y,title,col,w,shape in ((250,'Full output projection',b.blue,1000,'8 × 2048  @  2048 × 131072'),(515,'64 allowed outputs',b.orange,300,'8 × 2048  @  2048 × 64')):
            b.text(70,y,title,48,col)
            b.matrix(80,y+35,w,80,col)
            b.text(80 if col==b.blue else 440,y+180 if col==b.blue else y+95,shape,48)
        b.foot(['FP32 weights used: 1 GiB → 512 KiB','Full weight matrix stays resident','Derived sizes · traffic unmeasured'])

    return b


def platform(square):
    b=Board('aud-platform-operations','Does the batch share one set?','Only one shared allowed set was tested',square)
    if square:
        b.text(55,340,'Batch 8 → one allowed set',42,b.orange)
        for n in range(8):b.rect(80+n%4*95,390+n//4*70,65,45,b.orange)
        b.line(500,450,590,450,b.orange,3,True);b.text(810,450,['Gather rows','then score'],40,b.orange,'middle')
        b.text(55,675,'Different IDs per request?',42)
        for n in range(4):
            x=70+n*250;b.rect(x,730,205,85,b.ink);b.text(x+102,785,f'Set {n+1}',36,anchor='middle')
        b.foot(['Per-request sets need a different implementation','Measure preparation and full request time'])
    else:
        b.text(70,290,'Batch 8 → one allowed set',48,b.orange)
        for n in range(8):b.rect(80+n%4*95,345+n//4*70,65,45,b.orange)
        b.line(500,405,650,405,b.orange,3,True);b.text(1060,415,'Gather rows → score',54,b.orange,'middle')
        b.text(70,610,'Different IDs per request?',48)
        for n in range(4):
            x=80+n*220;b.rect(x,655,175,80,b.ink);b.text(x+87,710,f'Set {n+1}',48,anchor='middle')
        b.foot(['Per-request sets need a different implementation','Measure preparation and full request time'])
    return b


def business(square):
    b=Board('aud-business-capacity','Measure requests before promising capacity','A stage gain does not measure request capacity',square)
    pairs=[r for r in PAIRS if r['size']==64 and r['mode']=='changing'];ratios=[r['baseline_wall_ms']/r['candidate_wall_ms'] for r in pairs]
    if square:
        b.circle(270,440,135,b.orange);b.text(270,438,f"{next(c for c in SUMMARY['cells'] if c['size']==64 and c['mode']=='changing')['speedup']:.2f}×",60,b.orange,'middle');b.text(270,500,'Stage gain',36,anchor='middle')
        b.line(455,440,610,440,b.ink,3,True);b.text(820,415,['Measure','requests'],42,anchor='middle')
        b.text(540,680,f'Paired range: {min(ratios):.2f}–{max(ratios):.2f}×',40,anchor='middle')
        b.text(540,790,['How much request time','is in the output head?'],42,anchor='middle')
        b.foot(['64 changing IDs · gather included · 16-step blocks','FP32 · B8 · H2048 · V131072 · TF32 off','3 pairs in one process · requests unmeasured'])
    else:
        b.circle(300,400,140,b.orange);b.text(300,400,f"{next(c for c in SUMMARY['cells'] if c['size']==64 and c['mode']=='changing')['speedup']:.2f}×",64,b.orange,'middle');b.text(300,465,'Stage gain',48,anchor='middle')
        b.line(500,400,690,400,b.ink,3,True);b.text(1080,400,'Measure real requests',54,anchor='middle')
        b.text(800,645,f'Paired range: {min(ratios):.2f}–{max(ratios):.2f}×',48,anchor='middle')
        b.foot(['64 changing IDs · gather included · warm 16-step blocks',
                'FP32 · B8 · H2048 · V131072 · TF32 off',
                '3 paired blocks, one process · whole requests unmeasured'])
    return b


DRAW={'a':mechanism,'b':comparison,'c':inspection,'aud-curious':curious,'aud-ml-engineer':ml,
      'aud-performance-engineer':performance,'aud-kernel-engineer':kernel,
      'aud-platform-operations':platform,'aud-business-capacity':business,'creative-v2/matrix-cut':detail}


def sources(package):
    global SUMMARY,PAIRS,TOKENS,FONT
    for name,digest in PINS.items():require(sha(package/name)==digest,'Retained input changed: '+name)
    SUMMARY=json.loads((package/'evidence/summary.json').read_text())
    PAIRS=[json.loads(line) for line in (package/'evidence/raw.jsonl').read_text().splitlines() if json.loads(line).get('kind')=='pair']
    TOKENS=json.loads((HERE/'visual-tokens.json').read_text());FONT=package/'images/fonts/Inter.ttf'
    require(TOKENS['typography']['font_family']=='Inter' and TOKENS['typography']['allow_handwritten'] is False,'Inter/shared token authority required')
    require(len(SUMMARY['cells'])==6 and len(PAIRS)==18,'Retained exact six-cell/three-pair scope changed')


def tool(name):
    path=shutil.which(name)
    require(path, 'Required existing tool unavailable: '+name+'. No installation is performed.')
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root',type=Path,default=HERE.parent)
    parser.add_argument('--output-root',type=Path,default=HERE)
    parser.add_argument('--public-layout',action='store_true',help='Write the mapped comparison-detail names used by the public companion')
    args=parser.parse_args();package=args.package_root.resolve();out=args.output_root.resolve();out.mkdir(parents=True,exist_ok=True)
    target_stem=lambda stem: 'comparison-detail' if args.public_layout and stem=='creative-v2/matrix-cut' else stem
    sources(package);converter=tool('rsvg-convert');matcher=tool('fc-match')
    records={};dimensions={};checks=[];geometry_rows=[]
    with tempfile.TemporaryDirectory(prefix='inter-fontconfig-',dir=out) as temporary:
        config=Path(temporary)/'fonts.conf'
        config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><dir>'+escape(str(FONT.parent))+'</dir><cachedir>'+escape(temporary)+'</cachedir></fontconfig>')
        env=dict(os.environ,PANGOCAIRO_BACKEND='fc',FONTCONFIG_FILE=str(config))
        matched=subprocess.check_output([matcher,'-f','%{file}\n%{family}\n%{style}\n','Inter:style=Regular'],env=env,text=True).splitlines()
        require(Path(matched[0]).resolve()==FONT.resolve() and matched[1]=='Inter' and matched[2]=='Regular','Exact regular Inter font did not resolve')
        for stem,draw in DRAW.items():
            entries={};copy=[]
            for square in (False,True):
                board=draw(square);suffix='-square' if square else '';path=out/(target_stem(stem)+suffix+'.svg');path.parent.mkdir(parents=True,exist_ok=True)
                svg=board.finish();path.write_text(svg);tree=ET.fromstring(svg)
                require(all(t.get('font-family')=='Inter' and t.get('font-weight')=='400' for t in tree.iter() if t.tag.endswith('text')),'Nonregular Inter text')
                bars=[t for t in tree.iter() if t.get('data-science')=='bar']
                require(len(bars)==len(board.chart),'Scientific bar metadata missing')
                clearance=check_text_bar_clearance(board.boxes,bars) if board.stem=='b' else None
                paragraphs=check_paragraph_gaps(board)
                for node,row in zip(bars,board.chart):
                    require(abs(float(node.get('width'))-row['scale_px_per_ms']*row['median_ms'])<1e-9,'Median bar geometry changed')
                    require(node.get('fill')==(board.blue if row['series']=='baseline' else board.orange),'Series color changed')
                require(' Q' not in svg and 'font-weight="bold"' not in svg,'Sketched geometry or bold text found')
                minimum_css=min(float(t.get('font-size'))*400/board.w for t in tree.iter() if t.tag.endswith('text'))
                require(minimum_css>=TOKENS['typography']['minimum_phone_px']['material_condition'],'Text below shared 400px phone minimum')
                copies=[]
                for width,height,extra in ((board.w,board.h,''),(400,400 if square else 225,'-400')):
                    png=out/(target_stem(stem)+suffix+extra+'.png')
                    subprocess.run([converter,str(path),'-w',str(width),'-h',str(height),'-o',str(png)],env=env,check=True)
                    with Image.open(png) as image:require(image.size==(width,height),'PNG dimensions differ')
                    dimensions[str(png.relative_to(out))]={'width':width,'height':height,'sha256':sha(png)};copies.append(png)
                entries[str(path.relative_to(out))]={'sha256':sha(path),'bytes':path.stat().st_size}
                for png in copies:entries[str(png.relative_to(out))]={'sha256':sha(png),'bytes':png.stat().st_size}
                copy += board.copy
                checks.append({'stem':stem,'square':square,'text_boxes_checked':len(board.boxes),'regular_Inter_only':True,'text_nonoverlap':True,'text_bar_clearance':clearance,'paragraph_separation':paragraphs,'minimum_text_px_at_400':min(float(t.get('font-size'))*400/board.w for t in tree.iter() if t.tag.endswith('text'))})
                geometry_rows.append({'stem':stem,'square':square,'bars':board.chart})
            setup={c['size']:c['row_setup_wall_ms'] for c in SUMMARY['cells'] if c['mode']=='static'}
            side=out/(target_stem(stem)+'.txt')
            side.write_text('\n'.join(dict.fromkeys(copy))+f'\n\nRetained methods: synthetic strict FP32 output head. Batch 8, hidden 2048, vocabulary 131072. One shared device-resident allowed set. TF32 off. Median warm blocks contain 16 steps. Three paired blocks per cell in one process. Whiskers are observed min–max, not confidence intervals. Changing sets include each gather. Static sets reuse gathered weights after one-time setup. Static setup for 16/64/256 IDs: {setup[16]:.6f} / {setup[64]:.6f} / {setup[256]:.6f} ms. The 64-ID setup is an unexplained outlier. Cold-start benefit and break-even were not measured. Whole-model and serving gains were not measured. Different per-request sets and BF16 are untested. Ratios use retained unrounded medians.\n')
            entries[str(side.relative_to(out))]={'sha256':sha(side),'bytes':side.stat().st_size};records[stem]=entries
        # Contact sheets contain exact 400-pixel renders, with no scaling/cropping.
        for square in (False,True):
            suffix='-square' if square else '';tileh=400 if square else 225;rowh=tileh+45;rows=(len(DRAW)+2)//3
            parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1260" height="{rows*rowh+15}"><rect width="100%" height="100%" fill="#ECEDEA"/>']
            for n,stem in enumerate(DRAW):
                x=10+n%3*415;y=10+n//3*rowh;png=out/(target_stem(stem)+suffix+'-400.png');encoded=base64.b64encode(png.read_bytes()).decode()
                parts += [f'<image x="{x}" y="{y}" width="400" height="{tileh}" href="data:image/png;base64,{encoded}"/>',f'<text x="{x+5}" y="{y+tileh+29}" font-family="Inter" font-size="18" font-weight="400">{escape(stem)}</text>']
            parts.append('</svg>');svg=out/('contact-sheet'+suffix+'.svg');svg.write_text('\n'.join(parts));subprocess.run([converter,str(svg),'-o',str(svg.with_suffix('.png'))],env=env,check=True)
        versions={'rsvg':subprocess.check_output([converter,'--version'],text=True).strip()}
        for library in ('pango','cairo'):
            p=shutil.which('pkg-config')
            if p:
                result=subprocess.run([p,'--modversion',library],capture_output=True,text=True)
                versions[library]=result.stdout.strip() if result.returncode==0 else 'Unavailable from installed pkg-config'
            else:versions[library]='pkg-config unavailable'
        receipt={'renderer':'rsvg-convert','versions':versions,'font_resolution':matched,'font_sha256':sha(FONT),'PANGOCAIRO_BACKEND':'fc','tokens_sha256':sha(HERE/'visual-tokens.json'),'source_pins':PINS,'source_sha256':sha(Path(__file__)),'series_roles':{'baseline':TOKENS['colors']['blue'],'allowed_rows':TOKENS['colors']['orange']},'scope':'Graphics export only. No measurement, analyzer, model, account or publication action.','assets':records,'readability_checks':checks,'data_geometry_checks':True,'new_measurements':0}
        (out/'exports.json').write_text(json.dumps(receipt,indent=2)+'\n')
        (out/'dimension-check.json').write_text(json.dumps(dimensions,indent=2)+'\n')
        (out/'scientific-geometry.json').write_text(json.dumps({'retained_summary_sha256':PINS['evidence/summary.json'],'retained_raw_sha256':PINS['evidence/raw.jsonl'],'rows':geometry_rows},indent=2)+'\n')
    print(json.dumps({'families':len(DRAW),'regular_Inter':True,'data_geometry_exact':True,'output':str(out)}))


if __name__=='__main__':main()
