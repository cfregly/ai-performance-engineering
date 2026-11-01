#!/usr/bin/env python3
"""Editable attention reference diagrams and audience cards.

Python 3.10+ standard library; existing rsvg-convert, ImageMagick and Fontconfig.
Uses the retained Inter file, visual tokens and RESULTS.json. No science run,
network, model or GPU call. --check compares a fresh render without overwriting.
"""
from __future__ import annotations
import argparse,base64,hashlib,json,math,os,re,shutil,struct,subprocess,tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from html import escape
ROOT=Path(__file__).resolve().parents[1]
INPUTS=ROOT/'images/attention-scenes'
RESULTS=ROOT/'evidence/derivation/RESULTS.json'
C={}
EXPECTED_RESULTS='d1439661b3fb845fa556338b2779cdaa2a34da46b3ccf62b1d26fec87d667557'
EXPECTED_FONT='29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031'
BW_FLOORS=[('8','0.243269632'),('6.4','0.30408704')]
sha=lambda b:hashlib.sha256(b).hexdigest()

def require(ok,message):
    if not ok:raise ValueError(message)

def run(args,**kw):
    try:return subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,check=True,**kw).stdout
    except subprocess.CalledProcessError as exc:
        raise ValueError('Command failed ('+str(exc.returncode)+'): '+' '.join(map(str,args))+'\n'+(exc.stderr or b'').decode('utf8','replace')) from exc

def atomic_json(path,value):
    if path is None:return
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'.',suffix='.tmp',dir=path.parent);tmp=Path(name)
    try:
        with os.fdopen(fd,'w',encoding='utf8') as f:
            json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    except BaseException:
        tmp.unlink(missing_ok=True);raise

def font_env(temp):
    config=temp/'fonts.conf';config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "fonts.dtd"><fontconfig><dir>'+escape(str(INPUTS))+'</dir><cachedir>'+escape(str(temp/'cache'))+'</cachedir></fontconfig>')
    return dict(os.environ,FONTCONFIG_FILE=str(config),LC_ALL='C',TZ='UTC',PANGOCAIRO_BACKEND='fc')

class Diagram:
    def __init__(self,stem,title,w=1080,h=1080):
        self.stem,self.title,self.w,self.h=stem,title,w,h;self.labels=[];self.edges=[];self.bars=[];self.lines=[];self.nodes={};self.obstacles=[]
        self.parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><title>{escape(title)}</title>',f'<style>@font-face{{font-family:Inter;src:url("__RETAINED_INTER_FONT__") format("truetype");font-weight:100 900}}</style>',f'<rect width="{w}" height="{h}" fill="{C["background"]}"/>']
    def text(self,t,x,y,size=None,weight=400,color='ink',anchor='start'):
        size=size or (64 if self.w==1600 else 48 if self.w==1080 else 52)
        self.labels.append(dict(text=t,x=x,y=y,font_px=size,weight=weight,anchor=anchor))
        self.parts.append(f'<text x="{x}" y="{y}" font-family="Inter" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{C[color]}">{escape(t)}</text>')
    def rect(self,x,y,w,h,color='background',stroke='ink',extra=''):
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{C[color]}" stroke="{C[stroke]}" stroke-width="3" {extra}/>')
        if color != 'background':self.obstacles.append((x,y,x+w,y+h))
    def node(self,identifier,x,y,w,h):
        require(identifier not in self.nodes,'Duplicate route node')
        self.nodes[identifier]=(x,y,x+w,y+h)
        self.parts.append(f'<g data-node-id="{identifier}">');self.rect(x,y,w,h);self.parts.append('</g>')
    def line(self,x1,y1,x2,y2,color='ink',width=4,dash='',extra=''):
        self.parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C[color]}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+' '+extra+'/>')
        self.lines.append(dict(x1=x1,y1=y1,x2=x2,y2=y2,width=width))
    def arrow(self,x1,y1,x2,y2,color='blue',name=None,source=None,target=None):
        self.route([(x1,y1),(x2,y2)],color,name,source,target)
    def route(self,points,color='ink',name=None,source=None,target=None):
        x1,y1=points[0];x2,y2=points[-1]
        px,py=points[-2]
        dx,dy=x2-px,y2-py;length=math.hypot(dx,dy);require(length>0,'Zero edge');ux,uy=dx/length,dy/length
        self.parts.append(f'<g data-edge="{escape(name or str(len(self.edges)))}" data-source="{source or ""}" data-target="{target or ""}">')
        for start,end in zip(points,points[1:]):self.line(*start,*end,color,5)
        for sign in [-1,1]:self.line(x2-18*ux+sign*11*uy,y2-18*uy-sign*11*ux,x2,y2,color,5)
        self.parts.append('</g>');self.edges.append(dict(name=name,x1=x1,y1=y1,x2=x2,y2=y2,points=points,source=source,target=target))
    def title_lines(self,lines):
        size=76 if self.w==1600 else 64 if self.w==1200 else 60
        x=80 if self.w==1600 else 72 if self.w==1080 else 84
        for i,t in enumerate(lines):self.text(t,x,112+i*(size+16),size,700)
    def finish(self,asset_context=None):
        prefix=asset_prefix(self.stem,asset_context)
        return ('\n'.join(self.parts)+'\n</svg>\n').replace('__RETAINED_INTER_FONT__',prefix+'/Inter.ttf').encode()


def asset_prefix(stem,context=None):
    # Resolve both bases before counting parents across macOS /var -> /private/var.
    return Path(os.path.relpath(INPUTS.resolve(),(context or ROOT).resolve()/Path(stem).parent)).as_posix()


def operands():
    d=Diagram('images/scientific/02-operand-paths','Trace a Blackwell tensor-core path',1200,1200)
    d.title_lines(['Trace a Blackwell','tensor-core path'])
    d.text('Common tcgen05 SMEM-operand route',84,292)
    d.node('HBM',84,365,220,145);d.text('HBM',194,453,52,600,anchor='middle')
    d.node('SMEM',424,365,250,150);d.text('Shared',549,423,52,600,anchor='middle');d.text('memory',549,483,52,600,anchor='middle')
    d.node('MMA',856,365,260,150);d.text('Tensor',986,423,52,600,anchor='middle');d.text('cores',986,483,52,600,anchor='middle')
    d.arrow(320,400,408,400,'ink','HBM-to-SMEM','HBM','SMEM');d.arrow(408,480,320,480,'ink','SMEM-to-HBM','SMEM','HBM')
    d.arrow(690,437,840,437,'ink','SMEM-operands-to-MMA','SMEM','MMA')
    d.text('A + B',765,506,52,500,anchor='middle')
    d.node('TMEM',716,730,400,160);d.text('TMEM',916,792,52,600,anchor='middle');d.text('accumulator',916,854,52,600,anchor='middle')
    d.node('registers',84,730,360,160);d.text('Thread',264,792,52,600,anchor='middle');d.text('registers',264,854,52,600,anchor='middle')
    d.arrow(950,526,950,714,'ink','MMA-writes-TMEM','MMA','TMEM');d.arrow(1040,714,1040,526,'ink','MMA-reads-TMEM','TMEM','MMA')
    d.text('MMA reads + writes',925,645,52,500,anchor='end')
    d.arrow(700,768,460,768,'ink','TMEM-load-to-registers','TMEM','registers');d.arrow(460,842,700,842,'ink','register-store-to-TMEM','registers','TMEM')
    d.text('tcgen05.ld / st',580,953,52,500,anchor='middle')
    d.text('A can also come from TMEM.',84,1040)
    d.text('Use the required synchronization.',84,1110)
    return d


def sensitivity():
    d=Diagram('images/scientific/05-independent-budgets','Test assumed bandwidths separately',1200,1200)
    d.title_lines(['Test assumed bandwidths','separately'])
    d.text('Same traffic: 1,856 MiB (1.946 GB)',84,298)
    d.text('One layer · one device · FP32 results',84,365)
    d.text('Read + write payload ceiling',84,430)
    x0,width,axismax=84,1032,.35
    for y,(bw,val) in zip([505,730],BW_FLOORS):
        d.text('Assumed '+bw+' TB/s',x0,y,52,600)
        d.text('≥'+format(float(val),'.3f')+' ms',1116,y,52,600,'orange','end')
        barw=width*float(val)/axismax
        d.rect(x0,y+30,barw,44,'orange','orange',f'data-value-ms="{val}" data-axis-max-ms="0.35"')
        d.bars.append(dict(value_ms=val,x=x0,width=barw,plot_width=width,axis_max_ms=axismax,zero=0))
    targetx=x0+width*.2/axismax
    d.line(targetx,480,targetx,824,'blue',5,'12 10','data-target-ms="0.2" data-axis-max-ms="0.35"')
    d.line(x0,830,x0+width,830,extra='data-quantitative-axis="ms" data-axis-max-ms="0.35"')
    for value in [0,.2,.35]:
        d.parts.append(f'<g data-quantitative-tick-ms="{value}">')
        x=x0+width*value/axismax;d.line(x,830,x,842);d.text(format(value,'.2f'),x,910,52,400,'ink','start' if value==0 else 'end' if value==.35 else 'middle')
        d.parts.append('</g>')
    d.text('Time (ms) · zero-based scale',84,990)
    d.text('Blue line: 0.200 ms target.',84,1060,52,500,'blue')
    d.text('Calculated floors, not runtimes.',84,1130)
    return d


def concept(stem,wide):
    w,h=(1600,900) if wide else (1080,1080);d=Diagram('images/'+stem+('' if wide else '-square'),'Scratch traffic' if stem=='a' else 'Transfer floor and target',w,h)
    if stem=='a':
        d.title_lines(['Write now. Read later.'] if wide else ['Write now. Read later.','Pay twice.'])
        if wide:
            for name,x,lines in [('producers',80,['Producers','finish']),('scratch',620,['Scratch','in HBM']),('reducer',1160,['Reducer','combines'])]:
                d.node(name,x,300,360,225);d.text(lines[0],x+180,390,64,600,anchor='middle');d.text(lines[1],x+180,475,64,600,anchor='middle')
            d.arrow(460,410,600,410,'blue','write-out','producers','scratch');d.arrow(1000,410,1140,410,'orange','reread','scratch','reducer')
            d.text('Write results, then read them back.',80,675);d.text('Smaller batches need a different schedule.',80,790)
        else:
            for name,y,lines in [('producers',280,['Producers finish']),('scratch',515,['Scratch in HBM']),('reducer',750,['Reducer combines'])]:
                d.node(name,100,y,880,120);d.text(lines[0],540,y+78,48,600,anchor='middle')
            d.arrow(540,420,540,495,'blue','write-out','producers','scratch');d.arrow(540,655,540,730,'orange','reread','scratch','reducer')
            d.text('Write out, then read back.',72,972)
    else:
        d.title_lines(['Transfers alone miss the target'] if wide else ['Transfers alone','miss the target'])
        # Same retained floor as selected figure04, with explicit common zero axis.
        x0=80 if wide else 72;plot=1440 if wide else 936;maxval=.3
        for y,label,v,col in [(350 if wide else 390,'Target',.2,'blue'),(545 if wide else 600,'Calculated floor',float(BW_FLOORS[0][1]),'orange')]:
            d.text(label,x0,y-(4 if wide else 0),64 if wide else 48,600);d.text(f'{v:.3f} ms',w-x0,y-(4 if wide else 0),64 if wide else 48,600,col,'end')
            d.rect(x0,y+30,plot*v/maxval,40,col,col,f'data-value-ms="{v}" data-axis-max-ms="0.3"');d.bars.append(dict(value_ms=str(v),x=x0,width=plot*v/maxval,plot_width=plot,axis_max_ms=.3,zero=0))
        axis_y=680 if wide else 770;d.line(x0,axis_y,x0+plot,axis_y,extra='data-quantitative-axis="ms" data-axis-max-ms="0.3"')
        for val in [0,.15,.3]:
            d.parts.append(f'<g data-quantitative-tick-ms="{val}">')
            x=x0+plot*val/maxval;d.line(x,axis_y,x,axis_y+12);d.text(f'{val:.2f}',x,axis_y+78,64 if wide else 48,anchor='start' if val==0 else 'end' if val==.3 else 'middle')
            d.parts.append('</g>')
        if wide:
            d.text('One layer · one device',80,240)
            d.text('Time (ms) · assumed 8 TB/s · calculated',80,850)
        else:
            d.text('Time (ms) · assumed 8 TB/s',72,930)
            d.text('One layer · one device · calculated',72,1000)
    return d

AUDIENCES=['curious','ml-engineer','performance-engineer','kernel-engineer','platform-operations','business-capacity']

def audience(kind,wide):
    w,h=(1600,900) if wide else (1080,1080);suffix='' if wide else '-square';d=Diagram('images/aud-'+kind+suffix,kind.replace('-',' '),w,h)
    fs=64 if wide else 48;left=80 if wide else 72;right=w-left
    if kind=='curious':
        d.title_lines(['Write it out. Read it back.'] if wide else ['Write it out.','Read it back.'])
        # Large mechanism fills the wide canvas; no square thumbnail is shrunk.
        y=290 if wide else 340;bx=80 if wide else 72;bw=560 if wide else 390;rx=1000 if wide else 618
        d.node('on-chip',bx,y,bw,210);d.text('On-chip',bx+bw/2,y+85,fs,600,anchor='middle');d.text('workspace',bx+bw/2,y+165,fs,600,anchor='middle')
        d.node('HBM',rx,y,bw,210);d.text('Off-chip memory' if wide else 'Off-chip',rx+bw/2,y+85,fs,600,anchor='middle');d.text('(HBM)' if wide else 'memory (HBM)',rx+bw/2,y+165,fs,600,anchor='middle')
        d.arrow(bx+bw+20,y+55,rx-20,y+55,'blue','write-out','on-chip','HBM');d.arrow(rx-20,y+170,bx+bw+20,y+170,'orange','reread','HBM','on-chip')
        if wide:
            d.text('Write results out',80,650,fs,500,'blue');d.text('Reread them later',900,650,fs,500,'orange')
            d.text('Here, the final sum waits for every piece.',80,800)
        else:
            d.text('Write out. Reread later.',72,690,fs,500)
            d.text('Here, the final sum waits',72,850);d.text('until every piece is finished.',72,925)
    elif kind=='ml-engineer':
        d.title_lines(['Choose when to combine results'] if wide else ['Choose when to','combine results'])
        if wide:
            d.text('Keep outputs separate',80,320,64,600);d.text('Combine in batches',880,320,64,600)
            d.text('Simpler schedule',80,440);d.text('Less scratch space',880,440)
            d.text('More scratch + traffic',80,555,64,500,'orange');d.text('More coordination',880,555,64,500,'orange')
            d.text('Use the simpler path when it meets your budget.',80,790)
        else:
            for y,title,lines in [(335,'Keep outputs separate',['Simple producer schedule','More scratch + traffic']),(630,'Combine smaller batches',['Less scratch space','More coordination'])]:
                d.text(title,72,y,48,600);d.text(lines[0],72,y+80);d.text(lines[1],72,y+155,48,500,'orange')
            d.text('Keep the simpler path',72,940);d.text('when it meets your budget.',72,1005)
    elif kind=='performance-engineer':
        d.title_lines(['Check each resource separately'] if wide else ['Check each resource','separately'])
        if wide:
            d.text('Compute floor',80,340,64,600);d.text('Transfer floor',850,340,64,600)
            d.text('Work ÷ compute rate',80,470,64,500,'orange');d.text('Bytes ÷ bandwidth',850,470,64,500,'orange')
            d.text('Do not add floors that can overlap.',80,665);d.text('Dependencies can push real time above both.',80,790)
        else:
            d.text('Compute floor',72,345,48,600);d.text('Work ÷ compute rate',72,430,48,500,'orange')
            d.text('Transfer floor',72,590,48,600);d.text('Bytes ÷ bandwidth',72,675,48,500,'orange')
            d.text('Do not add overlapping floors.',72,850);d.text('Dependencies may make',72,935);d.text('real time longer than both.',72,1005)
    elif kind=='kernel-engineer':
        d.title_lines(['Add bytes while lifetimes overlap'] if wide else ['Add bytes while','lifetimes overlap'])
        x0=370 if wide else 310;plot=1110 if wide else 690;y1=250 if wide else 370;y2=y1+150
        d.text('Buffer A',left,y1+70,fs,600);d.text('Buffer B',left,y2+70,fs,600)
        # Qualitative timeline only: widths encode lifetime order, never byte size.
        d.rect(x0,y1,plot*.68,100,'blue','blue');d.rect(x0+plot*.4,y2,plot*.6,100,'orange','orange')
        d.line(x0+plot*.4,y1-25,x0+plot*.4,y2+135,'ink',3,'10 10');d.line(x0+plot*.68,y1-25,x0+plot*.68,y2+135,'ink',3,'10 10')
        ty=y2+175;d.arrow(x0,ty,x0+plot,ty,'ink','time');d.text('Time → (qualitative)',x0,ty+85,fs,500)
        if wide:
            d.text('In the overlap, count A + B.',80,750);d.text('Reuse storage only after its last reader.',80,830)
        else:
            d.text('In the overlap, count A + B.',72,885);d.text('Reuse after the last reader.',72,965)
    elif kind=='platform-operations':
        d.title_lines(['Compare the whole update'] if wide else ['Compare the','whole update'])
        if wide:
            d.text('Equal math + quality',80,340,64,600);d.text('Equal load + resources',820,340,64,600)
            d.line(80,410,1520,410,'blue',5)
            d.text('Include compute, transfers and synchronization.',80,560)
            d.text('Check the complete training step.',80,700);d.text('A smaller buffer is not a whole-model speedup.',80,815)
        else:
            for y,line in [(335,'Equal math and quality'),(420,'Equal load and resources'),(585,'Include compute, transfers'),(660,'and synchronization.'),(840,'Check the complete step.'),(925,'Smaller buffers alone'),(1000,'do not prove a speedup.')]:d.text(line,72,y)
    elif kind=='business-capacity':
        d.title_lines(['Screen the design before building'] if wide else ['Screen the design','before building'])
        # One decision diamond. Distinct directed pass/fail lanes never cross.
        if wide:
            cx,cy,rx,ry=800,270,440,120
            pts=f'{cx},{cy-ry} {cx+rx},{cy} {cx},{cy+ry} {cx-rx},{cy}'
            d.nodes['decision']=(cx-rx,cy-ry,cx+rx,cy+ry)
            d.parts.append(f'<g data-node-id="decision"><polygon points="{pts}" fill="{C["background"]}" stroke="{C["ink"]}" stroke-width="4"/></g>')
            d.text('Floor ≤ target?',cx,cy+22,64,600,anchor='middle')
            d.node('fail-outcome',80,470,340,280);d.node('pass-outcome',1150,470,400,280)
            d.route([(344,270),(250,270),(250,450)],'orange','fail-end','decision','fail-outcome')
            d.route([(1256,270),(1350,270),(1350,450)],'blue','pass-end','decision','pass-outcome')
            for x,label,lines,color in [(250,'NO',['Reject','design'],'orange'),(1350,'YES',['Implement','+ measure'],'blue')]:
                d.text(label,x,540,64,600,color,'middle');d.text(lines[0],x,620,64,600,color,'middle');d.text(lines[1],x,700,64,600,color,'middle')
            d.text('Passing is no runtime guarantee.',80,830)
        else:
            cx,cy,rx,ry=540,375,300,140
            pts=f'{cx},{cy-ry} {cx+rx},{cy} {cx},{cy+ry} {cx-rx},{cy}'
            d.nodes['decision']=(cx-rx,cy-ry,cx+rx,cy+ry)
            d.parts.append(f'<g data-node-id="decision"><polygon points="{pts}" fill="{C["background"]}" stroke="{C["ink"]}" stroke-width="4"/></g>')
            d.text('Floor ≤ target?',540,393,48,600,anchor='middle')
            d.node('fail-outcome',72,630,256,280);d.node('pass-outcome',728,630,304,280)
            d.route([(224,375),(200,375),(200,610)],'orange','fail-end','decision','fail-outcome')
            d.route([(856,375),(880,375),(880,610)],'blue','pass-end','decision','pass-outcome')
            for x,label,lines,color in [(200,'NO',['Reject','design'],'orange'),(880,'YES',['Implement','+ measure'],'blue')]:
                d.text(label,x,700,48,600,color,'middle');d.text(lines[0],x,790,48,600,color,'middle');d.text(lines[1],x,865,48,600,color,'middle')
            d.text('Passing is no runtime guarantee.',72,1000)
    return d


def all_diagrams():return [operands(),sensitivity()]+[concept(s,w) for s in ['a','b'] for w in [False,True]]+[audience(s,w) for s in AUDIENCES for w in [False,True]]

def validate_inputs():
    global C
    tokens=json.loads((INPUTS/'visual-tokens.json').read_text())
    require(tokens['colors']=={'background':'#FBFBF8','ink':'#202326','blue':'#3478BA','orange':'#E87823'}, 'Retained palette changed')
    C=tokens['colors']
    require(sha(RESULTS.read_bytes())==EXPECTED_RESULTS,'Retained RESULTS.json changed')
    require(sha((INPUTS/'Inter.ttf').read_bytes())==EXPECTED_FONT,'Retained Inter changed')
    data=json.loads(RESULTS.read_text())
    require(data['derivation']['hypothetical_bandwidth_floors_ms']==dict(BW_FLOORS),'Retained floor values changed')
    return data


def measure(label,env,cache):
    key=(label['text'],label['font_px'],label['weight'])
    if key not in cache:
        sample=f'<svg xmlns="http://www.w3.org/2000/svg" width="5000" height="300"><text x="100" y="200" font-family="Inter" font-size="{key[1]}" font-weight="{key[2]}">{escape(key[0])}</text></svg>'.encode()
        raster=run(['rsvg-convert','--format','png'],input=sample,env=env)
        geometry=run(['magick','png:-','-format','%@','info:'],input=raster,env=env).decode().strip()
        m=re.fullmatch(r'(\d+)x(\d+)\+(-?\d+)\+(-?\d+)',geometry);require(m is not None,'Unexpected glyph box '+geometry)
        width,height,x,y=map(int,m.groups());cache[key]=(width,height,x-100,y-200)
    return cache[key]


def validate_route_svg(d, svg):
    root=ET.fromstring(svg)
    ns={'s':'http://www.w3.org/2000/svg'}
    groups=[node for node in root.iter() if 'data-edge' in node.attrib]
    require(len(groups)==len(d.edges),'SVG directed edge count changed')
    for node,edge in zip(groups,d.edges):
        require(node.attrib['data-edge']==(edge['name'] or str(d.edges.index(edge))),'SVG edge identity changed')
        lines=node.findall('s:line',ns)
        require(len(lines)==len(edge['points'])+1,'SVG arrow needs a shaft and two head lines for its connected route')
        for shaft,(start,end) in zip(lines,zip(edge['points'],edge['points'][1:])):
            actual=[float(shaft.attrib[k]) for k in ['x1','y1','x2','y2']]
            require(actual==[*start,*end],'SVG arrow direction or endpoint differs from declared route')
        for head in lines[-2:]:
            require(float(head.attrib['x2'])==edge['x2'] and float(head.attrib['y2'])==edge['y2'],'SVG arrowhead does not reach shaft target')
    if d.nodes:
        bounds={}
        for node in root.iter():
            identifier=node.attrib.get('data-node-id')
            if identifier is None:continue
            require(identifier not in bounds and len(node)==1,'Drawn route box IDs must be unique')
            shape=node[0]
            if shape.tag.endswith('rect'):
                x,y,w,h=(float(shape.attrib[k]) for k in ['x','y','width','height']);bounds[identifier]=(x,y,x+w,y+h)
            else:
                require(shape.tag.endswith('polygon'),'Unsupported route node shape')
                points=[tuple(map(float,p.split(','))) for p in shape.attrib['points'].split()]
                bounds[identifier]=(min(p[0] for p in points),min(p[1] for p in points),max(p[0] for p in points),max(p[1] for p in points))
        require(bounds==d.nodes,'Emitted intended route boxes changed')
        for node,edge in zip(groups,d.edges):
            require(node.attrib.get('data-source')==edge['source'] and node.attrib.get('data-target')==edge['target'],'Emitted intended route identity changed')
            for identifier,point in [(edge['source'],edge['points'][0]),(edge['target'],edge['points'][-1])]:
                require(identifier in bounds,'Route endpoint lacks intended drawn box')
                distances={key:point_box_distance(point,box) for key,box in bounds.items()}
                require(distances[identifier]<=20,'Route endpoint detached from intended box: '+identifier)
                require(all(distances[identifier]<v-1e-6 for key,v in distances.items() if key!=identifier),'Route endpoint nearer a different drawn box: '+identifier)
    validate_bar_svg(d,root)
    return True


def point_box_distance(point,box):
    x,y=point;left,top,right,bottom=box
    if left<=x<=right and top<=y<=bottom:return min(x-left,right-x,y-top,bottom-y)
    return math.hypot(max(left-x,0,x-right),max(top-y,0,y-bottom))


def validate_bar_svg(d,root):
    bars=[node for node in root.iter() if 'data-value-ms' in node.attrib]
    if '05-independent' in d.stem:
        expected=[(.243269632,535,44),(.30408704,760,44)];x0,plot,maximum,axis_y=84,1032,.35,830
    elif d.stem in ['images/b','images/b-square']:
        expected=[(.2,380 if d.w==1600 else 420,40),(.243269632,575 if d.w==1600 else 630,40)]
        x0,plot,maximum,axis_y=(80,1440,.3,680) if d.w==1600 else (72,936,.3,770)
    else:
        require(not bars,'Unexpected quantitative bar');return
    require(len(bars)==len(expected),'Emitted quantitative bar inventory changed')
    for bar,(value,y,height) in zip(bars,expected):
        a=bar.attrib
        require(bar.tag.endswith('rect') and float(a['data-value-ms'])==value and float(a['data-axis-max-ms'])==maximum,'Emitted quantitative bar value or scale changed')
        geometry=[float(a[k]) for k in ['x','y','width','height']]
        require(all(math.isfinite(v) for v in geometry),'Non-finite emitted bar geometry')
        require(geometry[0]==x0 and geometry[1]==y and geometry[3]==height and abs(geometry[2]-plot*value/maximum)<1e-8,'Emitted quantitative bar differs from retained zero scale')
    axes=[node for node in root.iter() if 'data-quantitative-axis' in node.attrib]
    require(len(axes)==1,'Emitted quantitative axis inventory changed')
    a=axes[0].attrib
    require(a['data-quantitative-axis']=='ms' and float(a['data-axis-max-ms'])==maximum and [float(a[k]) for k in ['x1','y1','x2','y2']]==[x0,axis_y,x0+plot,axis_y],'Emitted quantitative axis origin or extent changed')
    tick_values=[0,.2,.35] if '05-independent' in d.stem else [0,.15,.3]
    tick_y=910 if '05-independent' in d.stem else axis_y+78
    ticks=[node for node in root.iter() if 'data-quantitative-tick-ms' in node.attrib]
    require(len(ticks)==len(tick_values),'Emitted quantitative tick inventory changed')
    for tick,value in zip(ticks,tick_values):
        require(float(tick.attrib['data-quantitative-tick-ms'])==value and len(tick)==2,'Emitted quantitative tick value or children changed')
        line,label=tick;position=x0+plot*value/maximum
        require(line.tag.endswith('line') and [float(line.attrib[k]) for k in ['x1','y1','x2','y2']]==[position,axis_y,position,axis_y+12],'Emitted quantitative tick position differs from axis scale')
        require(label.tag.endswith('text') and label.text==format(value,'.2f') and float(label.attrib['x'])==position and float(label.attrib['y'])==tick_y,'Emitted quantitative tick label differs from axis scale')
    if '05-independent' in d.stem:
        targets=[node for node in root.iter() if 'data-target-ms' in node.attrib]
        require(len(targets)==1,'Emitted target line inventory changed')
        a=targets[0].attrib;target=x0+plot*.2/maximum
        require(float(a['data-target-ms'])==.2 and float(a['data-axis-max-ms'])==maximum and float(a['x1'])==float(a['x2'])==target,'Emitted target line does not use the retained scale')
        require(float(a['y1'])<=expected[0][1]-16 and float(a['y2'])>=expected[-1][1]+expected[-1][2],'Target line must start above and cover both bars')


def segment_text_clearance(line,box):
    x1,y1,x2,y2,width=line
    dx,dy=x2-x1,y2-y1;enter,leave=0.,1.
    for p,q in [(-dx,x1-box[0]),(dx,box[2]-x1),(-dy,y1-box[1]),(dy,box[3]-y1)]:
        if p==0:
            if q<0:break
        elif p<0:enter=max(enter,q/p)
        else:leave=min(leave,q/p)
    else:
        if enter<=leave:return -width/2
    def corner_distance(x,y):
        length=dx*dx+dy*dy;t=max(0,min(1,((x-x1)*dx+(y-y1)*dy)/length)) if length else 0
        return math.hypot(x-x1-t*dx,y-y1-t*dy)
    def end_distance(x,y):return math.hypot(max(box[0]-x,0,x-box[2]),max(box[1]-y,0,y-box[3]))
    return min(end_distance(x1,y1),end_distance(x2,y2),*(corner_distance(x,y) for x in [box[0],box[2]] for y in [box[1],box[3]]))-width/2


def outline_edges(root,d):
    """Use actual emitted container strokes, including the decision diamond."""
    result=[]
    for node in root.iter():
        if node.attrib.get('fill')!=C['background'] or 'stroke' not in node.attrib:
            continue
        stroke=float(node.attrib.get('stroke-width','0'))
        if node.tag.endswith('rect'):
            x,y,w,h=(float(node.attrib[k]) for k in ['x','y','width','height'])
            points=[(x,y),(x+w,y),(x+w,y+h),(x,y+h)]
        elif node.tag.endswith('polygon'):
            points=[tuple(map(float,p.split(','))) for p in node.attrib['points'].split()]
            require(len(points)>=3,'Outline polygon needs three vertices')
        else:
            continue
        require(all(math.isfinite(v) for point in points for v in point),'Non-finite outline geometry')
        result.extend((*a,*b,stroke) for a,b in zip(points,points[1:]+points[:1]))
    return result


def embed_font(svg,prefix):
    reference=(prefix+'/Inter.ttf').encode()
    require(svg.count(reference)==1 and b'__RETAINED_INTER_FONT__' not in svg,'Expected one retained Inter reference before embedding')
    font=(INPUTS/'Inter.ttf').read_bytes()
    require(sha(font)==EXPECTED_FONT,'Retained Inter changed before embedding')
    result=svg.replace(reference,b'data:font/ttf;base64,'+base64.b64encode(font),1)
    require(reference not in result and result.count(b'data:font/ttf;base64,')==1,'Retained Inter substitution failed')
    return result


def validate(d,env,cache):
    boxes=[];minimum=64 if d.w==1600 else 48 if d.w==1080 else 52
    emitted=ET.fromstring(d.finish());text_nodes=[node for node in emitted.iter() if node.tag.endswith('text')]
    require(len(text_nodes)==len(d.labels),'Emitted label inventory differs')
    for label,node in zip(d.labels,text_nodes):
        require(node.text==label['text'] and [float(node.attrib[k]) for k in ['x','y','font-size','font-weight']]==[label['x'],label['y'],label['font_px'],label['weight']],'Emitted label geometry differs from declared layout')
        require(label['font_px']>=minimum,'Font too small')
        require(label['weight'] in [400,500,600,700],'Unexpected font weight')
        width,height,dx,dy=measure(label,env,cache)
        offset=0 if label['anchor']=='start' else width if label['anchor']=='end' else width/2
        x=label['x']-offset+dx;y=label['y']+dy;box=(x,y,x+width,y+height)
        require(x>=48 and x+width<=d.w-48 and y>=36 and y+height<=d.h-35,'Text outside safe margins: '+label['text']+' '+str(box))
        for old,oldlabel in boxes:
            require(not (min(box[2],old[2])>max(box[0],old[0])+1 and min(box[3],old[3])>max(box[1],old[1])+1),'Text overlaps: '+label['text']+' / '+oldlabel['text'])
        for old,oldlabel in boxes:
            if abs(label['y']-oldlabel['y'])<1e-6:
                gap=max(box[0]-old[2],old[0]-box[2])
                require(gap>=.75*max(label['font_px'],oldlabel['font_px']),'Same-row text gutter below0.75em: '+label['text']+' / '+oldlabel['text'])
        boxes.append((box,label))
    lines=[tuple(float(node.attrib[k]) for k in ['x1','y1','x2','y2','stroke-width']) for node in emitted.iter() if node.tag.endswith('line')]
    obstacles=[tuple(float(node.attrib[k]) for k in ['x','y','width','height']) for node in emitted.iter() if node.tag.endswith('rect') and node.attrib.get('fill') in [C['blue'],C['orange']]]
    outlines=outline_edges(emitted,d)
    for box,label in boxes:
        for line in lines:require(segment_text_clearance(line,box)>=16,'Text/line clearance below16px: '+d.stem+': '+label['text'])
        for edge in outlines:require(segment_text_clearance(edge,box)>=16,'Text/container padding or clearance below 16 px: '+d.stem+': '+label['text'])
        for x,y,width,height in obstacles:
            gap=math.hypot(max(x-box[2],box[0]-x-width,0),max(y-box[3],box[1]-y-height,0))-1.5
            require(gap>=16,'Text/bar clearance below16px: '+d.stem+': '+label['text'])
    if '02-operand' in d.stem:
        names={e['name'] for e in d.edges}
        require(names=={'HBM-to-SMEM','SMEM-to-HBM','SMEM-operands-to-MMA','MMA-writes-TMEM','MMA-reads-TMEM','TMEM-load-to-registers','register-store-to-TMEM'},'Operand route inventory changed')
    if 'business-capacity' in d.stem:
        require({e['name'] for e in d.edges}=={'fail-end','pass-end'},'Decision graph needs both outgoing directed paths')
        for fail in [e for e in d.edges if e['name'].startswith('fail')]:
            for passed in [e for e in d.edges if e['name'].startswith('pass')]:
                require(max(fail['x1'],fail['x2'])<min(passed['x1'],passed['x2']),'Decision paths cross')
    return {'minimum_font_px':minimum,'minimum_at_306_px':minimum*306/d.w,'text_boxes':[{**l,'glyph_box':list(b)} for b,l in boxes],'bars':d.bars,'directed_edges':d.edges,'all_text_inside_safe_margin':True,'no_text_text_overlap':True,'same_row_gutter_min_em':.75,'text_line_bar_clearance_min_px':16,'outline_container_clearance_min_px':16,'outline_edges_checked':len(outlines)}


def render(output,asset_context=None):
    validate_inputs();records={};pending={};cache={}
    with tempfile.TemporaryDirectory(prefix='attention-reference-') as tmp:
        env=font_env(Path(tmp));matched=run(['fc-match','-f','%{family}|%{file}','Inter'],env=env).decode()
        require(matched=='Inter|'+str(INPUTS/'Inter.ttf'),'Unexpected Inter resolution: '+matched)
        for d in all_diagrams():
            checks=validate(d,env,cache);svg=d.finish(asset_context or output);validate_route_svg(d,svg);prefix=asset_prefix(d.stem,asset_context or output)
            raster_svg=embed_font(svg,prefix)
            png=run(['rsvg-convert','--format','png'],input=raster_svg,env=env)
            png=run(['magick','png:-','-background',C['background'],'-alpha','off','-strip','-define','png:color-type=2','png:-'],input=png,env=env)
            require(struct.unpack('>II',png[16:24])==(d.w,d.h),'Wrong PNG size');require(png[25]==2,'PNG must be RGB')
            pending[d.stem+'.svg']=svg;pending[d.stem+'.png']=png;pending[d.stem+'.txt']=('\n'.join(t['text'] for t in d.labels)+'\n').encode()
            if d.w==1080:pending[d.stem.removesuffix('-square')+'-400.png']=run(['magick','png:-','-filter','Lanczos','-resize','400x400','-strip','png:-'],input=png,env=env)
            records[d.stem]={'width':d.w,'height':d.h,'svg_sha256':sha(svg),'png_sha256':sha(png),**checks}
    # All render/validation work finishes before any destination is replaced.
    for rel,data in pending.items():
        dest=output/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        fd,name=tempfile.mkstemp(prefix='.'+dest.name+'.',suffix='.tmp',dir=dest.parent)
        try:
            with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
            os.replace(name,dest)
        except BaseException:Path(name).unlink(missing_ok=True);raise
    return records,sorted(pending)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,default=ROOT);p.add_argument('--check',action='store_true');p.add_argument('--receipt',type=Path);a=p.parse_args()
    if a.receipt:a.receipt.unlink(missing_ok=True)
    atomic_json(a.receipt,{'status':'INCOMPLETE','stage':'started'})
    try:
        for tool in ['rsvg-convert','magick','fc-match']:require(shutil.which(tool),'Required existing binary unavailable: '+tool)
        if a.check:
            with tempfile.TemporaryDirectory(prefix='attention-reference-check-') as tmp:
                first,second=Path(tmp)/'first',Path(tmp)/'second';records,paths=render(first,a.output_dir);render(second,a.output_dir)
                for rel in sorted(paths,key=lambda s:(0 if s.endswith('.svg') else 1 if s.endswith('.txt') else 2,s)):
                    require((first/rel).read_bytes()==(second/rel).read_bytes(),'Non-deterministic output: '+rel)
                    require((a.output_dir/rel).is_file(),'Current output missing: '+rel)
                    require((a.output_dir/rel).read_bytes()==(first/rel).read_bytes(),'Current output drift: '+rel+(' (PNG-only drift may be rendering-stack dependent)' if rel.endswith('.png') else ''))
        else:records,paths=render(a.output_dir)
        receipt={'status':'PASS','mode':'read-only-check' if a.check else 'render','records':records,'outputs':paths,'inputs':{'results_sha256':EXPECTED_RESULTS,'font_sha256':EXPECTED_FONT},'dependencies':{tool:run([tool,'--version']).decode().strip() for tool in ['rsvg-convert','magick']},'limits':['Qualitative diagrams do not encode unknown byte quantities','Local preview only, native platform proof and current independent reviews remain required','No new measurement or analytical result']}
        atomic_json(a.receipt,receipt);print(json.dumps({'status':'PASS','diagrams':len(records),'outputs':len(paths)}))
    except BaseException as exc:
        atomic_json(a.receipt,{'status':'INTERRUPTED' if isinstance(exc,(KeyboardInterrupt,SystemExit)) or not isinstance(exc,Exception) else 'FAIL','exception_type':type(exc).__name__,'message':str(exc)})
        raise

if __name__=='__main__':main()
