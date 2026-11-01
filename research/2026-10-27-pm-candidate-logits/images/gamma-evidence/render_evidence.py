#!/usr/bin/env python3
"""Rebuild the square evidence page from retained data, without a benchmark.

Needs existing Pillow, rsvg-convert and Fontconfig(fc-match). --check regenerates
in temporary storage and compares SVG/PNG/TXT. --overwrite replaces only this
new evidence-page family. Includes the current retained portrait stamp upright in the header margin.
"""
from pathlib import Path
import argparse
import hashlib
import html
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from PIL import Image, ImageFont

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1]
W=H=1200
X0,X1,AXIS_MAX=72,1128,6
TARGET_WIDTH=306
STAMP_SHA256='f456e86072e455924bb45aca4500b27cf02d4210297c55d24179097700a946ec'
STAMP_BOX=(952,12,1140,200)
SOURCE_PINS={
 'evidence/summary.json':'f695fc48f67c11b2d2e0b38c9de6ce385a5877b9e49c4703bfd01b5b9f49d380',
 'evidence/raw.jsonl':'eca4fcb7e280baebb2dbc223fb9f55584144e797dcae419957e66fd91f7e4a4f',
 'images/fonts/Inter.ttf':'29160a80ff49ddcab2c97711247e08b1fab27a484a329ce8b813d820dc559031',
}
PALETTE={'background':'#FBFBF8','ink':'#202326','header':'#172432','header_ink':'#FFFFFF','header_accent':'#C3DDF4','baseline':'#3478BA','selected':'#E87823','catch':'#EDF3F8','grid':'#BCC8CF'}


def require(value,message):
 if not value:raise ValueError(message)


def sha(data):return hashlib.sha256(data).hexdigest()


def contrast(a,b):
 def lum(c):
  v=[int(c[i:i+2],16)/255 for i in (1,3,5)]
  v=[x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4 for x in v]
  return sum(x*w for x,w in zip(v,(.2126,.7152,.0722)))
 l,h=sorted((lum(a),lum(b)));return (h+.05)/(l+.05)


def build(package):
 for name,digest in SOURCE_PINS.items():require(sha((package/name).read_bytes())==digest,'Retained input SHA changed: '+name)
 tokens=json.loads((HERE/'gamma-visual-tokens.json').read_text())
 require(tokens['typography']['font_family']=='Inter' and tokens['typography']['allow_handwritten'] is False,'Gamma Inter authority required')
 require(sha((HERE/'fregly-take-stamp.png').read_bytes())==STAMP_SHA256,'Original upright stamp bytes changed')
 font_path=package/'images/fonts/Inter.ttf'
 summary=json.loads((package/'evidence/summary.json').read_text())
 cell=next(c for c in summary['cells'] if c['size']==64 and c['mode']=='changing')
 base,selected,ratio=[cell[k] for k in ('baseline_median_wall_ms','candidate_median_wall_ms','speedup')]
 require((base,selected)==(5.399248795583844,0.4205747973173857) and round(ratio,2)==12.84,'Exact retained changing64 result required')
 require(not summary['model_measured'] and not summary['serving_measured'],'No model/serving measurements permitted')
 pairs=[json.loads(line) for line in (package/'evidence/raw.jsonl').read_text().splitlines() if json.loads(line).get('kind')=='pair']
 pairs=[p for p in pairs if p['size']==64 and p['mode']=='changing'];require(len(pairs)==3,'Three retained paired blocks required')
 parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="1200" viewBox="0 0 1200 1200">',
        '<title>Retained B200 output-head comparison</title>',
        '<desc>Zero-based median timing bars. Weight gathering is included. Allowed-set creation and transfer are excluded from both paths. Whole-model gains are unmeasured.</desc>']
 texts=[];boxes=[];marks=[]
 def rect(x,y,w,h,fill,attrs=''):
  parts.append(f'<rect x="{x}" y="{y}" width="{w:.13f}" height="{h}" fill="{fill}" {attrs}/>')
 def line(x,y,a,b,color=PALETTE['ink'],width=2,attrs=''):
  parts.append(f'<line x1="{x:.13f}" y1="{y}" x2="{a:.13f}" y2="{b}" stroke="{color}" stroke-width="{width}" {attrs}/>')
 def text(x,y,value,size=48,weight=400,color=PALETTE['ink'],anchor='start',background=PALETTE['background']):
  require(';' not in value and '—' not in value,'Plain labels only')
  require(size*TARGET_WIDTH/W>=tokens['typography']['minimum_phone_px']['material_condition'],'Type below actual306px minimum')
  require(weight in tokens['typography']['allowed_weights'],'Unsupported Inter weight')
  face=ImageFont.truetype(str(font_path),size);face.set_variation_by_axes([14,weight]);bb=face.getbbox(value,anchor='ls');width=face.getlength(value)
  shift=0 if anchor=='start' else width if anchor=='end' else width/2
  bounds=(x-shift+bb[0],y+bb[1],x-shift+bb[2],y+bb[3])
  require(bounds[0]>=50 and bounds[2]<=1150 and bounds[1]>=25 and bounds[3]<=1150,'Text outside safe canvas: '+value)
  for old,label in boxes:
   require(not(min(bounds[2],old[2])-max(bounds[0],old[0])>1 and min(bounds[3],old[3])-max(bounds[1],old[1])>1),'Text overlap: '+value+' / '+label)
  require(contrast(color,background)>=tokens['palette_policy']['ordinary_text_contrast_target'],'Text contrast below4.5: '+value)
  boxes.append((bounds,value));texts.append({'text':value,'x':x,'baseline':y,'size':size,'weight':weight,'bounds':bounds,'css_px_at306':size*TARGET_WIDTH/W})
  parts.append(f'<text x="{x}" y="{y}" font-family="Inter" font-size="{size}" font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{html.escape(value)}</text>')
 rect(0,0,W,H,PALETTE['background']);rect(0,0,W,218,PALETTE['header'])
 text(72,101,"Skip scores you'll discard",58,600,PALETTE['header_ink'],background=PALETTE['header'])
 text(72,177,f'{ratio:.2f}× faster · output head only',52,600,PALETTE['header_accent'],background=PALETTE['header'])
 parts.append('<image x="952" y="12" width="188" height="188" href="fregly-take-stamp.png" preserveAspectRatio="xMidYMid meet"/>')
 text(72,281,'Synthetic strict FP32 · B200',48,500)
 for field,raw_field,value,y,label,color in (
  ('baseline','baseline_wall_ms',base,377,'Full scoring + filter',PALETTE['baseline']),
  ('selected','candidate_wall_ms',selected,487,'Allowed rows first',PALETTE['selected'])):
  text(72,y-27,label,52,500);text(1128,y-27,f'{value:.3f} ms',52,600,anchor='end')
  width=(X1-X0)*value/AXIS_MAX
  rect(X0,y,width,38,color,f'data-science="median" data-series="{field}"')
  values=[p[raw_field] for p in pairs];lo,hi=min(values),max(values);a=X0+(X1-X0)*lo/AXIS_MAX;b=X0+(X1-X0)*hi/AXIS_MAX;yy=y+19
  line(a,yy,b,yy,attrs=f'data-science="range" data-series="{field}"');line(a,yy-10,a,yy+10);line(b,yy-10,b,yy+10)
  marks.append({'series':field,'value_ms':value,'bar_x':X0,'bar_y':y,'bar_width':width,'scale_px_per_ms':(X1-X0)/AXIS_MAX,'observed_min_ms':lo,'observed_max_ms':hi,'whisker_x':[a,b],'color':color})
 line(X0,550,X1,550)
 for v in (0,2,4,6):
  x=X0+(X1-X0)*v/AXIS_MAX;line(x,550,x,562)
  text(x,613,str(v),48,400,anchor='start' if v==0 else 'end' if v==6 else 'middle')
 text(72,672,'Median ms per warm 16-step block',48,500)
 text(72,730,'Whiskers: min–max of 3 paired blocks',48,400)
 rect(48,764,1104,380,PALETTE['catch'])
 for y,value,weight,size in (
  (820,'Allowed set ready BEFORE scoring',600,52),
  (880,'Weight gathering included in timing',500,48),
  (940,'64 shared allowed tokens · new set each step',400,48),
  (1000,'Creating/transferring the set excluded',400,48),
  (1060,'from BOTH paths',600,48),
  (1120,'Whole-model gains unmeasured',500,48)):
  text(72,y,value,size,weight,background=PALETTE['catch'])
 for bounds,value in boxes:
  require(not(min(bounds[2],STAMP_BOX[2])-max(bounds[0],STAMP_BOX[0])>0 and min(bounds[3],STAMP_BOX[3])-max(bounds[1],STAMP_BOX[1])>0),'Stamp overlaps text: '+value)
 parts.append('</svg>');svg='\n'.join(parts)+'\n';ET.fromstring(svg)
 metadata={'dimensions':[W,H],'target_embedded_width_css_px':TARGET_WIDTH,'minimum_text_css_px':min(t['css_px_at306'] for t in texts),'text':texts,'marks':marks,'axis':{'minimum_ms':0,'maximum_ms':AXIS_MAX,'start_x':X0,'end_x':X1,'linear':True},'retained_cell':cell,'palette':PALETTE,'contrast_target':4.5,'all_text_nonoverlap_and_in_safe_canvas':True,'no_new_measurement':True,'stamp':{'relative_href':'fregly-take-stamp.png','sha256':STAMP_SHA256,'box':STAMP_BOX,'rotation_degrees':0,'mirror':False,'skew_degrees':0,'crop':False,'raw_bytes_preserved':True,'no_text_or_data_overlap':True},'Gamma_tokens_sha256':sha((HERE/'gamma-visual-tokens.json').read_bytes()),'source_pins':SOURCE_PINS}
 return svg, '\n'.join(t['text'] for t in texts)+'\n',metadata


def render(package):
 svg,text,metadata=build(package)
 converter=shutil.which('rsvg-convert');matcher=shutil.which('fc-match')
 require(converter and matcher,'Existing rsvg-convert and fc-match are required. No installation is performed.')
 with tempfile.TemporaryDirectory(prefix='evidence-render-',dir=HERE) as folder:
  tmp=Path(folder);config=tmp/'fonts.conf';font=package/'images/fonts/Inter.ttf'
  config.write_text('<?xml version="1.0"?><!DOCTYPE fontconfig SYSTEM "urn:fontconfig:fonts.dtd"><fontconfig><dir>'+html.escape(str(font.parent))+'</dir><cachedir>'+html.escape(str(tmp/'cache'))+'</cachedir></fontconfig>')
  env=dict(os.environ,PANGOCAIRO_BACKEND='fc',FONTCONFIG_FILE=str(config))
  match=subprocess.check_output([matcher,'-f','%{file}\n%{family}\n%{style}\n','Inter:style=Regular'],env=env,text=True).splitlines()
  require(Path(match[0]).resolve()==font.resolve() and match[1]=='Inter' and match[2]=='Regular','Exact regular Inter file did not resolve')
  shutil.copyfile(HERE/'fregly-take-stamp.png',tmp/'fregly-take-stamp.png')
  src=tmp/'comparison.svg';src.write_text(svg);png=tmp/'comparison.png';subprocess.run([converter,str(src),'-o',str(png)],env=env,check=True)
  outputs={'comparison.svg':svg.encode(),'comparison.txt':text.encode(),'fregly-take-stamp.png':(HERE/'fregly-take-stamp.png').read_bytes()}
  with Image.open(png) as decoded:
   require(decoded.size==(W,H),'Exact square dimensions required')
   image=decoded.convert('RGB')
   canonical=io.BytesIO();image.save(canonical,format='PNG');outputs['comparison.png']=canonical.getvalue()
   metadata['PNG_color_mode']='RGB'
   for width in (306,400):
    buffer=io.BytesIO();image.resize((width,width),Image.Resampling.LANCZOS).save(buffer,format='PNG');outputs[f'comparison-{width}px.png']=buffer.getvalue()
  metadata['font_resolution']=match;metadata['PANGOCAIRO_BACKEND']='fc';metadata['renderer_version']=subprocess.check_output([converter,'--version'],text=True).strip();metadata['renderer_sha256']=sha(Path(__file__).read_bytes())
 for name,digest in SOURCE_PINS.items():require(sha((package/name).read_bytes())==digest,'Retained input changed during rendering: '+name)
 require(sha((HERE/'fregly-take-stamp.png').read_bytes())==STAMP_SHA256,'Stamp changed during rendering')
 return outputs,metadata


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--package-root',type=Path,default=PACKAGE);parser.add_argument('--output-root',type=Path,default=HERE)
 mode=parser.add_mutually_exclusive_group();mode.add_argument('--check',action='store_true');mode.add_argument('--overwrite',action='store_true');args=parser.parse_args()
 package=args.package_root.resolve();out=args.output_root.resolve();names=['comparison.svg','comparison.txt','comparison.png','comparison-306px.png','comparison-400px.png','render-receipt.json']
 require(args.check or args.overwrite or not any((out/name).exists() for name in names),'Outputs exist. Use --check or --overwrite, or a new --output-root.')
 outputs,metadata=render(package)
 if args.check:
  require(all((out/name).is_file() and (out/name).read_bytes()==outputs[name] for name in outputs),'Canonical evidence outputs differ')
  print(json.dumps({'state':'BYTE_EXACT_CHECK_PASS','written':False,'checked':list(outputs)}));return
 out.mkdir(parents=True,exist_ok=True)
 for name,content in outputs.items():(out/name).write_bytes(content)
 metadata['outputs_sha256']={name:sha(content) for name,content in outputs.items()};metadata['scope']='Export/geometry/type proof only. No model, art generation, new measurement, native upload, publication or quality verdict.'
 (out/'render-receipt.json').write_text(json.dumps(metadata,indent=2)+'\n');print(json.dumps({'state':'RENDERED_NEW_EVIDENCE_PAGE','outputs':metadata['outputs_sha256'],'minimum_text_css_px_at306':metadata['minimum_text_css_px']}))


if __name__=='__main__':main()
