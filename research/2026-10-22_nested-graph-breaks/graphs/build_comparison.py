"""Render the acquired compiler graph fragments and the measured comparison."""

import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parent
GRAPHS = json.loads((ROOT / 'compiled-graphs.json').read_text(encoding='utf-8'))
COLORS = {'0': '#2f63bb', '1': '#167450'}
OPCOLORS = {'sin': '#e7efff', 'mul': '#e2f3ee', 'add': '#fff0da', 'cos': '#eee5fb'}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalize_svg(path):
    text = path.read_text(encoding='utf-8')
    normalized = re.sub(r'[ \t]+(?=\n|\Z)', '', text)
    if normalized != text:
        path.write_text(normalized, encoding='utf-8')


def verify_18_regions(graphs):
    expected_ids = {'0': list(range(16)), '1': [0, 1]}
    if set(graphs) != set(expected_ids):
        raise RuntimeError('compiled-graphs.json must contain False and True arms')

    root = ROOT.resolve()
    for flag, expected in expected_ids.items():
        records = graphs[flag]
        if [graph['aot_id'] for graph in records] != expected:
            raise RuntimeError(f'arm {flag} does not contain the expected AOT IDs')
        if sum(graph['operation_count'] for graph in records) != 64:
            raise RuntimeError(f'arm {flag} does not contain exactly 64 ATen operations')

        for graph in records:
            source = (ROOT / graph['source_path']).resolve()
            try:
                source.relative_to(root)
            except ValueError as error:
                raise RuntimeError(f'public source path escapes the graph root: {source}') from error
            if not source.is_file():
                raise RuntimeError(f'public source wrapper is missing: {source}')
            if sha256(source) != graph['source_sha256']:
                raise RuntimeError(f'public source wrapper digest differs: {source}')

            marker = '# Graph fragment:\n'
            source_text = source.read_text(encoding='utf-8')
            if marker not in source_text:
                raise RuntimeError(f'public source wrapper has no graph fragment: {source}')
            fragment = source_text.split(marker, 1)[1].rstrip()
            if fragment != graph['raw_graph_fragment']:
                raise RuntimeError(f'public source fragment differs from JSON: {source}')
            operation_nodes = [node for node in graph['nodes'] if node['kind'] == 'operation']
            if len(operation_nodes) != graph['operation_count']:
                raise RuntimeError(f'operation count differs from parsed nodes: {source}')


def render_region(graph):
    region = graph['aot_id']
    flag = str(int(graph['flag']))
    stem = f'region-flag{flag}-aot{region}'
    target = ROOT / 'graphs'
    target.mkdir(exist_ok=True)
    lines = [
        'digraph G {',
        'graph [rankdir=TB, bgcolor="transparent", pad="0.25", nodesep="0.17", ranksep="0.35", fontname="Arial", fontsize=17];',
        'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=12, color="#a9b5c4", margin="0.12,0.08"];',
        'edge [color="#607187", arrowsize=0.65];',
        'label=' + json.dumps(f'{"True" if graph["flag"] else "False"} · AOT {region} · {graph["operation_count"]} ATen operations', ensure_ascii=False) + ';',
    ]
    for node in graph['nodes']:
        if node['kind'] == 'input':
            label, fill = 'input\n%' + node['name'], '#f0f3f7'
        else:
            op = node['op']
            display = op
            if op == 'mul':
                const = re.search(r',\s*(0\.5|0\.25)\)', node['args'])
                display = 'multiply ' + const.group(1) if const else 'multiply'
            label, fill = display + '\n%' + node['name'], OPCOLORS[op]
        lines.append(f'{node["id"]} [label={json.dumps(label, ensure_ascii=False)}, fillcolor={json.dumps(fill)}, tooltip={json.dumps(node["raw"], ensure_ascii=False)}];')
    for edge in graph['edges']:
        lines.append(f'{edge["from"]} -> {edge["to"]};')
    ops = [node['id'] for node in graph['nodes'] if node['kind'] == 'operation']
    for offset in range(0, len(ops), 4):
        lines.append('{rank=same; ' + '; '.join(ops[offset:offset + 4]) + ';}')
    for index, identity in enumerate(graph['outputs']):
        lines.append(f'out{index} [label="return {index}", fillcolor="#f0f3f7"];')
        lines.append(f'{identity} -> out{index};')
    lines.append('}')
    dot = target / (stem + '.dot')
    dot.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    dot_executable = shutil.which('dot')
    if dot_executable is None:
        raise RuntimeError('Graphviz dot is required to render the region graphs')
    render_environment = os.environ.copy()
    render_environment['SOURCE_DATE_EPOCH'] = '1792566000'
    for fmt in ['svg', 'pdf']:
        subprocess.run(
            [dot_executable, '-T' + fmt, str(dot), '-o', str(target / (stem + '.' + fmt))],
            check=True,
            env=render_environment,
        )
    normalize_svg(target / (stem + '.svg'))
    return {
        'aot_id': region,
        'operation_count': graph['operation_count'],
        'svg_path': 'graphs/' + stem + '.svg',
        'pdf_path': 'graphs/' + stem + '.pdf',
        'dot_path': 'graphs/' + stem + '.dot',
        'svg_data': base64.b64encode((target / (stem + '.svg')).read_bytes()).decode(),
        'source_path': graph['source_path'],
        'source_sha256': graph['source_sha256'],
        'original_source_sha256': graph['original_source_sha256'],
        'fragment': graph['raw_graph_fragment'],
    }


def overview():
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 11,
        'svg.hashsalt': 'nested-graph-breaks-study',
    })
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.7), gridspec_kw={'wspace': .1})
    fig.patch.set_facecolor('white')
    fig.suptitle('Same 64 operations, fewer compiled regions', fontsize=23, fontweight='bold', y=.97, color='#15243a')
    fig.text(.5, .914, 'Actual generated ATen graph fragments · depth 8, seed 0 · PyTorch 2.14.1+cu130 on B200', ha='center', color='#526276', fontsize=12)
    for flag, ax in zip(['0', '1'], axes):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.text(.5, .985, ('False: 16 regions' if flag == '0' else 'True: 2 regions'), ha='center', va='top', fontsize=20, color=COLORS[flag], fontweight='bold')
        if flag == '0':
            for region in range(16):
                row, col = divmod(region, 4)
                x, y = .025 + col * .248, .69 - row * .195
                rect = FancyBboxPatch((x, y), .223, .145, boxstyle='round,pad=0.006,rounding_size=0.014', facecolor='#edf3ff', edgecolor='#96b3e4', linewidth=1.2)
                ax.add_patch(rect)
                ax.text(x + .1115, y + .103, f'AOT {region}', ha='center', color='#234d92', fontsize=12, fontweight='bold')
                ax.text(x + .1115, y + .055, '4 ATen ops', ha='center', color='#34475e', fontsize=11)
            ax.text(.5, .03, '16 replay launches + 16 copy kernels per call', ha='center', color='#34475e', fontsize=11)
        else:
            for region, y in [(0, .485), (1, .095)]:
                rect = FancyBboxPatch((.055, y), .89, .345, boxstyle='round,pad=0.006,rounding_size=0.018', facecolor='#e9f6ef', edgecolor='#81b69b', linewidth=1.4)
                ax.add_patch(rect)
                ax.text(.5, y + .267, f'AOT {region}: 32 ATen ops', ha='center', color='#125f40', fontsize=16, fontweight='bold')
                ax.text(.5, y + .183, 'sin → multiply → add residual → cos', ha='center', color='#34475e', fontsize=12)
                ax.text(.5, y + .105, '8 repeated blocks fused into one kernel', ha='center', color='#34475e', fontsize=12)
            ax.text(.5, .03, '2 replay launches + 2 copy kernels per call', ha='center', color='#34475e', fontsize=11)
    fig.text(.5, .048, 'Both arms keep the same explicit inner graph break. Region boxes summarize actual compiler output, ordered by AOT compile ID.', ha='center', fontsize=10, color='#526276')
    fig.text(.5, .021, 'This overview does not assert CUDA Graph dependency edges. Inspect the per-region operator DAGs and recorded replay timeline in the interactive view.', ha='center', fontsize=9.4, color='#526276')
    fig.subplots_adjust(top=.855, bottom=.12, left=.035, right=.965)
    fixed_date = datetime(2026, 10, 21, tzinfo=timezone.utc)
    metadata = {
        'svg': {'Date': fixed_date.isoformat()},
        'png': {'Title': 'Same 64 operations, fewer compiled regions'},
        'pdf': {
            'Title': 'Same 64 operations, fewer compiled regions',
            'Creator': 'build_comparison.py',
            'CreationDate': fixed_date,
            'ModDate': fixed_date,
        },
    }
    for fmt in ['svg', 'png', 'pdf']:
        fig.savefig(
            ROOT / ('comparison-overview.' + fmt),
            dpi=180,
            facecolor='white',
            metadata=metadata[fmt],
        )
    plt.close(fig)
    normalize_svg(ROOT / 'comparison-overview.svg')


def b64file(path):
    return base64.b64encode(path.read_bytes()).decode()


def build_html(regions):
    timeline_paths = [ROOT / f'replay-invocation-{i}.svg' for i in range(1, 6)]
    startup = ROOT / 'startup-decomposition.svg'
    missing = [path for path in timeline_paths + [startup] if not path.is_file()]
    if missing:
        raise RuntimeError('required retained visual is missing: ' + ', '.join(str(path) for path in missing))
    for path in timeline_paths + [startup]:
        normalize_svg(path)
    payload = {
        'arms': regions,
        'timelines': [b64file(path) for path in timeline_paths],
        'timeline_paths': [path.name for path in timeline_paths],
    }
    data = json.dumps(payload, ensure_ascii=False).replace('</', '<\\/')
    startup_markup = (
        '<img class="wide" src="data:image/svg+xml;base64,'
        + b64file(startup)
        + '" alt="First-call and calls two through ten decomposition">'
        + '<div class="downloads"><a href="startup-decomposition.svg">SVG</a>'
        + '<a href="startup-decomposition.png">PNG</a>'
        + '<a href="startup-decomposition-data.json">Data</a></div>'
    )
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>False / True: actual B200 compiler graphs</title>
<style>
*{box-sizing:border-box}body{margin:0;background:#f4f6fa;color:#17283d;font:16px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}main{max-width:1460px;margin:auto;padding:30px 32px 60px}h1{font-size:34px;line-height:1.15;margin:10px 0 12px}h2{font-size:23px;margin:0 0 10px}h3{margin:0;font-size:20px}p{margin:8px 0 14px}.sub,.note{color:#53677c}.tag{font-size:12px;font-weight:700;letter-spacing:.08em;color:#52667c;text-transform:uppercase}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:22px 0}.card,section{background:#fff;border:1px solid #d9e1ec;border-radius:14px}.card{padding:17px 19px}.value{font-size:27px;font-weight:750;line-height:1.3}.label{font-size:13px;color:#52667c;margin-top:6px}section{padding:25px;margin:18px 0}.wide{display:block;width:100%;height:auto}.columns{display:grid;grid-template-columns:1fr 1fr;gap:20px}.arm{border:1px solid #d9e1ec;border-radius:12px;overflow:hidden}.arm-head{padding:16px;background:#f8fafe}.false h3{color:#2f63bb}.true h3{color:#167450}select,button{font:inherit;border:1px solid #b4c3d5;border-radius:6px;background:#fff;padding:7px 10px}select{max-width:100%}.controls{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-top:10px}.scroll{height:520px;overflow:auto;padding:14px;background:white}.dag{display:block;width:100%;max-width:none;height:auto}.source{border-top:1px solid #d9e1ec;padding:12px 16px}.source summary{cursor:pointer;color:#405d80}pre{white-space:pre;overflow:auto;font-size:11px;background:#f4f7fb;padding:12px;max-height:260px}a{color:#225da5}.downloads{display:flex;gap:14px;font-size:14px;margin-top:10px}.note{font-size:14px}.scope{border-left:4px solid #8aa7cd;padding-left:13px}.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px;margin-bottom:12px}.dot{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;border:1px solid #b3c1d2}.footer{font-size:13px;color:#53677c}code{background:#edf1f6;padding:2px 4px;border-radius:4px}#trace-placeholder{padding:20px;color:#52667c}@media(max-width:850px){main{padding:18px 14px}h1{font-size:28px}.cards{grid-template-columns:1fr 1fr}.columns{grid-template-columns:1fr}section{padding:17px}.scroll{height:440px}}
</style><main>
<div class="tag">Public evidence projection · retained B200 run</div><h1>Actual compiled graphs: False versus True</h1>
<p class="sub"><code>nested_graph_breaks</code> · PyTorch 2.14.1+cu130 · Inductor reduce-overhead · depth 8, seed 0</p>
<p>Both programs perform the same 64 ATen operations on 1,048,576 FP32 elements. True combines the work into two larger compiled regions.</p>
<div class="cards"><div class="card"><div class="value">16 → 2</div><div class="label">Compiled regions</div></div><div class="card"><div class="value">4 → 32</div><div class="label">ATen operations per region</div></div><div class="card"><div class="value">16 → 2</div><div class="label">Recorded graph replays per call</div></div><div class="card"><div class="value">407.57 → 89.94 µs</div><div class="label">Steady median, four-seed geometric mean</div></div></div>
<section><h2>Same work, fewer region boundaries</h2><img class="wide" src="data:image/svg+xml;base64,__OVERVIEW__" alt="False has sixteen four-operation regions, True has two thirty-two-operation regions"><div class="downloads"><a href="comparison-overview.svg">SVG</a><a href="comparison-overview.pdf">PDF</a><a href="comparison-overview.png">PNG</a></div></section>
<section><h2>Inspect the actual operator DAGs</h2><p class="note">These nodes and edges come from graph-fragment comments in retained generated Inductor wrappers. Select a region, zoom, or inspect its exact emitted fragment. AOT IDs are local compile IDs, not CUDA graph IDs.</p>
<div class="legend"><span><i class="dot" style="background:#e7efff"></i>sin</span><span><i class="dot" style="background:#e2f3ee"></i>multiply</span><span><i class="dot" style="background:#fff0da"></i>add residual</span><span><i class="dot" style="background:#eee5fb"></i>cos</span></div>
<div class="columns">__ARMS__</div></section>
<section><h2>Compare the recorded CUDA replays</h2><p class="note">The trace retains 16 distinct replay graph IDs for False and 2 for True. Each replay has one observed Triton kernel node. Separate copy kernels have graph ID 0. Full CUDA Graph definitions and dependency edges were not recorded.</p><div class="controls"><label for="invocation">Compiled invocation</label><select id="invocation"><option value="0">1 of 5</option><option value="1">2 of 5</option><option value="2">3 of 5</option><option value="3">4 of 5</option><option value="4">5 of 5</option></select></div><div id="trace-placeholder"></div><img id="timeline" class="wide" alt="Recorded graph launches and GPU kernels"><div class="downloads"><a id="timeline-download" download>Selected timeline SVG</a></div><p class="note scope">Profiler durations are descriptive evidence. They are not the acceptance latency measurements. All five recorded invocations are available.</p></section>
<section><h2>Where startup time went</h2>__STARTUP__<p>The extra time is in the first compiled call: True adds 786–876 ms across the four seeds. Calls 2–10 are 18.6–19.3 ms faster. The no-break seed-1 excursion is also in call 1, adding 117.6 ms while calls 2–10 are effectively unchanged. It remains a failed startup control.</p><p class="note">The leading explanation is the cost of compiling two much larger fused programs. Recovered LLVM IR and PTX are about five times larger. Exact runtime source places CUDA Graph recording on call 2, which is faster with True. The retained run lacks compile-phase timings, so the precise split among compiler stages, loading, and initial warmup remains unmeasured. The isolated no-break cold-call excursion also has no proven phase-level cause.</p></section>
<section><h2>Evidence and scope</h2><p class="note">The graphs were recovered read-only from retained compiler-cache wrappers after the run. No workload was rerun to build this comparison.</p><p class="note scope">This public projection excludes internal hostnames, trace and profile identifiers, GPU UUIDs, PCI addresses, absolute owner paths, and thread IDs. The provenance file records original and public SHA-256 values. All numeric measurements and ATen graph content are unchanged.</p><ul class="note"><li><a href="compiled-graphs.json">All 18 extracted ATen graphs and source hashes</a></li><li><a href="trace-extraction.json">Five replay traces per arm, correlations, and graph and node IDs</a></li><li><a href="public-projection.json">Redactions, original hashes, and public hashes</a></li><li><a href="region-summary.svg">Portable region summary</a></li><li><a href="https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.reducing_compile_time.html">PyTorch: cold compilation and reuse</a></li></ul><p class="footer">Graph structure shown: depth 8, seed 0. Steady timing shown: all four paired seeds. This is a compiler microbenchmark, not serving throughput or a model-quality result. The diagrams do not assert dependencies between CUDA Graph nodes that were absent from the trace.</p></section>
</main><script>const DATA=__DATA__;
function renderArm(flag){const rows=DATA.arms[flag],select=document.getElementById('region'+flag),r=rows[Number(select.value)];document.getElementById('dag'+flag).src='data:image/svg+xml;base64,'+r.svg_data;document.getElementById('source'+flag).textContent=r.fragment;document.getElementById('identity'+flag).textContent='Public wrapper SHA256 '+r.source_sha256+' · Original wrapper SHA256 '+r.original_source_sha256;document.getElementById('wrapper'+flag).href=r.source_path;for(const ext of ['svg','pdf','dot'])document.getElementById(ext+flag).href=r[ext+'_path'];document.getElementById('dag'+flag).style.width=document.getElementById('zoom'+flag).value+'%';}
for(const flag of ['0','1']){const s=document.getElementById('region'+flag);DATA.arms[flag].forEach((r,i)=>{const o=document.createElement('option');o.value=i;o.textContent='AOT '+r.aot_id+' · '+r.operation_count+' operations';s.appendChild(o)});s.onchange=()=>renderArm(flag);document.getElementById('zoom'+flag).oninput=()=>renderArm(flag);renderArm(flag);}
function renderTimeline(){const img=document.getElementById('timeline'),n=Number(document.getElementById('invocation').value);img.src='data:image/svg+xml;base64,'+DATA.timelines[n];document.getElementById('timeline-download').href=DATA.timeline_paths[n];}document.getElementById('invocation').onchange=renderTimeline;renderTimeline();
</script></html>'''
    arm_markup = []
    for flag, title in [('0', 'False · 16 regions'), ('1', 'True · 2 regions')]:
        arm_markup.append(f'''<div class="arm {'false' if flag == '0' else 'true'}"><div class="arm-head"><h3>{title}</h3><div class="controls"><select id="region{flag}" aria-label="Select region for {title}"></select><label>Zoom <input id="zoom{flag}" type="range" min="60" max="180" value="100" step="10"></label></div></div><div class="scroll"><img id="dag{flag}" class="dag" alt="Actual emitted ATen operator graph for {title}"></div><div class="source"><details><summary>Exact emitted graph and source identity</summary><pre id="source{flag}"></pre><pre id="identity{flag}"></pre></details><div class="downloads"><a id="wrapper{flag}" download>Public wrapper</a><a id="svg{flag}" download>SVG</a><a id="pdf{flag}" download>PDF</a><a id="dot{flag}" download>Graphviz DOT</a></div></div></div>''')
    page = page.replace('__ARMS__', ''.join(arm_markup)).replace('__DATA__', data).replace('__OVERVIEW__', b64file(ROOT / 'comparison-overview.svg')).replace('__STARTUP__', startup_markup)
    (ROOT / 'comparison.html').write_text(page, encoding='utf-8')


if __name__ == '__main__':
    verify_18_regions(GRAPHS['arms'])
    projection = ROOT / 'public-projection.json'
    if not projection.is_file():
        raise RuntimeError(f'public projection manifest is missing: {projection}')
    normalize_svg(ROOT / 'region-summary.svg')
    regions = {flag: [render_region(graph) for graph in graphs] for flag, graphs in GRAPHS['arms'].items()}
    overview()
    build_html(regions)
    manifest = {
        'schema': 'aiperf.nested_graph_breaks.visualization.v2',
        'graph_counts': {flag: len(values) for flag, values in regions.items()},
        'operation_counts': {
            flag: sum(graph['operation_count'] for graph in values)
            for flag, values in regions.items()
        },
        'source_wrapper_count': sum(len(values) for values in regions.values()),
        'public_data_sha256': {
            'compiled-graphs.json': sha256(ROOT / 'compiled-graphs.json'),
            'trace-extraction.json': sha256(ROOT / 'trace-extraction.json'),
            'startup-decomposition-data.json': sha256(ROOT / 'startup-decomposition-data.json'),
            'public-projection.json': sha256(projection),
        },
        'rendered_asset_sha256': {
            'comparison.html': sha256(ROOT / 'comparison.html'),
            'comparison-overview.svg': sha256(ROOT / 'comparison-overview.svg'),
            'comparison-overview.png': sha256(ROOT / 'comparison-overview.png'),
            'startup-decomposition.svg': sha256(ROOT / 'startup-decomposition.svg'),
            'startup-decomposition.png': sha256(ROOT / 'startup-decomposition.png'),
            'region-summary.svg': sha256(ROOT / 'region-summary.svg'),
            'region-summary.png': sha256(ROOT / 'region-summary.png'),
            **{
                path.name: sha256(path)
                for path in sorted(ROOT.glob('replay-invocation-*.svg'))
            },
        },
        'no_workload_rerun': True,
    }
    (ROOT / 'visualization-manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    print(ROOT / 'comparison.html')
