"""Real wire/parser/CLI checks; GPU series below are declared input fixtures."""
import ast
from http.server import HTTPServer
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from urllib.request import urlopen

import pytest
import torch

from monitoring import prometheus_exporter as exporter
from core.whatif import get_scenarios

CODE_ROOT = Path(__file__).resolve().parents[1]


def gpu_series(count):
    return {f'{name}{{gpu="{gpu}"}}': value + gpu
            for gpu in range(count)
            for name, value in [('gpu_memory_allocated_gb', 2.0),
                                ('gpu_memory_reserved_gb', 3.0),
                                ('gpu_memory_total_gb', 80.0),
                                ('gpu_sm_count', 100.0)]}


def strict_parse(text):
    binary = shutil.which('promtool')
    if not binary:
        pytest.skip('strict Prometheus wire parser requires promtool')
    result = subprocess.run([binary, 'check', 'metrics', '--extended', '--lint=none'],
                            input=text, text=True, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr + result.stdout


@pytest.mark.parametrize('count', [0, 1, 2, 8])
def test_prometheus_families_parse_once_with_all_gpu_series(count):
    text = exporter.format_prometheus_metrics({**gpu_series(count), 'training_loss': 0.125,
                                               'inference_ttft_ms': 4.5})
    strict_parse(text)
    parser = pytest.importorskip('prometheus_client.parser')
    families = list(parser.text_string_to_metric_families(text))
    assert sum(len(f.samples) for f in families) == count * 4 + 2
    for name in ('gpu_memory_allocated_gb', 'gpu_memory_reserved_gb',
                 'gpu_memory_total_gb', 'gpu_sm_count'):
        assert text.count(f'# HELP {name} ') == (1 if count else 0)
        assert text.count(f'# TYPE {name} gauge\n') == (1 if count else 0)
        if count:
            samples = next(f for f in families if f.name == name).samples
            assert {s.labels['gpu'] for s in samples} == {str(i) for i in range(count)}


def test_prometheus_http_scrape_is_complete_and_parseable():
    class FixtureCollector:
        # Input fixture only: do not claim CUDA telemetry collection ran.
        def format_prometheus(self):
            return exporter.format_prometheus_metrics(gpu_series(8))

    class Handler(exporter.PrometheusHandler):
        collector = FixtureCollector()

    server = HTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with urlopen(f'http://127.0.0.1:{server.server_port}/metrics', timeout=5) as response:
            assert response.status == 200
            assert 'version=0.0.4' in response.headers['Content-Type']
            text = response.read().decode()
        strict_parse(text)
        assert len([line for line in text.splitlines() if not line.startswith('#')]) == 32
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def cli(*args):
    return subprocess.run([sys.executable, '-m', 'cli.aisp', *map(str, args)],
                          cwd=CODE_ROOT, text=True, capture_output=True, timeout=30,
                          env={**os.environ, 'NO_COLOR': '1'})


@pytest.fixture
def trace(tmp_path):
    path = tmp_path/'trace.json'
    path.write_text(json.dumps({'traceEvents':[
        {'ph':'X','cat':'cpu_op','name':'fixture_mm','ts':0,'dur':20},
        {'ph':'X','cat':'kernel','name':'fixture_gemm','ts':10,'dur':40},
        {'ph':'X','cat':'kernel','name':'fixture_gemm','ts':60,'dur':60},
    ]}))
    return path


@pytest.mark.parametrize('suffix', ['json', 'html'])
def test_real_flame_cli_creates_requested_artifact(trace, tmp_path, suffix):
    output = tmp_path/f'flame.{suffix}'
    result = cli('profile','flame',trace,'--output',output)
    assert result.returncode == 0, result.stderr + result.stdout
    assert output.is_file() and output.stat().st_size > 0
    if suffix == 'json':
        data = json.loads(output.read_text())
        assert data['value'] == 120
        kernel = next(c for c in data['children'] if c['name'] == 'kernel')
        assert kernel['children'] == [{'name':'fixture_gemm','value':100.0,'children':[]}]
    else:
        html = output.read_text()
        assert '<svg' in html and 'fixture_gemm' in html
        assert '<script src=' not in html  # offline artifact, no external scripts


def test_flame_html_escapes_trace_names(tmp_path):
    trace = tmp_path/'unsafe.json'
    trace.write_text(json.dumps([{'ph':'X','name':'</script><img src=x onerror=alert(1)>','dur':25}]))
    output = tmp_path/'safe.html'
    result = cli('profile','flame',trace,'--output',output)
    assert result.returncode == 0, result.stderr
    assert '<img ' not in output.read_text()
    assert '&lt;img ' in output.read_text()


@pytest.mark.parametrize('payload', ['{bad json', '{"traceEvents":[]}', '{"traceEvents":[{"ph":"X","name":"bad","dur":NaN}]}',
    '{"traceEvents":[{"ph":"X","name":"valid","dur":10},{"ph":"X","name":"negative","dur":-1}]}'])
def test_flame_rejects_invalid_or_empty_trace_without_success_artifact(tmp_path, payload):
    trace, output = tmp_path/'bad.json', tmp_path/'flame.html'
    trace.write_text(payload)
    result = cli('profile','flame',trace,'--output',output)
    assert result.returncode != 0
    assert not output.exists()


@pytest.mark.parametrize('command', ['flame', 'memory', 'kernels'])
def test_missing_or_unimplemented_profile_command_exits_nonzero(command, tmp_path):
    result = cli('profile',command,tmp_path/'missing.json')
    assert result.returncode != 0


def test_fp8_scenario_is_versioned_integration_guidance():
    scenario = next(s for s in get_scenarios()['scenarios'] if s['id'] == 'fp8')
    assert scenario['code_example_kind'] == 'integration_template'
    assert scenario['code_example_executable'] is False
    assert '2.18' in scenario['code_example_api_version']
    assert any('Transformer Engine' in s for s in scenario['requirements'])
    tree = ast.parse(scenario['code_example'])
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert any(isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
               and node.func.value.id == 'te' and node.func.attr == 'autocast' for node in calls)
    assert not any(isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name)
                   and node.func.value.id == 'torch' and node.func.attr == 'autocast' for node in calls)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FP8 template runtime requires real CUDA and TE2.18')
def test_fp8_template_executes_real_transformer_engine_linear():
    from importlib.metadata import version
    te = pytest.importorskip('transformer_engine.pytorch')
    if not version('transformer_engine').startswith('2.18.'):
        pytest.skip('example contract is Transformer Engine2.18.x')
    if torch.cuda.get_device_capability()[0] < 9:
        pytest.skip('FP8 example qualification requires Hopper or newer')
    model = te.Linear(256, 256, params_dtype=torch.bfloat16, device='cuda').eval()
    inputs = torch.randn(32, 256, dtype=torch.bfloat16, device='cuda')
    scenario = next(s for s in get_scenarios()['scenarios'] if s['id'] == 'fp8')
    context = {'model':model,'inputs':inputs}
    with torch.no_grad():
        exec(scenario['code_example'], context)
        reference = model(inputs)
    actual = context['output']
    assert actual.shape == reference.shape == (32, 256)
    assert torch.isfinite(actual).all()
    # Quantization quality must be assessed; this is a coarse numerical smoke gate.
    assert torch.linalg.vector_norm(actual.float()-reference.float()) / torch.linalg.vector_norm(reference.float()) < 0.15
