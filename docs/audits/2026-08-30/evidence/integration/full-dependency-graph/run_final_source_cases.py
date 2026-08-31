"""Resolve final source-derived inputs, retaining every declared package."""
from pathlib import Path
import importlib.util
import json
import re

import yaml

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
module_spec = importlib.util.spec_from_file_location('resolver_cases', OUT / 'run_resolver_cases.py')
resolver = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(resolver)
source = (ROOT / 'code/requirements_latest.txt').read_text()
assert source.count('-f ./third_party/wheels') == 1
full = source.replace('-f ./third_party/wheels', '# Diagnostic only: absent local find-links omitted; every package requirement retained.')
workflow = yaml.load((ROOT / '.github/workflows/benchmark-validation.yml').read_text(), Loader=yaml.BaseLoader)
command = next(s['run'] for s in workflow['jobs']['validate']['steps'] if s['name'] == 'Install CPU test dependencies')
pins = [part for part in command.split() if '==' in part]
assert len(pins) == 20
non_torch = '\n'.join(pin for pin in pins if not pin.startswith('torch==')) + '\n'
name = 'full-90-final-source-attempt-1'
resolver.CASES[name] = (full, ['--no-binary', 'gputil'])
print(json.dumps(resolver.run(name)), flush=True)
name = 'cpu-first-pypi-final-source-attempt-1'
resolver.CASES[name] = (non_torch, [])
result = resolver.run(name)
print(json.dumps(result), flush=True)
if result['exit_code'] != 0:
    raise SystemExit(result['exit_code'])
constraint = OUT / name / 'resolved-requirements.txt'
link = json.loads((OUT.parent / 'pinned-linux-readiness/torch-cpu-wheel-link.json').read_text())['href']
name = 'cpu-exact-sources-final-source-attempt-1'
resolver.CASES[name] = (non_torch + 'torch @ ' + link + '\n', ['--constraint', str(constraint)])
print(json.dumps(resolver.run(name)), flush=True)
