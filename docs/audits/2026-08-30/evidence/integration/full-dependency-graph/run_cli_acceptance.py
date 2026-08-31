"""Exercise real CLI parsing/registry operations without launching workloads."""
from pathlib import Path
import json
import os
import re
import subprocess
import time

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]
WORK = Path(json.loads((OUT / 'venv-create.json').read_text())['task_root'])
PYTHON = str(WORK / 'venv/bin/python')
CASES = [
    ('root-help', ['--help'], 0, 'distributed'),
    ('help-alias', ['help'], 0, 'benchmark'),
    ('bench-help', ['bench', '--help'], 0, 'list-targets'),
    ('nested-help', ['bench', 'list-targets', '--help'], 0, '--chapter'),
    ('ncu-help', ['profile', 'ncu', '--help'], 0, '--launch-count'),
    ('inference-help', ['inference', 'vllm', '--help'], 0, '--model-size'),
    ('registry', ['bench', 'list-targets', '--chapter', 'ch07'], 0, 'ch07:'),
    ('demos-registry', ['demos', 'list'], 0, 'ch11-stream-overlap'),
    ('missing-option-value', ['bench', 'list-targets', '--chapter'], 2, 'requires an argument'),
    ('integer-option-negative', ['profile', 'ncu', '--launch-count', 'not-an-int'], 2, 'not a valid integer'),
]
results = []
for name, args, expected_exit, expected_text in CASES:
    argv = [PYTHON, '-m', 'cli.aisp', *args]
    started = time.monotonic()
    process = subprocess.run(argv, cwd=ROOT / 'code', capture_output=True, text=True,
                             timeout=30, env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
    combined = process.stdout + process.stderr
    (OUT / f'cli-{name}.txt').write_text(combined)
    plain = re.sub(r'\x1b\[[0-9;]*m', '', combined)
    passed = process.returncode == expected_exit and expected_text in plain
    result = {'case': name, 'argv': argv, 'cwd': str(ROOT / 'code'),
              'exit_code': process.returncode, 'expected_exit': expected_exit,
              'expected_text': expected_text, 'status': 'PASS' if passed else 'FAIL',
              'elapsed_seconds': time.monotonic() - started,
              'output': f'cli-{name}.txt'}
    results.append(result)
    (OUT / f'cli-{name}-command.json').write_text(json.dumps(result, indent=2) + '\n')
    print(name, result['status'], flush=True)
(OUT / 'cli-acceptance-results.json').write_text(json.dumps(results, indent=2) + '\n')
raise SystemExit(0 if all(r['status'] == 'PASS' for r in results) else 1)
