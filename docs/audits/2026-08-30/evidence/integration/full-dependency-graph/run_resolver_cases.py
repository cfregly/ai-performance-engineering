"""Preserved orchestration; never installs target packages or runs GPU code."""
from pathlib import Path
import concurrent.futures
import datetime
import json
import os
import signal
import subprocess
import time

OUT = Path(__file__).resolve().parent
WORK = Path(json.loads((OUT / 'venv-create.json').read_text())['task_root'])
CASES = {
    'gputil-binary-only-negative': ('GPUtil==1.4.0\n', []),
    'gputil-selective-build-positive': ('GPUtil==1.4.0\n', ['--no-binary', 'gputil']),
    'full-90-specs-gputil-reviewed-attempt-1': ((OUT / 'requirements-public.in').read_text(), ['--no-binary', 'gputil']),
}

def run(name):
    content, flags = CASES[name]
    case = OUT / name
    case.mkdir(exist_ok=False)
    req = case / 'requirements.in'
    req.write_text(content)
    argv = [
        '/Users/admin/.local/bin/uv', 'pip', 'compile', '--verbose',
        '--python', str(WORK / 'venv/bin/python'), '--python-version', '3.12',
        '--python-platform', 'x86_64-manylinux_2_31',
        '--only-binary', ':all:', *flags,
        '--build-constraints', str(OUT / 'build-constraints.txt'),
        '--no-python-downloads', '--no-config', '--keyring-provider', 'disabled',
        '--default-index', 'https://pypi.org/simple',
        '--index-strategy', 'unsafe-best-match',
        '--cache-dir', str(WORK / (name + '-cache')),
        '--no-progress', '--color', 'never', '--generate-hashes',
        '--emit-index-annotation', '--emit-index-url', '--emit-build-options',
        '--output-file', str(case / 'resolved-requirements.txt'), str(req),
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith(('UV_', 'PIP_'))}
    env['UV_HTTP_TIMEOUT'] = '30'
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    start = time.monotonic()
    process = subprocess.Popen(argv, cwd=WORK, env=env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=180)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
    code = 124 if timed_out else process.returncode
    (case / 'stdout.txt').write_text(stdout)
    (case / 'stderr.txt').write_text(stderr)
    (case / 'command.json').write_text(json.dumps({
        'argv': argv, 'cwd': str(WORK), 'started_utc': started,
        'elapsed_seconds': time.monotonic() - start, 'exit_code': code,
        'timeout_seconds': 180, 'timed_out': timed_out,
        'build_policy': 'Only GPUtil explicitly allowed from reviewed source; every other distribution must have a wheel. Build backend packages constrained and isolated. No target install.',
        'environment_policy': 'Inherited UV_/PIP_ settings removed; explicit official indexes; keyring disabled for public metadata.',
    }, indent=2) + '\n')
    return {'case': name, 'exit_code': code, 'stderr_tail': stderr[-4000:]}

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(run, CASES):
            print(json.dumps(result), flush=True)
