"""Run the actual CUDA gates only with separately authorized hardware custody.

CPU execution exits 2/HOLD. CUDA success establishes bounded correctness only,
not benchmark speed, SASS attribution, a release, or a performance qualification.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[5]
TEST = 'tests/test_audit_wave1_attention_regressions.py'

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True,
                        help='Fresh directory; existing attempts cannot be overwritten')
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    import torch
    result = dict(captured_at=datetime.now(timezone.utc).isoformat(), python=sys.version,
        platform=platform.platform(), torch=torch.__version__, cuda_build=torch.version.cuda,
        cuda_available=torch.cuda.is_available(), nvcc=shutil.which('nvcc'),
        gpu_executed=False, status='HOLD', performance_qualified=False,
        head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip())
    manifest = json.loads((Path(__file__).parent / 'source_manifest.json').read_text())
    result['source_files'] = [dict(path=item['path'], sha256=hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest())
                              for item in manifest]
    mismatch = [item['path'] for item, recorded in zip(result['source_files'], manifest)
                if item['sha256'] != recorded['sha256']]
    if mismatch:
        result['reason'] = 'Source epoch differs from reviewed manifest; require a new reviewed receipt'
        result['mismatched_source'] = mismatch
    elif not torch.cuda.is_available() or not shutil.which('nvcc'):
        result['reason'] = 'Actual CUDA GPU and nvcc are required; skipped tests cannot qualify this package'
    else:
        result['gpu_name'] = torch.cuda.get_device_name()
        result['compute_capability'] = torch.cuda.get_device_capability()
        cmd = [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               TEST, '-k', 'test_real_', '--junitxml', str(output/'junit.xml'),
               '--basetemp', str(output/'pytest-tmp')]
        import os
        env = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        result['command'] = cmd
        result['gpu_executed'] = True
        with (output/'pytest.log').open('w') as log:
            completed = subprocess.run(cmd, cwd=ROOT/'code', env=env, stdout=log, stderr=subprocess.STDOUT)
        result['exit_code'] = completed.returncode
        suites = list(ET.parse(output/'junit.xml').getroot().iter('testsuite')) if (output/'junit.xml').exists() else []
        counts = {name:sum(int(suite.get(name, '0')) for suite in suites)
                  for name in ('tests','failures','errors','skipped')}
        result['junit_counts'] = counts
        accepted = completed.returncode == 0 and counts['tests'] >= 25 and not any(counts[n] for n in ('failures','errors','skipped'))
        result['status'] = 'PASS_BOUNDED_CUDA_CORRECTNESS' if accepted else 'FAIL'
        result['reason'] = 'No performance, profiler/SASS, full-size or release qualification is implied'
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('status','reason','gpu_executed','performance_qualified')}))
    return 0 if result['status'] == 'PASS_BOUNDED_CUDA_CORRECTNESS' else 2

if __name__ == '__main__':
    raise SystemExit(main())
