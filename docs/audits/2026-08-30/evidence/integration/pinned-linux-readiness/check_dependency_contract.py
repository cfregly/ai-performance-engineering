"""Check explicit published dependency contracts; not a Linux resolver run."""
from datetime import datetime, timezone
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import re

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

HERE = Path(__file__).resolve().parent
ROOT = next(parent for parent in HERE.parents if (parent / '.git').exists())
sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
metadata_path = HERE / 'torch-cuda-wheel.metadata.txt'
metadata = Parser().parsestr(metadata_path.read_text())
assert metadata['Name'] == 'torch' and metadata['Version'] == '2.9.1+cu130'
triton_requirement = next(Requirement(raw) for raw in metadata.get_all('Requires-Dist')
                          if Requirement(raw).name == 'triton')
original_report = json.loads((HERE / 'preserved-torch-wheel-metadata.json').read_text())
assert str(triton_requirement) in [str(Requirement(raw)) for raw in
                                  original_report['torch']['metadata']['requires_dist']]
requirements = (ROOT / 'code/requirements_latest.txt').read_text()
pins = {}
for line in requirements.splitlines():
    line = line.split('#', 1)[0].strip()
    if not line or line.startswith('-'):
        continue
    req = Requirement(line)
    if len(list(req.specifier)) == 1 and next(iter(req.specifier)).operator == '==':
        pins[req.name] = next(iter(req.specifier)).version
assert pins['torch'] == metadata['Version']
linux = {**default_environment(), 'platform_system': 'Linux', 'sys_platform': 'linux',
         'platform_machine': 'x86_64', 'python_version': '3.12', 'python_full_version': '3.12.0'}
assert triton_requirement.marker.evaluate(linux)
assert not triton_requirement.specifier.contains('3.5.0')
assert triton_requirement.specifier.contains(pins['triton'])
workflow = (ROOT / '.github/workflows/benchmark-validation.yml').read_text()
install = workflow.split('- name: Install CPU test dependencies', 1)[1].split('- name: Run core contract tests', 1)[0]
ci_specs = re.findall(r'[A-Za-z0-9][A-Za-z0-9_-]*(?:\[[A-Za-z0-9_,.-]+\])?==[A-Za-z0-9.+_-]+', install)
ci_pins = {Requirement(raw).name: next(iter(Requirement(raw).specifier)).version for raw in ci_specs}
assert ci_pins['triton'] == pins['triton'] == '3.5.1'
assert ci_pins['requests'] == pins['requests'] == '2.34.2'
requests = json.loads((HERE / 'requests-2.34.2-pypi.json').read_text())
assert requests['info']['version'] == pins['requests']
assert SpecifierSet(requests['info']['requires_python']).contains('3.12')
assert any(item['filename'] == 'requests-2.34.2-py3-none-any.whl' for item in requests['urls'])
triton = json.loads((HERE / 'triton-3.5.1-pypi.json').read_text())
assert SpecifierSet(triton['info']['requires_python']).contains('3.12')
wheel_tags = {}
for architecture in ('x86_64', 'aarch64'):
    wheels = [item for item in triton['urls'] if item['packagetype'] == 'bdist_wheel'
              and '-cp312-cp312-' in item['filename'] and architecture in item['filename']]
    assert wheels
    wheel_tags[architecture] = [{'filename': item['filename'], 'url': item['url'],
                                'sha256': item['digests']['sha256']} for item in wheels]
result = {
    'captured_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'PASS_EXPLICIT_METADATA_CONTRACT__REAL_LINUX_RESOLVER_HOLD',
    'official_torch_wheel_metadata_sha256': sha(metadata_path),
    'torch_pin': pins['torch'], 'requires_dist': str(triton_requirement),
    'explicit_target_marker_context': {key: linux[key] for key in ('platform_system', 'sys_platform', 'platform_machine', 'python_version', 'python_full_version')},
    'original_triton_3_5_0_satisfies_required_specifier': False,
    'current_triton_pin': pins['triton'], 'current_pin_satisfies_required_specifier': True,
    'fresh_metadata_requirement_matches_preserved_original_resolver_metadata': True,
    'published_cp312_triton_wheels': wheel_tags,
    'requests_pin': pins['requests'], 'requests_python_3_12_supported': True,
    'cpu_workflow_pins_match_shared_requests_and_triton': True,
    'cpu_workflow_package_specs': ci_specs,
    'source_sha256': {str(path.relative_to(ROOT)): sha(path) for path in
                      (ROOT/'code/requirements_latest.txt', ROOT/'code/setup.sh', ROOT/'.github/workflows/benchmark-validation.yml')},
    'limits': ['No installed CUDA stack, native ABI import or GPU execution.',
               'Explicit PEP508 marker/specifier evaluation is not a full dependency resolution or Linux environment.',
               'Darwin pip --platform changes compatibility tags but does not establish Linux dependency-marker execution.',
               'ARM still has no official exact torchao CUDA wheel; existing binary-bootstrap rejection remains.',
               'Full supported Linux requirements resolution, installation and CI tests remain pending.'],
}
output = HERE / 'dependency-contract.json'
if output.exists():
    raise RuntimeError('Refusing to overwrite a prior dependency-contract attempt')
output.write_text(json.dumps(result, indent=2)+'\n')
(HERE/'cpu-workflow-package-pins.txt').write_text('\n'.join(ci_specs)+'\n')
print(json.dumps({key:result[key] for key in ('status','torch_pin','requires_dist','original_triton_3_5_0_satisfies_required_specifier','current_triton_pin','current_pin_satisfies_required_specifier','requests_pin')},indent=2))
