from pathlib import Path
import json
import re
import hashlib
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[5]

def parse(text):
    result = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith(('#', '-')):
            continue
        requirement = Requirement(line.split('#', 1)[0].strip())
        result[canonicalize_name(requirement.name)] = requirement
    return result

def versions(path):
    return {canonicalize_name(name): version for name, version in re.findall(
        r'^([A-Za-z0-9_.-]+)(?:\[[^]]+\])?==([^\s]+)', path.read_text(), re.M
    )}

original = parse((OUT / 'original-requirements_latest.txt').read_text())
source = ROOT / 'code/requirements_latest.txt'
final = parse(source.read_text())
actual_input = parse((OUT / 'full-90-final-source-attempt-1/requirements.in').read_text())
assert len(original) == len(final) == len(actual_input) == 90
assert original.keys() == final.keys()
assert {k: str(v) for k, v in final.items()} == {k: str(v) for k, v in actual_input.items()}
changed = {k: {'old': str(original[k]), 'new': str(final[k])}
           for k in final if str(final[k]) != str(original[k])}
assert set(changed) == {'typer', 'typer-slim', 'rich'}
resolved = versions(OUT / 'full-90-final-source-attempt-1/resolved-requirements.txt')
assert len(resolved) == 327
for name, requirement in final.items():
    assert requirement.specifier.contains(resolved[name]), (name, requirement, resolved[name])
first = versions(OUT / 'cpu-first-pypi-final-source-attempt-1/resolved-requirements.txt')
union_path = OUT / 'cpu-exact-sources-final-source-attempt-1/resolved-requirements.txt'
union = versions(union_path)
assert len(first) == 49 and len(union) == 55
assert all(union[name] == version for name, version in first.items())
assert 'torch @ https://download-r2.pytorch.org/whl/cpu/' in union_path.read_text()
report = {
    'status': 'PASS_EXACT_FINAL_REQUIREMENTS_METADATA_CONTRACT',
    'direct_spec_count': 90, 'resolved_package_count': 327,
    'all_direct_specifiers_satisfied': True, 'changed_specs': changed,
    'unchanged_package_names_and_extras': all(original[k].extras == final[k].extras for k in final),
    'final_input_source_equivalence': 'Only missing local find-links omitted; every final package specifier is identical.',
    'cpu_first_pypi_packages': 49, 'cpu_constrained_union_packages': 56,
    'cpu_all_first_versions_retained': True,
    'source_requirements_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
}
(OUT / 'final-specifier-contract.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
