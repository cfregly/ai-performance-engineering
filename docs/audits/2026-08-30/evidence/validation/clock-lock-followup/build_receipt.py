"""Assemble this bounded source/CPU receipt; never overwrite an existing receipt."""
import hashlib
import importlib.metadata
import json
import pathlib
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

import torch


EVIDENCE = pathlib.Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[5]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def junit(name):
    suite = next(ET.parse(EVIDENCE / name).iter("testsuite")).attrib
    result = {key: int(suite[key]) for key in ("tests", "failures", "errors", "skipped")}
    result["passed"] = result["tests"] - result["failures"] - result["errors"] - result["skipped"]
    result["seconds"] = float(suite["time"])
    return result


source_paths = ["code/tests/test_anti_cheat_protections.py"]
dependency_paths = [
    "code/core/harness/benchmark_harness.py", "code/core/benchmark/defaults.py",
    ".github/workflows/tier1-nightly.yml", "code/tests/test_anti_cheat_edge_cases.py",
    "code/tests/protection_test_utils.py", "code/core/harness/validity_checks.py",
    "code/core/benchmark/verify_runner.py",
]
base = "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/miniconda3/bin/python -m pytest "
five_suites = (
    "tests/test_anti_cheat_edge_cases.py tests/test_anti_cheat_protections.py "
    "tests/test_protection_effectiveness.py tests/test_verification.py "
    "tests/test_runtime_transparency_warnings.py -q -rs -p no:cacheprovider"
)
target = "tests/test_anti_cheat_protections.py::TestEnvironmentProtections::test_frequency_boost_clock_locking"
xml_arg = " --junitxml=../docs/audits/2026-08-30/evidence/validation/clock-lock-followup/"
attempts = [{
    "command": base + "tests/test_anti_cheat_protections.py::TestClockLockControlPlane " + target
        + " -q -rs -p no:cacheprovider" + xml_arg + "focused.xml",
    "exit_code": 0, "artifact": "focused.txt", "result": junit("focused.xml"),
    "scope": "Initial control set before additional typed-NVML/return-code/unrelated-tool controls; not final source",
}]
for stem in ("combined-final", "combined-final-v2", "combined-final-v3"):
    attempts.append({
        "command": base + five_suites + xml_arg + stem + ".xml", "exit_code": 0,
        "artifact": stem + ".txt", "result": junit(stem + ".xml"),
        "final_source_gate": stem == "combined-final-v3",
    })
for stem in ("attested-cpu-negative", "attested-cpu-negative-final"):
    attempts.append({
        "command": "TIER1_EXPECTED_GPU_NAME='NVIDIA B200' " + base + target
            + " -q -p no:cacheprovider" + xml_arg + stem + ".xml",
        "exit_code": 1, "artifact": stem + ".txt", "result": junit(stem + ".xml"),
        "expected_negative": True, "final_source_gate": stem.endswith("-final"),
        "scope": "Actual pytest invocation on CPU rejects unavailable required capability; not GPU qualification",
    })

receipt = {
    "scope": ["W1-040 residual failed-clock-lock-as-pass repair", "W1-040/W1-093 bounded disposition review"],
    "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    "prior_receipt": "../protection-coverage-receipt.json",
    "prior_receipt_sha256": sha(EVIDENCE.parent / "protection-coverage-receipt.json"),
    "prior_source_sha256": json.loads((EVIDENCE / "before.json").read_text())["source_sha256"],
    "source_status": "Source fixed; no production runtime/default changes; CUDA/NVML integration pending",
    "environment": {
        "python": sys.version.split()[0], "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pynvml_distributions": {name: importlib.metadata.version(name)
            for name in importlib.metadata.packages_distributions().get("pynvml", [])},
        "qualification": "Existing CPU environment, not the pinned GPU stack",
    },
    "source_files": {path: sha(ROOT / path) for path in source_paths},
    "read_only_dependency_hashes": {path: sha(ROOT / path) for path in dependency_paths},
    "reproduction": {
        "artifact": "before.json",
        "result": "Three failed-lock examples complete the exact original AST exception handler successfully",
        "scope": "Source/control-flow proof only; no simulated CUDA availability or successful lock",
    },
    "changes": [
        "Use real NVML maximum targets and existing harness lock context; inspect actual application clocks before/after real CUDA work",
        "Use existing production50MHz application-clock contract; no new timing/accuracy tolerance",
        "Typed permission/NVML capability errors, missing NVIDIA tools and documented nvidia-smi codes3/4/12/13 skip locally, never pass",
        "Existing TIER1_EXPECTED_GPU_NAME requires failure for unavailable clock capability; no new bypass flag",
        "Generic lock, driver, unrelated-tool and observed-clock mismatch errors propagate",
        "42 added CPU cases check exception dispositions and diagnostic comparisons; no fake successful GPU/runtime-validity result",
    ],
    "attempts": attempts,
    "static_validation": {
        "ruff": {"command": "/opt/miniconda3/bin/python -m ruff check code/tests/test_anti_cheat_protections.py --select E9,F63,F7,F82,B006,B023", "exit_code": 0, "artifact": "ruff-final-v3.txt"},
        "ast_parse": "PASS", "scoped_git_diff_check": "PASS",
    },
    "recommended_dispositions": {
        "W1-040": "awaiting_runtime after documentation reconciliation: source test-quality defects corrected, including final failed-lock-as-pass path. Real CUDA protection/observed-clock-lock cases still need attested runner evidence.",
        "W1-093": "source_fixed; verify original assertion-free/pass-only test cleanup after final-source CPU/CI integration and accurate documentation. Explicit skips correct misleading test passes but establish no protection coverage.",
        "unsupported_policies": "The61 unsupported/obsolete cases are explicit requirements/coverage limitations, not61 implied new production implementations. Replace blanket95-protection claims with actual scopes.",
        "required_production_guards": "This bounded read-only interface/default review did not reproduce a broken required production guard. It found documentation overclaims and the remaining test exception-handling defect repaired here.",
        "jitter_limit": "Prior Dynamo-count and post-perturbation jitter source fixes retain earlier evidence. Pre-perturbation unsupported/advisory exits remain unchanged; no universally fail-closed claim.",
    },
    "documentation_proposal": "documentation-proposal.md",
    "official_sources": [{
        "url": "https://docs.nvidia.com/deploy/nvidia-smi/index.html#return-value",
        "verified_on": "2026-08-30",
        "purpose": "Return codes3/4/12/13 identify unavailable operation, permission, missing NVML library/function. Generic or driver failures must not be swallowed.",
    }],
    "remaining_gates": [
        "Actual CUDA/NVML successful lock observation; no GPU lease or clock changes performed here",
        "35 real-CUDA skips remain runtime coverage gaps;61 unsupported/obsolete skips remain limitations",
        "Root final-source CPU/CI integration and README/generator/AGENTS factual coverage reconciliation",
    ],
    "limitations": [
        "No CUDA compilation/execution, GPU measurement or performance claim",
        "Attested CPU failure is expected capability rejection, not GPU qualification",
        "Optional typed-NVML CPU controls explicitly skip if the Python wrapper is absent; actual GPU test fails missing dependency in Tier-1",
        "No whole-repository immutable-state test during concurrent writers",
        "The prior frozen receipt describes its earlier source epoch and is unchanged",
        "Receipt assembly first failed because importable pynvml was supplied by pynvml11.5.0, not a distribution named nvidia-ml-py; failure retained separately",
    ],
}
receipt["artifacts"] = {
    path.name: sha(path) for path in sorted(EVIDENCE.iterdir())
    if path.is_file() and path.name != "receipt.json"
}
with (EVIDENCE / "receipt.json").open("x") as output:
    output.write(json.dumps(receipt, indent=2) + "\n")
print("receipt SHA256:", sha(EVIDENCE / "receipt.json"))
print("source SHA256:", receipt["source_files"])
print("final CPU:", junit("combined-final-v3.xml"))
