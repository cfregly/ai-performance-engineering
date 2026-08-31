"""Freeze this LOCAL-019 receipt; preserve all earlier attempts without overwrite."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def result(name):
    suite = ET.parse(HERE / f"{name}.xml").getroot().find("testsuite")
    counts = {key: int(suite.attrib[key]) for key in ("tests", "failures", "errors", "skipped")}
    counts["passed"] = counts["tests"] - counts["failures"] - counts["errors"] - counts["skipped"]
    counts["seconds"] = float(suite.attrib["time"])
    counts["skips"] = [
        {"test": case.attrib["classname"] + "::" + case.attrib["name"], "reason": case.find("skipped").attrib["message"]}
        for case in suite.findall("testcase") if case.find("skipped") is not None
    ]
    return counts


def main():
    expected = {
        "code/labs/train_distributed/training_utils/torchrun_harness.py": "faf2903b81f3825422764a9819031375a287bbec28f1327899dcabd3bfd548b0",
        "code/core/harness/benchmark_harness.py": "97aafb1fe60e84d18f16a25882aa4cb49e715e70b3217baf3a700881311128e9",
        "code/tests/test_audit_wave1_torchrun_verification.py": "5512a4fd6d6252ee1dac2e3c5d86e42f45abdcba9b8d6a94d64c80a9867c6de8",
    }
    actual = {path: digest(ROOT / path) for path in expected}
    assert actual == expected, "Frozen LOCAL-019 source changed before receipt assembly"
    before = json.loads((HERE / "before-source.json").read_text())
    cli = json.loads((HERE / "direct-cli-help.json").read_text())
    assert all(item["returncode"] == 0 for item in cli)
    test_prefix = "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/miniconda3/bin/python -m pytest "
    focused = "tests/test_audit_wave1_torchrun_verification.py"
    combined = focused + " tests/test_audit_wave1_harness.py tests/test_seed_config_immutability.py"
    arguments = " -q -rs -p no:cacheprovider"
    definitions = [
        ("before-tests", focused, 1, "Original production sources; initial test revision. 26 contract failures plus one real c10d rendezvous timeout, not a production diagnosis."),
        ("after-tests", focused, 1, "Repaired production sources; sole failing new fixture assumed spawn errors returned a result. Fixed expectation, not production."),
        ("integration", combined, 0, "Before replacing temporary local launcher skip with the actual static-loopback test."),
        ("static-loopback", focused + "::test_explicit_none_spec_retains_real_cpu_module_launch", 0, "Real torchrun to real CPU child; supported static loopback, no fake launch success."),
        ("final-integration", combined + " tests/test_audit_wave1_timing_provenance.py", 0, "Frozen source. 32 LOCAL-019 cases pass; existing two Linux-only validity checks skip."),
    ]
    attempts = []
    for name, tests, exit_code, interpretation in definitions:
        attempts.append({
            "command": test_prefix + tests + arguments + f" --junitxml=../docs/audits/2026-08-30/evidence/torchrun-verification/{name}.xml > ../docs/audits/2026-08-30/evidence/torchrun-verification/{name}.txt 2>&1",
            "cwd": "code", "exit_code": exit_code, "result": result(name),
            "interpretation": interpretation,
        })
    final = attempts[-1]["result"]
    assert final["passed"] == 66 and final["skipped"] == 2 and final["failures"] == final["errors"] == 0
    dependencies = [
        "code/core/harness/torchrun_wrapper.py", "code/tests/protection_test_utils.py",
        "code/tests/test_audit_wave1_harness.py", "code/tests/test_seed_config_immutability.py",
        "code/tests/test_audit_wave1_timing_provenance.py",
    ]
    receipt = {
        "scope": ["LOCAL-019 withdraw parent-side toy child-training verification", "LOCAL-019 declared torchrun-spec failures propagate before subprocess launch"],
        "source_audit_count_unchanged": 128,
        "base_commit": "b57e4c6a9e261c09ac09208705d040c81b03d35e",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_status": "Source repaired and CPU control-plane verified by explicit withdrawal; actual generic training qualification remains unsupported",
        "environment": {"python": platform.python_version(), "executable": sys.executable, "platform": platform.platform(), "torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "qualification": "Existing CPU environment, not pinned GPU stack"},
        "source_files_before": before["source_files"], "source_files": actual,
        "read_only_dependencies": {path: digest(ROOT / path) for path in dependencies},
        "affected_wrapper_count": 61, "affected_wrapper_inventory": "source-validation.json",
        "reproduction": {"initial": "before-mechanism.json", "repeat_command": "PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/bin/python docs/audits/2026-08-30/evidence/torchrun-verification/reproduce_archived_surrogate.py > docs/audits/2026-08-30/evidence/torchrun-verification/before-replay.json 2> docs/audits/2026-08-30/evidence/torchrun-verification/before-replay.stderr.txt", "repeat_exit_code": 0, "repeat_result": json.loads((HERE / "before-replay.json").read_text()), "limitation": "Actual archived CPU toy math only; no child training or GPU evidence"},
        "changes": ["Remove unrelated Linear/meta model, inputs, cached output, signature and fabricated tolerance", "Reject all generic wrapper execution/verification hooks even with stale payload injection", "Preserve factory/discovery metadata and configuration method ASTs", "Propagate callable launch-spec getter failures; reject noncallable getter; only explicit None may select legacy module fallback"],
        "attempts": attempts,
        "cli_import_and_help": {"artifact": "direct-cli-help.json", "commands": [item["command"] for item in cli], "passed": len(cli), "scope": "Import and parser availability only; no actual training"},
        "static_validation": {"artifact": "source-validation.json", "ruff_command": "/opt/miniconda3/bin/python -m ruff check code/labs/train_distributed/training_utils/torchrun_harness.py code/core/harness/benchmark_harness.py code/tests/test_audit_wave1_torchrun_verification.py --select E9,F63,F7,F82,B006,B023", "ruff_exit_code": 0, "ruff_artifact": "ruff-final.txt"},
        "remaining_gates": ["Implement actual child-produced training result/state protocol plus independent workload-matched oracle before re-enabling any generic wrapper", "Require rejection of corrupted/unrelated child results through actual comparison path", "Run target-specific training/numerical/distributed checks on allocated compatible GPU stack before any accuracy, memory or performance acceptance", "Existing two Linux-only seed/config validity cases remain skipped locally"],
        "non_claims": ["No CUDA/NCCL/multi-GPU training executed", "No trained model, loss, gradient or optimizer-state numerical acceptance", "No memory or speedup claim", "No generic training qualification inferred from real CPU launcher success or independent ZeRO gates", "No old receipt rewritten; historical shared-harness dependency hashes remain tied to their earlier source epoch"],
        "artifacts": {str(path.relative_to(HERE)): digest(path) for path in sorted(HERE.rglob("*")) if path.is_file() and path.name != "receipt.json" and "__pycache__" not in path.parts},
    }
    with (HERE / "receipt.json").open("x") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"receipt": str((HERE / "receipt.json").relative_to(ROOT)), "sha256": digest(HERE / "receipt.json"), "artifacts": len(receipt["artifacts"]), "source_files": len(actual)}))


if __name__ == "__main__":
    main()
