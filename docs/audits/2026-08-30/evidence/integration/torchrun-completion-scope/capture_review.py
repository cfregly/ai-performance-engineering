"""Read source/receipts and retain this bounded review; never import repository code."""

import ast
import collections
import hashlib
import json
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[5]
BASE = "b57e4c6a9e261c09ac09208705d040c81b03d35e"
AUDIT = ROOT / "docs/audits/2026-08-30"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(name, value):
    path = HERE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2) + "\n")


def original(path):
    return subprocess.run(
        ["git", "show", f"{BASE}:{path}"], cwd=ROOT, check=True, capture_output=True
    ).stdout


source = json.loads((AUDIT / "wave-1-source.json").read_text())
write("original-findings.json", {
    "source_file_sha256": sha((AUDIT / "wave-1-source.json").read_bytes()),
    "findings": [x for x in source["findings"] if x["id"] in ("W1-004", "W1-091")],
})

listed = json.loads((AUDIT / "evidence/torchrun-verification/source-validation.json").read_text())
rows = []
for item in listed["affected_wrappers"]:
    path = item["path"]
    data = original(path)
    text = data.decode()
    tree = ast.parse(text)
    factory = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "get_benchmark")
    call = next(n for n in ast.walk(factory) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Name) and n.func.id == "TorchrunScriptBenchmark")
    kwargs = {kw.arg: kw.value for kw in call.keywords}
    basename = Path(path).name
    if basename in {"baseline_symmem_training.py", "optimized_symmem_training.py"}:
        classification = "preexisting_unconditional_refusal_for_declared_arguments"
        reason = "--allow-single-gpu reaches original symmetric child init_distributed lines105-106 unconditional refusal."
    elif basename == "baseline_symmem_training_multigpu.py":
        classification = "preexisting_unconditional_refusal_for_declared_arguments"
        reason = "--disable-symmetric sets SYMMETRIC_MEMORY_DISABLED at880-881; capability query98-99 returnsFalse and init109-110 refuses."
    elif basename == "optimized_zero2.py":
        classification = "original_W1_091_missing_API_refusal"
        reason = "Original source77-79 requests batched_reduce_scatter_hook; missing API verified in original finding and retained prior CPU receipts, not rerun here."
    elif basename == "optimized_zero2_multigpu.py":
        classification = "original_W1_004_launchable_but_no_optimizer_update"
        reason = "overlap_with_ddp=True with manual step and ordinary RS/AG hook; original audit and prior reproductions establish skipped optimizer math, not a universal launch refusal."
    elif basename in {"baseline_zero2.py", "baseline_zero2_multigpu.py"}:
        classification = "zero2_baseline_no_unconditional_refusal_identified"
        reason = "Direct baseline training existed; original workload mismatches repaired under LOCAL-006. No runtime replay in this review."
    else:
        classification = "other_child_path_present_runtime_support_not_established"
        reason = "Source has an existing child dispatcher/implementation; no universal refusal established by bounded source review. This does not prove runtime correctness or dependency/target support."
    rows.append({
        "path": path,
        "base_sha256": sha(data),
        "current_sha256": sha((ROOT / path).read_bytes()),
        "factory_line_at_base": factory.lineno,
        "factory_source_at_base": ast.get_source_segment(text, factory),
        "script_expression": ast.unparse(kwargs["script_path"]),
        "arguments_at_base": ast.literal_eval(kwargs["base_args"]),
        "classification": classification,
        "reason": reason,
        "base_verification": "unrelated parent Linear surrogate; no child-result correctness evidence",
        "current_generic_harness_execution": "unconditionally refused before child launch",
    })
assert len(rows) == 61 and len({row["path"] for row in rows}) == 61
write("factory-inventory.json", {
    "base": BASE, "count": 61,
    "classification_counts": dict(collections.Counter(row["classification"] for row in rows)),
    "limits": "Static source inventory, not 61 runtime tests or a complete audit of every child implementation. Explicit refusal is classified only for declared default arguments.",
    "factories": rows,
})

original_paths = [
    "code/labs/train_distributed/training_utils/torchrun_harness.py",
    "code/labs/train_distributed/README.md",
    "code/ch04/symmetric_memory_training_advanced.py",
    *[row["path"] for row in rows if row["classification"] != "other_child_path_present_runtime_support_not_established"],
]
original_hashes = {}
for path in original_paths:
    data = original(path)
    destination = HERE / "originals" / (path.replace("/", "__") + ".txt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(data)
    original_hashes[path] = sha(data)

current_paths = [
    "AUDIT_REMEDIATION_PLAN.md",
    "docs/audits/2026-08-30/remediation-ledger.json",
    "docs/audits/2026-08-30/wave-1-source.json",
    "code/labs/train_distributed/training_utils/torchrun_harness.py",
    "code/labs/train_distributed/zero2_common.py",
    "code/labs/train_distributed/zero2.py",
    "code/labs/train_distributed/baseline_zero2.py",
    "code/labs/train_distributed/optimized_zero2.py",
    "code/labs/train_distributed/baseline_zero2_multigpu.py",
    "code/labs/train_distributed/optimized_zero2_multigpu.py",
    "code/core/harness/benchmark_harness.py",
    "code/core/harness/torchrun_wrapper.py",
    "code/core/benchmark/verify_runner.py",
    "code/tests/test_audit_wave1_zero2_parity.py",
    "code/tests/test_audit_wave1_zero2_parity_cuda.py",
    "code/tests/test_audit_wave1_torchrun_verification.py",
    "docs/audits/2026-08-30/evidence/zero2-parity/receipt.json",
    "docs/audits/2026-08-30/evidence/zero2-parity/cuda-gate/validation-receipts.json",
    "docs/audits/2026-08-30/evidence/torchrun-verification/receipt.json",
]
write("source-evidence.json", {
    "original_base": BASE,
    "original_sha256": original_hashes,
    "current_sha256": {p: sha((ROOT / p).read_bytes()) for p in current_paths},
    "snapshot_limit": "Plan/ledger may be updated by their owner after this review. This is an observation epoch, not a source freeze or acceptance receipt.",
    "executed_repository_code": False,
    "tests_run": False,
    "gpu_or_remote_work": False,
    "install_or_git_mutation": False,
})

ledger = json.loads((AUDIT / "remediation-ledger.json").read_text())
write("observed-status.json", {
    "original_findings": [x for x in ledger["findings"] if x["id"] in ("W1-004", "W1-091")],
    "adjacent": [x for x in ledger["adjacent_discoveries"] if x["id"] == "LOCAL-019"],
    "meaning": "Historical observed status only. Owner decides updates; this review recommends tracking functional restoration as open implementation work.",
})
print(json.dumps({"factory_count": len(rows), "classification_counts": dict(collections.Counter(r["classification"] for r in rows)), "new_directory": str(HERE)}, indent=2))
