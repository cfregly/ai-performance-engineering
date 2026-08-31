"""Reproduce original CPU-visible defects without running GPU/benchmark timing code."""

from pathlib import Path
import gc
import json
import subprocess
import sys
from types import SimpleNamespace
import weakref

ROOT = Path(__file__).resolve().parents[5]
CODE = ROOT / "code"
sys.path.insert(0, str(CODE))
import torch
from labs.nvfp4_gemm.utils import make_match_reference
from labs.nvfp4_gemm.local_eval_submission import _verify_submission

BASE = "b57e4c6a9"


def original(path):
    return subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT, text=True)


report = {"reviewed_base": BASE, "scope": "CPU verifier/components only; no GPU timing"}
report["W1-030_old_allzero_passes"] = {
    str(scale): torch.allclose(torch.zeros(10), torch.arange(1., 11.) * scale, rtol=1, atol=10)
    for scale in (1e-5, 1, 1e6)}


def reference(data):
    data[2].copy_(data[0] @ data[1])
    return data[2]


def zero_submission(data):
    return data[2].zero_()


checker = make_match_reference(reference, rtol=1e-3, atol=1e-3)
data = (torch.ones(2, 2), torch.ones(2, 2), torch.empty(2, 2))
old_ok, _ = checker(data, zero_submission(data))
new_ok, _ = _verify_submission(data, SimpleNamespace(custom_kernel=zero_submission),
                              SimpleNamespace(check_implementation=checker))
report["W1-037"] = {"original_shared_C_zero_passes": old_ok, "new_isolated_zero_passes": new_ok}

path = "code/labs/nvfp4_gemv/optimized_submission.py"
ns = {"__name__": "audit_original_gemv", "__file__": str(ROOT / path)}
exec(compile(original(path), str(ROOT / path), "exec"), ns)
scale = torch.ones(128, 4, 1)
other = torch.ones_like(scale)
ns["_get_packed_scales"](scale, other, 1)
weak = weakref.ref(scale)
del scale
gc.collect()
report["W1-122_original_cache_releases_source_storage"] = weak() is None

ref = torch.tensor([1e-5, -2e-5, 3e-5, -4e-5], dtype=torch.float64)
report["W1-087_old_allzero_passes"] = torch.allclose(torch.zeros_like(ref), ref, rtol=.01, atol=.01)
corrupt = ref.clone(); corrupt[0] += 1e-5; corrupt[1] -= 1e-5
report["W1-087_checksum_cancelled_corruption_passes"] = torch.allclose(corrupt.sum(), ref.sum(), rtol=.01, atol=.01)
report["W1-086_original_baselines_use_custom_kernel"] = {
    p.name: "custom_kernel=custom_kernel_custom_cuda" in original(str(p.relative_to(ROOT)))
    for p in sorted((CODE / "labs/nvfp4_group_gemm").glob("baseline_nvfp4_group_gemm*.py"))}
report["W1-123_old_documented_path_missing"] = (
    "cutlass_extension.py" in original("code/labs/nvfp4_group_gemm/README.md")
    and not (CODE / "labs/nvfp4_group_gemm/cutlass_extension.py").exists())
print(json.dumps(report, indent=2))
