"""Reproduce preserved payload/copy defects with actual CPU tensor methods only."""

import ast
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[6]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))
import torch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


controls = load("prefill_frozen_controls", ROOT / "code/tests/test_audit_wave1_prefill_full_output.py")
results = []
for variant, (filename, class_name) in controls.VARIANTS.items():
    old_module = load("prefill_preserved_" + variant, HERE / ("original-" + filename + ".py"))
    current = controls._payload_control(variant, torch.float32)
    old = object.__new__(getattr(old_module, class_name))
    old.__dict__.update(current.__dict__)
    old.output = old._output_view = old.inputs.out[:1, :min(8, old.inputs.out.shape[1])]
    old._verify_output_buffer = torch.empty_like(old.output, dtype=torch.float32)
    old.capture_verification_payload()
    before = old.get_verify_output().clone()
    old.inputs.out[-1, -1, -1] += 1024
    old.prefill_dst[-1, -1] += 1024
    old.capture_verification_payload()
    torch.testing.assert_close(before, old.get_verify_output(), rtol=0, atol=0)
    assert old.validate_result() is None
    assert "prefill_src" not in old.get_verify_inputs()
    results.append({"variant": variant, "full_decode_shape": list(old.inputs.out.shape),
                    "original_payload_shape": list(before.shape),
                    "last_decode_and_prefill_corruption_omitted": True,
                    "original_validate_result_after_corruption": None})

old_module = load("prefill_preserved_copy", HERE / "original-baseline_tma_prefill_decode.py")
old = object.__new__(old_module.BaselineTmaPrefillDecodeBenchmark)
src = torch.arange(34, dtype=torch.float32).reshape(2, 17) / 8
dst = torch.empty_like(src)
old._prefill_work = tuple(zip(src.unbind(0), dst.unbind(0), strict=True))
old._prefill_sequential()
torch.testing.assert_close(dst, 2 * src, rtol=0, atol=0)
assert not torch.equal(dst, src)

unchanged = []
for filename, method in [
    ("baseline_tma_prefill_decode.py", "_decode_host_loop"),
    ("baseline_native_tma_prefill_decode.py", "_decode_host_loop"),
    ("optimized_tma_prefill_decode.py", "_decode_body"),
    ("optimized_native_tma_prefill_decode.py", "_decode_body"),
    ("optimized_tma_prefill_decode.py", "_load_cp_async_tma_ext"),
    ("persistent_decode_common.py", "validate_decode_output"),
]:
    old_tree = ast.parse((HERE / ("original-" + filename)).read_text())
    current_path = ROOT / "code/labs/persistent_decode" / filename
    current_tree = ast.parse(current_path.read_text())
    old_node = next(node for node in ast.walk(old_tree) if isinstance(node, ast.FunctionDef) and node.name == method)
    current_node = next(node for node in ast.walk(current_tree) if isinstance(node, ast.FunctionDef) and node.name == method)
    assert ast.dump(old_node, include_attributes=False) == ast.dump(current_node, include_attributes=False)
    unchanged.append({"path": str(current_path.relative_to(ROOT)), "function": method})

paths = [ROOT / "code/labs/persistent_decode" / item["path"].split("/")[-1]
         for item in json.loads((HERE / "original-source-manifest.json").read_text())]
paths.append(ROOT / "code/tests/test_audit_wave1_prefill_full_output.py")
for path in paths:
    compile(path.read_text(), str(path), "exec")
report = {
    "status": "PASS", "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
    "scope": "Actual preserved Python payload and copy methods on real CPU tensors. CUDA-only constructors, setup, streams, extensions and graphs were neither executed nor mocked. Current CPU fixture supplies data and storage only.",
    "original_capture_reproductions": results,
    "original_copy_result": "dst equals 2*src, differs from copy-only peer contract",
    "unchanged_decode_math_extension_and_inherited_validator_ast": unchanged,
    "python_compile": [str(path.relative_to(ROOT)) for path in paths],
}
(HERE / "original-mechanisms-rechecked.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
