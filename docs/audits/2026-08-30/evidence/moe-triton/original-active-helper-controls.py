"""Actual original host expression and CPU address controls; no GPU is simulated."""
import ast
import hashlib
import json
from pathlib import Path

import torch


root = Path(__file__).resolve().parent
source = (root / "original-triton_kernels.py").read_text()
node = next(item for item in ast.parse(source).body
            if isinstance(item, ast.FunctionDef) and item.name == "fused_silu_mul")
segment = ast.get_source_segment(source, node)
assignment = next(item for item in node.body if isinstance(item, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == "out" for target in item.targets))
gate = torch.linspace(-2, 2, 12, requires_grad=True).reshape(3, 4)
allocated = eval(compile(ast.Expression(assignment.value), "original-output-allocation", "eval"),
                 {"torch": torch, "gate": gate})
base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
view = base.t()
flat_storage = base.reshape(-1).reshape(view.shape)
result = {
    "classification": "Original allocation AST and actual CPU address controls; no original Triton compilation or GPU execution",
    "original_wrapper_sha256": hashlib.sha256(segment.encode()).hexdigest(),
    "original_allocation_expression": ast.unparse(assignment.value),
    "original_allocated_output_requires_grad": allocated.requires_grad,
    "original_input_requires_grad": gate.requires_grad,
    "logical_transposed_values": view.tolist(),
    "raw_contiguous_storage_values_reshaped": flat_storage.tolist(),
    "flat_storage_matches_logical_view": torch.equal(view, flat_storage),
    "cuda_available": torch.cuda.is_available(),
}
assert gate.requires_grad and not allocated.requires_grad
assert not torch.equal(view, flat_storage)
(root / "original-active-helper-mechanisms.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
