"""Preserve the reviewed sources and exercise their real model builders on CPU."""
import ast
import hashlib
import json
from pathlib import Path
import subprocess

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
BASE = "b57e4c6a9e261c09ac09208705d040c81b03d35e"
records = []
outputs = []
for name in ("baseline_zero2", "optimized_zero2", "baseline_zero2_multigpu", "optimized_zero2_multigpu"):
    path = f"code/labs/train_distributed/{name}.py"
    data = subprocess.check_output(["git", "show", f"{BASE}:{path}"], cwd=ROOT)
    archive = OUT / f"original-{name}.py.txt"
    if archive.exists():
        assert archive.read_bytes() == data, "Never overwrite an earlier source capture"
    else:
        archive.write_bytes(data)
    source = data.decode()
    tree = ast.parse(source)
    builder = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_build_model")
    scope = {"nn": nn, "torch": torch}
    exec(compile(ast.Module(body=[builder], type_ignores=[]), path, "exec"), scope)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(25)
        model = scope["_build_model"](8, torch.device("cpu"))
    outputs.append(model(torch.arange(24.).reshape(3, 8)).detach())
    records.append({"path": path, "sha256": hashlib.sha256(data).hexdigest(),
                    "archive": archive.name, "relu_count": sum(isinstance(m, nn.ReLU) for m in model),
                    "gelu_count": sum(isinstance(m, nn.GELU) for m in model)})
delta = float((outputs[0] - outputs[1]).abs().max())
assert delta > 0
report = {"reviewed_base": BASE, "kind": "actual original CPU model builders; not CUDA training",
          "source_files": records, "single_baseline_vs_optimized_max_abs_difference": delta,
          "all_optimized_and_multigpu_builders_equal": all(torch.equal(outputs[1], x) for x in outputs[2:])}
print(json.dumps(report, indent=2))
