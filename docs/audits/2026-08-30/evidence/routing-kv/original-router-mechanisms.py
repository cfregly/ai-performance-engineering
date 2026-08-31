"""Execute the original pinned Python payload parser; no vLLM/GPU execution."""
import ast
import hashlib
import json
import subprocess
from types import SimpleNamespace
from typing import List, Tuple
from labs.dynamic_router.vllm_runner import _RequestRuntime, _VllmWrapper
from labs.dynamic_router.router_round_robin import Request
from pathlib import Path

revision = "b57e4c6a9e261c09ac09208705d040c81b03d35e"
path = "code/labs/dynamic_router/vllm_runner.py"
source = subprocess.check_output(["git", "show", f"{revision}:{path}"], text=True)
tree = ast.parse(source)
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_VllmWrapper")
step = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "step")
namespace = {"List": List, "Tuple": Tuple}
exec(compile(ast.Module(body=[step], type_ignores=[]), "original-vllm-wrapper-step", "exec"), namespace)
runtime = lambda: _RequestRuntime(Request(req_id="request", prompt_tokens=4, expected_new_tokens=3), "gpu0", 100.0)
original = SimpleNamespace(_inflight={"request": runtime()})
current = _VllmWrapper.__new__(_VllmWrapper)
current._inflight = {"request": runtime()}
old_counts, new_counts, old_ttft, new_ttft = [], [], [], []
for count in (1, 2, 3):
    payload = [SimpleNamespace(request_id="request", outputs=[SimpleNamespace(token_ids=list(range(count)))], finished=count == 3)]
    # Serialized request-output fixture only. This exercises the original parser,
    # not engine throughput, kernel math, API compatibility, or CUDA behavior.
    original.engine = SimpleNamespace(step=lambda: payload)
    _, ttft, tokens = namespace["step"](original, 100 + count * .25)
    old_counts.append(tokens); old_ttft.extend(ttft)
    _, ttft, tokens = current._consume_request_outputs(payload, 100 + count * .25)
    new_counts.append(tokens); new_ttft.extend(ttft)
assert old_counts == [1, 2, 3] and new_counts == [1, 1, 1]
assert old_ttft == new_ttft == [("request", 250.0)]
wiring = []
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "snapshot_metrics":
        args = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
        if args.get("ttft_ema") == args.get("tpot_ema"):
            wiring.append({"line": node.lineno, "args": args})
assert len(wiring) == 1
result = {"baseline": revision, "source_sha256": hashlib.sha256(source.encode()).hexdigest(), "original_token_deltas": old_counts, "current_token_deltas": new_counts, "first_token_samples": new_ttft, "original_latency_wiring": wiring, "scope": "Actual Python parser and AST source reproduction with cumulative response fixtures; no vLLM/GPU runtime"}
print(json.dumps(result, indent=2))
