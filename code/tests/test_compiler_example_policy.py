"""The compiler configuration must leave actual compilation enabled."""

import os
from pathlib import Path
import subprocess
import sys


CODE = Path(__file__).resolve().parents[1]


def test_configuration_does_not_suppress_compilation_or_force_triton_rebuilds():
    # Execute in a fresh process: configuration intentionally changes GPU tuning
    # settings, but must not leak an eager-only stance or disable cache reuse.
    script = """
import os
import torch
from ch14.torch_compiler_examples import configure_for_blackwell_peak_performance

configure_for_blackwell_peak_performance()
assert 'TRITON_ALWAYS_COMPILE' not in os.environ
graphs = []
def backend(graph, inputs):
    graphs.append(graph)
    return graph.forward

def fn(x):
    return x.sin() + x

compiled = torch.compile(fn, backend=backend, dynamic=False, fullgraph=True)
with torch.inference_mode():
    for size in (3, 4):
        x = torch.arange(size, dtype=torch.float32)
        torch.testing.assert_close(compiled(x), fn(x))
assert len(graphs) == 2, len(graphs)
with torch.inference_mode(), torch.compiler.set_stance('fail_on_recompile'):
    torch.testing.assert_close(compiled(x), fn(x))
    try:
        compiled(torch.ones(5))
    except RuntimeError as error:
        assert 'recompil' in str(error).lower(), str(error)
    else:
        raise AssertionError('unexpected signature did not fail')
assert len(graphs) == 2
"""
    env = {**os.environ, "PYTHONPATH": str(CODE)}
    env.pop("TRITON_ALWAYS_COMPILE", None)
    result = subprocess.run(
        [sys.executable, "-c", script], cwd=CODE, env=env,
        text=True, capture_output=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
