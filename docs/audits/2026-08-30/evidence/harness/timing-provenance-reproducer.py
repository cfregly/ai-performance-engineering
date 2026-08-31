"""Replay the original Timer body in a separate process; never edit the checkout.

Run from code/ with the CPU interpreter. An exit1/three failed tests is the
expected original-source reproduction, not a current-source gate.
"""
import ast
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path.cwd()))
import pytest
from core.harness import benchmark_harness as harness

base = "b57e4c6a9e261c09ac09208705d040c81b03d35e"
source = subprocess.check_output(
    ["git", "show", f"{base}:code/core/harness/benchmark_harness.py"], text=True,
)
cls = next(node for node in ast.parse(source).body if isinstance(node, ast.ClassDef) and node.name == "BenchmarkHarness")
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_benchmark_pytorch")
namespace = dict(vars(harness))
exec(compile(ast.Module(body=[method], type_ignores=[]), f"{base}:_benchmark_pytorch", "exec"), namespace)
harness.BenchmarkHarness._benchmark_pytorch = namespace["_benchmark_pytorch"]
raise SystemExit(pytest.main([
    "tests/test_audit_wave1_timing_provenance.py", "-q", "-p", "no:cacheprovider",
    "-k", "not torchrun", "--tb=short",
]))
