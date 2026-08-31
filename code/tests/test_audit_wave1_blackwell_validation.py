"""Failure/skip routing checks; these are not GPU numerical evidence.

Only hardware availability and pre-work failures are injected. The companion
suite performs real CUDA work on hardware and reports explicit skips on CPU.
"""

import builtins
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch


SOURCE = Path(__file__).with_name("test_blackwell_stack.py")


@pytest.fixture
def stack():
    spec = importlib.util.spec_from_file_location("audit_blackwell_stack", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SINGLE_GPU_CHECKS = (
    "test_architecture_detection", "test_pytorch_29_features", "test_cuda_130_features",
    "test_profiling_tools", "test_triton_35", "test_performance",
)


@pytest.mark.parametrize("name", SINGLE_GPU_CHECKS)
def test_missing_device_is_a_pytest_skip(stack, monkeypatch, name):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(pytest.skip.Exception, match="requires a real CUDA device"):
        getattr(stack, name)()


@pytest.mark.parametrize("name,owner,attribute", [
    ("test_architecture_detection", torch.cuda, "get_device_properties"),
    ("test_pytorch_29_features", torch.nn, "Linear"),
    ("test_cuda_130_features", torch.cuda, "Stream"),
    ("test_profiling_tools", torch, "randn"),
    ("test_performance", torch, "randn"),
])
def test_unexpected_prework_error_is_not_swallowed(stack, monkeypatch, name, owner, attribute):
    def fail(*args, **kwargs):
        raise RuntimeError("sentinel pre-work failure")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    monkeypatch.setattr(owner, attribute, fail)
    with pytest.raises(RuntimeError, match="sentinel pre-work failure"):
        getattr(stack, name)()


def test_missing_declared_triton_dependency_fails_on_supported_path(stack, monkeypatch):
    original_import = builtins.__import__

    def import_without_triton(name, *args, **kwargs):
        if name == "triton":
            raise ImportError("sentinel missing Triton dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(builtins, "__import__", import_without_triton)
    with pytest.raises(ImportError, match="sentinel missing Triton dependency"):
        stack.test_triton_35()


def test_standalone_command_reports_unavailable_without_success():
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
    env.pop("PYTORCH_NVML_BASED_CUDA_CHECK", None)
    result = subprocess.run(
        [sys.executable, "-m", "tests.test_blackwell_stack"],
        cwd=SOURCE.parents[1], env=env, text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    assert "UNAVAILABLE" in result.stderr
    assert "Test completed" not in result.stdout
    assert "passed" not in result.stdout
