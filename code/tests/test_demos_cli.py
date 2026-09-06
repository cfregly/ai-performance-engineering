import shutil
import subprocess
import sys
import os
from types import SimpleNamespace

import pytest


def test_demos_list_exits_cleanly():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "demos", "list"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "ch11-stream-overlap" in result.stdout
    assert "ch12-graph-capture" in result.stdout


def test_tools_list_does_not_include_demos():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "tools", "list"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert "ch11-stream-overlap:" not in result.stdout
    assert "ch12-graph-capture:" not in result.stdout


def test_no_backward_compat_demo_names_under_tools():
    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "tools", "ch11-stream-overlap-demo", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0


def test_torchrun_required_demo_requires_nproc_per_node():
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not available in PATH")

    result = subprocess.run(
        [sys.executable, "-m", "cli.aisp", "demos", "ch15-tensor-parallel"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert result.returncode != 0
    combined = (result.stdout + result.stderr).lower()
    assert "nproc-per-node" in combined


def test_torchrun_required_demo_runs_under_aisp_demos():
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not available in PATH")

    try:
        import torch
    except ImportError:
        pytest.skip("torch not available")

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cli.aisp",
            "demos",
            "ch15-tensor-parallel",
            "--nproc-per-node",
            "1",
            "--",
            "--batch",
            "1",
            "--in-features",
            "16",
            "--out-features",
            "16",
            "--iters",
            "1",
            "--warmup",
            "0",
            "--dtype",
            "fp16",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0


def test_module_backed_demo_launches_via_python_module(monkeypatch):
    import core.demos.demos_commands as demos_commands

    recorded = {}

    def fake_run(cmd, env=None):
        recorded["cmd"] = cmd
        recorded["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(demos_commands.subprocess, "run", fake_run)

    exit_code = demos_commands._run_demo("ch11-stream-overlap", ["--iterations", "1"])

    assert exit_code == 0
    assert recorded["cmd"][:3] == [sys.executable, "-m", "ch11.stream_overlap_demo"]
    assert recorded["cmd"][3:] == ["--iterations", "1"]
    assert str(demos_commands.REPO_ROOT) in recorded["env"]["PYTHONPATH"].split(os.pathsep)


def test_module_backed_torchrun_demo_launches_via_module(monkeypatch):
    import core.demos.demos_commands as demos_commands

    recorded = {}

    def fake_run(cmd, env=None):
        recorded["cmd"] = cmd
        recorded["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(demos_commands.subprocess, "run", fake_run)

    exit_code = demos_commands._run_demo(
        "ch15-tensor-parallel",
        ["--batch", "1"],
        nproc_per_node=2,
    )

    assert exit_code == 0
    assert recorded["cmd"][:7] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "2",
        "-m",
        "ch15.tensor_parallel_demo",
    ]
    assert recorded["cmd"][7:] == ["--batch", "1"]
    assert str(demos_commands.REPO_ROOT) in recorded["env"]["PYTHONPATH"].split(os.pathsep)
