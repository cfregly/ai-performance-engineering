from __future__ import annotations

import os
import sys
from types import SimpleNamespace


def test_tool_launch_injects_repo_root_pythonpath(monkeypatch):
    import core.tools.tools_commands as tools_commands

    recorded = {}

    def fake_run(cmd, env=None):
        recorded["cmd"] = cmd
        recorded["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tools_commands.subprocess, "run", fake_run)

    exit_code = tools_commands._run_tool("kv-cache", ["--help"])

    assert exit_code == 0
    assert recorded["cmd"] == [
        sys.executable,
        str(tools_commands.TOOLS["kv-cache"].script_path.resolve()),
        "--help",
    ]
    assert str(tools_commands.REPO_ROOT) in recorded["env"]["PYTHONPATH"].split(os.pathsep)


def test_tool_launch_prefers_module_name_when_available(monkeypatch):
    import core.tools.tools_commands as tools_commands

    recorded = {}

    def fake_run(cmd, env=None):
        recorded["cmd"] = cmd
        recorded["env"] = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tools_commands.subprocess, "run", fake_run)

    exit_code = tools_commands._run_tool("kernel-verification", ["--help"])

    assert exit_code == 0
    assert recorded["cmd"] == [
        sys.executable,
        "-m",
        "ch20.kernel_verification_tool",
        "--help",
    ]
    assert str(tools_commands.REPO_ROOT) in recorded["env"]["PYTHONPATH"].split(os.pathsep)


def test_new_chapter_parity_tools_are_registered() -> None:
    import core.tools.tools_commands as tools_commands

    expected = {
        "ch04-nixl-tier-handoff": "ch04.nixl_tier_handoff_tool",
        "ch19-adaptive-parallelism": "ch19.adaptive_parallelism_strategy",
        "ch19-dynamic-precision": "ch19.dynamic_precision_switching",
        "ch19-dynamic-quantized-cache": "ch19.dynamic_quantized_cache",
        "ch20-ai-kernel-generator": "ch20.ai_kernel_generator",
        "ch20-ai-kernel-workflow": "ch20.ai_kernel_workflow_tool",
    }

    for tool_name, module_name in expected.items():
        spec = tools_commands.TOOLS[tool_name]
        assert spec.module_name == module_name
        assert spec.script_path.exists()


def test_multi_gpu_communication_tool_runs_real_help(capfd) -> None:
    import core.tools.tools_commands as tools_commands

    spec = tools_commands.TOOLS["multi-gpu-communication"]
    assert spec.module_name == "labs.multi_gpu_communication.run_lab_multi_gpu_communication"
    assert tools_commands._run_tool("multi-gpu-communication", ["--help"]) == 0
    output = capfd.readouterr().out
    assert "--registered-buffers" in output
    assert "--graph" in output
    assert "nvshmem" in output


def test_nvshmem_device_tool_runs_real_help(capfd) -> None:
    import core.tools.tools_commands as tools_commands

    spec = tools_commands.TOOLS["nvshmem-device"]
    assert spec.module_name == "labs.nvshmem_device.run_lab_nvshmem_device"
    assert tools_commands._run_tool("nvshmem-device", ["--help"]) == 0
    output = capfd.readouterr().out
    for mode in ("scalar-halo", "persistent-halo", "block-signal", "persistent-signal", "persistent-atomic"):
        assert mode in output
