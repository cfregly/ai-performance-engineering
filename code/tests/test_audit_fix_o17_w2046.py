"""Regression tests for audit finding W2-046.

NCCL has no NCCL_NVLINK_TCE_ENABLE, NCCL_NVLINK_C2C_ENABLE or
NCCL_GRACE_BLACKWELL control. NCCL ignores unknown variables, so exporting
them only makes launch logs and run records list settings that never took
effect. These tests run the real repo scripts and configuration helpers and
check the environment they hand to NCCL.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

import ch04.nccl_blackwell_config as nccl_config

CODE_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = CODE_ROOT / "core" / "scripts"
FABRICATED_NCCL_CONTROLS = (
    "NCCL_NVLINK_TCE_ENABLE",
    "NCCL_NVLINK_C2C_ENABLE",
    "NCCL_GRACE_BLACKWELL",
)

_NVIDIA_SMI_STUB = """#!/bin/bash
if [ "$*" = "--query-gpu=name --format=csv,noheader" ]; then
    for _ in 1 2 3 4 5 6 7 8; do echo "NVIDIA B200"; done
fi
exit 0
"""

_TORCHRUN_STUB = """#!/bin/bash
env > "${LAUNCH_ENV_CAPTURE}"
exit 0
"""

_LSCPU_STUB = """#!/bin/bash
echo "Model name: stub CPU"
"""

_NOOP_STUB = """#!/bin/bash
exit 0
"""


def _write_stub(directory: Path, name: str, body: str) -> None:
    path = directory / name
    path.write_text(body)
    path.chmod(0o755)


def _run_launch_script(
    script_name: str, tmp_path: Path
) -> tuple[str, dict[str, str], Path]:
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    _write_stub(stub_dir, "nvidia-smi", _NVIDIA_SMI_STUB)
    _write_stub(stub_dir, "torchrun", _TORCHRUN_STUB)
    _write_stub(stub_dir, "lscpu", _LSCPU_STUB)
    # The scripts start the NVLink probe and power monitor through python.
    # Both would touch a GPU, so they are replaced along with the waits.
    for name in ("python", "python3", "sleep", "free"):
        _write_stub(stub_dir, name, _NOOP_STUB)

    capture = tmp_path / "launch_env.txt"
    output_dir = tmp_path / "results"
    env = {key: value for key, value in os.environ.items() if not key.startswith("NCCL_")}
    env["PATH"] = f"{stub_dir}{os.pathsep}{env['PATH']}"
    env["LAUNCH_ENV_CAPTURE"] = str(capture)

    result = subprocess.run(
        ["bash", str(SCRIPTS_DIR / script_name), "1", "5", str(output_dir)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert capture.is_file(), f"{script_name} never reached the torchrun launch"

    launch_env: dict[str, str] = {}
    for line in capture.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key.startswith("NCCL_"):
            launch_env[key] = value
    return result.stdout + result.stderr, launch_env, output_dir


def _assert_no_fabricated_controls(text: str) -> None:
    for name in FABRICATED_NCCL_CONTROLS:
        assert name not in text


def test_orchestrated_load_test_launch_has_no_fabricated_nccl_controls(
    tmp_path: Path,
) -> None:
    console, launch_env, output_dir = _run_launch_script(
        "orchestrate_multigpu_load_test.sh", tmp_path
    )

    assert launch_env["NCCL_PROTO"] == "Simple"
    assert launch_env["NCCL_ALGO"] == "Tree,Ring,NVLS"
    assert launch_env["NCCL_NVLS_ENABLE"] == "1"
    assert launch_env["NCCL_P2P_LEVEL"] == "NVL"
    assert set(FABRICATED_NCCL_CONTROLS).isdisjoint(launch_env)
    _assert_no_fabricated_controls(console)

    system_info = (output_dir / "system_info.txt").read_text()
    assert "=== NCCL Environment ===" in system_info
    assert "NCCL_NVLS_ENABLE=1" in system_info
    _assert_no_fabricated_controls(system_info)


def test_inference_server_nccl_benchmark_launch_has_no_fabricated_nccl_controls(
    tmp_path: Path,
) -> None:
    console, launch_env, output_dir = _run_launch_script(
        "run_inference_server_benchmark_nccl.sh", tmp_path
    )

    assert launch_env["NCCL_P2P_LEVEL"] == "NVL"
    assert launch_env["NCCL_P2P_DISABLE"] == "0"
    assert launch_env["NCCL_NVLS_ENABLE"] == "1"
    assert launch_env["NCCL_ALGO"] == "Tree,Ring,NVLS"
    assert launch_env["NCCL_NCHANNELS_PER_NET_PEER"] == "8"
    assert set(FABRICATED_NCCL_CONTROLS).isdisjoint(launch_env)
    _assert_no_fabricated_controls(console)

    system_info = (output_dir / "system_info.txt").read_text()
    assert "=== NCCL Environment ===" in system_info
    assert "NCCL_P2P_DISABLE=0" in system_info
    _assert_no_fabricated_controls(system_info)
    _assert_no_fabricated_controls((output_dir / "BENCHMARK_SUMMARY.md").read_text())


@pytest.mark.parametrize(
    "configure",
    [
        pytest.param(
            lambda: nccl_config.configure_nccl_for_blackwell(verbose=False),
            id="blackwell",
        ),
        pytest.param(
            lambda: nccl_config.configure_nccl_for_multigpu(num_gpus=8, verbose=False),
            id="multigpu",
        ),
        pytest.param(
            lambda: nccl_config.configure_nccl_for_gb200_gb300(verbose=False),
            id="gb200_gb300",
        ),
    ],
)
def test_nccl_config_helpers_set_no_fabricated_controls(
    monkeypatch: pytest.MonkeyPatch,
    configure: Callable[[], dict[str, str]],
) -> None:
    isolated_environment: dict[str, str] = {}
    monkeypatch.setattr(nccl_config.os, "environ", isolated_environment)
    # The GB200/GB300 helper sizes itself from the visible GPU count. Hiding
    # CUDA makes it use its fixed four-GPU value on any host.
    monkeypatch.setattr(nccl_config.torch.cuda, "is_available", lambda: False)

    configured = configure()

    assert configured["NCCL_PROTO"] == isolated_environment["NCCL_PROTO"]
    assert set(FABRICATED_NCCL_CONTROLS).isdisjoint(configured)
    assert set(FABRICATED_NCCL_CONTROLS).isdisjoint(isolated_environment)
