"""Real CLI regressions for registered demo entrypoints."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("demo_name", "cli_args", "expected_diagnostic"),
    [
        (
            "ch10-warpgroup-specialization",
            [],
            "CUDA required for ch10 warpgroup specialization demo",
        ),
        (
            "ch13-fp8-perchannel",
            [],
            "CUDA required for ch13 FP8 per-channel demo",
        ),
        (
            "labs-decode-multigpu",
            ["--nproc-per-node", "2"],
            "CUDA required for multi-GPU decode demo",
        ),
    ],
)
def test_registered_demo_reaches_explicit_cuda_gate(
    demo_name: str,
    cli_args: list[str],
    expected_diagnostic: str,
) -> None:
    """A CUDA-hidden real CLI launch must execute the script, not exit zero after import."""
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "PATH": f"{Path(sys.executable).parent}{os.pathsep}{env.get('PATH', '')}",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(CODE_ROOT),
        }
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cli.aisp",
            "demos",
            demo_name,
            *cli_args,
        ],
        cwd=CODE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail(f"Demo process timed out:\n{stdout}{stderr}")

    combined_output = stdout + stderr
    assert process.returncode != 0, combined_output
    assert expected_diagnostic in combined_output
