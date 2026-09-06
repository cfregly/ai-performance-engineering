"""Real CLI regressions for registered demo entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

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
    completed = subprocess.run(
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
        capture_output=True,
        text=True,
        timeout=45,
        check=False,
    )

    combined_output = completed.stdout + completed.stderr
    assert completed.returncode != 0, combined_output
    assert expected_diagnostic in combined_output
