"""Real CLI regressions for registered tool workload entrypoints."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("tool_name", "expected_diagnostic"),
    [
        ("dtensor-mesh", "CUDA required for ch13 DTensor mesh tool"),
        (
            "fp8-calibration-free",
            "CUDA required for ch19 calibration-free FP8 tool",
        ),
        ("fp8-perchannel-bench", "CUDA required for ch13 FP8 per-channel tool"),
        ("kernel-verification", "CUDA required for ch20 kernel verification tool"),
        ("kv-cache-math", "CUDA required for ch15 KV-cache math tool"),
        ("nvfp4-trtllm", "CUDA required for ch18 NVFP4/TRT-LLM tool"),
        (
            "proofwright-verify",
            "CUDA required for ch20 ProofWright verification tool",
        ),
    ],
)
def test_registered_tool_reaches_explicit_cuda_gate(
    tool_name: str,
    expected_diagnostic: str,
) -> None:
    """A CUDA-hidden real CLI launch must execute the workload entrypoint."""
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
            "tools",
            tool_name,
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
