from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from labs.nvfp4_group_gemm.custom_cuda_submission import (
    _cuda_gencode_flags_for_capability,
)

CODE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("capability", "target"),
    [
        ((10, 0), "100a"),
        ((10, 3), "103a"),
    ],
)
def test_nvfp4_group_gemm_selects_only_active_tcgen05_architecture(
    capability: tuple[int, int],
    target: str,
) -> None:
    assert _cuda_gencode_flags_for_capability(capability) == [
        f"-gencode=arch=compute_{target},code=sm_{target}",
        f"-gencode=arch=compute_{target},code=compute_{target}",
    ]


@pytest.mark.parametrize("capability", [(9, 0), (10, 2), (12, 0), (12, 1)])
def test_nvfp4_group_gemm_rejects_unvalidated_architectures(
    capability: tuple[int, int],
) -> None:
    with pytest.raises(RuntimeError, match="SKIPPED: NVFP4 grouped GEMM custom CUDA extension"):
        _cuda_gencode_flags_for_capability(capability)


def test_nvfp4_group_gemm_requires_a_visible_cuda_device() -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_ROOT)
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from labs.nvfp4_group_gemm.custom_cuda_submission "
                "import _active_cuda_gencode_flags; _active_cuda_gencode_flags()"
            ),
        ],
        cwd=CODE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=30,
    )

    assert result.returncode != 0
    assert (
        "SKIPPED: NVFP4 grouped GEMM custom CUDA extension requires a visible CUDA device"
        in result.stderr
    )
