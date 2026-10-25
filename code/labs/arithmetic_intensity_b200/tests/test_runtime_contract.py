"""Current-stack contracts for the B200 arithmetic-intensity lab."""

from __future__ import annotations

import pytest
from labs.arithmetic_intensity_b200 import verification_tool


def test_current_torch_2141_runtime_is_accepted() -> None:
    observed = verification_tool._validate_lab_runtime_versions(
        pytorch="2.14.1+cu130",
        pytorch_cuda="13.0",
        triton="3.8.0",
    )

    assert observed == {
        "qualification_role": "current_source_target",
        "pytorch": "2.14.1+cu130",
        "pytorch_cuda": "13.0",
        "triton": "3.8.0",
        "triton_role": "recorded_only_nonbinding_unused_by_lab",
    }


def test_historical_torch_213_runtime_is_rejected() -> None:
    with pytest.raises(
        verification_tool.DiagnosticGateError,
        match=r"PyTorch expected 2\.14\.1\+cu130, observed 2\.13\.0\+cu130",
    ):
        verification_tool._validate_lab_runtime_versions(
            pytorch="2.13.0+cu130",
            pytorch_cuda="13.0",
            triton="3.6.0",
        )
