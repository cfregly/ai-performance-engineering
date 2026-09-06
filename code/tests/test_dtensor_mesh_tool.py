"""Launch-contract regressions for the DTensor mesh chapter tool."""

from __future__ import annotations

import pytest

from ch13.dtensor_mesh_tool import _resolve_torchrun_context, get_benchmark


def test_dtensor_tool_does_not_report_an_unexecuted_precision_mode() -> None:
    assert get_benchmark().get_custom_metrics() is None


def test_plain_python_launch_reports_exact_torchrun_remedy() -> None:
    with pytest.raises(RuntimeError, match="requires torchrun") as exc_info:
        _resolve_torchrun_context({}, cuda_device_count=2)

    message = str(exc_info.value)
    assert "--nproc_per_node 2" in message
    assert "--rdzv_backend static" in message
    assert "-m ch13.dtensor_mesh_tool" in message


@pytest.mark.parametrize(
    ("environ", "device_count", "expected"),
    [
        ({"RANK": "0", "WORLD_SIZE": "1", "LOCAL_RANK": "0"}, 1, (0, 1, 0)),
        ({"RANK": "1", "WORLD_SIZE": "2", "LOCAL_RANK": "1"}, 2, (1, 2, 1)),
    ],
)
def test_supported_one_and_two_rank_launches_are_resolved(
    environ: dict[str, str],
    device_count: int,
    expected: tuple[int, int, int],
) -> None:
    assert (
        _resolve_torchrun_context(environ, cuda_device_count=device_count) == expected
    )


@pytest.mark.parametrize(
    ("environ", "device_count", "expected_error"),
    [
        ({"RANK": "0", "WORLD_SIZE": "3", "LOCAL_RANK": "0"}, 4, "one or two ranks"),
        ({"RANK": "2", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}, 2, "0 <= RANK"),
        (
            {"RANK": "0", "WORLD_SIZE": "2", "LOCAL_RANK": "2"},
            2,
            "invalid for cuda.device_count",
        ),
        ({"RANK": "rank0", "WORLD_SIZE": "2", "LOCAL_RANK": "0"}, 2, "Invalid RANK"),
    ],
)
def test_invalid_torchrun_topology_is_rejected_before_dtensor_initialization(
    environ: dict[str, str],
    device_count: int,
    expected_error: str,
) -> None:
    with pytest.raises(RuntimeError, match=expected_error):
        _resolve_torchrun_context(environ, cuda_device_count=device_count)
