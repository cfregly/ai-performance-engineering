"""Exercise the actual command builders and public replay-mode contracts."""

from pathlib import Path

import pytest

from core.benchmark.bench_commands import _validate_ncu_replay_mode
from core.profiling.nsight_automation import NsightAutomation
from core.profiling.profiler_config import (
    MINIMAL_METRICS,
    NCU_REPLAY_MODES,
    ProfilerConfig,
    validate_ncu_replay_mode,
)


RANGE_NAME = "compute_kernel:profile"


def _assert_full_range(command: list[str]) -> None:
    assert command[command.index("--replay-mode") + 1] == "app-range"
    assert command.count("--nvtx-include") == 1
    assert command[command.index("--nvtx-include") + 1] == RANGE_NAME
    assert command[command.index("--metrics") + 1].split(",") == MINIMAL_METRICS
    assert command[command.index("--target-processes") + 1] == "all"
    assert "--nvtx" in command
    assert not set(command).intersection(
        {"--launch-count", "--launch-skip", "--kernel-name", "--pm-sampling-interval"}
    )


@pytest.mark.parametrize("honor_override", [False, True])
def test_harness_builder_preserves_full_range(honor_override: bool) -> None:
    config = ProfilerConfig(ncu_replay_mode="app-range", honor_replay_mode_in_minimal=honor_override)
    command = config.get_ncu_command_for_target(
        "out", ["python", "benchmark.py"], nvtx_includes=[RANGE_NAME]
    )
    _assert_full_range(command)


@pytest.mark.parametrize(
    ("mode", "honor", "expected"),
    [("kernel", False, "kernel"), ("application", False, "kernel"),
     ("application", True, "application"), ("range", True, "range")],
)
def test_existing_lower_level_replay_policies(mode: str, honor: bool, expected: str) -> None:
    config = ProfilerConfig(ncu_replay_mode=mode, honor_replay_mode_in_minimal=honor)
    command = config.get_ncu_command_for_target("out", ["program"], nvtx_includes=[RANGE_NAME])
    assert command[command.index("--replay-mode") + 1] == expected


@pytest.mark.parametrize("mode", NCU_REPLAY_MODES)
def test_public_replay_modes(mode: str) -> None:
    assert validate_ncu_replay_mode(mode) == mode
    assert _validate_ncu_replay_mode(mode) == mode


@pytest.mark.parametrize("mode", ["range", "unknown", ""])
def test_unqualified_modes_are_not_new_public_options(mode: str) -> None:
    with pytest.raises(ValueError, match="Invalid Nsight Compute replay mode"):
        validate_ncu_replay_mode(mode)


@pytest.mark.parametrize(
    ("filters", "metrics", "sampling"),
    [([], MINIMAL_METRICS, None), ([RANGE_NAME, "another"], MINIMAL_METRICS, None),
     ([RANGE_NAME], MINIMAL_METRICS[:-1], None), ([RANGE_NAME], MINIMAL_METRICS, 1000)],
)
def test_harness_builder_rejects_incomplete_range_contract(filters, metrics, sampling) -> None:
    config = ProfilerConfig(ncu_replay_mode="app-range", pm_sampling_interval=sampling)
    with pytest.raises(ValueError, match="app-range requires"):
        config.get_ncu_command_for_target("out", ["program"], metrics=metrics, nvtx_includes=filters)


def _automation(tmp_path: Path) -> NsightAutomation:
    automation = NsightAutomation(tmp_path)
    # Isolate only installed-tool discovery; run the real repository builder.
    automation._ncu_sets_cache = {"basic", "full", "roofline"}
    return automation


def test_standalone_builder_keeps_one_exact_start_stop_filter(tmp_path: Path) -> None:
    command = _automation(tmp_path).build_ncu_command(
        command=["program"], output_path=tmp_path / "out.ncu-rep",
        metric_set="minimal", replay_mode="app-range", nvtx_includes=[RANGE_NAME],
    )
    _assert_full_range(command)
    assert f"{RANGE_NAME}/" not in command


@pytest.mark.parametrize(
    "override",
    [{"kernel_filter": "some_kernel"}, {"launch_skip": 0}, {"launch_count": 1},
     {"sampling_interval": 1000}, {"profile_from_start": "off"},
     {"metric_set": "full"}, {"nvtx_includes": []}],
)
def test_standalone_builder_rejects_partial_range_options(tmp_path: Path, override: dict) -> None:
    options = {"metric_set": "minimal", "replay_mode": "app-range", "nvtx_includes": [RANGE_NAME]}
    options.update(override)
    with pytest.raises(ValueError, match="app-range requires"):
        _automation(tmp_path).build_ncu_command(
            command=["program"], output_path=tmp_path / "out.ncu-rep", **options
        )


def test_mcp_replay_schemas_share_modes_and_preserve_defaults() -> None:
    from mcp.mcp_server import TOOLS

    for tool, field, default in (
        ("run_benchmarks", "ncu_replay_mode", "kernel"),
        ("profile_ncu", "replay_mode", "application"),
    ):
        schema = TOOLS[tool].input_schema
        option = schema["properties"][field]
        assert option["enum"] == list(NCU_REPLAY_MODES)
        assert option["default"] == default


@pytest.mark.parametrize(
    ("mode", "expected_error"),
    [("range", "Invalid Nsight Compute replay mode"),
     ("app-range", "app-range requires metric_set='minimal'")],
)
def test_cli_rejects_unsupported_capture_before_execution(tmp_path: Path, mode: str, expected_error: str) -> None:
    from cli.aisp import app
    from typer.testing import CliRunner

    result = CliRunner().invoke(
        app,
        ["profile", "ncu", "--command", "/bin/true", "--output-dir", str(tmp_path),
         "--replay-mode", mode],
    )
    assert result.exit_code != 0
    assert expected_error in result.stdout


def test_standalone_zero_exit_cannot_accept_a_stale_report(tmp_path: Path, monkeypatch) -> None:
    import os
    import subprocess

    automation = _automation(tmp_path)
    automation.ncu_available = True
    report = automation.output_dir / "stale.ncu-rep"
    report.write_bytes(b"retained incomplete capture")
    os.utime(report, ns=(1, 1))
    # Simulate only a tool's zero exit without a capture; the real acceptance
    # path must reject it. This is not a successful GPU/profiler fixture.
    monkeypatch.setattr(
        subprocess, "run",
        lambda command, **kwargs: subprocess.CompletedProcess(command, 0, stdout="", stderr=""),
    )
    result = automation.profile_ncu(
        ["program"], "stale", replay_mode="app-range", metric_set="minimal",
        nvtx_includes=[RANGE_NAME],
    )
    assert result is None
    assert "not fresh" in automation.last_error
    assert report.read_bytes() == b"retained incomplete capture"
