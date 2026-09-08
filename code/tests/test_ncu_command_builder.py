from pathlib import Path

import pytest

from core.profiling.nsight_automation import NsightAutomation
from core.profiling.profiler_config import MINIMAL_METRICS


def _build_cmd(**kwargs) -> list[str]:
    automation = NsightAutomation(Path("artifacts/runs"))
    return automation.build_ncu_command(
        command=["/bin/true"],
        output_path=Path("out.ncu-rep"),
        **kwargs,
    )


@pytest.mark.parametrize("replay_mode", ["kernel", "application"])
def test_ncu_command_minimal_uses_exact_metrics_without_extra_sections(replay_mode):
    cmd = _build_cmd(metric_set="minimal", replay_mode=replay_mode)
    assert "--set" not in cmd
    assert cmd[cmd.index("--metrics") + 1].split(",") == MINIMAL_METRICS


def test_explicit_basic_still_selects_nvidia_sections():
    cmd = _build_cmd(metric_set="basic")
    assert cmd[cmd.index("--set") + 1] in {"basic", "speed-of-light"}
    assert "--metrics" not in cmd


def test_ncu_command_full_includes_metrics():
    cmd = _build_cmd(metric_set="full", workload_type="memory_bound")
    set_idx = cmd.index("--set")
    assert cmd[set_idx + 1] == "full"
    metrics_idx = cmd.index("--metrics")
    metrics = cmd[metrics_idx + 1]
    assert "dram__bytes_read.sum" in metrics


def test_ncu_command_roofline_skips_custom_metrics():
    cmd = _build_cmd(metric_set="roofline")
    set_idx = cmd.index("--set")
    assert cmd[set_idx + 1] == "roofline"
    assert "--metrics" not in cmd


def test_ncu_command_kernel_filter_does_not_auto_limit():
    cmd = _build_cmd(kernel_filter="simple_warp_specialized_kernel")
    assert "--kernel-name" in cmd
    assert "--launch-skip" not in cmd
    assert "--launch-count" not in cmd


def test_ncu_command_explicit_launch_limits():
    cmd = _build_cmd(launch_skip=5, launch_count=2)
    skip_idx = cmd.index("--launch-skip")
    count_idx = cmd.index("--launch-count")
    assert cmd[skip_idx + 1] == "5"
    assert cmd[count_idx + 1] == "2"


def test_ncu_command_replay_mode_kernel():
    cmd = _build_cmd(replay_mode="kernel")
    replay_idx = cmd.index("--replay-mode")
    assert cmd[replay_idx + 1] == "kernel"


def test_ncu_command_invalid_metric_set_raises():
    automation = NsightAutomation(Path("artifacts/runs"))
    with pytest.raises(ValueError):
        automation.build_ncu_command(
            command=["/bin/true"],
            output_path=Path("out.ncu-rep"),
            metric_set="bogus",
        )


def test_ncu_command_invalid_workload_type_raises():
    automation = NsightAutomation(Path("artifacts/runs"))
    with pytest.raises(ValueError):
        automation.build_ncu_command(
            command=["/bin/true"],
            output_path=Path("out.ncu-rep"),
            workload_type="not_a_real_type",
        )
