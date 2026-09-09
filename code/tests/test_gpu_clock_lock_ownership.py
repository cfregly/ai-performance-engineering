"""Stateful control-plane tests for nested GPU application-clock ownership."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from core.harness import benchmark_harness


@dataclass
class _FakeGpuControlPlane:
    application_sm_mhz: int = 1965
    application_memory_mhz: int = 3996
    current_sm_mhz: int = 1965
    current_memory_mhz: int = 3996
    persistence_enabled: bool = False
    fail_next_application_clock_command: bool = False
    commands: list[tuple[str, ...]] = field(default_factory=list)

    @property
    def application_clocks(self) -> tuple[int, int]:
        return self.application_sm_mhz, self.application_memory_mhz

    def nvidia_smi(self, args: list[str]) -> bytes:
        command = tuple(args)
        self.commands.append(command)
        if any(
            argument in {"-rgc", "-rmc", "-rac"}
            or argument.startswith("--lock-gpu-clocks")
            or argument.startswith("--lock-memory-clocks")
            for argument in args
        ):
            raise AssertionError(f"unrestorable hard-clock command used: {command}")
        if "-pm" in args:
            self.persistence_enabled = args[args.index("-pm") + 1] == "1"
            return b""
        application_arg = next(
            (argument for argument in args if argument.startswith("--applications-clocks=")),
            None,
        )
        if application_arg is not None:
            memory_text, sm_text = application_arg.split("=", 1)[1].split(",", 1)
            self.application_sm_mhz = int(sm_text)
            self.application_memory_mhz = int(memory_text)
            self.current_sm_mhz = int(sm_text)
            self.current_memory_mhz = int(memory_text)
            if self.fail_next_application_clock_command:
                self.fail_next_application_clock_command = False
                raise subprocess.CalledProcessError(
                    9,
                    ["nvidia-smi", *args],
                    output=b"command failed after changing application clocks",
                )
            return b""
        if any(argument.startswith("--query-gpu=clocks.max") for argument in args):
            return b"1965, 3996\n"
        raise AssertionError(f"unexpected nvidia-smi command: {command}")


@pytest.fixture
def gpu_control_plane(monkeypatch: pytest.MonkeyPatch) -> _FakeGpuControlPlane:
    control = _FakeGpuControlPlane()
    clock_sm = 0
    clock_mem = 1
    fake_pynvml = SimpleNamespace(
        NVML_CLOCK_SM=clock_sm,
        NVML_CLOCK_MEM=clock_mem,
        NVML_FEATURE_DISABLED=0,
        NVML_FEATURE_ENABLED=1,
        nvmlInit=lambda: None,
        nvmlShutdown=lambda: None,
        nvmlDeviceGetHandleByIndex=lambda index: index,
        nvmlDeviceGetApplicationsClock=lambda _handle, clock: (
            control.application_sm_mhz if clock == clock_sm else control.application_memory_mhz
        ),
        nvmlDeviceGetClockInfo=lambda _handle, clock: (
            control.current_sm_mhz if clock == clock_sm else control.current_memory_mhz
        ),
        nvmlDeviceGetPersistenceMode=lambda _handle: int(control.persistence_enabled),
    )
    monkeypatch.setitem(sys.modules, "pynvml", fake_pynvml)
    monkeypatch.setattr(benchmark_harness.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        benchmark_harness.torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(
            multi_processor_count=132,
            memory_clock_rate=1_000_000,
            memory_bus_width=512,
        ),
    )
    monkeypatch.setattr(benchmark_harness, "_resolve_physical_device_index", lambda _device: 0)
    monkeypatch.setattr(benchmark_harness, "_nvidia_smi", control.nvidia_smi)
    return control


def _application_clock_commands(control: _FakeGpuControlPlane) -> list[tuple[str, ...]]:
    return [
        command
        for command in control.commands
        if any(argument.startswith("--applications-clocks=") for argument in command)
    ]


def _persistence_commands(control: _FakeGpuControlPlane) -> list[tuple[str, ...]]:
    return [command for command in control.commands if "-pm" in command]


def test_nested_matching_clock_context_borrows_outer_state(gpu_control_plane) -> None:
    control = gpu_control_plane
    entry_clocks = control.application_clocks

    with benchmark_harness.lock_gpu_clocks(0, 1500, 3996):
        assert control.application_clocks == (1500, 3996)
        assert control.persistence_enabled
        outer_commands = list(control.commands)

        with benchmark_harness.lock_gpu_clocks(0, 1500, 3996):
            assert control.application_clocks == (1500, 3996)
            assert control.persistence_enabled

        assert control.commands == outer_commands
        assert control.application_clocks == (1500, 3996)
        assert control.persistence_enabled

    assert control.application_clocks == entry_clocks
    assert not control.persistence_enabled
    assert len(_application_clock_commands(control)) == 2
    assert _persistence_commands(control) == [
        ("-i", "0", "-pm", "1"),
        ("-i", "0", "-pm", "0"),
    ]


def test_nested_different_clock_context_restores_outer_state(gpu_control_plane) -> None:
    control = gpu_control_plane
    entry_clocks = control.application_clocks

    with benchmark_harness.lock_gpu_clocks(0, 1500, 3996):
        with benchmark_harness.lock_gpu_clocks(0, 1200, 3500):
            assert control.application_clocks == (1200, 3500)
            assert control.persistence_enabled
        assert control.application_clocks == (1500, 3996)
        assert control.persistence_enabled

    assert control.application_clocks == entry_clocks
    assert not control.persistence_enabled
    application_commands = _application_clock_commands(control)
    assert [
        next(argument for argument in command if argument.startswith("--applications-clocks="))
        for command in application_commands
    ] == [
        "--applications-clocks=3996,1500",
        "--applications-clocks=3500,1200",
        "--applications-clocks=3996,1500",
        "--applications-clocks=3996,1965",
    ]


def test_body_exception_restores_entry_clocks_and_enabled_persistence(gpu_control_plane) -> None:
    control = gpu_control_plane
    control.application_sm_mhz = 1710
    control.application_memory_mhz = 3650
    control.current_sm_mhz = 1710
    control.current_memory_mhz = 3650
    control.persistence_enabled = True

    class WorkloadError(RuntimeError):
        pass

    with (
        pytest.raises(WorkloadError, match="workload failed"),
        benchmark_harness.lock_gpu_clocks(0, 1500, 3996),
    ):
        raise WorkloadError("workload failed")

    assert control.application_clocks == (1710, 3650)
    assert control.persistence_enabled
    assert _persistence_commands(control) == []


def test_partial_application_clock_failure_restores_entry_state_without_hard_lock_fallback(
    gpu_control_plane,
) -> None:
    control = gpu_control_plane
    entry_clocks = control.application_clocks
    control.fail_next_application_clock_command = True

    with (
        pytest.raises(RuntimeError, match="hard-clock fallback is not restorable"),
        benchmark_harness.lock_gpu_clocks(0, 1500, 3996),
    ):
        pytest.fail("the workload must not run after a partial lock failure")

    assert control.application_clocks == entry_clocks
    assert not control.persistence_enabled
    assert len(_application_clock_commands(control)) == 2
    assert _persistence_commands(control)[-1] == ("-i", "0", "-pm", "0")


def test_borrowed_clock_drift_fails_without_claiming_restoration_ownership(
    gpu_control_plane,
) -> None:
    control = gpu_control_plane
    control.application_sm_mhz = 1500
    control.current_sm_mhz = 1500
    control.persistence_enabled = True

    with (
        pytest.raises(RuntimeError, match="borrowed GPU application clocks changed"),
        benchmark_harness.lock_gpu_clocks(0, 1500, 3996),
    ):
        control.application_sm_mhz = 1400
        control.current_sm_mhz = 1400

    assert control.application_clocks == (1400, 3996)
    assert control.persistence_enabled
    assert control.commands == []
