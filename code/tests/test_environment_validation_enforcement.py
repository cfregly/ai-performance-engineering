"""Harness-level enforcement tests for validate_environment().

These tests use a synthetic /proc + /sys snapshot via EnvironmentProbe and verify
that BenchmarkHarness FAILS for chapter/lab benchmarks when the environment is
invalid (i.e., validate_environment() returns errors).
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from core.harness import validity_checks
from core.benchmark.models import BenchmarkResult as PydanticBenchmarkResult, TimingStats
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    ExecutionMode,
    _format_environment_invalid_message,
)
from core.harness.validity_checks import (
    EnvironmentProbe,
    EnvironmentValidationResult,
    validate_environment,
)


@pytest.fixture(autouse=True)
def _use_synthetic_linux_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(validity_checks, "sys", SimpleNamespace(platform="linux"))


def _write_file(root: Path, relpath: str, content: str) -> None:
    path = root / relpath.lstrip("/")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _make_base_env(root: Path, *, governor: str = "performance") -> None:
    _write_file(root, "/proc/swaps", "Filename\tType\tSize\tUsed\tPriority\n")
    _write_file(root, "/proc/sys/vm/swappiness", "0\n")
    _write_file(root, "/proc/cpuinfo", "processor\t: 0\n")
    _write_file(root, "/sys/devices/virtual/dmi/id/product_name", "BareMetal\n")
    _write_file(root, "/sys/devices/system/node/node0/cpulist", "0-3\n")
    _write_file(root, "/sys/devices/system/cpu/cpufreq/policy0/scaling_governor", governor + "\n")


def _load_ch_fake_benchmark(module_dir: Path) -> object:
    """Load a benchmark class from a file path containing '/ch' so the harness enforces."""
    module_path = module_dir / "ch_fake_env_bench.py"
    module_path.write_text(
        textwrap.dedent(
            """
            import torch
            from core.harness.benchmark_harness import BaseBenchmark

            class EnvBench(BaseBenchmark):
                allow_cpu = True

                def __init__(self):
                    super().__init__()
                    self.x = None

                def setup(self) -> None:
                    self.x = torch.ones(1, device=self.device)

                def benchmark_fn(self) -> None:
                    self.x.add_(1)
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    spec = importlib.util.spec_from_file_location("ch_fake_env_bench", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.EnvBench()


def _run_harness(env_root: Path, *, probe: EnvironmentProbe, allow_virtualization: Optional[bool] = None) -> list[str]:
    bench_dir = env_root / "bench_mod"
    bench_dir.mkdir(parents=True, exist_ok=True)
    bench = _load_ch_fake_benchmark(bench_dir)
    harness = BenchmarkHarness(environment_probe=probe)
    config_kwargs = dict(iterations=1, warmup=5, use_subprocess=False)
    if allow_virtualization is not None:
        config_kwargs["allow_virtualization"] = allow_virtualization
    config = BenchmarkConfig(**config_kwargs)
    result = harness._benchmark_with_threading(bench, config)
    return list(result.errors)


class _CpuOnlyBenchmark(BaseBenchmark):
    allow_cpu = True

    def setup(self) -> None:
        return None

    def benchmark_fn(self) -> None:
        return None


def _valid_environment_result() -> EnvironmentValidationResult:
    return EnvironmentValidationResult(
        is_valid=True,
        errors=[],
        warnings=[],
        details={},
        notices=[],
    )


def test_environment_invalid_message_does_not_offer_ineffective_portable_recovery() -> None:
    message = _format_environment_invalid_message(
        ["Non-Linux platform 'darwin' is not supported for benchmark validity checks."]
    )

    assert message == (
        "ENVIRONMENT INVALID: Non-Linux platform 'darwin' is not supported for benchmark validity checks."
    )
    assert "--validity-profile portable" not in message
    assert "--allow-portable-expectations-update" not in message


def _make_subprocess_result(*, seeds: Optional[dict] = None) -> PydanticBenchmarkResult:
    return PydanticBenchmarkResult(
        timing=TimingStats(
            mean_ms=1.0,
            median_ms=1.0,
            std_ms=0.0,
            min_ms=1.0,
            max_ms=1.0,
            p50_ms=1.0,
            p90_ms=1.0,
            p95_ms=1.0,
            p99_ms=1.0,
            percentiles={50.0: 1.0},
            iterations=1,
            warmup_iterations=0,
            raw_times_ms=[1.0],
            schemaVersion="1.0",
        ),
        errors=[],
        benchmark_name="cpu-only-bench",
        device="cuda",
        mode="custom",
        seeds=seeds,
        schemaVersion="1.0",
    )


def test_environment_enforcement_cpu_governor_mismatch() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root, governor="powersave")
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}))
        assert any("ENVIRONMENT INVALID" in e and "CPU governor mismatch" in e for e in errors), errors


def test_environment_enforcement_cgroup_cpu_quota() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/proc/self/cgroup", "0::/test.slice\n")
        _write_file(env_root, "/sys/fs/cgroup/test.slice/cpu.max", "100000 100000\n")
        _write_file(env_root, "/sys/fs/cgroup/test.slice/memory.max", "max\n")
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}))
        assert any("ENVIRONMENT INVALID" in e and "cpu.max" in e for e in errors), errors


def test_environment_enforcement_cgroup_memory_limit() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/proc/self/cgroup", "0::/test.slice\n")
        _write_file(env_root, "/sys/fs/cgroup/test.slice/cpu.max", "max 100000\n")
        _write_file(env_root, "/sys/fs/cgroup/test.slice/memory.max", "1073741824\n")
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}))
        assert any("ENVIRONMENT INVALID" in e and "memory.max" in e for e in errors), errors


def test_environment_enforcement_numa_affinity_spans_nodes() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/sys/devices/system/node/node0/cpulist", "0-1\n")
        _write_file(env_root, "/sys/devices/system/node/node1/cpulist", "2-3\n")
        probe = EnvironmentProbe(root=env_root, env={}, cpu_affinity={0, 2})
        result = validate_environment(probe=probe)
        assert result.is_valid, result
        assert any("CPU affinity spans multiple NUMA nodes" in w for w in result.warnings), result.warnings


def test_environment_enforcement_swap_enabled() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(
            env_root,
            "/proc/swaps",
            "Filename\tType\tSize\tUsed\tPriority\n/swapfile\tfile\t1024\t0\t-2\n",
        )
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}))
        assert any("ENVIRONMENT INVALID" in e and "Swap is enabled" in e for e in errors), errors


def test_environment_virtualization_strict_is_warning_only() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/proc/cpuinfo", "processor\t: 0\nflags\t: hypervisor\n")
        errors = _run_harness(env_root, probe=EnvironmentProbe(root=env_root, env={}), allow_virtualization=False)
        assert not any("ENVIRONMENT INVALID" in e and "Virtualization detected" in e for e in errors), errors


def test_environment_warning_virtualization_portable_message() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/proc/cpuinfo", "processor\t: 0\nflags\t: hypervisor\n")
        result = validate_environment(
            probe=EnvironmentProbe(root=env_root, env={}),
            allow_virtualization=True,
        )
        assert result.is_valid, result
        assert not result.errors, result.errors
        assert any("validity_profile=portable" in warning for warning in result.warnings), result.warnings


def test_environment_warning_virtualization_strict_loud_message() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        _write_file(env_root, "/proc/cpuinfo", "processor\t: 0\nflags\t: hypervisor\n")
        result = validate_environment(
            probe=EnvironmentProbe(root=env_root, env={}),
            allow_virtualization=False,
        )
        assert result.is_valid, result
        assert not result.errors, result.errors
        assert any("STRICT VALIDITY NOTICE [VIRTUALIZATION]" in notice for notice in result.notices), result.notices


def test_environment_probe_errors_are_reported_in_details() -> None:
    with tempfile.TemporaryDirectory() as env_dir:
        env_root = Path(env_dir)
        _make_base_env(env_root)
        swappiness_path = env_root / "proc" / "sys" / "vm" / "swappiness"
        swappiness_path.unlink()
        swappiness_path.mkdir(parents=True, exist_ok=True)

        result = validate_environment(probe=EnvironmentProbe(root=env_root, env={}))

        assert result.is_valid, result
        assert any("Environment probe encountered" in warning for warning in result.warnings), result.warnings
        assert any("swappiness" in error for error in result.details.get("probe_errors", [])), result.details


def test_environment_enforcement_foreign_gpu_processes_fail_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 4321, "process_name": "vllm", "used_memory_mb": 32768.0}],
            None,
        ),
    )
    monkeypatch.setattr(
        validity_checks,
        "_pid_is_live_process",
        lambda pid, proc_root=Path("/proc"): True,
    )

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={"AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0"},
        ),
    )
    assert any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors


def test_environment_enforcement_foreign_gpu_processes_allow_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 9876, "process_name": "vllm", "used_memory_mb": 12288.0}],
            None,
        ),
    )
    monkeypatch.setattr(
        validity_checks,
        "_pid_is_live_process",
        lambda pid, proc_root=Path("/proc"): True,
    )

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={"AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0"},
        ),
        allow_foreign_gpu_processes=True,
    )
    assert not any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors
    assert any("Foreign CUDA compute process(es) detected on benchmark GPU" in warning for warning in result.warnings), result.warnings


def test_environment_enforcement_foreign_gpu_processes_ignore_owned_descendants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid()), 1111},
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 1111, "process_name": "VLLM::EngineCore", "used_memory_mb": 6114.0}],
            None,
        ),
    )

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={"AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0"},
        ),
    )
    assert not any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors
    assert result.details.get("foreign_cuda_compute_processes") == [], result.details


def test_environment_enforcement_foreign_gpu_processes_ignore_owned_ancestors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid()), 2222},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 2222, "process_name": "python", "used_memory_mb": 612.0}],
            None,
        ),
    )

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={"AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0"},
        ),
    )
    assert not any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors
    assert result.details.get("foreign_cuda_compute_processes") == [], result.details


def test_environment_enforcement_foreign_gpu_processes_ignore_dead_pids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 3333, "process_name": "python", "used_memory_mb": 2048.0}],
            None,
        ),
    )
    monkeypatch.setattr(
        validity_checks,
        "_pid_is_live_process",
        lambda pid, proc_root=Path("/proc"): False,
    )

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={"AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0"},
        ),
    )
    assert not any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors
    assert result.details.get("foreign_cuda_compute_processes") == [], result.details


def test_environment_enforcement_foreign_gpu_processes_ignore_transient_recheck_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )

    calls = {"count": 0}

    def _fake_list_foreign_cuda_compute_processes(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return ([{"pid": 3333, "process_name": "python", "used_memory_mb": 2048.0}], None)
        return ([], None)

    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        _fake_list_foreign_cuda_compute_processes,
    )
    monkeypatch.setattr(
        validity_checks,
        "_pid_is_live_process",
        lambda pid, proc_root=Path("/proc"): True,
    )
    monkeypatch.setattr(validity_checks.time, "sleep", lambda _seconds: None)

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={"AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0"},
        ),
    )

    assert calls["count"] == 2
    assert not any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors
    assert result.details.get("foreign_cuda_compute_processes") == [], result.details
    assert result.details.get("foreign_cuda_compute_processes_reconfirmed") is True, result.details
    assert result.details.get("transient_foreign_cuda_compute_process_pids") == [3333], result.details


def test_environment_enforcement_foreign_gpu_processes_ignore_same_run_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 4444, "process_name": "python", "used_memory_mb": 2048.0}],
            None,
        ),
    )
    monkeypatch.setattr(
        validity_checks,
        "_read_process_environ_value",
        lambda pid, key, proc_root=Path("/proc"): "tier1-owner" if int(pid) == 4444 else None,
    )
    monkeypatch.setattr(
        validity_checks,
        "_pid_is_live_process",
        lambda pid, proc_root=Path("/proc"): True,
    )

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={
                "AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0",
                "AISP_BENCHMARK_OWNER_RUN_ID": "tier1-owner",
            },
        ),
    )
    assert not any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors
    assert result.details.get("foreign_cuda_compute_processes") == [], result.details


def test_environment_enforcement_foreign_gpu_processes_ignore_same_owner_pid_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 5555, "process_name": "python", "used_memory_mb": 2048.0}],
            None,
        ),
    )
    monkeypatch.setattr(
        validity_checks,
        "_read_process_environ_value",
        lambda pid, key, proc_root=Path("/proc"): "4242" if int(pid) == 5555 and key == "AISP_BENCHMARK_OWNER_PID" else None,
    )
    monkeypatch.setattr(
        validity_checks,
        "_pid_is_live_process",
        lambda pid, proc_root=Path("/proc"): True,
    )

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={
                "AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0",
                "AISP_BENCHMARK_OWNER_PID": "4242",
            },
        ),
    )
    assert not any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors
    assert result.details.get("foreign_cuda_compute_processes") == [], result.details


def test_environment_enforcement_foreign_gpu_processes_ignore_same_owner_pid_process_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)

    current_pid = int(os.getpid())

    def _fake_collect_process_tree_pids(pid: int, proc_root: Path = Path("/proc")) -> set[int]:
        if int(pid) == current_pid:
            return {current_pid}
        if int(pid) == 4242:
            return {4242, 5555}
        return {int(pid)}

    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        _fake_collect_process_tree_pids,
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda pid, proc_root=Path("/proc"): {int(pid)},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 5555, "process_name": "python", "used_memory_mb": 2048.0}],
            None,
        ),
    )
    monkeypatch.setattr(
        validity_checks,
        "_read_process_environ_value",
        lambda pid, key, proc_root=Path("/proc"): None,
    )
    monkeypatch.setattr(
        validity_checks,
        "_pid_is_live_process",
        lambda pid, proc_root=Path("/proc"): True,
    )

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={
                "AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0",
                "AISP_BENCHMARK_OWNER_PID": "4242",
            },
        ),
    )
    assert not any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors
    assert result.details.get("foreign_cuda_compute_processes") == [], result.details


def test_environment_enforcement_foreign_gpu_processes_ignore_same_owner_pid_from_live_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)

    current_pid = int(os.getpid())

    def _fake_collect_process_tree_pids(pid: int, proc_root: Path = Path("/proc")) -> set[int]:
        if int(pid) == current_pid:
            return {current_pid}
        if int(pid) == 4242:
            return {4242, 7777}
        return {int(pid)}

    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        _fake_collect_process_tree_pids,
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda pid, proc_root=Path("/proc"): {int(pid)},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 7777, "process_name": "python", "used_memory_mb": 2048.0}],
            None,
        ),
    )
    monkeypatch.setattr(
        validity_checks,
        "_read_process_environ_value",
        lambda pid, key, proc_root=Path("/proc"): None,
    )
    monkeypatch.setattr(
        validity_checks,
        "_pid_is_live_process",
        lambda pid, proc_root=Path("/proc"): True,
    )
    monkeypatch.setenv("AISP_BENCHMARK_OWNER_PID", "4242")

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={
                "AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0",
            },
        ),
    )
    assert not any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors
    assert result.details.get("foreign_cuda_compute_processes") == [], result.details
    assert result.details.get("owned_benchmark_owner_pid") == "4242", result.details


def test_environment_enforcement_foreign_gpu_processes_ignore_same_owner_pid_cmdline_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(validity_checks.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(validity_checks.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_tree_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_collect_process_lineage_pids",
        lambda _pid, proc_root=Path("/proc"): {int(os.getpid())},
    )
    monkeypatch.setattr(
        validity_checks,
        "_list_foreign_cuda_compute_processes",
        lambda **kwargs: (
            [{"pid": 6666, "process_name": "python", "used_memory_mb": 2048.0}],
            None,
        ),
    )
    monkeypatch.setattr(
        validity_checks,
        "_read_process_environ_value",
        lambda pid, key, proc_root=Path("/proc"): None,
    )
    monkeypatch.setattr(
        validity_checks,
        "_read_process_cmdline_arg_value",
        lambda pid, flag, proc_root=Path("/proc"): "4242" if int(pid) == 6666 and flag == "--aisp-owner-pid" else None,
    )
    monkeypatch.setattr(
        validity_checks,
        "_pid_is_live_process",
        lambda pid, proc_root=Path("/proc"): True,
    )

    result = validate_environment(
        device=torch.device("cuda"),
        probe=EnvironmentProbe(
            root=Path("/"),
            env={
                "AISP_FOREIGN_GPU_PROCESS_MIN_MB": "0",
                "AISP_BENCHMARK_OWNER_PID": "4242",
            },
        ),
    )
    assert not any("Foreign CUDA compute process(es) detected on benchmark GPU" in err for err in result.errors), result.errors
    assert result.details.get("foreign_cuda_compute_processes") == [], result.details


def test_benchmark_subprocess_mode_defers_parent_foreign_gpu_process_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    env_result = EnvironmentValidationResult(
        is_valid=False,
        errors=[
            "Foreign CUDA compute process(es) detected on benchmark GPU before run: "
            "pid=4321 name=python mem=612.0MiB. Threshold=0MiB. "
            "Stop concurrent GPU workloads (for example vLLM serve/bench) before strict benchmarking."
        ],
        warnings=[],
        details={},
        notices=[],
    )
    monkeypatch.setattr(validity_checks, "validate_environment", lambda **_: env_result)

    harness = BenchmarkHarness()
    benchmark = _CpuOnlyBenchmark()
    config = BenchmarkConfig(
        execution_mode=ExecutionMode.SUBPROCESS,
        enforce_environment_validation=True,
    )

    calls: dict[str, bool] = {"subprocess": False}

    def _fake_subprocess(self, bench, cfg):
        calls["subprocess"] = True
        return "subprocess-ok"

    monkeypatch.setattr(BenchmarkHarness, "_benchmark_with_subprocess", _fake_subprocess)

    result = harness.benchmark(benchmark, config)

    assert result == "subprocess-ok"
    assert calls["subprocess"] is True


def test_benchmark_subprocess_mode_skips_parent_cuda_seed_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    monkeypatch.setattr(validity_checks, "validate_environment", lambda **_: _valid_environment_result())
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "manual_seed_all",
        lambda _seed: (_ for _ in ()).throw(AssertionError("parent should not seed CUDA in subprocess mode")),
    )
    monkeypatch.setattr(
        torch.cuda,
        "initial_seed",
        lambda: (_ for _ in ()).throw(AssertionError("parent should not read CUDA seed in subprocess mode")),
    )

    harness = BenchmarkHarness()
    benchmark = _CpuOnlyBenchmark()
    config = BenchmarkConfig(
        execution_mode=ExecutionMode.SUBPROCESS,
        enforce_environment_validation=True,
        seed=123,
    )

    monkeypatch.setattr(
        BenchmarkHarness,
        "_benchmark_with_subprocess",
        lambda self, bench, cfg: _make_subprocess_result(),
    )

    result = harness.benchmark(benchmark, config)

    assert result.device == "cuda"
    assert harness._seed_info is None


def test_benchmark_subprocess_mode_does_not_start_parent_gpu_memory_logger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks
    import core.harness.benchmark_harness as benchmark_harness_mod

    monkeypatch.setattr(validity_checks, "validate_environment", lambda **_: _valid_environment_result())

    class _ForbiddenGpuMemoryLogger:
        def __init__(self, *args, **kwargs) -> None:
            raise AssertionError("parent should not start GPU memory logging in subprocess mode")

    monkeypatch.setattr(benchmark_harness_mod, "GpuMemoryLogger", _ForbiddenGpuMemoryLogger)
    monkeypatch.setattr(
        BenchmarkHarness,
        "_benchmark_with_subprocess",
        lambda self, bench, cfg: _make_subprocess_result(),
    )

    harness = BenchmarkHarness()
    benchmark = _CpuOnlyBenchmark()
    config = BenchmarkConfig(
        execution_mode=ExecutionMode.SUBPROCESS,
        enforce_environment_validation=True,
        enable_gpu_memory_logging=True,
    )

    result = harness.benchmark(benchmark, config)

    assert result.device == "cuda"


def test_benchmark_with_manifest_subprocess_mode_uses_child_seed_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    expected_seeds = {
        "random_seed": 11,
        "numpy_seed": 11,
        "torch_seed": 22,
        "cuda_seed": 33,
        "deterministic_mode": True,
    }
    monkeypatch.setattr(validity_checks, "validate_environment", lambda **_: _valid_environment_result())
    monkeypatch.setattr(
        BenchmarkHarness,
        "_setup_reproducibility",
        lambda self: (_ for _ in ()).throw(AssertionError("parent runtime init should remain deferred")),
    )
    monkeypatch.setattr(
        BenchmarkHarness,
        "_benchmark_with_subprocess",
        lambda self, bench, cfg: _make_subprocess_result(seeds=expected_seeds),
    )

    harness = BenchmarkHarness()
    benchmark = _CpuOnlyBenchmark()
    harness.config = BenchmarkConfig(
        execution_mode=ExecutionMode.SUBPROCESS,
        enforce_environment_validation=True,
    )

    run = harness.benchmark_with_manifest(benchmark, run_id="unit-test-run")

    assert run.manifest is not None
    assert run.manifest.seeds is not None
    assert run.manifest.seeds.random_seed == expected_seeds["random_seed"]
    assert run.manifest.seeds.numpy_seed == expected_seeds["numpy_seed"]
    assert run.manifest.seeds.torch_seed == expected_seeds["torch_seed"]
    assert run.manifest.seeds.cuda_seed == expected_seeds["cuda_seed"]
    assert run.manifest.seeds.deterministic_mode is True


def test_benchmark_subprocess_mode_still_fails_non_foreign_environment_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import core.harness.validity_checks as validity_checks

    env_result = EnvironmentValidationResult(
        is_valid=False,
        errors=[
            "Foreign CUDA compute process(es) detected on benchmark GPU before run: "
            "pid=4321 name=python mem=612.0MiB. Threshold=0MiB. "
            "Stop concurrent GPU workloads (for example vLLM serve/bench) before strict benchmarking.",
            "Swap is enabled; disable swap for canonical benchmark runs.",
        ],
        warnings=[],
        details={},
        notices=[],
    )
    monkeypatch.setattr(validity_checks, "validate_environment", lambda **_: env_result)

    harness = BenchmarkHarness()
    benchmark = _CpuOnlyBenchmark()
    config = BenchmarkConfig(
        execution_mode=ExecutionMode.SUBPROCESS,
        enforce_environment_validation=True,
    )

    with pytest.raises(RuntimeError, match="Swap is enabled"):
        harness.benchmark(benchmark, config)
