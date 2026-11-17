"""Keep explicit execution deadlines intact through runner configuration."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
import json
from pathlib import Path
import subprocess
import sys

import pytest

from core.benchmark.defaults import BenchmarkDefaults
from core.harness.benchmark_harness import BenchmarkConfig
from core.harness import run_benchmarks


EXECUTION_FIELDS = (
    "setup_timeout_seconds",
    "warmup_timeout_seconds",
    "measurement_timeout_seconds",
    "timeout_seconds",
)
PROFILE_FIELDS = (
    "profiling_timeout_seconds",
    "nsys_timeout_seconds",
    "ncu_timeout_seconds",
    "proton_timeout_seconds",
)
CODE_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class _ConfiguredBenchmark:
    """Supply a real configuration to the runner's merge path."""

    config: BenchmarkConfig

    def get_config(self) -> BenchmarkConfig:
        return self.config


def _locked(config: BenchmarkConfig, *, provided: bool) -> set[str]:
    return run_benchmarks._compute_locked_fields(
        base_config=config,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        cli_execution_timeout_provided=provided,
        enable_profiling=False,
    )


def test_execution_timeout_is_optional_on_both_runner_entrypoints() -> None:
    for function in (run_benchmarks.test_chapter, run_benchmarks._test_chapter_impl):
        assert signature(function).parameters["execution_timeout_seconds"].default is None


def test_omitted_execution_timeout_preserves_defaults_and_existing_merge_behavior() -> None:
    base = BenchmarkConfig(timeout_multiplier=1.0)
    explicit_null = BenchmarkConfig(execution_timeout_seconds=None, timeout_multiplier=1.0)
    for field in EXECUTION_FIELDS + PROFILE_FIELDS:
        assert getattr(base, field) == getattr(explicit_null, field)
    assert base.execution_timeout_seconds is None
    assert not _locked(base, provided=False).intersection(EXECUTION_FIELDS)

    override = BenchmarkConfig(
        setup_timeout_seconds=701, warmup_timeout_seconds=702,
        measurement_timeout_seconds=703, timeout_seconds=704, timeout_multiplier=1.0,
    )
    merged = run_benchmarks._merge_benchmark_config(
        base_config=base, benchmark_obj=_ConfiguredBenchmark(override),
        defaults_obj=BenchmarkDefaults(), locked_fields=_locked(base, provided=False),
    )
    assert [getattr(merged, field) for field in EXECUTION_FIELDS] == [701, 702, 703, 704]


@pytest.mark.parametrize("seconds", [0, 7, 180, 300, 1200])
@pytest.mark.parametrize("multiplier", [1.0, 3.0, 5.0])
def test_explicit_execution_timeout_survives_merge_and_json_reconstruction(
    seconds: int, multiplier: float,
) -> None:
    base = BenchmarkConfig(execution_timeout_seconds=seconds, timeout_multiplier=multiplier)
    locked = _locked(base, provided=True)
    assert set(EXECUTION_FIELDS) | {"execution_timeout_seconds"} <= locked
    assert not locked.intersection(PROFILE_FIELDS)
    override = BenchmarkConfig(
        setup_timeout_seconds=701, warmup_timeout_seconds=702,
        measurement_timeout_seconds=703, timeout_seconds=704, timeout_multiplier=1.0,
    )
    merged = run_benchmarks._merge_benchmark_config(
        base_config=base, benchmark_obj=_ConfiguredBenchmark(override),
        defaults_obj=BenchmarkDefaults(), locked_fields=locked,
    )
    payload = json.loads(json.dumps({
        field: getattr(merged, field)
        for field in EXECUTION_FIELDS + ("execution_timeout_seconds", "timeout_multiplier")
    }))
    assert payload["execution_timeout_seconds"] == seconds
    restored = BenchmarkConfig(**payload)
    for config in (base, merged, restored):
        assert config.execution_timeout_seconds == seconds
        assert [getattr(config, field) for field in EXECUTION_FIELDS] == [seconds] * 4

    inherited = BenchmarkConfig(timeout_multiplier=multiplier)
    for field in PROFILE_FIELDS:
        assert getattr(base, field) == getattr(inherited, field)
        assert getattr(restored, field) == getattr(inherited, field)


@pytest.mark.parametrize("value", [-1, True, False, 0.0, 1.5, "0", float("inf"), float("nan")])
def test_invalid_timeout_is_rejected_before_environment_or_benchmark_execution(tmp_path: Path, value) -> None:
    with pytest.raises(ValueError, match="execution_timeout_seconds"):
        run_benchmarks.test_chapter(tmp_path, execution_timeout_seconds=value)


def test_uncapped_torchrun_is_rejected_before_environment_or_benchmark_execution(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="owned subprocess"):
        run_benchmarks.test_chapter(tmp_path, execution_timeout_seconds=0, launch_via="torchrun")


@pytest.mark.parametrize("launch_via,execution_mode", [("python", "thread"), ("torchrun", "subprocess")])
def test_uncapped_operator_timeout_cannot_merge_an_unowned_execution_mode(
    launch_via: str, execution_mode: str,
) -> None:
    base = BenchmarkConfig(execution_timeout_seconds=0, timeout_multiplier=1.0)
    override = BenchmarkConfig(launch_via=launch_via, execution_mode=execution_mode, timeout_multiplier=1.0)
    with pytest.raises(ValueError, match="owned subprocess"):
        run_benchmarks._merge_benchmark_config(
            base_config=base, benchmark_obj=_ConfiguredBenchmark(override),
            defaults_obj=BenchmarkDefaults(), locked_fields=_locked(base, provided=True),
        )


def test_positive_operator_timeout_keeps_existing_thread_support() -> None:
    base = BenchmarkConfig(execution_timeout_seconds=7, timeout_multiplier=1.0)
    override = BenchmarkConfig(execution_mode="thread", timeout_multiplier=1.0)
    merged = run_benchmarks._merge_benchmark_config(
        base_config=base, benchmark_obj=_ConfiguredBenchmark(override),
        defaults_obj=BenchmarkDefaults(), locked_fields=_locked(base, provided=True),
    )
    assert merged.execution_mode == "thread"
    assert merged.measurement_timeout_seconds == 7


def test_standalone_runner_describes_explicit_execution_timeout() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "core.harness.run_benchmarks", "--help"],
        cwd=CODE_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--execution-timeout-seconds" in result.stdout
    assert "Profiling timeouts are unchanged" in " ".join(result.stdout.split())


def test_standalone_runner_rejects_negative_timeout_before_execution() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "core.harness.run_benchmarks", "--execution-timeout-seconds", "-1"],
        cwd=CODE_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "execution_timeout_seconds" in result.stderr


def test_standalone_runner_rejects_uncapped_torchrun_before_execution() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "core.harness.run_benchmarks", "--execution-timeout-seconds", "0",
         "--launch-via", "torchrun"],
        cwd=CODE_ROOT, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 2
    assert "owned subprocess" in result.stderr
