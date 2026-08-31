"""Tests for run_benchmarks config merge invariants (no mocks)."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import signature
import os
from pathlib import Path
import sys

from core.benchmark.defaults import BenchmarkDefaults
from core.harness.benchmark_harness import BenchmarkConfig, LaunchVia
from core.harness import run_benchmarks as run_benchmarks_mod
from core.harness.run_benchmarks import (
    _allow_mixed_provenance_for_expectation_writes,
    _apply_preferred_ncu_profile_overrides,
    _compute_locked_fields,
    _is_retryable_ncu_driver_resource_error,
    _merge_benchmark_config,
    _resolve_nsys_profile_mode,
    _resolve_expectation_validation_policy,
)


@dataclass
class _DummyBenchmark:
    override: BenchmarkConfig

    def get_config(self) -> BenchmarkConfig:
        return self.override


def _base_config(**overrides) -> BenchmarkConfig:
    # Keep defaults explicit for fields that interact with locking/merging.
    kwargs = dict(
        timeout_multiplier=1.0,
        enforce_environment_validation=True,
        enable_memory_tracking=True,
        enable_profiling=False,
        enable_nsys=False,
        enable_ncu=False,
        profile_type="none",
    )
    kwargs.update(overrides)
    return BenchmarkConfig(**kwargs)


def test_merge_respects_cli_iterations_and_warmup_locks() -> None:
    base = _base_config(iterations=20, warmup=5)
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=True,
        cli_warmup_provided=True,
        enable_profiling=False,
    )
    override = BenchmarkConfig(iterations=999, warmup=99, timeout_multiplier=1.0)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.iterations == base.iterations
    assert merged.warmup == base.warmup


def test_merge_locks_runner_protections_when_enabled() -> None:
    base = _base_config()
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=False,
    )
    override = BenchmarkConfig(enable_memory_tracking=False, detect_setup_precomputation=False, timeout_multiplier=1.0)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.enable_memory_tracking is True
    assert merged.detect_setup_precomputation is True


def test_merge_preserves_allow_foreign_gpu_processes_from_run_level_policy() -> None:
    base = _base_config(allow_foreign_gpu_processes=True)
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=False,
    )
    override = BenchmarkConfig(allow_foreign_gpu_processes=False, timeout_multiplier=1.0)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.allow_foreign_gpu_processes is True


def test_merge_locks_profiling_knobs_when_cli_enabled() -> None:
    base = _base_config(enable_profiling=True, enable_nsys=True, enable_ncu=True, profile_type="minimal")
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=True,
    )
    override = BenchmarkConfig(
        enable_profiling=False,
        enable_nsys=False,
        enable_ncu=False,
        enable_nvtx=False,
        profile_type="none",
        timeout_multiplier=1.0,
    )

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.enable_profiling is True
    assert merged.enable_nsys is True
    assert merged.enable_ncu is True
    assert merged.enable_nvtx is True
    assert merged.profile_type == "minimal"


def test_merge_preserves_cli_launch_via_when_benchmark_supplies_default() -> None:
    defaults_obj = BenchmarkDefaults(launch_via="python")
    base = _base_config(launch_via="torchrun")
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=False,
    )
    override = BenchmarkConfig(launch_via="python", timeout_multiplier=1.0)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=defaults_obj,
        locked_fields=locked,
    )

    assert merged.launch_via == LaunchVia.TORCHRUN


def test_merge_target_extra_args_merges_dicts() -> None:
    base = _base_config(target_extra_args={"a": ["--foo"], "b": ["--bar"]})
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=False,
    )
    override = BenchmarkConfig(target_extra_args={"b": ["--override"], "c": ["--baz"]}, timeout_multiplier=1.0)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.target_extra_args == {"a": ["--foo"], "b": ["--override"], "c": ["--baz"]}


def test_merge_env_passthrough_ignores_empty_override() -> None:
    base = _base_config(env_passthrough=["CUDA_VISIBLE_DEVICES", "FOO"])
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=False,
    )
    override = BenchmarkConfig(env_passthrough=[], timeout_multiplier=1.0)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.env_passthrough == ["CUDA_VISIBLE_DEVICES", "FOO"]


def test_merge_allows_benchmark_timeouts_to_widen_defaults() -> None:
    base = _base_config(measurement_timeout_seconds=10, setup_timeout_seconds=10)
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=False,
    )
    override = BenchmarkConfig(measurement_timeout_seconds=999, setup_timeout_seconds=999, timeout_multiplier=1.0)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.measurement_timeout_seconds == 999
    assert merged.setup_timeout_seconds == 999


def test_merge_locks_explicit_cli_profiler_timeouts() -> None:
    base = _base_config(nsys_timeout_seconds=540, ncu_timeout_seconds=7200)
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        cli_nsys_timeout_provided=True,
        cli_ncu_timeout_provided=True,
        enable_profiling=False,
    )
    override = BenchmarkConfig(nsys_timeout_seconds=1200, ncu_timeout_seconds=10800, timeout_multiplier=1.0)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.nsys_timeout_seconds == 540
    assert merged.ncu_timeout_seconds == 7200


def test_resolve_nsys_profile_mode_honors_benchmark_override() -> None:
    config = BenchmarkConfig(profile_type="deep_dive", nsys_preset_override="light")

    preset, full_timeline = _resolve_nsys_profile_mode(config)

    assert preset == "light"
    assert full_timeline is False


def test_merge_enforces_policy_invariants() -> None:
    base = _base_config(timeout_multiplier=1.0, enforce_environment_validation=True)
    locked = _compute_locked_fields(
        base_config=base,
        cli_iterations_provided=False,
        cli_warmup_provided=False,
        enable_profiling=False,
    )
    override = BenchmarkConfig(timeout_multiplier=5.0, enforce_environment_validation=False)

    merged = _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_DummyBenchmark(override),
        defaults_obj=BenchmarkDefaults(),
        locked_fields=locked,
    )

    assert merged.timeout_multiplier == base.timeout_multiplier
    assert merged.enforce_environment_validation == base.enforce_environment_validation


def test_run_benchmarks_entrypoints_leave_ncu_replay_mode_unset_by_default() -> None:
    impl_param = signature(run_benchmarks_mod._test_chapter_impl).parameters["ncu_replay_mode"]
    public_param = signature(run_benchmarks_mod.test_chapter).parameters["ncu_replay_mode"]

    assert impl_param.default is None
    assert public_param.default is None


def test_apply_preferred_ncu_profile_overrides_honors_benchmark_locals() -> None:
    config = BenchmarkConfig(ncu_metric_set="deep_dive", ncu_replay_mode="application")

    class _Benchmark:
        preferred_ncu_replay_mode = "kernel"
        preferred_ncu_metric_set = "minimal"

    updated = _apply_preferred_ncu_profile_overrides(config, _Benchmark())

    assert updated.ncu_replay_mode == "kernel"
    assert updated.ncu_replay_mode_override is True
    assert updated.ncu_metric_set == "minimal"


def test_harden_profile_env_prepends_startup_stub() -> None:
    env = run_benchmarks_mod._harden_profile_env({}, repo_root=Path.cwd(), chapter_dir=Path.cwd() / "labs")
    pythonpath_entries = [entry for entry in env["PYTHONPATH"].split(os.pathsep) if entry]

    assert pythonpath_entries
    assert pythonpath_entries[0].endswith("aisp_profile_python_startup")
    assert (Path(pythonpath_entries[0]) / "sitecustomize.py").exists()
    assert (Path(pythonpath_entries[0]) / "usercustomize.py").exists()


def test_run_repo_python_module_uses_module_entrypoint_from_repo_root(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(run_benchmarks_mod.subprocess, "run", _fake_run)

    run_benchmarks_mod._run_repo_python_module(
        "scripts.utilities.dump_hardware_capabilities",
        "--fast",
        timeout=15,
    )

    assert captured["cmd"] == [
        sys.executable,
        "-m",
        "scripts.utilities.dump_hardware_capabilities",
        "--fast",
    ]
    kwargs = captured["kwargs"]
    assert kwargs["cwd"] == str(run_benchmarks_mod.repo_root)
    assert kwargs["check"] is True
    assert kwargs["timeout"] == 15
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is True
    pythonpath_entries = [entry for entry in kwargs["env"]["PYTHONPATH"].split(os.pathsep) if entry]
    assert pythonpath_entries
    assert Path(pythonpath_entries[0]) == run_benchmarks_mod.repo_root.resolve()


def test_reset_parent_execution_state_uses_launch_mode_guards(monkeypatch) -> None:
    calls: list[tuple[str, object]] = []

    def _fake_reset_cuda_state(*, allow_cuda_context: bool) -> None:
        calls.append(("cuda", allow_cuda_context))

    def _fake_reset_gpu_state() -> None:
        calls.append(("gpu", None))

    monkeypatch.setattr(run_benchmarks_mod, "reset_cuda_state", _fake_reset_cuda_state)
    monkeypatch.setattr(run_benchmarks_mod, "reset_gpu_state", _fake_reset_gpu_state)

    run_benchmarks_mod._reset_parent_execution_state(
        launch_via="python",
        cold_start=False,
        include_gpu_state=True,
    )
    assert calls == [("cuda", False)]

    calls.clear()
    run_benchmarks_mod._reset_parent_execution_state(
        launch_via="torchrun",
        cold_start=True,
        include_gpu_state=True,
    )
    assert calls == [("cuda", True), ("gpu", None), ("gpu", None)]


def test_repo_relative_path_handles_relative_artifact_paths() -> None:
    repo_root = Path.cwd()
    artifact_path = Path("artifacts/runs/example/profiles/test.nsys-rep")

    relative_path = run_benchmarks_mod._repo_relative_path(artifact_path, repo_root)

    assert relative_path == artifact_path.as_posix()


def test_repo_relative_path_handles_absolute_artifact_paths() -> None:
    repo_root = Path.cwd()
    artifact_path = repo_root / "artifacts/runs/example/profiles/test.nsys-rep"

    relative_path = run_benchmarks_mod._repo_relative_path(artifact_path, repo_root)

    assert relative_path == "artifacts/runs/example/profiles/test.nsys-rep"


def test_resolve_profile_output_dir_absolutizes_relative_paths() -> None:
    relative_output_dir = Path("artifacts/runs/example/profiles")

    resolved = run_benchmarks_mod._resolve_profile_output_dir(relative_output_dir)

    assert resolved == (Path.cwd() / relative_output_dir).resolve()


def test_expectation_policy_strict_without_write_flags_is_preview_only() -> None:
    validation_enabled, writes_enabled = _resolve_expectation_validation_policy(
        validity_profile="strict",
        update_expectations=False,
        accept_regressions=False,
        allow_mixed_provenance=False,
        allow_portable_expectations_update=False,
    )

    assert validation_enabled is True
    assert writes_enabled is False


def test_expectation_policy_strict_with_update_flag_enables_writes() -> None:
    validation_enabled, writes_enabled = _resolve_expectation_validation_policy(
        validity_profile="strict",
        update_expectations=True,
        accept_regressions=False,
        allow_mixed_provenance=False,
        allow_portable_expectations_update=False,
    )

    assert validation_enabled is True
    assert writes_enabled is True


def test_expectation_policy_portable_requires_explicit_override_for_validation() -> None:
    validation_enabled, writes_enabled = _resolve_expectation_validation_policy(
        validity_profile="portable",
        update_expectations=True,
        accept_regressions=False,
        allow_mixed_provenance=False,
        allow_portable_expectations_update=False,
    )

    assert validation_enabled is False
    assert writes_enabled is False


def test_update_expectations_allows_mixed_provenance_writes() -> None:
    assert (
        _allow_mixed_provenance_for_expectation_writes(
            update_expectations=True,
            allow_mixed_provenance=False,
        )
        is True
    )


def test_explicit_mixed_provenance_flag_allows_writes() -> None:
    assert (
        _allow_mixed_provenance_for_expectation_writes(
            update_expectations=False,
            allow_mixed_provenance=True,
        )
        is True
    )


def test_preview_mode_keeps_mixed_provenance_blocked() -> None:
    assert (
        _allow_mixed_provenance_for_expectation_writes(
            update_expectations=False,
            allow_mixed_provenance=False,
        )
        is False
    )


def test_ncu_driver_resource_unavailable_is_retryable(tmp_path: Path) -> None:
    stdout_log = tmp_path / "ncu.stdout.log"
    stderr_log = tmp_path / "ncu.stderr.log"
    stdout_log.write_text(
        "==ERROR== Profiling failed because a driver resource was unavailable. "
        "Ensure that no other tool (like DCGM) is concurrently collecting profiling data.\n",
        encoding="utf-8",
    )
    stderr_log.write_text("", encoding="utf-8")

    assert _is_retryable_ncu_driver_resource_error(
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    ) is True


def test_unrelated_ncu_failure_is_not_retryable(tmp_path: Path) -> None:
    stdout_log = tmp_path / "ncu.stdout.log"
    stderr_log = tmp_path / "ncu.stderr.log"
    stdout_log.write_text("==ERROR== Failed to profile kernel due to invalid argument\n", encoding="utf-8")
    stderr_log.write_text("", encoding="utf-8")

    assert _is_retryable_ncu_driver_resource_error(
        stdout_log=stdout_log,
        stderr_log=stderr_log,
    ) is False
