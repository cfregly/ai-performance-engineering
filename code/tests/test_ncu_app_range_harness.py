"""CPU control-plane tests for NCU app-range acceptance and provenance.

These tests exercise command selection, report binding, and result plumbing. They
do not represent a successful GPU or profiler capture.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.harness import run_benchmarks
from core.harness.benchmark_harness import BenchmarkConfig
from core.profiling.profiler_config import MINIMAL_METRICS


class _RangeMetrics:
    def to_dict(self) -> dict[str, float]:
        return {
            "ncu_range_time_ms": 12.5,
            "ncu_sm_throughput_pct": 61.0,
            "ncu_dram_throughput_pct": 42.0,
            "ncu_l2_throughput_pct": 31.0,
            "ncu_occupancy_pct": 73.0,
        }


def _provenance(report: Path) -> dict[str, object]:
    digest = hashlib.sha256(report.read_bytes()).hexdigest()
    return {
        "schema_version": "1.0",
        "report_sha256": digest,
        "report_bytes": report.stat().st_size,
        "replay_mode": "app-range",
        "result_scope": "range",
        "result_id": "fixture-result",
        "result_count": 1,
        "nvtx_range": "compute_kernel:profile",
        "nvtx_range_raw": "compute_kernel:profile",
        "coverage_policy": "full_selected_nvtx_range",
        "constituent_kernels_enumerated": False,
        "duration_semantics": "ncu_aggregate_range",
        "requested_metrics": list(MINIMAL_METRICS),
        "observed_metrics": list(MINIMAL_METRICS),
        "metric_units": {},
        "session_capture": {
            "replay_mode": "app-range",
            "nvtx_enabled": True,
            "nvtx_includes": ["compute_kernel:profile"],
            "target_processes": "all",
            "metrics": list(MINIMAL_METRICS),
            "limiting_flags": [],
            "profile_from_start": "default",
            "command_sha256": "1" * 64,
        },
    }


def _valid_command() -> list[str]:
    return [
        "ncu",
        "--metrics",
        ",".join(MINIMAL_METRICS),
        "--replay-mode",
        "app-range",
        "--target-processes",
        "all",
        "--nvtx",
        "--nvtx-include",
        "compute_kernel:profile",
        "-o",
        "/tmp/report",
        "python",
        "workload.py",
    ]


def test_app_range_sidecar_binds_report_and_maps_range_metrics(tmp_path: Path) -> None:
    report = tmp_path / "capture.ncu-rep"
    report.write_bytes(b"fixture NCU report")

    capture = run_benchmarks._materialize_ncu_app_range_capture(
        report,
        _RangeMetrics(),
        _provenance(report),
    )

    sidecar = Path(capture["sidecar_path"])
    stored = json.loads(sidecar.read_text(encoding="utf-8"))
    assert stored["report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert stored["report_path"] == report.name
    assert stored["result_scope"] == "range"
    assert stored["constituent_kernels_enumerated"] is False
    assert stored["duration_semantics"] == "ncu_aggregate_range"
    assert stored["metrics"]["ncu_range_time_ms"] == 12.5
    assert "ncu_kernel_time_ms" not in stored["metrics"]
    assert run_benchmarks._ncu_metrics_for_result(report, capture) == {
        "range_time_ms": 12.5,
        "sm_throughput_percent": 61.0,
        "dram_throughput_percent": 42.0,
        "l2_throughput_percent": 31.0,
        "occupancy": 73.0,
    }


def test_app_range_sidecar_rejects_report_hash_mismatch(tmp_path: Path) -> None:
    report = tmp_path / "capture.ncu-rep"
    report.write_bytes(b"fixture NCU report")
    provenance = _provenance(report)
    provenance["report_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="report_sha256 does not match"):
        run_benchmarks._materialize_ncu_app_range_capture(
            report,
            _RangeMetrics(),
            provenance,
        )

    assert not report.with_name("capture.capture.json").exists()


def test_app_range_command_requires_full_single_range_capture() -> None:
    expected_range, requested_metrics = run_benchmarks._validate_ncu_app_range_command(
        _valid_command()
    )

    assert expected_range == "compute_kernel:profile"
    assert requested_metrics == list(MINIMAL_METRICS)

    with pytest.raises(ValueError, match="limiting options"):
        run_benchmarks._validate_ncu_app_range_command(
            [*_valid_command(), "--launch-count", "1"]
        )
    with pytest.raises(ValueError, match="limiting options"):
        run_benchmarks._validate_ncu_app_range_command(
            [*_valid_command(), "--pm-sampling-interval", "1000"]
        )
    with pytest.raises(ValueError, match="profile from range start"):
        run_benchmarks._validate_ncu_app_range_command(
            [*_valid_command(), "--profile-from-start", "off"]
        )
    with pytest.raises(ValueError, match="exactly one NVTX"):
        run_benchmarks._validate_ncu_app_range_command(
            [*_valid_command(), "--nvtx-include", "inner:range"]
        )


def _mock_completed_capture(tmp_path: Path, observed: dict[str, object]):
    def run_capture(**kwargs):
        command = kwargs["command"]
        observed["command"] = list(command)
        report = Path(command[command.index("-o") + 1]).with_suffix(".ncu-rep")
        report.write_bytes(b"fixture NCU report")
        stdout_log = tmp_path / "ncu.stdout.log"
        stderr_log = tmp_path / "ncu.stderr.log"
        stdout_log.write_text("", encoding="utf-8")
        stderr_log.write_text("", encoding="utf-8")
        return SimpleNamespace(
            process=SimpleNamespace(returncode=0),
            stdout_log=stdout_log,
            stderr_log=stderr_log,
            timed_out=False,
            failure_warning=None,
        )

    return run_capture


def test_python_app_range_control_plane_uses_inspector_and_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_path = tmp_path / "fixture.py"
    benchmark_path.write_text("def get_benchmark():\n    raise AssertionError\n", encoding="utf-8")
    observed: dict[str, object] = {}

    def inspect(report_path, **kwargs):
        observed["inspection"] = kwargs
        return _RangeMetrics(), _provenance(report_path)

    monkeypatch.setattr(run_benchmarks, "check_ncu_available", lambda: True)
    monkeypatch.setattr(
        run_benchmarks,
        "_run_profile_subprocess",
        _mock_completed_capture(tmp_path, observed),
    )
    monkeypatch.setattr(run_benchmarks, "inspect_ncu_app_range_report", inspect)
    monkeypatch.setattr(
        run_benchmarks,
        "extract_from_ncu_report",
        lambda _path: (_ for _ in ()).throw(AssertionError("kernel fallback used")),
    )
    capture: dict[str, object] = {}

    report = run_benchmarks.profile_python_benchmark_ncu(
        SimpleNamespace(),
        benchmark_path,
        tmp_path,
        tmp_path / "profiles",
        BenchmarkConfig(
            profile_type="minimal",
            ncu_metric_set="minimal",
            ncu_replay_mode="app-range",
            ncu_replay_mode_override=True,
            validity_profile="portable",
        ),
        capture_out=capture,
    )

    assert report is not None and report.is_file()
    assert observed["inspection"] == {
        "expected_nvtx_range": "compute_kernel:profile",
        "requested_metrics": list(MINIMAL_METRICS),
        "timeout": 300,
    }
    assert Path(capture["sidecar_path"]).is_file()
    assert capture["result_scope"] == "range"
    assert capture["metrics"]["ncu_range_time_ms"] == 12.5


def test_python_app_range_rejects_invalid_inspection_without_kernel_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_path = tmp_path / "fixture.py"
    benchmark_path.write_text("def get_benchmark():\n    raise AssertionError\n", encoding="utf-8")
    observed: dict[str, object] = {}
    monkeypatch.setattr(run_benchmarks, "check_ncu_available", lambda: True)
    monkeypatch.setattr(
        run_benchmarks,
        "_run_profile_subprocess",
        _mock_completed_capture(tmp_path, observed),
    )
    monkeypatch.setattr(
        run_benchmarks,
        "inspect_ncu_app_range_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("Invalid NCU app-range report fixture: wrong result scope")
        ),
    )
    monkeypatch.setattr(
        run_benchmarks,
        "extract_from_ncu_report",
        lambda _path: (_ for _ in ()).throw(AssertionError("kernel fallback used")),
    )
    capture: dict[str, object] = {"stale": True}

    report = run_benchmarks.profile_python_benchmark_ncu(
        SimpleNamespace(),
        benchmark_path,
        tmp_path,
        tmp_path / "profiles",
        BenchmarkConfig(
            profile_type="minimal",
            ncu_metric_set="minimal",
            ncu_replay_mode="app-range",
            ncu_replay_mode_override=True,
            validity_profile="portable",
        ),
        capture_out=capture,
    )

    assert report is None
    assert capture == {}
    assert "wrong result scope" in (run_benchmarks._get_profile_failure_detail("ncu") or "")


def test_python_application_replay_keeps_legacy_kernel_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark_path = tmp_path / "fixture.py"
    benchmark_path.write_text("def get_benchmark():\n    raise AssertionError\n", encoding="utf-8")
    observed: dict[str, object] = {}
    monkeypatch.setattr(run_benchmarks, "check_ncu_available", lambda: True)
    monkeypatch.setattr(
        run_benchmarks,
        "_run_profile_subprocess",
        _mock_completed_capture(tmp_path, observed),
    )
    monkeypatch.setattr(
        run_benchmarks,
        "inspect_ncu_app_range_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("app-range inspector used for application replay")
        ),
    )
    monkeypatch.setattr(
        run_benchmarks,
        "extract_from_ncu_report",
        lambda _path: {"kernel_time_ms": 1.0},
    )
    capture: dict[str, object] = {}

    report = run_benchmarks.profile_python_benchmark_ncu(
        SimpleNamespace(),
        benchmark_path,
        tmp_path,
        tmp_path / "profiles",
        BenchmarkConfig(
            profile_type="minimal",
            ncu_metric_set="minimal",
            ncu_replay_mode="application",
            ncu_replay_mode_override=True,
            validity_profile="portable",
        ),
        capture_out=capture,
    )

    assert report is not None and report.is_file()
    command = observed["command"]
    assert isinstance(command, list)
    assert command[command.index("--replay-mode") + 1] == "application"
    assert capture == {}


def test_explicit_replay_override_wins_while_internal_range_preference_remains() -> None:
    class _Benchmark:
        preferred_ncu_replay_mode = "range"

    explicit = BenchmarkConfig(
        ncu_replay_mode="kernel",
        ncu_replay_mode_override=True,
    )
    assert run_benchmarks._apply_preferred_ncu_profile_overrides(
        explicit,
        _Benchmark(),
    ).ncu_replay_mode == "kernel"

    benchmark_local = BenchmarkConfig(
        ncu_replay_mode="kernel",
        ncu_replay_mode_override=False,
    )
    updated = run_benchmarks._apply_preferred_ncu_profile_overrides(
        benchmark_local,
        _Benchmark(),
    )
    assert updated.ncu_replay_mode == "range"
    assert updated.ncu_replay_mode_override is True
