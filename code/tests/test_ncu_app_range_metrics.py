from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from core.analysis.metric_extractor import extract_from_ncu_report
from core.benchmark.models import NcuMetrics
from core.profiling import metrics_extractor
from core.profiling.metrics_extractor import (
    NCU_APP_RANGE_DEFAULT_METRICS,
    extract_ncu_metrics,
    inspect_ncu_app_range_report,
)

FIXTURES = Path(__file__).parent / "fixtures"
DETAILS = (FIXTURES / "ncu_app_range_details.csv").read_text(encoding="utf-8")
SESSION = (FIXTURES / "ncu_app_range_session.csv").read_text(encoding="utf-8")


def _report(tmp_path: Path) -> Path:
    report = tmp_path / "capture.ncu-rep"
    report.write_bytes(b"sanitized-ncu-app-range-report")
    return report


def _install_imports(
    monkeypatch: pytest.MonkeyPatch,
    *,
    details: str = DETAILS,
    session: str = SESSION,
    failed_page: str | None = None,
) -> list[list[str]]:
    calls: list[list[str]] = []

    def _run(command, **_kwargs):
        argv = [str(part) for part in command]
        calls.append(argv)
        page = argv[argv.index("--page") + 1]
        if page == failed_page:
            return subprocess.CompletedProcess(argv, 7, stdout="", stderr="failed")
        output = details if page == "details" else session
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr="")

    monkeypatch.setattr(metrics_extractor.subprocess, "run", _run)
    return calls


def _replace_first_command_metric_line(details: str, old: str, new: str) -> str:
    lines = details.splitlines()
    for index, line in enumerate(lines):
        if '"Command line profiler metrics"' in line and old in line:
            lines[index] = line.replace(old, new, 1)
            return "\n".join(lines) + "\n"
    raise AssertionError(f"fixture line containing {old!r} was not found")


def _duplicate_first_command_metric_line(details: str) -> str:
    lines = details.splitlines()
    duplicate = next(line for line in lines if '"Command line profiler metrics"' in line)
    return "\n".join([*lines, duplicate]) + "\n"


def _change_first_analysis_record(
    details: str,
    *,
    old: str,
    new: str,
) -> str:
    lines = details.splitlines()
    for index, line in enumerate(lines):
        if '"GPU Speed Of Light Throughput"' in line:
            lines[index] = line.replace(old, new, 1)
            return "\n".join(lines) + "\n"
    raise AssertionError("analysis fixture row was not found")


def _add_overlong_details_row(details: str) -> str:
    lines = details.splitlines()
    lines[1] += '"","","","","","extra"'
    return "\n".join(lines) + "\n"


def test_real_schema_fixture_is_sanitized() -> None:
    fixture_text = DETAILS + SESSION
    assert "/root/" not in fixture_text
    assert "/Users/" not in fixture_text
    assert "2005734" not in fixture_text
    assert "fregly-dev" not in fixture_text


def test_inspect_ncu_app_range_report_extracts_range_metrics_and_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(tmp_path)
    calls = _install_imports(monkeypatch)

    metrics, provenance = inspect_ncu_app_range_report(report)

    assert [call[call.index("--page") + 1] for call in calls] == ["details", "session"]
    assert all(call[call.index("--print-units") + 1] == "base" for call in calls)
    assert calls[0][calls[0].index("--print-metric-name") + 1] == "name"
    assert "--print-metric-name" not in calls[1]
    assert calls[0][calls[0].index("--metrics") + 1].split(",") == list(
        NCU_APP_RANGE_DEFAULT_METRICS
    )
    assert metrics.kernel_time_ms is None
    assert metrics.range_time_ms == pytest.approx(123.480704)
    assert metrics.sm_throughput_pct == pytest.approx(0.14)
    assert metrics.dram_throughput_pct == pytest.approx(0.06)
    assert metrics.l2_throughput_pct == pytest.approx(0.05)
    assert metrics.occupancy_pct == pytest.approx(6.23)
    assert metrics.to_dict()["ncu_range_time_ms"] == pytest.approx(123.480704)
    assert "ncu_kernel_time_ms" not in metrics.to_dict()

    assert provenance["report_sha256"] == hashlib.sha256(report.read_bytes()).hexdigest()
    assert provenance["report_bytes"] == report.stat().st_size
    assert provenance["replay_mode"] == "app-range"
    assert provenance["result_scope"] == "range"
    assert provenance["result_id"] == 0
    assert provenance["result_count"] == 1
    assert provenance["nvtx_range"] == "compute_kernel:profile"
    assert "compute_kernel:profile" in provenance["nvtx_range_raw"]
    assert provenance["coverage_policy"] == "full_selected_nvtx_range"
    assert provenance["constituent_kernels_enumerated"] is False
    assert provenance["duration_semantics"] == "ncu_aggregate_range"
    assert provenance["requested_metrics"] == list(NCU_APP_RANGE_DEFAULT_METRICS)
    assert provenance["observed_metrics"] == list(NCU_APP_RANGE_DEFAULT_METRICS)
    assert provenance["metric_units"]["gpu__time_duration.avg"] == "ns"
    assert provenance["session_capture"] == {
        "replay_mode": "app-range",
        "nvtx_enabled": True,
        "nvtx_includes": ["compute_kernel:profile"],
        "target_processes": "all",
        "metrics": list(NCU_APP_RANGE_DEFAULT_METRICS),
        "limiting_flags": [],
        "profile_from_start": "default",
        "command_sha256": provenance["session_capture"]["command_sha256"],
    }


@pytest.mark.parametrize(
    ("details", "error"),
    [
        (DETAILS.replace(',"range",', ',"kernel",'), "result scope"),
        (
            DETAILS.replace("compute_kernel:profile", "setup_only"),
            "NVTX range",
        ),
        (
            _replace_first_command_metric_line(DETAILS, '"123480704"', '"0"'),
            "duration must be positive",
        ),
        (
            _replace_first_command_metric_line(DETAILS, '"ns","123480704"', '"us","123480704"'),
            "uses unit",
        ),
        (_duplicate_first_command_metric_line(DETAILS), "duplicate metrics"),
        (
            _change_first_analysis_record(DETAILS, old='"0",', new='"1",'),
            "logical result IDs",
        ),
        (
            _change_first_analysis_record(DETAILS, old=',"range",', new=',"kernel",'),
            "result scope",
        ),
        (_add_overlong_details_row(DETAILS), "columns; expected at most"),
    ],
)
def test_inspect_ncu_app_range_report_rejects_invalid_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    details: str,
    error: str,
) -> None:
    report = _report(tmp_path)
    _install_imports(monkeypatch, details=details)

    with pytest.raises(ValueError, match=error):
        inspect_ncu_app_range_report(report)


@pytest.mark.parametrize(
    ("session", "error"),
    [
        (SESSION.replace("--replay-mode app-range", "--replay-mode range"), "replay mode"),
        (SESSION.replace(" -o test_report", " --launch-count 1 -o test_report"), "limits range"),
        (
            SESSION.replace(" -o test_report", " --profile-from-start off -o test_report"),
            "limits range",
        ),
        (SESSION.replace(" -o test_report", " --nvtx-exclude setup -o test_report"), "limits range"),
        (
            SESSION.replace(" -o test_report", " --pm-sampling-interval 7 -o test_report"),
            "limits range",
        ),
    ],
)
def test_inspect_ncu_app_range_report_rejects_non_full_session_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    session: str,
    error: str,
) -> None:
    report = _report(tmp_path)
    _install_imports(monkeypatch, session=session)

    with pytest.raises(ValueError, match=error):
        inspect_ncu_app_range_report(report)


def test_inspect_ncu_app_range_report_rejects_import_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(tmp_path)
    _install_imports(monkeypatch, failed_page="session")

    with pytest.raises(ValueError, match="session import exited with code 7"):
        inspect_ncu_app_range_report(report)


def test_ordinary_extraction_recognizes_app_range_without_kernel_relabeling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(tmp_path)
    _install_imports(monkeypatch)

    metrics = extract_ncu_metrics(report)
    flat_metrics = extract_from_ncu_report(report)

    assert metrics.range_time_ms == pytest.approx(123.480704)
    assert metrics.kernel_time_ms is None
    assert flat_metrics["range_time_ms"] == pytest.approx(123.480704)
    assert "kernel_time_ms" not in flat_metrics


def test_ordinary_extraction_rejects_non_app_range_and_companion_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = _report(tmp_path)
    range_session = SESSION.replace("--replay-mode app-range", "--replay-mode range")
    _install_imports(monkeypatch, session=range_session)

    metrics = extract_ncu_metrics(report)
    assert metrics.kernel_time_ms is None
    assert metrics.range_time_ms is None

    report.with_suffix(".csv").write_text(
        '"ID","Kernel Name","gpu__time_duration.avg"\n'
        '"","","ns"\n'
        '"0","range","123480704"\n',
        encoding="utf-8",
    )
    _install_imports(monkeypatch, failed_page="details")
    metrics = extract_ncu_metrics(report)
    assert metrics.kernel_time_ms is None
    assert metrics.range_time_ms is None


def test_ncu_metrics_range_field_is_numeric_and_separate() -> None:
    metrics = NcuMetrics(range_time_ms=1.25, kernel_time_ms=None)

    assert metrics.to_dict() == {"ncu_range_time_ms": 1.25}
