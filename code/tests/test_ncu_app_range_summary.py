from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest

from core import profile_insights
from core.profiling.ncu_summary import summarize_ncu_report

NVTX_RANGE_COLUMN = "Id:Domain:Start/Stop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg"
RANGE_IDENTITY = "0:<default domain>:compute_kernel:profile:none:none:none:none:none:none"
METRICS = {
    "gpu__time_duration.avg": ("ns", "123480704"),
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": ("%", "0.14"),
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": ("%", "0.06"),
    "lts__throughput.avg.pct_of_peak_sustained_elapsed": ("%", "0.05"),
    "sm__warps_active.avg.pct_of_peak_sustained_active": ("%", "6.23"),
}
FIXTURES = Path(__file__).parent / "fixtures"


def _write_raw_csv(path: Path, *, include_range_identity: bool = True) -> None:
    header = ["ID", "Kernel Name", NVTX_RANGE_COLUMN, *METRICS]
    units = ["", "", "", *(unit for unit, _ in METRICS.values())]
    values = [
        "0",
        "range",
        RANGE_IDENTITY if include_range_identity else "",
        *(value for _, value in METRICS.values()),
    ]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows((header, units, values))


def _write_details_csv(path: Path, *, time_ns: str, occupancy: str) -> None:
    header = [
        "ID",
        "Kernel Name",
        NVTX_RANGE_COLUMN,
        "Metric Name",
        "Metric Unit",
        "Metric Value",
    ]
    rows = []
    for name, (unit, value) in METRICS.items():
        if name == "gpu__time_duration.avg":
            value = time_ns
        elif name == "sm__warps_active.avg.pct_of_peak_sustained_active":
            value = occupancy
        rows.append(["0", "range", RANGE_IDENTITY, name, unit, value])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def test_raw_app_range_summary_is_not_a_kernel_summary(tmp_path: Path) -> None:
    report = tmp_path / "capture.csv"
    _write_raw_csv(report)

    summary = summarize_ncu_report(report)

    assert summary["success"] is True
    assert summary["result_scope"] == "range"
    assert "kernels" not in summary
    assert "kernel_count" not in summary
    assert "total_time_sum_ms" not in summary
    assert summary["range_summary"]["nvtx_range"] == "compute_kernel:profile"
    assert summary["range_summary"]["range_time_ms"] == pytest.approx(123.480704)
    assert summary["range_summary"]["metrics"][
        "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    ] == pytest.approx(0.14)
    assert summary["provenance"]["coverage_policy"] == "range_row_only_unverified"
    assert "metrics" not in summary["provenance"]


def test_kernel_named_range_without_nvtx_identity_remains_a_kernel(tmp_path: Path) -> None:
    report = tmp_path / "kernel.csv"
    _write_raw_csv(report, include_range_identity=False)

    summary = summarize_ncu_report(report)

    assert summary["success"] is True
    assert summary["result_scope"] == "kernel"
    assert summary["kernel_count"] == 1
    assert summary["kernels"][0]["kernel_name"] == "range"
    assert summary["kernels"][0]["time_avg_ms"] == pytest.approx(123.480704)
    assert "range_summary" not in summary


def test_report_summary_preserves_strict_app_range_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    report = tmp_path / "capture.ncu-rep"
    report.write_bytes(b"sanitized-ncu-app-range-report")
    raw_csv = tmp_path / "raw.csv"
    _write_raw_csv(raw_csv)
    page_output = {
        "raw": raw_csv.read_text(),
        "details": (FIXTURES / "ncu_app_range_details.csv").read_text(),
        "session": (FIXTURES / "ncu_app_range_session.csv").read_text(),
    }

    def _run(command, **_kwargs):
        argv = [str(part) for part in command]
        page = argv[argv.index("--page") + 1]
        return subprocess.CompletedProcess(argv, 0, stdout=page_output[page], stderr="")

    monkeypatch.setattr(subprocess, "run", _run)

    summary = summarize_ncu_report(report)

    assert summary["success"] is True
    assert summary["result_scope"] == "range"
    assert "kernels" not in summary
    assert summary["range_summary"]["range_time_ms"] == pytest.approx(123.480704)
    assert summary["provenance"]["replay_mode"] == "app-range"
    assert summary["provenance"]["coverage_policy"] == "full_selected_nvtx_range"
    assert summary["provenance"]["session_capture"]["limiting_flags"] == []


def test_range_pair_has_no_kernel_or_speedup_claims(tmp_path: Path) -> None:
    baseline_dir = tmp_path / "baseline"
    optimized_dir = tmp_path / "optimized"
    baseline_dir.mkdir()
    optimized_dir.mkdir()
    _write_details_csv(
        baseline_dir / "baseline_app_range_ncu.csv",
        time_ns="123480704",
        occupancy="6.23",
    )
    _write_details_csv(
        optimized_dir / "optimized_app_range_ncu.csv",
        time_ns="784690848",
        occupancy="6.29",
    )

    comparison = profile_insights.compare_ncu_files(tmp_path)

    assert comparison is not None
    assert comparison["success"] is True
    assert comparison["result_scope"] == "range"
    assert "kernel_comparison" not in comparison
    assert "aggregate" not in comparison
    assert "metrics" not in comparison
    range_comparison = comparison["range_comparison"]
    assert range_comparison["nvtx_range"] == "compute_kernel:profile"
    assert range_comparison["baseline_range_time_ms"] == pytest.approx(123.480704)
    assert range_comparison["optimized_range_time_ms"] == pytest.approx(784.690848)
    assert range_comparison["canonical_speedup_eligible"] is False
    assert range_comparison["coverage_policy"] == "range_row_only_unverified"
    time_row = next(
        row for row in range_comparison["metrics"] if row["name"] == "gpu__time_duration.avg"
    )
    assert time_row["delta"] is None
    assert time_row["ratio"] is None
    assert time_row["speedup_eligible"] is False

    summary = profile_insights._summarize_ncu_side_by_side(comparison)
    assert summary["kernel"] is None
    assert summary["range_summary"]["canonical_speedup_eligible"] is False
    recommendations = profile_insights.generate_recommendations_from_profiles(
        {"ncu_comparison": comparison}
    )
    assert all("block size" not in recommendation.lower() for recommendation in recommendations)
    assert recommendations == [
        "NCU metrics cover the selected range; collect a kernel-scoped report "
        "before making per-kernel tuning recommendations"
    ]
