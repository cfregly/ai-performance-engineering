from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from core.analysis.performance_analyzer import PerformanceAnalyzer, load_benchmark_data
from core.analysis.reporting import BenchmarkData, BenchmarkReport, generate_report


@pytest.mark.parametrize("source", ["missing.json", "http://127.0.0.1:6970"])
def test_generate_report_requires_local_json_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str
) -> None:
    def reject_network(*_args, **_kwargs):
        pytest.fail("Report generation attempted a network request")

    monkeypatch.setattr(urllib.request, "urlopen", reject_network)
    input_path = tmp_path / source if source == "missing.json" else source
    output_path = tmp_path / "report.html"

    with pytest.raises(FileNotFoundError, match="Benchmark JSON file not found"):
        generate_report(input_path, output_path, format="html")

    assert not output_path.exists()


def _write_results(path: Path, timestamp: str, run_id: str | None, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "run_id": run_id,
        "timestamp": timestamp,
        "results": [{"chapter": "ch01", "benchmarks": rows}],
    }), encoding="utf-8")


def test_latest_qualified_measurement_is_selected_and_failures_stay_visible(tmp_path: Path) -> None:
    old = tmp_path / "artifacts" / "a" / "benchmark_test_results.json"
    new = tmp_path / "artifacts" / "b" / "benchmark_test_results.json"
    _write_results(old, "2026-01-01T00:00:00Z", "earlier", [
        {
            "example": "gemm", "status": "succeeded", "best_speedup": 8.0, "baseline_time_ms": 16.0,
            "optimizations": [{"status": "succeeded", "time_ms": 2.0, "speedup": 8.0}],
        },
    ])
    legacy = tmp_path / "artifacts" / "legacy" / "benchmark_test_results.json"
    _write_results(legacy, "2026-01-01T00:00:00Z", None, [
        {"example": "gemm", "type": "cuda", "best_speedup": 1.5},
    ])
    failure_statuses = [
        "failed_verification", "failed_profiler", "failed_expectation_invalid",
        "failed_regression", "failed_no_speedup",
    ]
    _write_results(new, "2026-01-02T00:00:00Z", "later", [
        {
            "example": "gemm", "status": "succeeded", "baseline_time_ms": 10.0, "best_speedup": 2.0,
            "optimizations": [{"status": "succeeded", "time_ms": 5.0, "speedup": 2.0}],
        },
        *[
            {"example": "gemm", "status": status, "baseline_time_ms": 10.0, "best_speedup": 99.0}
            for status in failure_statuses
        ],
    ])

    data = load_benchmark_data(bench_roots=[tmp_path])

    assert [(row["type"], row["speedup"]) for row in data["benchmarks"]] == [
        ("python", 2.0), ("cuda", 1.5),
    ]
    assert data["benchmarks"][0]["run_id"] == "later"
    assert data["benchmarks"][1]["status_source"] == "legacy_unrecorded"
    assert data["summary"]["avg_speedup"] == 1.75
    assert data["summary"]["max_speedup"] == 2.0
    assert {row["status"] for row in data["diagnostics"]} == set(failure_statuses)
    assert all(row["source_file"] == str(new) for row in data["diagnostics"])

    output = generate_report(new, tmp_path / "report.html", format="html")
    ranking, diagnostics = output.read_text(encoding="utf-8").split("Run Diagnostics")
    assert "99.00x" not in ranking
    assert "Top Performers" in ranking
    assert "99.00x" in diagnostics
    assert "failed_profiler" in diagnostics


def test_explicit_variants_must_back_the_reported_speedup(tmp_path: Path) -> None:
    result = tmp_path / "benchmark_test_results.json"
    measured = {"status": "succeeded", "speedup": 2.0, "time_ms": 5.0}
    _write_results(result, "2026-01-02T00:00:00Z", "measured", [
        {
            "example": name,
            "status": "succeeded",
            "baseline_file": "baseline.py",
            "baseline_time_ms": 10.0,
            "best_speedup": speedup,
            "optimizations": optimizations,
        }
        for name, speedup, optimizations in [
            ("rejected_variant", 99.0, [{**measured, "status": "failed_verification", "speedup": 99.0}]),
            ("baseline_only", 1.0, []),
            ("invalid_variant_time", 1.0, [{**measured, "time_ms": 0.0, "speedup": 1.0}]),
            ("unsupported_best", 99.0, [measured]),
            ("inconsistent_pair", 99.0, [{**measured, "speedup": 99.0}]),
            ("measured_equal", 1.0, [{**measured, "time_ms": 10.0, "speedup": 1.0}]),
        ]
    ])

    data = load_benchmark_data(data_file=result)
    assert [row["name"] for row in data["benchmarks"]] == ["measured_equal"]
    assert data["summary"]["max_speedup"] == 1.0
    assert len(data["diagnostics"]) == 5
    assert all(row["status"] == "unqualified" for row in data["diagnostics"])
    assert all(row["optimized_time_ms"] is None for row in data["diagnostics"])
    assert {row["name"]: row["speedup"] for row in data["diagnostics"]}["rejected_variant"] == 99.0

    output = generate_report(result, tmp_path / "report.html", format="html")
    ranking, diagnostics = output.read_text(encoding="utf-8").split("Run Diagnostics")
    assert "measured_equal" in ranking
    assert "99.00x" not in ranking
    assert "No successful optimization with measured timings and speedup" in diagnostics
    assert "Recorded best speedup disagrees with the best successful optimization" in diagnostics


def test_normalized_rows_are_filtered_at_analysis_and_report_boundaries(tmp_path: Path) -> None:
    rows = [
        {"name": "legacy", "chapter": "ch01", "speedup": 2.0},
        {"name": "reject", "chapter": "ch01", "status": "failed_profiler", "speedup": 99.0},
        {"name": "baseline_only", "chapter": "ch01", "status": "succeeded", "speedup": 1.0, "optimizations": []},
    ]
    analyzer = PerformanceAnalyzer(lambda: {"benchmarks": rows})

    entries = analyzer.get_categorized_leaderboards()["leaderboards"]["speed"]["entries"]
    assert [row["name"] for row in entries] == ["ch01:legacy"]
    assert [row["name"] for row in analyzer.get_pareto_frontier()["all_points"]] == ["ch01:legacy"]
    report = BenchmarkReport([BenchmarkData(**row) for row in rows[:2]])
    assert report.summary()["avg_speedup"] == 2.0

    output = generate_report({"benchmarks": rows}, tmp_path / "report.html", format="html")
    ranking, diagnostics = output.read_text(encoding="utf-8").split("Run Diagnostics")
    assert "99.00x" not in ranking
    assert "99.00x" in diagnostics


@pytest.mark.parametrize("payload", ['{not-json', '{}', '{"results": [{"chapter": "ch01", "benchmarks": [42]}]}'])
def test_malformed_input_is_explicit_and_discovered_scan_retains_error(tmp_path: Path, payload: str) -> None:
    data_file = tmp_path / "benchmark_test_results.json"
    data_file.write_text(payload, encoding="utf-8")

    with pytest.raises(ValueError, match=data_file.name):
        load_benchmark_data(data_file=data_file)
    with pytest.raises(ValueError, match=data_file.name):
        PerformanceAnalyzer(lambda: load_benchmark_data(data_file=data_file)).get_pareto_frontier()
    with pytest.raises(ValueError, match=data_file.name):
        generate_report(data_file, tmp_path / "report.html", format="html")
    assert not (tmp_path / "report.html").exists()

    data = load_benchmark_data(bench_roots=[tmp_path])
    assert data["benchmarks"] == []
    assert data["ingest_errors"][0]["source_file"] == str(data_file)


def test_discovery_counts_latest_target_outcome_and_exposes_later_failure(tmp_path: Path) -> None:
    old = tmp_path / "artifacts/a/benchmark_test_results.json"
    new = tmp_path / "artifacts/b/benchmark_test_results.json"
    alternate = tmp_path / "artifacts/c/benchmark_test_results.json"
    qualified = {
        "status": "succeeded", "baseline_time_ms": 10.0, "best_speedup": 2.0,
        "optimizations": [{"status": "succeeded", "time_ms": 5.0, "speedup": 2.0}],
    }
    failed = {"status": "failed_verification", "best_speedup": 99.0}
    _write_results(old, "2026-01-01T00:00:00Z", "earlier", [
        {"example": "broke", **qualified}, {"example": "fixed", **failed},
    ])
    _write_results(new, "2026-01-02T00:00:00Z", "later", [
        {"example": "broke", **failed}, {"example": "fixed", **qualified},
    ])
    _write_results(alternate, "2026-01-03T00:00:00Z", "latest", [{"example": "broke", **failed}])

    data = load_benchmark_data(bench_roots=[tmp_path])

    assert data["summary"]["total_benchmarks"] == 2
    assert data["summary"]["successful"] == 1
    assert data["summary"]["failed"] == 1
    assert data["summary"]["qualified"] == 2
    assert len(data["diagnostics"]) == 3
    by_target = {row["name"]: row for row in data["benchmarks"]}
    assert by_target["broke"]["run_id"] == "earlier"
    assert by_target["broke"]["newer_diagnostic"]["run_id"] == "latest"
    assert "newer_diagnostic" not in by_target["fixed"]

    output = generate_report(data, tmp_path / "report.html", format="html")
    assert "Newer result failed_verification from latest" in output.read_text(encoding="utf-8")


def test_modern_rows_require_status_and_variants(tmp_path: Path) -> None:
    path = tmp_path / "benchmark_test_results.json"
    _write_results(path, "2026-01-01T00:00:00Z", "modern", [
        {"example": "missing_variants", "status": "succeeded", "best_speedup": 99.0},
        {"example": "missing_status", "best_speedup": 2.0, "baseline_time_ms": 10.0,
         "optimizations": [{"status": "succeeded", "time_ms": 5.0, "speedup": 2.0}]},
    ])
    data = load_benchmark_data(data_file=path)

    assert data["benchmarks"] == []
    assert [row["error"] for row in data["diagnostics"]] == [
        "Current benchmark result is missing its optimization records",
        "Current benchmark result is missing its final status",
    ]


def test_scanned_malformed_rows_keep_valid_siblings_and_report_errors(tmp_path: Path) -> None:
    path = tmp_path / "benchmark_test_results.json"
    _write_results(path, "2026-01-01T00:00:00Z", None, [
        None,
        {"example": "legacy", "best_speedup": 2.0},
        {"example": "broken", "best_speedup": "unknown"},
    ])
    with pytest.raises(ValueError, match="Invalid benchmark"):
        load_benchmark_data(data_file=path)

    data = load_benchmark_data(bench_roots=[tmp_path])

    assert [row["name"] for row in data["benchmarks"]] == ["legacy"]
    assert len(data["ingest_errors"]) == 2
    output = generate_report(data, tmp_path / "report.html", format="html")
    rendered = output.read_text(encoding="utf-8")
    assert "Input Errors" in rendered and "Invalid best_speedup" in rendered


def test_legacy_normalized_export_keeps_internally_consistent_pairs(tmp_path: Path) -> None:
    data = {"benchmarks": [{
        "name": "old_export", "status": "succeeded", "speedup": 2.0, "baseline_time_ms": 10.0,
        "optimized_time_ms": 5.0, "optimizations": [{"time_ms": 5.0, "speedup": 2.0}],
    }]}
    path = tmp_path / "old-export.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    output = generate_report(path, tmp_path / "report.html", format="html")

    assert "old_export" in output.read_text(encoding="utf-8").split("Top Performers")[1]
    assert BenchmarkData(
        name="incomplete", run_id="modern", speedup=2.0, baseline_time_ms=10.0
    ).qualified is False
    assert BenchmarkData(name="no_measurement").qualified is False


@pytest.mark.parametrize("has_success", [True, False])
def test_cli_triage_counts_failed_results_without_admitting_them(tmp_path: Path, has_success: bool) -> None:
    from typer.testing import CliRunner
    from core.benchmark.bench_commands import app

    path = tmp_path / "benchmark_test_results.json"
    rows = [{"example": "failed", "status": "failed_profiler", "best_speedup": 99.0}]
    if has_success:
        rows.append({
            "example": "qualified", "status": "succeeded", "baseline_time_ms": 10.0, "best_speedup": 2.0,
            "optimizations": [{"status": "succeeded", "time_ms": 5.0, "speedup": 2.0}],
        })
    _write_results(path, "2026-01-01T00:00:00Z", "triage", rows)

    result = CliRunner().invoke(app, ["triage", "--data-file", str(path), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["summary"]["failed"] == 1
    assert payload["summary"]["passed"] == int(has_success)
    assert payload["summary"]["avg_speedup"] == (2.0 if has_success else 0)
    assert [row["benchmark"] for row in payload["improvements"]] == (["ch01:qualified"] if has_success else [])
    assert payload["diagnostics"][0]["name"] == "failed"


def test_cli_triage_surfaces_discovered_ingest_errors_without_discarding_good_rows(tmp_path: Path, monkeypatch) -> None:
    from typer.testing import CliRunner
    from core.benchmark import bench_commands

    path = tmp_path / "benchmark_test_results.json"
    _write_results(path, "2026-01-01T00:00:00Z", None, [
        {"example": "legacy", "best_speedup": 2.0},
        {"example": "broken", "best_speedup": "unknown"},
    ])
    monkeypatch.setattr(
        bench_commands, "load_benchmark_results",
        lambda data_file: load_benchmark_data(data_file, bench_roots=[tmp_path]),
    )

    result = CliRunner().invoke(bench_commands.app, ["triage", "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["summary"]["passed"] == 1
    assert payload["improvements"][0]["baseline_ms"] is None
    assert payload["ingest_errors"][0]["source_file"] == str(path)
    assert "Invalid best_speedup" in result.stderr
