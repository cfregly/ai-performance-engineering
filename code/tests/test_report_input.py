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


def _write_results(path: Path, timestamp: str, run_id: str, rows: list[dict]) -> None:
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
        {"example": "gemm", "status": "succeeded", "best_speedup": 8.0},
    ])
    failure_statuses = [
        "failed_verification", "failed_profiler", "failed_expectation_invalid",
        "failed_regression", "failed_no_speedup",
    ]
    _write_results(new, "2026-01-02T00:00:00Z", "later", [
        {"example": "gemm", "status": "succeeded", "baseline_time_ms": 10.0, "best_speedup": 2.0},
        {"example": "gemm", "type": "cuda", "best_speedup": 1.5},
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
