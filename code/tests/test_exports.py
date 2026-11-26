import csv
import io
import json
from pathlib import Path


def test_export_csv_html_pdf_minimal():
    from core.engine import get_engine

    engine = get_engine()
    csv_basic = engine.export.csv()
    assert isinstance(csv_basic, str) and len(csv_basic) > 0

    csv_detailed = engine.export.csv_detailed()
    assert isinstance(csv_detailed, str) and len(csv_detailed) > 0

    html = engine.export.html()
    assert isinstance(html, str) and "<html" in html.lower()

    # PDF may be bytes or empty fallback; accept both but ensure type is bytes/str
    pdf_bytes = engine.export.pdf()
    assert isinstance(pdf_bytes, (bytes, str))



def test_legacy_speedup_only_exports_keep_missing_timings_blank(tmp_path: Path, monkeypatch):
    import core.engine as engine_module
    from core.perf_core import PerfCore
    from core.report_export import generate_html_report

    source = tmp_path / "benchmark_test_results.json"
    source.write_text(json.dumps({
        "results": [{"chapter": "ch01", "benchmarks": [
            {"example": "legacy_cuda", "type": "cuda", "best_speedup": 1.5},
            {"example": "with_baseline", "baseline_time_ms": 10.0, "best_speedup": 2.0},
        ]}],
    }), encoding="utf-8")
    core = PerfCore(data_file=source, bench_root=tmp_path)
    monkeypatch.setattr(engine_module, "_get_handler", lambda: core)

    engine = engine_module.get_engine()
    for exported in (engine.export.csv(), engine.export.csv_detailed()):
        rows = {row["Benchmark"]: row for row in csv.DictReader(io.StringIO(exported))}
        assert rows["legacy_cuda"]["Baseline Time (ms)"] == ""
        assert rows["legacy_cuda"]["Optimized Time (ms)"] == ""
        assert rows["legacy_cuda"]["Speedup"] == "1.50"
        assert rows["with_baseline"]["Baseline Time (ms)"] == "10.0000"
        assert rows["with_baseline"]["Optimized Time (ms)"] == ""

    html = generate_html_report(core.load_benchmark_data(), {}, {}, {}, {}, {})
    assert "legacy_cuda</td><td>n/a</td><td>n/a</td>" in html
    assert "with_baseline</td><td>10.00 ms</td><td>n/a</td>" in html


def test_compile_analysis_keeps_legacy_compile_speedup_without_inventing_compile_time(tmp_path: Path):
    from core.perf_core import PerfCore

    source = tmp_path / "benchmark_test_results.json"
    source.write_text(json.dumps({
        "results": [{"chapter": "ch01", "benchmarks": [
            {"example": "model_compile", "best_speedup": 1.5},
        ]}],
    }), encoding="utf-8")
    core = PerfCore(data_file=source, bench_root=tmp_path)

    analysis = core.get_compile_analysis()

    assert analysis["speedup"] == 1.5
    assert analysis["compile_time_ms"] is None
    assert analysis["has_real_data"]
    assert analysis["compile_benchmarks"][0]["baseline_time_ms"] is None

    (tmp_path / "compile_report.json").write_text('{"compile_time_ms": 42.0}', encoding="utf-8")
    assert core.get_compile_analysis()["compile_time_ms"] == 42.0


def test_csv_and_html_exports_label_a_later_rejection(tmp_path: Path) -> None:
    from core.analysis.performance_analyzer import load_benchmark_data
    from core.profile_artifacts import export_benchmarks_csv, export_detailed_csv
    from core.report_export import generate_html_report

    for run_id, date, outcome in (("earlier", "2026-04-01", "succeeded"), ("later", "2026-04-02", "failed_verification")):
        target = tmp_path / "artifacts" / run_id / "benchmark_test_results.json"
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({
            "run_id": run_id,
            "timestamp": date,
            "results": [{"chapter": "ch01", "benchmarks": [{
                "example": "x",
                "status": outcome,
                "baseline_time_ms": 20.0,
                "best_speedup": 2.0,
                "optimizations": [{"status": "succeeded", "speedup": 2.0, "time_ms": 10.0}],
            }]}],
        }), encoding="utf-8")

    data = load_benchmark_data(bench_roots=[tmp_path])
    for export in (export_benchmarks_csv, export_detailed_csv):
        (row,) = csv.DictReader(io.StringIO(export(data)))
        assert row["Speedup"] == "2.00"
        assert "failed_verification (later" in row["Newer Result"]
    assert "Earlier qualified result" in export_benchmarks_csv(data)
    assert "Newer result: failed_verification (later" in generate_html_report(data, {}, {}, {}, {}, {})
