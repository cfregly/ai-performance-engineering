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
