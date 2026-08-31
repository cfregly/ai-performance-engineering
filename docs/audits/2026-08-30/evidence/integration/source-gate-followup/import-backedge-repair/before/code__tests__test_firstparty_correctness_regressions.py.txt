from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

from core.scripts import benchmark_coverage
from core.utils import extension_prewarm


def test_benchmark_coverage_records_metric_helper(tmp_path: Path) -> None:
    benchmark = tmp_path / "baseline_example.py"
    benchmark.write_text(
        "def get_custom_metrics():\n" "    return compute_matmul_metrics(measured_time_ms=1.0)\n",
        encoding="utf-8",
    )

    result = benchmark_coverage.analyze_file(benchmark)

    assert result["uses_helper"] is True
    assert result["helper_name"] == "compute_matmul_metrics"


def test_benchmark_coverage_main_scans_repository_root(monkeypatch) -> None:
    scanned_roots: list[Path] = []

    def _capture_root(root: Path) -> benchmark_coverage.CoverageReport:
        scanned_roots.append(root)
        return benchmark_coverage.CoverageReport()

    monkeypatch.setattr(benchmark_coverage, "generate_report", _capture_root)
    monkeypatch.setattr(benchmark_coverage, "print_text_report", lambda _report: None)
    monkeypatch.setattr("sys.argv", ["benchmark_coverage.py"])

    benchmark_coverage.main()

    expected_root = Path(benchmark_coverage.__file__).resolve().parents[2]
    assert scanned_roots == [expected_root]


def test_extension_health_check_uses_temporary_repo_import_path(monkeypatch) -> None:
    events: list[object] = []

    @contextmanager
    def _tracked_import_context():
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    def _check_with_imports(*, verbose: bool) -> bool:
        events.append(verbose)
        return True

    monkeypatch.setattr(extension_prewarm, "_repo_import_context", _tracked_import_context)
    monkeypatch.setattr(
        extension_prewarm,
        "_health_check_with_repo_imports",
        _check_with_imports,
    )

    assert extension_prewarm.health_check(verbose=False) is True
    assert events == ["enter", False, "exit"]
