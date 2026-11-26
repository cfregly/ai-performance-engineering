"""Regression coverage for profiler artifact links in benchmark Markdown reports."""

import re
from pathlib import Path
from urllib.parse import unquote

from core.harness.run_benchmarks import generate_markdown_report


def test_profiler_links_are_report_relative_and_url_escaped(tmp_path: Path) -> None:
    bench_root = tmp_path / "bench root"
    run_root = bench_root / "artifact run"
    profile_dir = run_root / "profiles" / "bench role"
    report_path = run_root / "reports" / "benchmark results.md"
    profile_dir.mkdir(parents=True)
    report_path.parent.mkdir(parents=True)

    baseline_paths = [
        profile_dir / "baseline system #1.nsys-rep",
        profile_dir / "baseline compute (1).ncu-rep",
        profile_dir / "baseline torch trace.json",
    ]
    optimized_paths = [
        profile_dir / "optimized system #2.nsys-rep",
        profile_dir / "optimized compute (2).ncu-rep",
        profile_dir / "optimized torch trace.json",
    ]
    expected_paths = baseline_paths + optimized_paths
    for path in expected_paths:
        path.touch()

    results = [
        {
            "chapter": "labs_example",
            "status": "completed",
            "benchmarks": [
                {
                    "example": "example",
                    "type": "python",
                    "baseline_file": "baseline_example.py",
                    "baseline_time_ms": 2.0,
                    "baseline_nsys_rep": str(baseline_paths[0]),
                    "baseline_ncu_rep": str(baseline_paths[1]),
                    "baseline_torch_trace": str(baseline_paths[2]),
                    "optimizations": [
                        {
                            "file": "optimized_example.py",
                            "status": "succeeded",
                            "time_ms": 1.0,
                            "speedup": 2.0,
                            "optimized_nsys_rep": str(optimized_paths[0].relative_to(bench_root)),
                            "optimized_ncu_rep": str(optimized_paths[1].relative_to(bench_root)),
                            "optimized_torch_trace": str(optimized_paths[2].relative_to(bench_root)),
                        }
                    ],
                    "best_speedup": 2.0,
                    "status": "succeeded",
                }
            ],
            "summary": {
                "total_benchmarks": 1,
                "successful": 1,
                "failed": 0,
                "average_speedup": 2.0,
                "max_speedup": 2.0,
            },
        }
    ]

    generate_markdown_report(results, report_path, bench_root=bench_root)

    report = report_path.read_text(encoding="utf-8")
    links = re.findall(r"\[(nsys|ncu|torch)\]\(([^)]+)\)", report)
    assert [role for role, _ in links] == ["nsys", "ncu", "torch"] * 2
    assert len(links) == len(expected_paths)
    for (_, link), expected_path in zip(links, expected_paths):
        assert not link.startswith(".//")
        assert " " not in link
        assert "#" not in link
        decoded = unquote(link)
        linked_path = Path(decoded)
        if not linked_path.is_absolute():
            linked_path = report_path.parent / linked_path
        assert linked_path.resolve() == expected_path.resolve()
        assert linked_path.is_file()
    assert any("%20" in link for _, link in links)
    assert any("%23" in link for _, link in links)


def test_report_treats_profiler_rejection_as_diagnostic(tmp_path: Path) -> None:
    def measured_result(name: str, time_ms: float, speedup: float, status: str | None) -> dict:
        result = {
            "example": name,
            "baseline_file": "baseline_example.py",
            "baseline_time_ms": speedup * time_ms,
            "best_speedup": speedup,
            "optimizations": [
                {"file": f"optimized_{name}.py", "status": "succeeded", "time_ms": time_ms, "speedup": speedup}
            ],
        }
        if status is not None:
            result["status"] = status
        return result

    rejected = measured_result("profiler_rejected", 1.0, 99.0, "failed_profiler")
    rejected["error"] = "Required profiler capture failed"
    rejected["best_llm_patch"] = {"variant_name": "rejected_patch", "actual_speedup": 98.0}
    provisional = measured_result("provisional", 1.0, 50.0, "provisional")
    results = [
        {
            "chapter": "ch01",
            "status": "completed",
            "benchmarks": [measured_result("qualified", 5.0, 2.0, "succeeded"), rejected, provisional],
            "summary": {
                "total_benchmarks": 3, "successful": 1, "failed": 1, "average_speedup": 50.5, "max_speedup": 99.0,
            },
        },
        {
            "chapter": "ch01",
            "status": "completed",
            "benchmarks": [measured_result("legacy", 4.0, 4.0, None)],
            "summary": {
                "total_benchmarks": 1, "successful": 1, "failed": 0, "average_speedup": 4.0, "max_speedup": 4.0,
            },
        },
    ]
    report_path = tmp_path / "benchmark_test_results.md"
    generate_markdown_report(results, report_path, bench_root=tmp_path)

    report = report_path.read_text(encoding="utf-8")
    assert "**Average speedup:** 3.00x" in report
    assert "**Best speedup:** 4.00x" in report
    assert "| ch01 | PASS | 3 | 1 | 1 | 2.00x | 2.00x |" in report
    assert "| ch01 | PASS | 1 | 1 | 0 | 4.00x | 4.00x |" in report
    assert "Run status `failed_profiler`.\n- Required profiler capture failed" in report
    assert "Run status `provisional`." in report
    assert "Recorded speedup was 99.00x." in report
    assert "Recorded speedup was 50.00x." in report
    assert "optimized_profiler_rejected.py" not in report
    assert "optimized_provisional.py" not in report
    assert "rejected_patch" not in report
    assert "optimized_legacy.py" in report


def test_markdown_report_excludes_speedups_that_disagree_with_observed_pairs(tmp_path: Path) -> None:
    valid = {"file": "valid.py", "status": "succeeded", "time_ms": 5.0, "speedup": 2.0}
    incomplete = {"file": "incomplete.py", "status": "succeeded", "speedup": 99.0}
    results = [{
        "chapter": "ch01",
        "status": "completed",
        "summary": {"total_benchmarks": 2, "successful": 2, "failed": 0},
        "benchmarks": [
            {
                "example": "qualified", "status": "succeeded", "baseline_file": "baseline.py", "baseline_time_ms": 10.0,
                "best_speedup": 2.0, "optimizations": [valid, incomplete],
            },
            {
                "example": "unsupported", "status": "succeeded", "baseline_file": "baseline.py", "baseline_time_ms": 10.0,
                "best_speedup": 99.0, "optimizations": [valid, incomplete],
            },
        ],
    }]
    path = tmp_path / "report.md"
    generate_markdown_report(results, path, bench_root=tmp_path)
    report = path.read_text(encoding="utf-8")

    assert "**Best speedup:** 2.00x" in report
    assert "Recorded speedup was 99.00x" in report
    assert "`incomplete.py`: No qualified optimization measurement." in report
