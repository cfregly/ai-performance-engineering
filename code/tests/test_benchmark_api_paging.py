from __future__ import annotations

import json

import core.engine as engine_module
from core.api.handlers import benchmark_data
from core.engine import get_engine
from core.perf_core import PerfCore


def _configure_with_sample_data(tmp_path, monkeypatch):
    data = {
        "timestamp": "2025-01-02 00:00:00",
        "results": [
            {
                "chapter": "ch01",
                "benchmarks": [
                    {
                        "example": "gemm",
                        "best_speedup": 2.0,
                        "baseline_time_ms": 100.0,
                        "optimizations": [],
                        "status": "succeeded",
                    },
                    {
                        "example": "mem",
                        "best_speedup": 1.0,
                        "baseline_time_ms": 50.0,
                        "optimizations": [],
                        "status": "skipped",
                    },
                ],
            },
            {
                "chapter": "ch02",
                "benchmarks": [
                    {
                        "example": "transfer",
                        "best_speedup": 3.0,
                        "baseline_time_ms": 200.0,
                        "optimizations": [],
                        "status": "failed",
                    },
                    {
                        "example": "verify_only",
                        "best_speedup": 0.9,
                        "baseline_time_ms": 120.0,
                        "optimizations": [],
                        "status": "failed_verification",
                    },
                    {
                        "example": "runtime_boom",
                        "best_speedup": 0.8,
                        "baseline_time_ms": 80.0,
                        "optimizations": [],
                        "status": "failed_error",
                    },
                    {
                        "example": "flat_speed",
                        "best_speedup": 1.02,
                        "baseline_time_ms": 90.0,
                        "optimizations": [],
                        "status": "failed_no_speedup",
                    }
                ],
            },
        ],
    }
    path = tmp_path / "benchmark_test_results.json"
    path.write_text(json.dumps(data))
    core = PerfCore(data_file=path, bench_root=tmp_path)
    monkeypatch.setattr(engine_module, "_get_handler", lambda: core)


def test_benchmark_data_pagination_and_filters(tmp_path, monkeypatch):
    _configure_with_sample_data(tmp_path, monkeypatch)
    result = benchmark_data(
        {"page": 1, "page_size": 1, "sort_field": "speedup", "sort_dir": "desc"}
    )
    assert result["pagination"]["total"] == 6
    assert len(result["benchmarks"]) == 1
    assert result["benchmarks"][0]["name"] == "transfer"
    assert result["summary"]["total"] == 6

    filtered_payload = benchmark_data({"status": "succeeded"})
    assert filtered_payload["pagination"]["total"] == 1
    assert filtered_payload["benchmarks"][0]["name"] == "gemm"


def test_benchmark_overview_summary(tmp_path, monkeypatch):
    _configure_with_sample_data(tmp_path, monkeypatch)
    overview = get_engine().benchmark.overview()
    assert overview["summary"]["total"] == 6
    assert overview["status_counts"]["succeeded"] == 1
    assert overview["status_counts"]["failed"] == 4
    assert overview["status_counts"]["skipped"] == 1
    assert overview["top_speedups"][0]["name"] == "gemm"
