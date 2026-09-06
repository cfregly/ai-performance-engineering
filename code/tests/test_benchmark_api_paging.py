from __future__ import annotations

import json

from core.engine import reset_engine
from dashboard.api import server
from tests.http_client import asgi_request, asgi_stream_text


def _configure_with_sample_data(tmp_path):
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
    reset_engine()
    server._configure_engine(path)


def test_benchmark_data_pagination_and_filters(tmp_path):
    _configure_with_sample_data(tmp_path)
    response = asgi_request(
        server.fastapi_app,
        "GET",
        "/api/benchmark/data?page=1&page_size=1&sort_field=speedup&sort_dir=desc",
    )
    assert response.status_code == 200
    payload = response.json()
    result = payload["result"]
    assert result["pagination"]["total"] == 6
    assert len(result["benchmarks"]) == 1
    assert result["benchmarks"][0]["name"] == "transfer"
    assert result["summary"]["total"] == 6

    filtered = asgi_request(server.fastapi_app, "GET", "/api/benchmark/data?status=succeeded")
    filtered_payload = filtered.json()["result"]
    assert filtered_payload["pagination"]["total"] == 1
    assert filtered_payload["benchmarks"][0]["name"] == "gemm"

    reset_engine()


def test_benchmark_overview_summary(tmp_path):
    _configure_with_sample_data(tmp_path)
    response = asgi_request(server.fastapi_app, "GET", "/api/benchmark/overview")
    assert response.status_code == 200
    overview = response.json()["result"]
    assert overview["summary"]["total"] == 6
    assert overview["status_counts"]["succeeded"] == 1
    assert overview["status_counts"]["failed"] == 4
    assert overview["status_counts"]["skipped"] == 1
    assert overview["top_speedups"][0]["name"] == "gemm"

    reset_engine()


def test_gpu_stream_single_event(tmp_path):
    _configure_with_sample_data(tmp_path)
    status_code, stream_text = asgi_stream_text(
        server.fastapi_app,
        "GET",
        "/api/gpu/stream?max_events=1&interval=0.1",
    )
    assert status_code == 200

    data_lines = [line for line in stream_text.splitlines() if line.startswith("data:")]
    assert data_lines
    payload = json.loads(data_lines[0].split("data:", 1)[1].strip())
    assert "gpu" in payload

    reset_engine()
