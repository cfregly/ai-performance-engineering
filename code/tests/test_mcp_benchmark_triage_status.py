from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

import mcp.mcp_server as mcp_server


@pytest.mark.parametrize("has_qualified_result", [True, False])
def test_benchmark_triage_counts_diagnostics_without_ranking_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, has_qualified_result: bool
) -> None:
    data = {
        "timestamp": "2026-03-01T00:00:00",
        "results": [
            {
                "chapter": "ch_test",
                "benchmarks": [
                    {
                        "example": "ok_case",
                        "best_speedup": 1.20,
                        "baseline_time_ms": 10.0,
                        "status": "succeeded",
                        "optimizations": [
                            {"status": "succeeded", "speedup": 1.20, "time_ms": 10.0 / 1.20}
                        ],
                    },
                    {
                        "example": "baseline_only_case",
                        "best_speedup": 12.0,
                        "baseline_time_ms": 10.0,
                        "status": "succeeded",
                        "optimizations": [],
                    },
                    {
                        "example": "missing_variant_timing_case",
                        "best_speedup": 5.0,
                        "baseline_time_ms": 10.0,
                        "status": "succeeded",
                        "optimizations": [{"status": "succeeded", "speedup": 5.0}],
                    },
                    {
                        "example": "verify_case",
                        "best_speedup": 99.0,
                        "baseline_time_ms": 110.0,
                        "status": "failed_verification",
                        "optimizations": [],
                    },
                    {
                        "example": "error_case",
                        "best_speedup": 0.25,
                        "baseline_time_ms": 12.0,
                        "status": "failed_error",
                        "optimizations": [],
                    },
                    {
                        "example": "flat_case",
                        "best_speedup": 1.02,
                        "baseline_time_ms": 12.5,
                        "status": "failed_no_speedup",
                        "optimizations": [],
                    },
                    {
                        "example": "skipped_case",
                        "best_speedup": 1.00,
                        "baseline_time_ms": 13.0,
                        "status": "skipped_unsupported_hardware",
                        "optimizations": [],
                    },
                ],
            }
        ],
    }
    if not has_qualified_result:
        data["results"][0]["benchmarks"].pop(0)
    data_path = tmp_path / "benchmark_test_results.json"
    data_path.write_text(json.dumps(data), encoding="utf-8")

    monkeypatch.setattr(mcp_server, "_context_snapshot", lambda: {})
    response = asyncio.run(
        mcp_server.MCPServer().handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "benchmark_triage", "arguments": {"data_file": str(data_path)}},
            }
        )
    )
    assert response is not None
    tool_response = response["result"]
    assert tool_response["isError"] is False
    result = json.loads(tool_response["content"][0]["text"])["result"]
    summary = result["summary"]

    assert summary["total_benchmarks"] == (7 if has_qualified_result else 6)
    assert summary["passed"] == int(has_qualified_result)
    assert summary["failed"] == 3
    assert summary["skipped"] == 1
    assert summary["failure_classes"]["failed_error"] == 1
    assert summary["failure_classes"]["failed_verification"] == 1
    assert summary["failure_classes"]["failed_no_speedup"] == 1
    assert summary["avg_speedup"] == (1.20 if has_qualified_result else 0)
    assert result["regressions"] == []
    assert [row["benchmark"] for row in result["improvements"]] == (
        ["ch_test:ok_case"] if has_qualified_result else []
    )
    tools = {item["tool"] for item in result["recommended_tools"]}
    assert "profile_nsys" in tools
    assert ("benchmark_compare_runs" in tools) is has_qualified_result
    assert "Review the failed benchmark results." in result["next_steps_summary"]
