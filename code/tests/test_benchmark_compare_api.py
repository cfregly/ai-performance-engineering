from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from core.engine import get_engine


def _write_run(path, timestamp, results):
    payload = {"timestamp": timestamp, "results": results}
    path.write_text(json.dumps(payload))


def test_compare_runs_python_api(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"

    _write_run(
        baseline_path,
        "2025-01-01 00:00:00",
        [
            {
                "chapter": "ch01",
                "benchmarks": [
                    {
                        "example": "gemm",
                        "best_speedup": 2.0,
                        "baseline_time_ms": 100.0,
                        "status": "succeeded",
                    },
                    {
                        "example": "mem",
                        "best_speedup": 1.0,
                        "baseline_time_ms": 50.0,
                        "status": "skipped",
                    },
                ],
            }
        ],
    )

    _write_run(
        candidate_path,
        "2025-01-02 00:00:00",
        [
            {
                "chapter": "ch01",
                "benchmarks": [
                    {
                        "example": "gemm",
                        "best_speedup": 3.0,
                        "baseline_time_ms": 100.0,
                        "status": "succeeded",
                    }
                ],
            },
            {
                "chapter": "ch02",
                "benchmarks": [
                    {
                        "example": "transfer",
                        "best_speedup": 1.5,
                        "baseline_time_ms": 200.0,
                        "status": "failed",
                    }
                ],
            },
        ],
    )

    result = get_engine().benchmark.compare(
        {"baseline": str(baseline_path), "candidate": str(candidate_path), "top": 5}
    )

    assert result["overlap"]["common"] == 1
    assert result["overlap"]["added"] == 1
    assert result["overlap"]["removed"] == 1
    assert result["improvements"][0]["name"] == "gemm"
    assert result["improvements"][0]["delta"] == 1.0
    assert result["improvements"][0]["baseline_optimized_time_ms"] is None
    assert result["improvements"][0]["candidate_optimized_time_ms"] is None
    assert result["added_benchmarks"][0]["name"] == "transfer"
    assert result["removed_benchmarks"][0]["name"] == "mem"


def _write_measured_run(path: Path, run_id: str, rows: list[dict]) -> None:
    path.write_text(json.dumps({
        "run_id": run_id,
        "results": [{"chapter": "ch01", "benchmarks": rows}],
    }), encoding="utf-8")


def _measured_row(name: str, speedup: float, *, status: str = "succeeded", kind: str = "python") -> dict:
    return {
        "example": name,
        "type": kind,
        "status": status,
        "baseline_time_ms": 20.0,
        "best_speedup": speedup,
        "optimizations": [{"status": "succeeded", "speedup": speedup, "time_ms": 20.0 / speedup}],
    }


def test_compare_excludes_rejected_candidate_and_keeps_observed_timings(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    _write_measured_run(baseline, "baseline", [
        _measured_row("x", 2.0),
        _measured_row("y", 2.0),
        _measured_row("y", 1.0, kind="cuda"),
    ])
    _write_measured_run(candidate, "candidate", [
        _measured_row("x", 99.0, status="failed_expectation_invalid"),
        _measured_row("y", 4.0),
        _measured_row("y", 2.0, kind="cuda"),
    ])

    result = get_engine().benchmark.compare({"baseline": str(baseline), "candidate": str(candidate)})

    assert result["overlap"]["common"] == 3
    assert result["overlap"]["comparable"] == 2
    assert [(row["name"], row["type"]) for row in result["improvements"]] == [
        ("y", "python"), ("y", "cuda"),
    ]
    assert result["improvements"][0]["baseline_optimized_time_ms"] == 10.0
    assert result["improvements"][0]["candidate_optimized_time_ms"] == 5.0
    assert result["regressions"] == []
    assert result["status_transitions"] == {"succeeded->failed_expectation_invalid": 1}
    assert result["diagnostics"]["baseline"] == []
    (rejected,) = result["diagnostics"]["candidate"]
    assert rejected["name"] == "x"
    assert rejected["status"] == "failed_expectation_invalid"
    assert rejected["recorded_speedup"] == 99.0
    assert rejected["speedup"] is None
    assert rejected["optimized_time_ms"] is None

    from core.benchmark.bench_commands import app

    cli = CliRunner().invoke(app, ["compare-runs", "--baseline", str(baseline), "--candidate", str(candidate)])
    assert cli.exit_code == 0, cli.output
    ranked, diagnostics = cli.stdout.split("Run diagnostics")
    assert "ch01:x" not in ranked
    assert "ch01:y: 2.00x → 4.00x" in ranked
    assert "ch01:y (cuda): 1.00x → 2.00x" in ranked
    assert "candidate ch01:x (python): failed_expectation_invalid" in diagnostics


def test_compare_keeps_legacy_name_alias_and_speedup_only_rows(tmp_path: Path) -> None:
    baseline, candidate = tmp_path / "baseline.json", tmp_path / "candidate.json"
    _write_run(baseline, "2025-01-01 00:00:00", [
        {"chapter": "ch01", "benchmarks": [{"name": "legacy", "best_speedup": 2.0}]},
    ])
    _write_run(candidate, "2025-01-02 00:00:00", [
        {"chapter": "ch01", "benchmarks": [{"example": "legacy", "best_speedup": 1.25}]},
    ])

    result = get_engine().benchmark.compare({"baseline": str(baseline), "candidate": str(candidate)})

    (regression,) = result["regressions"]
    assert regression["key"] == "ch01:legacy"
    assert regression["delta"] == -0.75
    assert regression["baseline_time_ms"] is None
    assert regression["candidate_optimized_time_ms"] is None
