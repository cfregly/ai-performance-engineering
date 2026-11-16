"""Regression coverage for benchmark history through the Python Engine."""

import json

import core.engine as engine_module
from core.engine import get_engine, reset_engine
from core.perf_core import PerfCore


class _TestPerfCore(PerfCore):
    def __init__(self, *, history_root, data_file=None, bench_root=None):
        super().__init__(data_file=data_file, bench_root=bench_root)
        self._test_history_root = history_root

    def _tier1_history_root(self):
        return self._test_history_root


def test_engine_exposes_tier1_history_and_trends(tmp_path, sample_benchmark_results_file):
    history_root = tmp_path / "artifacts" / "history" / "tier1"
    run_dir = history_root / "20260309_010000_tier1_local"
    run_dir.mkdir(parents=True)

    summary_path = run_dir / "summary.json"
    regression_path = run_dir / "regression_summary.json"
    trend_path = run_dir / "trend_snapshot.json"
    index_path = history_root / "index.json"

    summary_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "suite_version": 1,
                "run_id": "20260309_010000_tier1_local",
                "generated_at": "2026-03-09T01:00:00",
                "targets": [
                    {
                        "key": "flashattention4_alibi",
                        "target": "labs/flashattention4:flashattention4_alibi",
                        "category": "attention",
                        "status": "succeeded",
                        "best_speedup": 12.5,
                        "artifacts": {
                            "baseline_nsys_rep": "artifacts/runs/demo/profiles/flash.nsys-rep",
                        },
                    }
                ],
                "summary": {
                    "target_count": 1,
                    "succeeded": 1,
                    "failed": 0,
                    "skipped": 0,
                    "missing": 0,
                    "avg_speedup": 12.5,
                    "median_speedup": 12.5,
                    "geomean_speedup": 12.5,
                    "representative_speedup": 12.5,
                    "max_speedup": 12.5,
                },
            }
        ),
        encoding="utf-8",
    )
    regression_path.write_text(
        json.dumps(
            {
                "baseline_run_id": "20260308_225441_tier1_manual",
                "current_run_id": "20260309_010000_tier1_local",
                "regressions": [],
                "improvements": [{"key": "flashattention4_alibi", "delta_pct": 4.2}],
                "new_targets": [],
                "missing_targets": [],
            }
        ),
        encoding="utf-8",
    )
    trend_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "run_count": 1,
                "latest_run_id": "20260309_010000_tier1_local",
                "best_speedup_seen": 12.5,
                "history": [
                    {
                        "run_id": "20260309_010000_tier1_local",
                        "run_accepted": True,
                        "baseline_eligible": True,
                        "generated_at": "2026-03-09T01:00:00",
                        "avg_speedup": 12.5,
                        "median_speedup": 12.5,
                        "geomean_speedup": 12.5,
                        "representative_speedup": 12.5,
                        "max_speedup": 12.5,
                        "succeeded": 1,
                        "failed": 0,
                        "skipped": 0,
                        "missing": 0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    index_path.write_text(
        json.dumps(
            {
                "suite_name": "tier1",
                "suite_version": 1,
                "history_root": ".",
                "runs": [
                        {
                            "run_id": "20260309_010000_tier1_local",
                            "run_accepted": True,
                            "baseline_eligible": True,
                            "summary_path": "20260309_010000_tier1_local/summary.json",
                        "regression_summary_path": (
                            "20260309_010000_tier1_local/regression_summary.md"
                        ),
                        "regression_json_path": (
                            "20260309_010000_tier1_local/regression_summary.json"
                        ),
                        "trend_snapshot_path": ("20260309_010000_tier1_local/trend_snapshot.json"),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    reset_engine()
    engine_module._handler_instance = _TestPerfCore(
        history_root=history_root,
        data_file=sample_benchmark_results_file,
        bench_root=tmp_path,
    )
    history = get_engine().benchmark.tier1_history()
    trends = get_engine().benchmark.tier1_trends()
    target_history = get_engine().benchmark.tier1_target_history(key="flashattention4_alibi")

    assert history["total_runs"] == 1
    assert history["latest_run_id"] == "20260309_010000_tier1_local"
    assert history["latest"]["run"]["representative_speedup"] == 12.5
    assert history["latest"]["improvements"][0]["key"] == "flashattention4_alibi"
    assert history["latest"]["run"]["regression_summary_json_path"] == (
        "20260309_010000_tier1_local/regression_summary.json"
    )
    assert trends["latest_run_id"] == "20260309_010000_tier1_local"
    assert trends["best_speedup_seen"] == 12.5
    assert target_history["selected_key"] == "flashattention4_alibi"
    assert target_history["run_count"] == 1
    assert target_history["history"][0]["target"] == "labs/flashattention4:flashattention4_alibi"
    assert target_history["history"][0]["best_speedup"] == 12.5
    assert target_history["history"][0]["artifacts"]["baseline_nsys_rep"] == (
        "artifacts/runs/demo/profiles/flash.nsys-rep"
    )

    reset_engine()
