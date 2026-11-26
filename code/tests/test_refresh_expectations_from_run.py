import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.analysis.refresh_expectations_from_run import refresh_expectations_from_run

TEST_REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_refresh_expectations_from_run_updates_selected_target(tmp_path: Path):
    repo_root = tmp_path
    chapter_dir = repo_root / "ch01"
    chapter_dir.mkdir(parents=True)
    _write_json(
        chapter_dir / "expectations_b200.json",
        {
            "schema_version": 2,
            "hardware_key": "b200",
            "examples": {
                "gemm": {
                    "example": "gemm",
                    "type": "python",
                    "metrics": {
                        "baseline_time_ms": 10.0,
                        "best_optimized_time_ms": 5.0,
                        "best_speedup": 2.0,
                        "best_optimized_speedup": 2.0,
                        "is_regression": False,
                    },
                    "provenance": {
                        "git_commit": "oldsha",
                        "hardware_key": "b200",
                        "profile_name": "none",
                        "timestamp": "2026-03-10T00:00:00Z",
                        "iterations": 20,
                        "warmup_iterations": 5,
                    },
                    "metadata": {
                        "optimization_goal": "speed",
                        "best_optimization_speedup": 2.0,
                    },
                }
            },
        },
    )

    results_json = repo_root / "artifacts" / "runs" / "rerun" / "results" / "benchmark_test_results.json"
    _write_json(
        results_json,
        {
            "timestamp": "2026-03-11T00:00:00Z",
            "results": [
                {
                    "chapter": "ch01",
                    "status": "completed",
                    "benchmarks": [
                        {
                            "example": "gemm",
                            "type": "python",
                            "expectation": {
                                "entry": {
                                    "example": "gemm",
                                    "type": "python",
                                    "metrics": {
                                        "baseline_time_ms": 10.0,
                                        "best_optimized_time_ms": 4.0,
                                        "best_speedup": 2.5,
                                        "best_optimized_speedup": 2.5,
                                        "is_regression": False,
                                    },
                                    "provenance": {
                                        "git_commit": "newsha",
                                        "hardware_key": "b200",
                                        "profile_name": "minimal",
                                        "timestamp": "2026-03-11T00:00:00Z",
                                        "iterations": 20,
                                        "warmup_iterations": 5,
                                    },
                                    "metadata": {
                                        "optimization_goal": "speed",
                                        "best_optimization_speedup": 2.5,
                                    },
                                }
                            },
                        }
                    ],
                }
            ],
        },
    )

    summary = refresh_expectations_from_run(
        results_json=results_json,
        repo_root=repo_root,
        targets={"ch01:gemm"},
        validity_profile="strict",
        accept_regressions=True,
        dry_run=False,
    )

    assert summary["counts"]["applied"] == 1
    assert summary["counts"]["updated"] == 1
    updated = json.loads((chapter_dir / "expectations_b200.json").read_text(encoding="utf-8"))
    entry = updated["examples"]["gemm"]
    assert entry["metrics"]["best_speedup"] == 2.5
    assert entry["provenance"]["git_commit"] == "newsha"
    assert entry["provenance"]["validity_profile"] is None
    assert entry["metadata"]["validity_profile_override"] == "strict"
    assert entry["metadata"]["validity_profile_source"] == "user_fallback"
    assert summary["validity_profile"] is None
    assert entry["provenance"]["execution_environment"] == "unknown"
    assert entry["provenance"]["dmi_product_name"] is None
    assert summary["records"][0]["validity_profile_source"] == "user_fallback"
    assert any("status was unrecorded" in warning for warning in summary["warnings"])


def test_refresh_expectations_from_run_skips_missing_preview_entry(tmp_path: Path):
    repo_root = tmp_path
    (repo_root / "ch01").mkdir(parents=True)
    results_json = repo_root / "artifacts" / "runs" / "rerun" / "results" / "benchmark_test_results.json"
    _write_json(
        results_json,
        {
            "results": [
                {
                    "chapter": "ch01",
                    "benchmarks": [
                        {
                            "example": "gemm",
                            "type": "python",
                        }
                    ],
                }
            ]
        },
    )

    summary = refresh_expectations_from_run(
        results_json=results_json,
        repo_root=repo_root,
        targets={"ch01:gemm"},
        dry_run=True,
    )

    assert summary["counts"]["skipped"] == 1


def test_refresh_expectations_from_run_infers_validity_profile_from_run_events(tmp_path: Path):
    repo_root = tmp_path
    chapter_dir = repo_root / "ch01"
    chapter_dir.mkdir(parents=True)
    _write_json(
        chapter_dir / "expectations_b200.json",
        {
            "schema_version": 2,
            "hardware_key": "b200",
            "examples": {
                "gemm_cuda": {
                    "example": "gemm",
                    "type": "cuda",
                    "metrics": {
                        "baseline_time_ms": 10.0,
                        "best_optimized_time_ms": 5.0,
                        "best_speedup": 2.0,
                        "best_optimized_speedup": 2.0,
                        "is_regression": False,
                    },
                    "provenance": {
                        "git_commit": "oldsha",
                        "hardware_key": "b200",
                        "profile_name": "none",
                        "timestamp": "2026-03-10T00:00:00Z",
                        "iterations": 20,
                        "warmup_iterations": 5,
                    },
                    "metadata": {
                        "optimization_goal": "speed",
                        "best_optimization_speedup": 2.0,
                    },
                }
            },
        },
    )

    run_dir = repo_root / "artifacts" / "runs" / "rerun"
    results_json = run_dir / "results" / "benchmark_test_results.json"
    _write_json(
        results_json,
        {
            "timestamp": "2026-03-11T00:00:00Z",
            "results": [
                {
                    "chapter": "ch01",
                    "status": "completed",
                    "benchmarks": [
                        {
                            "example": "gemm",
                            "type": "cuda",
                            "expectation": {
                                "entry": {
                                    "example": "gemm",
                                    "type": "cuda",
                                    "metrics": {
                                        "baseline_time_ms": 10.0,
                                        "best_optimized_time_ms": 5.0,
                                        "best_speedup": 2.0,
                                        "best_optimized_speedup": 2.0,
                                        "is_regression": False,
                                    },
                                    "provenance": {
                                        "git_commit": "newsha",
                                        "hardware_key": "b200",
                                        "profile_name": "minimal",
                                        "timestamp": "2026-03-11T00:00:00Z",
                                        "iterations": 20,
                                        "warmup_iterations": 5,
                                    },
                                    "metadata": {
                                        "optimization_goal": "speed",
                                        "best_optimization_speedup": 2.0,
                                    },
                                }
                            },
                        }
                    ],
                }
            ],
        },
    )
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "benchmark_events.jsonl").write_text(
        json.dumps({"event_type": "run_start", "validity_profile": "strict"}) + "\n",
        encoding="utf-8",
    )

    summary = refresh_expectations_from_run(
        results_json=results_json,
        repo_root=repo_root,
        targets={"ch01:gemm_cuda"},
        dry_run=False,
    )

    assert summary["validity_profile"] is None
    assert summary["legacy_run_start_validity_profile"] == "strict"
    assert summary["validity_profile_source"] == "legacy_unbound_run_start"
    updated = json.loads((chapter_dir / "expectations_b200.json").read_text(encoding="utf-8"))
    entry = updated["examples"]["gemm_cuda"]
    assert entry["provenance"]["validity_profile"] is None
    assert entry["metadata"]["legacy_run_start_validity_profile"] == "strict"
    assert entry["metadata"]["validity_profile_source"] == "legacy_unbound_run_start"


def test_refresh_expectations_from_run_surfaces_event_log_parse_warnings(tmp_path: Path):
    repo_root = tmp_path
    chapter_dir = repo_root / "ch01"
    chapter_dir.mkdir(parents=True)
    _write_json(
        chapter_dir / "expectations_b200.json",
        {
            "schema_version": 2,
            "hardware_key": "b200",
            "examples": {
                "gemm": {
                    "example": "gemm",
                    "type": "python",
                    "metrics": {
                        "baseline_time_ms": 10.0,
                        "best_optimized_time_ms": 5.0,
                        "best_speedup": 2.0,
                        "best_optimized_speedup": 2.0,
                        "is_regression": False,
                    },
                    "provenance": {
                        "git_commit": "oldsha",
                        "hardware_key": "b200",
                        "profile_name": "none",
                        "timestamp": "2026-03-10T00:00:00Z",
                        "iterations": 20,
                        "warmup_iterations": 5,
                    },
                    "metadata": {
                        "optimization_goal": "speed",
                        "best_optimization_speedup": 2.0,
                    },
                }
            },
        },
    )
    run_dir = repo_root / "artifacts" / "runs" / "rerun"
    results_json = run_dir / "results" / "benchmark_test_results.json"
    _write_json(
        results_json,
        {
            "results": [
                {
                    "chapter": "ch01",
                    "benchmarks": [
                        {
                            "example": "gemm",
                            "type": "python",
                            "expectation": {
                                "entry": {
                                    "example": "gemm",
                                    "type": "python",
                                    "metrics": {
                                        "baseline_time_ms": 10.0,
                                        "best_optimized_time_ms": 4.0,
                                        "best_speedup": 2.5,
                                        "best_optimized_speedup": 2.5,
                                        "is_regression": False,
                                    },
                                    "provenance": {
                                        "git_commit": "newsha",
                                        "hardware_key": "b200",
                                        "profile_name": "minimal",
                                        "timestamp": "2026-03-11T00:00:00Z",
                                        "iterations": 20,
                                        "warmup_iterations": 5,
                                    },
                                    "metadata": {
                                        "optimization_goal": "speed",
                                        "best_optimization_speedup": 2.5,
                                    },
                                }
                            },
                        }
                    ],
                }
            ]
        },
    )
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs" / "benchmark_events.jsonl").write_text("{not-json\n", encoding="utf-8")

    summary = refresh_expectations_from_run(
        results_json=results_json,
        repo_root=repo_root,
        targets={"ch01:gemm"},
        dry_run=True,
    )

    assert summary["validity_profile"] is None
    assert summary["warnings"]
    assert "Failed to parse benchmark events log" in summary["warnings"][0]


def test_refresh_expectations_from_run_rejects_malformed_results(tmp_path: Path):
    repo_root = tmp_path
    (repo_root / "ch01").mkdir(parents=True)
    results_json = repo_root / "artifacts" / "runs" / "rerun" / "results" / "benchmark_test_results.json"
    results_json.parent.mkdir(parents=True, exist_ok=True)
    results_json.write_text("{not-json", encoding="utf-8")

    try:
        refresh_expectations_from_run(
            results_json=results_json,
            repo_root=repo_root,
            dry_run=True,
        )
    except ValueError as exc:
        assert "Failed to read benchmark results" in str(exc)
        assert str(results_json) in str(exc)
    else:
        raise AssertionError("Expected malformed benchmark results to raise ValueError")


def test_refresh_expectations_from_run_cli_rejects_unreadable_targets_file(tmp_path: Path):
    repo_root = tmp_path
    results_json = repo_root / "artifacts" / "runs" / "rerun" / "results" / "benchmark_test_results.json"
    _write_json(results_json, {"results": []})
    targets_file = repo_root / "missing_targets.txt"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.analysis.refresh_expectations_from_run",
            "--repo-root",
            str(repo_root),
            "--results-json",
            str(results_json),
            "--targets-file",
            str(targets_file),
        ],
        cwd=TEST_REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "Failed to read targets file" in completed.stderr
    assert str(targets_file) in completed.stderr
    assert "Traceback" not in completed.stderr


def _preview_row(status: str, *, optimize_ms: float = 5.0) -> dict:
    return {
        "example": "gemm",
        "status": status,
        "type": "python",
        "baseline_time_ms": 10.0,
        "optimizations": [{"status": "succeeded", "time_ms": optimize_ms}],
        "expectation": {
            "status": "updated",
            "entry": {
                "example": "gemm",
                "type": "python",
                "metrics": {"baseline_time_ms": 10.0, "best_optimized_time_ms": optimize_ms},
                "provenance": {
                    "git_commit": "runsha",
                    "hardware_key": "b200",
                    "profile_name": "none",
                    "timestamp": "2026-03-11T00:00:00Z",
                    "execution_environment": "virtualized",
                    "dmi_product_name": "Measuring host",
                    "validity_profile": "strict",
                    "iterations": 20,
                    "warmup_iterations": 5,
                },
            },
        },
    }


def _write_preview(root: Path, row: dict, *, manifest: dict | None = None) -> Path:
    (root / "ch01").mkdir(exist_ok=True)
    run = root / "artifacts" / "runs" / "measured"
    results = run / "results" / "benchmark_test_results.json"
    _write_json(results, {
        "run_id": "measured",
        "results": [{"chapter": "ch01", "benchmarks": [row]}],
    })
    if manifest is not None:
        _write_json(run / "manifest.json", manifest)
    return results


@pytest.mark.parametrize("status", [
    "failed_verification", "failed_error", "failed_profiler", "failed_preflight",
    "failed_expectation_invalid", "skipped", "unrecognized",
])
def test_refresh_rejects_ineligible_rows_even_with_a_preview(tmp_path: Path, status: str) -> None:
    results = _write_preview(tmp_path, _preview_row(status))

    summary = refresh_expectations_from_run(results_json=results, repo_root=tmp_path)

    assert summary["counts"]["rejected"] == 1
    assert status in summary["records"][0]["message"]
    assert summary["updated_files"] == []
    assert not (tmp_path / "ch01" / "expectations_b200.json").exists()


@pytest.mark.parametrize("status", ["failed_regression", "failed_no_speedup"])
def test_refresh_allows_documented_rebaselining(tmp_path: Path, status: str) -> None:
    baseline = _write_preview(tmp_path, _preview_row("succeeded"))
    refresh_expectations_from_run(results_json=baseline, repo_root=tmp_path)
    expectation_file = tmp_path / "ch01" / "expectations_b200.json"
    before = expectation_file.read_bytes()

    row = _preview_row(status, optimize_ms=12.5)
    if status == "failed_regression":
        row["expectation"].update(status="rejected", validation_issues=[{"issue_type": "regression"}])
    results = _write_preview(tmp_path, row)

    summary = refresh_expectations_from_run(results_json=results, repo_root=tmp_path)
    assert not summary["accept_regressions"]
    assert summary["counts"]["rejected"] == 1
    assert not summary["updated_files"]
    assert expectation_file.read_bytes() == before

    summary = refresh_expectations_from_run(results_json=results, repo_root=tmp_path, accept_regressions=True)
    assert summary["counts"]["regressed"] == 1
    updated = json.loads(expectation_file.read_bytes())["examples"]["gemm"]
    assert updated["metrics"]["best_speedup"] == 0.8
    receipt = updated["metadata"]["expectation_refresh"]
    assert receipt["accept_regressions"]
    assert receipt["rebaselined"]
    assert receipt["benchmark_status"] == status


def test_refresh_rebaselines_a_documented_dropped_throughput_metric(tmp_path: Path) -> None:
    first = _preview_row("succeeded")
    first["expectation"]["entry"]["metrics"]["best_optimized_throughput.tokens_per_s"] = 100.0
    results = _write_preview(tmp_path, first)
    refresh_expectations_from_run(results_json=results, repo_root=tmp_path)

    second = _preview_row("failed_expectation_invalid")
    second["expectation"].update(
        status="rejected", validation_issues=[{"issue_type": "tracked_metric_missing"}]
    )
    _write_preview(tmp_path, second)
    denied = refresh_expectations_from_run(results_json=results, repo_root=tmp_path, accept_regressions=False)
    assert denied["counts"]["rejected"] == 1

    applied = refresh_expectations_from_run(results_json=results, repo_root=tmp_path, accept_regressions=True)
    assert applied["counts"]["applied"] == 1
    store = json.loads((tmp_path / "ch01" / "expectations_b200.json").read_bytes())
    assert "best_optimized_throughput.tokens_per_s" not in store["examples"]["gemm"]["metrics"]


@pytest.mark.parametrize("issue", [
    {"status": "skipped"},
    {"required_profiler_failures": ["ncu"]},
    {"validation_issues": [{"issue_type": "timing_outlier_skew"}]},
])
def test_refresh_rejects_preview_that_failed_its_own_gate(tmp_path: Path, issue: dict) -> None:
    row = _preview_row("succeeded")
    row["expectation"].update(issue)
    results = _write_preview(tmp_path, row)

    summary = refresh_expectations_from_run(results_json=results, repo_root=tmp_path)

    assert summary["counts"]["rejected"] == 1
    assert summary["updated_files"] == []
    assert not (tmp_path / "ch01" / "expectations_b200.json").exists()


def test_refresh_checks_measured_manifest_and_keeps_recorded_environment(tmp_path: Path) -> None:
    manifest = {
        "run_id": "measured",
        "git": {"commit": "runsha"},
        "manifests": [{
            "target_label": "ch01:gemm",
            "variant": "baseline",
            "manifest": {"git": {"commit": "runsha"}, "config": {"validity_profile": "strict"}},
        }],
    }
    row = _preview_row("succeeded")
    row["expectation"].update(status="rejected", validation_issues=[{"issue_type": "provenance_mismatch"}])
    results = _write_preview(tmp_path, row, manifest=manifest)
    summary = refresh_expectations_from_run(results_json=results, repo_root=tmp_path, validity_profile="portable")

    assert summary["manifest_run_id_match"]
    assert summary["validity_profile"] == "strict"
    assert summary["validity_profile_override"] == "portable"
    assert summary["records"][0]["validity_profile_source"] == "entry"
    expectation_file = tmp_path / "ch01" / "expectations_b200.json"
    provenance = json.loads(expectation_file.read_bytes())["examples"]["gemm"]["provenance"]
    assert provenance["execution_environment"] == "virtualized"
    assert provenance["dmi_product_name"] == "Measuring host"
    assert provenance["validity_profile"] == "strict"

    before = expectation_file.read_bytes()
    manifest["git"]["commit"] = "different-sha"
    _write_preview(tmp_path, row, manifest=manifest)
    summary = refresh_expectations_from_run(results_json=results, repo_root=tmp_path)
    assert summary["counts"]["rejected"] == 1
    assert "Git commit" in summary["records"][0]["message"]

    manifest["run_id"] = "another-run"
    _write_preview(tmp_path, row, manifest=manifest)
    with pytest.raises(ValueError, match="run_id.*another-run.*measured"):
        refresh_expectations_from_run(results_json=results, repo_root=tmp_path)
    assert expectation_file.read_bytes() == before


@pytest.mark.parametrize(("hardware_key", "measured_model", "applied"), [
    ("h100", "NVIDIA B200", False),
    ("2x_b200", "NVIDIA B200", True),
    ("b200", None, True),
])
def test_refresh_reconciles_hardware_key_with_the_matched_measuring_gpu_model(
    tmp_path: Path, hardware_key: str, measured_model: str | None, applied: bool
) -> None:
    row = _preview_row("succeeded")
    row["expectation"]["entry"]["provenance"]["hardware_key"] = hardware_key
    manifest = {
        "run_id": "measured",
        "git": {"commit": "runsha"},
        "manifests": [
            {
                "target_label": "ch01:unrelated",
                "variant": "baseline",
                "manifest": {"hardware": {"gpu_model": "NVIDIA A100"}},
            },
            {
                "target_label": "ch01:gemm",
                "variant": "baseline",
                "manifest": {"hardware": {"gpu_model": measured_model}},
            },
        ],
    }
    results = _write_preview(tmp_path, row, manifest=manifest)

    summary = refresh_expectations_from_run(results_json=results, repo_root=tmp_path)

    assert summary["counts"]["applied"] == int(applied)
    assert (tmp_path / "ch01" / f"expectations_{hardware_key}.json").exists() == applied
    if not applied:
        assert "'h100' disagrees with the measuring GPU model 'NVIDIA B200'" in summary["records"][0]["message"]


def test_refresh_marks_measurement_manifests_absent_for_cuda(tmp_path: Path) -> None:
    row = _preview_row("succeeded")
    row["type"] = "cuda"
    row["expectation"]["entry"]["type"] = "cuda"
    manifest = {"run_id": "measured", "git": {"commit": "runsha"}, "manifests": []}
    results = _write_preview(tmp_path, row, manifest=manifest)

    summary = refresh_expectations_from_run(results_json=results, repo_root=tmp_path)

    assert summary["manifest_run_id_match"]
    assert any("No matching measurement manifest" in warning for warning in summary["warnings"])
    expectations = json.loads((tmp_path / "ch01" / "expectations_b200.json").read_bytes())
    receipt = expectations["examples"]["gemm_cuda"]["metadata"]["expectation_refresh"]
    assert receipt["measurement_manifest_variants"] == []


def test_refresh_cli_requires_explicit_rebaseline_opt_in(tmp_path: Path) -> None:
    results = _write_preview(tmp_path, _preview_row("failed_no_speedup", optimize_ms=12.5))
    args = [
        sys.executable, "-m", "core.analysis.refresh_expectations_from_run",
        "--repo-root", str(tmp_path), "--results-json", str(results),
    ]
    denied = subprocess.run(args, cwd=TEST_REPO_ROOT, capture_output=True, text=True, check=True)
    assert json.loads(denied.stdout)["counts"]["rejected"] == 1
    assert not (tmp_path / "ch01" / "expectations_b200.json").exists()

    allowed = subprocess.run(
        [*args, "--accept-regressions"], cwd=TEST_REPO_ROOT, capture_output=True, text=True, check=True
    )
    assert json.loads(allowed.stdout)["counts"]["applied"] == 1
    assert (tmp_path / "ch01" / "expectations_b200.json").exists()
