"""Regressions for W2-077 (throughput regression gate) and W2-133 (outlier timing gate)."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from core.analysis.refresh_expectations_from_run import refresh_expectations_from_run
from core.benchmark.expectations import (
    MAX_MEAN_SKEW_RATIO,
    RELATIVE_TOLERANCE,
    ExpectationEntry,
    ExpectationsStore,
    RunProvenance,
)
from core.benchmark.models import ThroughputStats
from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness
from core.harness.run_benchmarks import (
    EXPECTATION_THROUGHPUT_FIELDS,
    _classify_benchmark_status,
    _expectation_entry_from_result,
    _expectation_preview_message,
    _expectation_rejection_status,
    _record_timing_fields,
    _update_expectation_entry_after_profiler_gate,
    collect_expectation_metrics,
    generate_markdown_report,
    serialize_throughput,
)

# Values from the retired 4x_gb200 llama_3_1_8b row. The baseline mean was
# 6.2x its p75, which turned a 2.4x p75 speedup into a stored 15.06x bar.
CORRUPTED_BASELINE_MEAN_MS = 52.50560550689697
CORRUPTED_BASELINE_P75_MS = 8.510496139526367
CORRUPTED_OPTIMIZED_MEAN_MS = 3.487565254342967
CORRUPTED_OPTIMIZED_P75_MS = 3.5392959117889404


def _provenance(git_commit: str = "abc123") -> RunProvenance:
    return RunProvenance(
        git_commit=git_commit,
        hardware_key="b200",
        profile_name="none",
        timestamp="2026-10-12T00:00:00",
        iterations=20,
        warmup_iterations=5,
        execution_environment="bare_metal",
        validity_profile="strict",
    )


def _throughput(
    mean_ms: float,
    *,
    bytes_per_iteration: float | None,
    tokens_per_iteration: float | None = None,
    flops_per_iteration: float | None = None,
    custom_unit_name: str | None = None,
) -> dict:
    seconds = mean_ms / 1000.0
    payload = serialize_throughput(
        ThroughputStats(
            requests_per_s=1.0 / seconds,
            tokens_per_s=None if tokens_per_iteration is None else tokens_per_iteration / seconds,
            bytes_per_s=None if bytes_per_iteration is None else bytes_per_iteration / seconds,
            custom_unit_per_s=None if flops_per_iteration is None else flops_per_iteration / seconds,
            custom_unit_name=custom_unit_name,
            latency_ms=mean_ms,
        )
    )
    assert payload is not None
    return payload


def _entry(
    baseline_ms: float,
    optimized_ms: float,
    *,
    bytes_per_iteration: float | None = 4.0e9,
    tokens_per_iteration: float | None = None,
    flops_per_iteration: float | None = None,
    custom_unit_name: str | None = None,
    baseline_p75_ms: float | None = None,
    optimized_p75_ms: float | None = None,
    git_commit: str = "abc123",
) -> ExpectationEntry:
    def throughput(mean_ms: float) -> dict:
        return _throughput(
            mean_ms,
            bytes_per_iteration=bytes_per_iteration,
            tokens_per_iteration=tokens_per_iteration,
            flops_per_iteration=flops_per_iteration,
            custom_unit_name=custom_unit_name,
        )

    return ExpectationEntry(
        example="stream_copy",
        type="python",
        optimization_goal="speed",
        baseline_time_ms=baseline_ms,
        best_optimized_time_ms=optimized_ms,
        provenance=_provenance(git_commit),
        baseline_p75_ms=baseline_p75_ms,
        best_optimized_p75_ms=optimized_p75_ms,
        baseline_throughput=throughput(baseline_ms),
        best_optimized_throughput=throughput(optimized_ms),
    )


def _store(
    tmp_path: Path,
    *,
    accept_regressions: bool = False,
    allow_mixed_provenance: bool = True,
) -> ExpectationsStore:
    return ExpectationsStore(
        tmp_path,
        "b200",
        accept_regressions=accept_regressions,
        allow_mixed_provenance=allow_mixed_provenance,
    )


def _write(store: ExpectationsStore, entry: ExpectationEntry, example_key: str = "stream_copy"):
    update_result, payload = _update_expectation_entry_after_profiler_gate(
        store, example_key, entry, profiler_failures=[]
    )
    assert update_result is not None
    return update_result, payload


def _seed(tmp_path: Path, entry: ExpectationEntry) -> None:
    store = _store(tmp_path)
    update_result, _ = _write(store, entry)
    assert update_result.status == "updated"
    store.save()


def _issue_metrics(update_result, issue_type: str) -> set:
    return {
        issue.message.split(" ", 1)[0]
        for issue in update_result.validation_issues
        if issue.issue_type == issue_type
    }


def _issue_types(update_result) -> set:
    return {issue.issue_type for issue in update_result.validation_issues}


def _benchmark_status(entry: ExpectationEntry, update_result) -> str:
    """Return the status both harness status sites assign to this result."""
    result_entry = {"optimization_goal": entry.optimization_goal, "best_speedup": entry.best_speedup}
    return _classify_benchmark_status(result_entry, update_result, profiler_failures=[])


def _cpu_harness(percentiles: list[float]) -> tuple[BenchmarkHarness, BenchmarkConfig]:
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=10,
        warmup=5,
        use_subprocess=False,
        enable_profiling=False,
        enable_memory_tracking=False,
        # Only the statistics step runs here. No CUDA is emulated.
        enforce_environment_validation=False,
        percentiles=percentiles,
    )
    harness = BenchmarkHarness(config=config)
    harness._ensure_runtime_initialized()
    return harness, config


def _harness_result_entry(baseline_samples: list[float], optimized_samples: list[float]):
    """Fill result fields the way run_benchmarks does for a decode-style config."""
    harness, config = _cpu_harness([50, 90, 99])
    baseline = harness._compute_stats(baseline_samples, config).timing
    optimized = harness._compute_stats(optimized_samples, config).timing
    result_entry: dict = {
        "example": "decode",
        "type": "python",
        "optimization_goal": "speed",
        "baseline_time_ms": baseline.mean_ms,
    }
    _record_timing_fields(
        result_entry,
        "baseline_",
        median_ms=baseline.median_ms,
        percentiles=baseline.percentiles,
        p90_ms=baseline.p90_ms,
    )
    best_opt: dict = {"file": "optimized_decode.py", "status": "succeeded", "time_ms": optimized.mean_ms}
    _record_timing_fields(
        best_opt,
        "",
        median_ms=optimized.median_ms,
        percentiles=optimized.percentiles,
        p90_ms=optimized.p90_ms,
    )
    entry = _expectation_entry_from_result(
        result_entry,
        best_opt,
        example_key="decode",
        default_type="python",
        provenance=_provenance(),
    )
    return result_entry, best_opt, entry


def test_w2_077_collector_keeps_every_fixed_unit_throughput_field() -> None:
    fixed_unit_fields = set(ThroughputStats.model_fields) - {
        "custom_unit_per_s",
        "custom_unit_name",
        "schemaVersion",
    }
    assert set(EXPECTATION_THROUGHPUT_FIELDS) == fixed_unit_fields

    result_entry = {
        "optimization_goal": "speed",
        "baseline_time_ms": 10.0,
        "baseline_throughput": _throughput(10.0, bytes_per_iteration=4.0e9),
        "optimizations": [
            {
                "status": "succeeded",
                "time_ms": 5.0,
                "speedup": 2.0,
                "throughput": _throughput(5.0, bytes_per_iteration=4.0e9),
            }
        ],
    }
    metrics, _ = collect_expectation_metrics(result_entry)

    assert metrics["baseline_throughput.bytes_per_s"] == 4.0e11
    assert metrics["best_optimized_throughput.bytes_per_s"] == 8.0e11


def test_w2_077_bytes_per_s_regression_fails_the_benchmark_when_timing_is_unchanged(tmp_path: Path) -> None:
    _seed(tmp_path, _entry(10.0, 5.0, bytes_per_iteration=4.0e9))

    store = _store(tmp_path)
    update_result, payload = _write(store, _entry(10.0, 5.0, bytes_per_iteration=2.4e9))

    assert update_result.status == "rejected"
    assert payload["status"] == "rejected"
    assert _issue_metrics(update_result, "regression") == {
        "baseline_throughput.bytes_per_s",
        "best_optimized_throughput.bytes_per_s",
    }
    assert _expectation_rejection_status(update_result) == "failed_regression"
    stored = store.get_entry("stream_copy")
    assert stored is not None
    assert stored.baseline_throughput["bytes_per_s"] == 4.0e11


def test_w2_077_shared_slowdown_is_rejected_although_speedup_is_unchanged(tmp_path: Path) -> None:
    _seed(tmp_path, _entry(10.0, 5.0))

    update_result, _ = _write(_store(tmp_path), _entry(14.0, 7.0))

    assert update_result.status == "rejected"
    regressed = _issue_metrics(update_result, "regression")
    for prefix in ("baseline_throughput", "best_optimized_throughput"):
        assert {
            f"{prefix}.bytes_per_s",
            f"{prefix}.requests_per_s",
            f"{prefix}.latency_ms",
        } <= regressed
    assert _expectation_rejection_status(update_result) == "failed_regression"


def test_w2_077_accepted_throughput_regression_is_stored_and_reported(tmp_path: Path) -> None:
    _seed(tmp_path, _entry(10.0, 5.0, bytes_per_iteration=4.0e9))

    store = _store(tmp_path, accept_regressions=True)
    update_result, payload = _write(store, _entry(10.0, 5.0, bytes_per_iteration=2.4e9))

    assert update_result.status == "regressed"
    assert "baseline_throughput.bytes_per_s" in update_result.message
    assert {issue["issue_type"] for issue in payload["validation_issues"]} == {"regression"}
    assert _expectation_rejection_status(update_result) is None
    stored = store.get_entry("stream_copy")
    assert stored is not None
    assert stored.baseline_throughput["bytes_per_s"] == 2.4e11


def test_w2_077_dropping_a_tracked_throughput_metric_fails_as_invalid_evidence(tmp_path: Path) -> None:
    _seed(tmp_path, _entry(10.0, 5.0, bytes_per_iteration=4.0e9))

    no_bytes = _entry(10.0, 5.0, bytes_per_iteration=None)
    update_result, _ = _write(_store(tmp_path), no_bytes)

    assert update_result.status == "rejected"
    assert _issue_metrics(update_result, "tracked_metric_missing") == {
        "baseline_throughput.bytes_per_s",
        "best_optimized_throughput.bytes_per_s",
    }
    assert _benchmark_status(no_bytes, update_result) == "failed_expectation_invalid"
    # A deliberate refresh may drop the metric, so the preview names the flags.
    assert "--update-expectations" in _expectation_preview_message(update_result)


def test_w2_077_custom_unit_throughput_is_compared_only_in_the_same_unit(tmp_path: Path) -> None:
    _seed(
        tmp_path,
        _entry(10.0, 5.0, flops_per_iteration=2.0e12, custom_unit_name="FLOPs"),
    )

    slower = _write(
        _store(tmp_path),
        _entry(10.0, 5.0, flops_per_iteration=1.2e12, custom_unit_name="FLOPs"),
    )[0]
    assert slower.status == "rejected"
    assert "best_optimized_throughput.custom_unit_per_s" in _issue_metrics(slower, "regression")

    renamed = _write(
        _store(tmp_path),
        _entry(10.0, 5.0, flops_per_iteration=2.0e12, custom_unit_name="tokens"),
    )[0]
    assert renamed.status == "rejected"
    assert _issue_metrics(renamed, "tracked_metric_missing") == {
        "baseline_throughput.custom_unit_per_s",
        "best_optimized_throughput.custom_unit_per_s",
    }


def test_w2_077_stored_zero_rate_is_not_a_tracked_measurement(tmp_path: Path) -> None:
    # Checked-in continuous_batching rows store tokens_per_s = 0 because the
    # workload declared zero tokens. A completed run cannot have a zero rate.
    _seed(tmp_path, _entry(10.0, 5.0, tokens_per_iteration=0.0))

    no_tokens, _ = _write(_store(tmp_path), _entry(10.0, 5.0, tokens_per_iteration=None))
    assert no_tokens.status == "unchanged"

    _seed(tmp_path / "positive", _entry(10.0, 5.0, tokens_per_iteration=512.0))
    zero_tokens, _ = _write(
        _store(tmp_path / "positive"), _entry(10.0, 5.0, tokens_per_iteration=0.0)
    )
    assert zero_tokens.status == "rejected"
    assert _issue_metrics(zero_tokens, "tracked_metric_missing") == {
        "baseline_throughput.tokens_per_s",
        "best_optimized_throughput.tokens_per_s",
    }


def test_w2_133_outlier_skewed_mean_is_refused_even_with_forced_update(tmp_path: Path) -> None:
    # accept_regressions and allow_mixed_provenance match --update-expectations.
    store = _store(tmp_path, accept_regressions=True)
    corrupted = _entry(
        CORRUPTED_BASELINE_MEAN_MS,
        CORRUPTED_OPTIMIZED_MEAN_MS,
        baseline_p75_ms=CORRUPTED_BASELINE_P75_MS,
        optimized_p75_ms=CORRUPTED_OPTIMIZED_P75_MS,
    )

    update_result, payload = _write(store, corrupted)

    assert update_result.status == "rejected"
    assert payload["status"] == "rejected"
    assert _issue_metrics(update_result, "timing_outlier_skew") == {"baseline_time_ms"}
    assert _expectation_rejection_status(update_result) == "failed_expectation_invalid"
    assert "No write flag stores this entry" in _expectation_preview_message(update_result)
    assert store.data["examples"] == {}
    store.save()
    assert not store.path.exists()


def test_w2_133_regression_with_outlier_timing_still_fails_as_regression(tmp_path: Path) -> None:
    _seed(tmp_path, _entry(10.0, 2.5, baseline_p75_ms=10.0, optimized_p75_ms=2.5))
    # 4x falls to 1.25x, and the optimized mean is 1.6x its p75.
    regressed_with_outliers = _entry(10.0, 8.0, baseline_p75_ms=10.0, optimized_p75_ms=5.0)

    update_result, _ = _write(_store(tmp_path), regressed_with_outliers)

    assert update_result.status == "rejected"
    assert {"regression", "timing_outlier_skew"} <= _issue_types(update_result)
    assert "best_optimized_throughput.bytes_per_s" in _issue_metrics(update_result, "regression")
    assert "Regression detected (best_speedup)" in update_result.message
    assert "Timing rejected" in update_result.message
    assert _expectation_rejection_status(update_result) == "failed_regression"
    # 1.25x clears the 1.05x minimum-speedup contract, so a status that
    # ignored the rejection would be "succeeded".
    assert _benchmark_status(regressed_with_outliers, update_result) == "failed_regression"

    accepted, _ = _write(_store(tmp_path, accept_regressions=True), regressed_with_outliers)
    assert accepted.status == "rejected"
    assert _issue_types(accepted) == {"timing_outlier_skew"}
    assert _benchmark_status(regressed_with_outliers, accepted) == "failed_expectation_invalid"


def test_w2_133_outlier_timing_fails_on_first_write_and_with_provenance_mismatch(tmp_path: Path) -> None:
    skewed = _entry(10.0, 8.0, baseline_p75_ms=10.0, optimized_p75_ms=5.0, git_commit="def456")

    first, _ = _write(_store(tmp_path / "first"), skewed)
    assert first.status == "rejected"
    assert _benchmark_status(skewed, first) == "failed_expectation_invalid"

    _seed(tmp_path, _entry(10.0, 2.5, baseline_p75_ms=10.0, optimized_p75_ms=2.5))
    clean_new_commit = _entry(10.0, 2.5, baseline_p75_ms=10.0, optimized_p75_ms=2.5, git_commit="def456")
    provenance_only, _ = _write(_store(tmp_path, allow_mixed_provenance=False), clean_new_commit)
    assert provenance_only.status == "rejected"
    assert _issue_types(provenance_only) == {"provenance_mismatch"}
    assert _expectation_rejection_status(provenance_only) is None
    assert _benchmark_status(clean_new_commit, provenance_only) == "succeeded"

    mismatch_and_skew, _ = _write(_store(tmp_path, allow_mixed_provenance=False), skewed)
    assert _issue_types(mismatch_and_skew) == {"provenance_mismatch", "timing_outlier_skew"}
    assert _benchmark_status(skewed, mismatch_and_skew) == "failed_expectation_invalid"


def test_w2_133_harness_zero_p75_placeholder_is_refused(tmp_path: Path) -> None:
    update_result, _ = _write(
        _store(tmp_path, accept_regressions=True),
        _entry(14.0, 2.9, baseline_p75_ms=0, optimized_p75_ms=2.9),
    )

    assert update_result.status == "rejected"
    assert _issue_types(update_result) == {"invalid_timing_percentile"}
    assert _expectation_rejection_status(update_result) == "failed_expectation_invalid"


def test_w2_133_memory_goal_without_memory_metrics_fails_as_invalid_evidence(tmp_path: Path) -> None:
    entry = _entry(10.0, 5.0)
    entry.optimization_goal = "memory"

    update_result, _ = _write(_store(tmp_path, accept_regressions=True), entry)

    assert update_result.status == "rejected"
    assert _issue_types(update_result) == {"missing_memory_metrics"}
    assert _benchmark_status(entry, update_result) == "failed_expectation_invalid"
    assert "No write flag stores this entry" in _expectation_preview_message(update_result)


def test_w2_133_representative_timing_is_still_stored(tmp_path: Path) -> None:
    assert MAX_MEAN_SKEW_RATIO == 1.0 + RELATIVE_TOLERANCE
    store = _store(tmp_path)
    near_limit_ms = 10.0 * MAX_MEAN_SKEW_RATIO

    with_p75, _ = _write(
        store,
        _entry(near_limit_ms, 5.0, baseline_p75_ms=10.0, optimized_p75_ms=5.2),
    )
    assert with_p75.status == "updated"

    without_reference, _ = _write(_store(tmp_path / "single_sample"), _entry(10.0, 5.0))
    assert without_reference.status == "updated"


def test_w2_133_harness_records_median_instead_of_a_zero_p75(tmp_path: Path) -> None:
    # decode_common.py configures percentiles [50, 90, 99], so no p75 exists.
    steady = [1.0, 1.1, 0.9, 1.0, 1.05, 0.95, 1.0, 1.0, 1.1, 0.9]
    result_entry, best_opt, entry = _harness_result_entry(steady, [v / 2 for v in steady])

    assert "baseline_p75_ms" not in result_entry
    assert "p75_ms" not in best_opt
    assert result_entry["baseline_median_ms"] == 1.0
    assert best_opt["median_ms"] == 0.5
    assert result_entry["baseline_p90_ms"] == 1.1

    store = _store(tmp_path)
    update_result, _ = _write(store, entry, "decode")
    assert update_result.status == "updated"
    stored = store.data["examples"]["decode"]["metrics"]
    assert "baseline_p75_ms" not in stored
    assert "best_optimized_p75_ms" not in stored
    assert stored["baseline_median_ms"] == 1.0
    assert stored["best_optimized_median_ms"] == 0.5


def test_w2_133_median_catches_outliers_when_no_p75_was_recorded(tmp_path: Path) -> None:
    steady = [1.0] * 10
    # Two 20 ms stalls in ten samples give a 4.8 ms mean against a 1 ms median.
    stalled = [1.0] * 8 + [20.0, 20.0]
    _, _, entry = _harness_result_entry(stalled, [v / 2 for v in steady])

    update_result, _ = _write(_store(tmp_path, accept_regressions=True), entry, "decode")

    assert update_result.status == "rejected"
    assert _issue_metrics(update_result, "timing_outlier_skew") == {"baseline_time_ms"}
    assert "baseline_median_ms" in update_result.message
    assert _expectation_rejection_status(update_result) == "failed_expectation_invalid"


def test_w2_133_validator_flags_the_stored_corrupted_row(tmp_path: Path) -> None:
    store = _store(tmp_path)
    row = _entry(
        CORRUPTED_BASELINE_MEAN_MS,
        CORRUPTED_OPTIMIZED_MEAN_MS,
        baseline_p75_ms=CORRUPTED_BASELINE_P75_MS,
        optimized_p75_ms=CORRUPTED_OPTIMIZED_P75_MS,
    ).to_dict()

    issues = store.validate_entry("llama_3_1_8b", row)

    skew = [issue for issue in issues if issue.issue_type == "timing_outlier_skew"]
    assert len(skew) == 1
    assert skew[0].message.startswith("baseline_time_ms ")


def test_w2_133_refresh_from_run_refuses_outlier_skewed_preview(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "labs" / "real_world_models"
    chapter_dir.mkdir(parents=True)
    corrupted = _entry(
        CORRUPTED_BASELINE_MEAN_MS,
        CORRUPTED_OPTIMIZED_MEAN_MS,
        baseline_p75_ms=CORRUPTED_BASELINE_P75_MS,
        optimized_p75_ms=CORRUPTED_OPTIMIZED_P75_MS,
    ).to_dict()
    corrupted["example"] = "llama_3_1_8b"
    results_json = tmp_path / "artifacts" / "results" / "benchmark_test_results.json"
    results_json.parent.mkdir(parents=True)
    results_json.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "chapter": "labs/real_world_models",
                        "benchmarks": [
                            {
                                "example": "llama_3_1_8b",
                                "type": "python",
                                "expectation": {"entry": corrupted},
                            }
                        ],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    summary = refresh_expectations_from_run(results_json=results_json, repo_root=tmp_path)

    assert summary["counts"]["rejected"] == 1
    assert summary["counts"]["applied"] == 0
    stored = json.loads((chapter_dir / "expectations_b200.json").read_text(encoding="utf-8"))
    assert stored["examples"] == {}


def test_w2_133_markdown_report_lists_refused_expectation_evidence(tmp_path: Path) -> None:
    error = "Timing rejected (accept_regressions does not override this): baseline_time_ms 4.8 ms ..."
    results = [
        {
            "chapter": "labs/decode_optimization",
            "status": "completed",
            "benchmarks": [
                {
                    "example": "decode",
                    "type": "python",
                    "baseline_file": "baseline_decode.py",
                    "baseline_time_ms": 4.8,
                    "status": "failed_expectation_invalid",
                    "error": error,
                    "optimizations": [],
                }
            ],
            "summary": {
                "total_benchmarks": 1,
                "successful": 0,
                "failed": 1,
                "failed_expectation_invalid": 1,
            },
        }
    ]
    report = tmp_path / "report.md"

    generate_markdown_report(results, report, bench_root=tmp_path)

    text = report.read_text(encoding="utf-8")
    assert "  - Expectation evidence refused: 1\n" in text
    assert f"- Expectation evidence refused: {error}\n" in text
