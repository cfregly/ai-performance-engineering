"""Reporting arithmetic regressions from audit findings W1-021 and W1-054.

These fixtures exercise the real result models and comparison/report pipeline;
they are not measured benchmark results or evidence of GPU performance.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.benchmark.comparison import (
    MetricDirection,
    compare_all_metrics,
    format_metric_comparison_table,
)
from core.benchmark.models import (
    BenchmarkResult,
    InferenceTimingStats,
    NsysMetrics,
    ProfilerMetrics,
    TimingStats,
)
from core.benchmark.performance_targets import compute_status, get_chapter_metrics


def _result(
    mean_ms: float = 100.0,
    *,
    bandwidth_gbps: float | None = None,
    ttft_p99_ms: float | None = None,
) -> BenchmarkResult:
    profiler_metrics = None
    if bandwidth_gbps is not None:
        profiler_metrics = ProfilerMetrics(
            nsys=NsysMetrics(raw_metrics={"memory_throughput_gb_per_s": bandwidth_gbps})
        )
    inference_timing = None
    if ttft_p99_ms is not None:
        inference_timing = InferenceTimingStats(
            ttft_mean_ms=ttft_p99_ms,
            tpot_mean_ms=1.0,
            ttft_p99_ms=ttft_p99_ms,
            num_requests=1,
            total_tokens_generated=1,
        )
    return BenchmarkResult(
        timing=TimingStats(
            mean_ms=mean_ms,
            median_ms=mean_ms,
            min_ms=mean_ms,
            max_ms=mean_ms,
            std_ms=0.0,
            iterations=1,
            warmup_iterations=0,
        ),
        profiler_metrics=profiler_metrics,
        inference_timing=inference_timing,
    )


@pytest.mark.parametrize("chapter", ["ch11", "ch13", "ch14", "ch16", "ch19"])
@pytest.mark.parametrize("candidate_ms", [200.0, 50.0])
def test_speedup_targets_do_not_invert_measured_latency(
    chapter: str, candidate_ms: float
) -> None:
    comparison = compare_all_metrics(_result(), _result(candidate_ms), chapter=chapter)
    timing = next(
        metric for metric in comparison.metric_comparisons if metric.metric_name == "timing.mean_ms"
    )

    assert timing.direction is MetricDirection.LOWER_IS_BETTER
    assert timing.unit == "ms"
    assert timing.baseline_value == 100.0
    assert timing.optimized_value == candidate_ms
    assert timing.ratio == pytest.approx(100.0 / candidate_ms)
    assert timing.ratio == comparison.timing_comparison.speedup
    assert timing.regression is (candidate_ms > 100.0)
    if candidate_ms > 100.0:
        assert timing.regression_pct == pytest.approx(100.0)
        assert timing.improvement_pct is None
    else:
        assert timing.improvement_pct == pytest.approx(50.0)
        assert timing.regression_pct is None

    report = format_metric_comparison_table(comparison)
    row = next(line for line in report.splitlines() if line.startswith(timing.display_name))
    assert ("REGRESS" if candidate_ms > 100.0 else "IMPROVE") in row
    assert "100.000 ms" in row


@pytest.mark.parametrize("chapter", ["ch04", "ch17"])
def test_chapter_latency_aliases_preserve_raw_timing_contract(chapter: str) -> None:
    comparison = compare_all_metrics(_result(), _result(110.0), chapter=chapter)
    timing = next(
        metric for metric in comparison.metric_comparisons if metric.metric_name == "timing.mean_ms"
    )
    assert timing.unit == "ms"
    assert timing.direction is MetricDirection.LOWER_IS_BETTER
    assert timing.regression is True
    assert timing.regression_pct == pytest.approx(10.0)


@pytest.mark.parametrize("candidate_gbps, regressed", [(100.0, True), (800.0, False)])
def test_bandwidth_absolute_target_does_not_set_percent_threshold(
    candidate_gbps: float, regressed: bool
) -> None:
    comparison = compare_all_metrics(
        _result(bandwidth_gbps=1000.0),
        _result(bandwidth_gbps=candidate_gbps),
        chapter="ch02",
        include_raw_metrics=True,
    )
    bandwidth = next(
        metric for metric in comparison.metric_comparisons
        if metric.metric_name == "profiler_metrics.nsys.raw.memory_throughput_gb_per_s"
    )
    assert bandwidth.regression is regressed
    assert bandwidth.regression_pct == pytest.approx((1000.0 - candidate_gbps) / 10.0)
    assert bandwidth.significant_change is regressed
    # Both candidates miss the absolute target, independently of relative change.
    target = get_chapter_metrics("ch02")["hbm3e_bandwidth_tbs"]
    assert compute_status(candidate_gbps, target) == "FAIL"
    report = format_metric_comparison_table(comparison, show_only_significant=True)
    assert ("REGRESS" in report) is regressed


@pytest.mark.parametrize("candidate_ms, regressed", [(103.0, False), (110.0, True)])
def test_small_routing_target_does_not_replace_timing_percent_threshold(
    candidate_ms: float, regressed: bool
) -> None:
    comparison = compare_all_metrics(_result(), _result(candidate_ms), chapter="ch17")
    timing = next(
        metric for metric in comparison.metric_comparisons if metric.metric_name == "timing.mean_ms"
    )
    assert timing.regression is regressed
    assert timing.significant_change is regressed


def test_ttft_absolute_target_and_relative_regression_are_independent() -> None:
    comparison = compare_all_metrics(
        _result(ttft_p99_ms=100.0), _result(ttft_p99_ms=110.0), chapter="ch17"
    )
    ttft = next(
        metric for metric in comparison.metric_comparisons
        if metric.metric_name == "inference_timing.ttft_p99_ms"
    )
    target = get_chapter_metrics("ch17")["ttft_p99_ms"]
    assert compute_status(110.0, target) == "PASS"
    assert ttft.regression is True
    assert ttft.regression_pct == pytest.approx(10.0)


@pytest.mark.parametrize("threshold_pct, regressed", [(10.0, True), (30.0, False)])
def test_explicit_percent_override_remains_authoritative(
    threshold_pct: float, regressed: bool
) -> None:
    comparison = compare_all_metrics(
        _result(bandwidth_gbps=1000.0),
        _result(bandwidth_gbps=800.0),
        chapter="ch02",
        include_raw_metrics=True,
        regression_threshold_pct=threshold_pct,
    )
    bandwidth = next(
        metric for metric in comparison.metric_comparisons
        if metric.metric_name == "profiler_metrics.nsys.raw.memory_throughput_gb_per_s"
    )
    assert bandwidth.regression is regressed


def test_file_backed_comparison_entrypoint_reports_slower_latency(tmp_path: Path) -> None:
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    baseline_path.write_text(_result().model_dump_json(), encoding="utf-8")
    candidate_path.write_text(_result(200.0).model_dump_json(), encoding="utf-8")
    script = """
import json
import sys
from pathlib import Path
from core.benchmark.models import BenchmarkResult
from core.benchmark.comparison import compare_and_display_all_metrics
baseline, candidate = [
    BenchmarkResult.model_validate_json(Path(path).read_text()) for path in sys.argv[1:]
]
comparison = compare_and_display_all_metrics(baseline, candidate, chapter='ch11', format_style='both')
timing = next(item for item in comparison.metric_comparisons if item.metric_name == 'timing.mean_ms')
print(json.dumps({'regression': timing.regression, 'ratio': timing.ratio, 'unit': timing.unit}))
"""
    process = subprocess.run(
        [sys.executable, "-c", script, str(baseline_path), str(candidate_path)],
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "REGRESS" in process.stdout
    assert "Mean Execution Time" in process.stdout
    assert json.loads(process.stdout.splitlines()[-1]) == {
        "regression": True, "ratio": 0.5, "unit": "ms"
    }
