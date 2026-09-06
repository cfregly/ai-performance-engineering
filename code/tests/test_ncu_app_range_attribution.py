"""CPU regressions for keeping NCU app-range duration out of speedup claims."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.benchmark.comparison import compare_all_metrics, compare_metric
from core.benchmark.models import BenchmarkResult, NcuMetrics, ProfilerMetrics, TimingStats
from core.harness.benchmark_harness import _ncu_metrics_from_flat
from core.scripts.harness.example_registry import Example
from core.scripts.harness.profile_harness import (
    RunResult,
    generate_baseline_optimized_comparison,
)


def _result(ncu: NcuMetrics) -> BenchmarkResult:
    return BenchmarkResult(
        timing=TimingStats(
            mean_ms=10.0,
            median_ms=10.0,
            std_ms=0.0,
            min_ms=10.0,
            max_ms=10.0,
            iterations=1,
            warmup_iterations=0,
        ),
        profiler_metrics=ProfilerMetrics(ncu=ncu),
    )


def test_flat_adapter_preserves_range_without_comparing_it_as_speedup() -> None:
    baseline_ncu = _ncu_metrics_from_flat(
        {
            "ncu_range_time_ms": 120.0,
            "ncu_kernel_time_ms": 6.0,
            "ncu_custom_counter": 2.0,
        }
    )
    optimized_ncu = _ncu_metrics_from_flat(
        {
            "ncu_range_time_ms": 80.0,
            "ncu_kernel_time_ms": 3.0,
            "ncu_custom_counter": 4.0,
        }
    )

    assert baseline_ncu.range_time_ms == pytest.approx(120.0)
    assert "range_time_ms" not in baseline_ncu.raw_metrics
    assert baseline_ncu.raw_metrics == {"custom_counter": 2.0}

    comparison = compare_all_metrics(
        _result(baseline_ncu),
        _result(optimized_ncu),
        include_raw_metrics=True,
    )
    assert comparison.timing_comparison.speedup == pytest.approx(1.0)
    by_name = {item.metric_name: item for item in comparison.metric_comparisons}
    assert not any("range_time_ms" in name for name in by_name)
    assert by_name["profiler_metrics.ncu.kernel_time_ms"].ratio == pytest.approx(2.0)
    assert by_name["profiler_metrics.ncu.raw.custom_counter"].ratio == pytest.approx(2.0)


def test_historical_raw_range_duration_is_not_generic_comparison_input() -> None:
    comparison = compare_all_metrics(
        _result(
            NcuMetrics(
                raw_metrics={
                    "range_time_ms": 120.0,
                    "ncu_range_time_ms": 120.0,
                    "other_time_ms": 12.0,
                }
            )
        ),
        _result(
            NcuMetrics(
                raw_metrics={
                    "range_time_ms": 80.0,
                    "ncu_range_time_ms": 80.0,
                    "other_time_ms": 6.0,
                }
            )
        ),
        include_raw_metrics=True,
    )
    names = {item.metric_name for item in comparison.metric_comparisons}

    assert "profiler_metrics.ncu.raw.range_time_ms" not in names
    assert "profiler_metrics.ncu.raw.ncu_range_time_ms" not in names
    assert "profiler_metrics.ncu.raw.other_time_ms" in names
    assert (
        compare_metric(
            "profiler_metrics.ncu.raw.ncu_range_time_ms",
            120.0,
            80.0,
        )
        is None
    )


def _run_result(tmp_path: Path, name: str) -> RunResult:
    output_dir = tmp_path / name
    return RunResult(
        profiler="ncu",
        example=Example(name=name, path=Path("fixtures") / f"{name}.py", description="fixture"),
        command=[],
        output_dir=output_dir,
        stdout_path=output_dir / "stdout.log",
        stderr_path=output_dir / "stderr.log",
        duration=5.0,
        exit_code=0,
        skipped=False,
    )


def test_profile_harness_renders_range_duration_without_faster_claim(tmp_path: Path) -> None:
    (tmp_path / "metrics_summary.json").write_text(
        json.dumps(
            {
                "by_source": {
                    "ncu": {
                        "ncu_baseline_fixture.ncu-rep": {
                            "range_time_ms": 120.0,
                            "kernel_time_ms": 6.0,
                        },
                        "ncu_optimized_fixture.ncu-rep": {
                            "range_time_ms": 80.0,
                            "kernel_time_ms": 3.0,
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    rendered = generate_baseline_optimized_comparison(
        [
            _run_result(tmp_path, "baseline_fixture"),
            _run_result(tmp_path, "optimized_fixture"),
        ],
        tmp_path,
    )

    assert rendered is not None
    range_row = next(line for line in rendered.splitlines() if "| range_time_ms |" in line)
    kernel_row = next(line for line in rendered.splitlines() if "| kernel_time_ms |" in line)
    assert "Descriptive only (NCU selected-range duration)" in range_row
    assert "x faster" not in range_row
    assert "2.00x faster" in kernel_row
