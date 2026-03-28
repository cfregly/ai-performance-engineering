from core.benchmark.expectations import ExpectationEntry, RunProvenance
from core.harness.run_benchmarks import (
    _format_failed_no_speedup,
    _should_fail_no_speedup,
    build_expectation_metadata,
)


def _provenance() -> RunProvenance:
    return RunProvenance(
        git_commit="deadbee",
        hardware_key="b200",
        profile_name="minimal",
        timestamp="2026-03-28T00:00:00+00:00",
        iterations=10,
        warmup_iterations=5,
        execution_environment="virtualized",
        validity_profile="portable",
    )


def test_should_fail_no_speedup_respects_local_minimum_speedup() -> None:
    assert _should_fail_no_speedup(
        {
            "optimization_goal": "speed",
            "best_speedup": 1.01,
            "minimum_required_speedup": 1.02,
        }
    ) is True
    assert _should_fail_no_speedup(
        {
            "optimization_goal": "speed",
            "best_speedup": 1.03,
            "minimum_required_speedup": 1.02,
        }
    ) is False


def test_format_failed_no_speedup_uses_local_minimum_speedup() -> None:
    rendered = _format_failed_no_speedup(
        {
            "optimization_goal": "speed",
            "best_speedup": 1.01,
            "minimum_required_speedup": 1.02,
        }
    )
    assert "1.01x" in rendered
    assert "1.02x threshold" in rendered


def test_build_expectation_metadata_includes_minimum_required_speedup() -> None:
    metadata = build_expectation_metadata(
        {
            "example": "cache_aware_disagg",
            "type": "python",
            "optimization_goal": "speed",
            "baseline_time_ms": 10.0,
            "minimum_required_speedup": 1.02,
        },
        {"file": "optimized_cache_aware_disagg.py", "technique": "optimized", "time_ms": 9.2},
        git_commit="deadbee",
    )
    assert metadata["minimum_required_speedup"] == 1.02


def test_expectation_entry_round_trip_preserves_minimum_required_speedup() -> None:
    entry = ExpectationEntry(
        example="cache_aware_disagg",
        type="python",
        optimization_goal="speed",
        baseline_time_ms=10.0,
        best_optimized_time_ms=9.2,
        provenance=_provenance(),
        best_optimization_name="optimized",
        best_optimization_file="optimized_cache_aware_disagg.py",
        best_optimization_technique="optimized",
        minimum_required_speedup=1.02,
    )

    serialized = entry.to_dict()
    assert serialized["metadata"]["minimum_required_speedup"] == 1.02

    restored = ExpectationEntry.from_dict(serialized)
    assert restored.minimum_required_speedup == 1.02
