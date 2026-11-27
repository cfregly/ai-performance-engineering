"""Check applied timing configuration receipts without executing benchmarks."""

import json
from dataclasses import dataclass

import pytest

from core.benchmark.defaults import BenchmarkDefaults
from core.benchmark.expectations import select_best_optimization
from core.benchmark.models import TimingStats
from core.harness.benchmark_harness import BenchmarkConfig, _benchmark_config_receipt
from core.harness.run_benchmarks import (
    CudaBenchmarkResult,
    _compute_locked_fields,
    _expectation_provenance_from_result,
    _merge_benchmark_config,
    _timing_configuration_receipt,
)


@dataclass
class _BenchmarkConfiguration:
    config: BenchmarkConfig

    def get_config(self):
        return self.config


def _merged(iterations, warmup, *, cli_locked=False):
    base = BenchmarkConfig(iterations=20, warmup=5)
    if cli_locked:
        base = BenchmarkConfig(iterations=7, warmup=2)
    locked = _compute_locked_fields(
        base_config=base, cli_iterations_provided=cli_locked,
        cli_warmup_provided=cli_locked, enable_profiling=False,
    )
    return _merge_benchmark_config(
        base_config=base,
        benchmark_obj=_BenchmarkConfiguration(BenchmarkConfig(iterations=iterations, warmup=warmup)),
        defaults_obj=BenchmarkDefaults(), locked_fields=locked,
    )


def _receipt(config, timing=None):
    return _timing_configuration_receipt(
        iterations=config.iterations, warmup=config.warmup,
        adaptive_iterations=config.adaptive_iterations, timing=timing,
        configured_config=_benchmark_config_receipt(config, source="configured"),
        applied_config=_benchmark_config_receipt(config, source="worker"),
    )


def _provenance(result):
    return _expectation_provenance_from_result(
        result, git_commit="a" * 40, hardware_key="metadata-test",
        profile_name="none", timestamp="2026-11-15T00:00:00+00:00",
        execution_environment="unknown", validity_profile="strict", dmi_product_name=None,
    )


@pytest.mark.parametrize("iterations,warmup", [(30, 10), (10, 5)])
def test_expectation_provenance_uses_applied_benchmark_configuration(iterations, warmup):
    config = _merged(iterations, warmup)
    result = {"baseline_timing_configuration": _receipt(config)}

    provenance = _provenance(result)
    retained = json.loads(json.dumps(provenance.to_dict()))

    assert retained["iterations"] == iterations
    assert retained["warmup_iterations"] == warmup
    assert result["baseline_timing_configuration"]["observed_timing_samples"] is None


def test_expectation_provenance_preserves_actual_cli_locked_configuration():
    with pytest.warns(UserWarning, match="Automatically raising to 5"):
        config = _merged(30, 10, cli_locked=True)
    provenance = _provenance({"baseline_timing_configuration": _receipt(config)})

    assert provenance.iterations == config.iterations == 7
    assert provenance.warmup_iterations == config.warmup == 5


def test_configured_iterations_do_not_replace_observed_adaptive_sample_count():
    config = _merged(30, 10)
    timing = TimingStats(
        mean_ms=1.0, median_ms=1.0, std_ms=0.0, min_ms=1.0, max_ms=1.0,
        iterations=47, warmup_iterations=10, raw_times_ms=[1.0] * 47,
    )
    receipt = _receipt(config, timing)
    provenance = _provenance({"baseline_timing_configuration": receipt})

    assert receipt["configured_iterations"] == provenance.iterations == 30
    assert receipt["adaptive_iterations"] is True
    assert receipt["observed_timing_samples"] == len(timing.raw_times_ms) == 47
    assert receipt["observed_warmup_iterations"] == 10


def test_selected_candidate_keeps_its_own_configuration():
    first = {"status": "succeeded", "speedup": 2.0, "timing_configuration": _receipt(_merged(30, 10))}
    later = {"status": "succeeded", "speedup": 1.1, "timing_configuration": _receipt(_merged(12, 8))}
    result = {
        "baseline_timing_configuration": _receipt(_merged(25, 6)),
        "optimizations": [first, later],
    }
    selected = select_best_optimization(result["optimizations"])
    retained = json.loads(json.dumps(result))
    provenance = _provenance(result)

    assert selected is first
    assert selected["timing_configuration"]["configured_iterations"] == 30
    assert selected["timing_configuration"]["configured_warmup_iterations"] == 10
    assert retained["optimizations"][1]["timing_configuration"]["configured_iterations"] == 12
    assert retained["optimizations"][1]["timing_configuration"]["configured_warmup_iterations"] == 8
    assert provenance.iterations == 25
    assert provenance.warmup_iterations == 6


def test_cuda_provenance_uses_dispatched_counts_instead_of_run_defaults():
    timing = CudaBenchmarkResult(
        mean_ms=1.0, median_ms=1.0, std_ms=0.0, min_ms=1.0, max_ms=1.0,
        percentiles={}, iterations=3, warmup_iterations=0,
    )
    receipt = _timing_configuration_receipt(
        iterations=20, warmup=5, adaptive_iterations=False, timing=timing,
    )
    provenance = _provenance({"baseline_timing_configuration": receipt})

    assert provenance.iterations == receipt["observed_timing_samples"] == 3
    assert provenance.warmup_iterations == receipt["observed_warmup_iterations"] == 0
    assert receipt["config_receipt_source"] == "unavailable"
    assert receipt["applied_config"] is None
    assert receipt["applied_iterations"] is None
    assert receipt["applied_warmup_iterations"] is None
    assert receipt["adaptive_iterations"] is False


def test_launcher_config_is_separate_and_expectations_use_observed_counts():
    timing = TimingStats(
        mean_ms=1.0, median_ms=1.0, std_ms=0.0, min_ms=1.0, max_ms=1.0,
        iterations=4, warmup_iterations=2, raw_times_ms=[1.0] * 4,
    )
    launcher_config = {
        "receipt_source": "launcher",
        "iterations": 20,
        "warmup": 5,
        "adaptive_iterations": False,
        "launch_via": "torchrun",
    }
    receipt = _timing_configuration_receipt(
        iterations=20,
        warmup=5,
        adaptive_iterations=False,
        timing=timing,
        applied_config=launcher_config,
    )
    provenance = _provenance({"baseline_timing_configuration": receipt})

    assert receipt["config_receipt_source"] == "launcher"
    assert receipt["launcher_config"] == launcher_config
    assert receipt["applied_config"] is None
    assert receipt["applied_iterations"] is None
    assert provenance.iterations == 4
    assert provenance.warmup_iterations == 2


def test_missing_config_receipt_is_not_backfilled_as_worker_applied():
    receipt = _timing_configuration_receipt(
        iterations=20,
        warmup=5,
        adaptive_iterations=False,
        timing=None,
    )

    assert receipt["config_receipt_source"] == "unavailable"
    assert receipt["configured_config"]["iterations"] == 20
    assert receipt["applied_config"] is None
    assert receipt["applied_iterations"] is None
    assert receipt["applied_warmup_iterations"] is None
    with pytest.raises(
        ValueError,
        match="worker config receipt or observed timing counts",
    ):
        _provenance({"baseline_timing_configuration": receipt})


def test_unattributed_config_receipt_is_not_treated_as_worker_evidence():
    timing = TimingStats(
        mean_ms=1.0, median_ms=1.0, std_ms=0.0, min_ms=1.0, max_ms=1.0,
        iterations=4, warmup_iterations=2, raw_times_ms=[1.0] * 4,
    )
    unattributed = {
        "iterations": 20,
        "warmup": 5,
        "adaptive_iterations": False,
    }
    receipt = _timing_configuration_receipt(
        iterations=20,
        warmup=5,
        adaptive_iterations=False,
        timing=timing,
        applied_config=unattributed,
    )
    provenance = _provenance({"baseline_timing_configuration": receipt})

    assert receipt["config_receipt_source"] == "unattributed"
    assert receipt["unattributed_config_receipt"] == unattributed
    assert receipt["applied_config"] is None
    assert receipt["applied_iterations"] is None
    assert provenance.iterations == 4
    assert provenance.warmup_iterations == 2


def test_missing_applied_configuration_cannot_create_provenance():
    with pytest.raises(KeyError, match="baseline_timing_configuration"):
        _provenance({})
