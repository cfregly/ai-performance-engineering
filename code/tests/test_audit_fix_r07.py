"""Regression checks for audit work unit r07 (W1-041 residual).

The statistical reporting tests in `tests/test_protection_effectiveness.py`
check that the timing reporter keeps every sample. No outlier or cherry-picking
classifier exists, so a test name must not claim that one catches anything.
These checks inspect the collected test names and run the real
`BenchmarkHarness._compute_stats` path that the renamed tests describe.
"""

from __future__ import annotations

import inspect
import statistics

import pytest

import tests.test_protection_effectiveness as protection_tests
from tests.protection_test_utils import cpu_harness

UNSUPPORTED_DETECTION_TERMS = ("outlier", "cherry_picking")


def _protection_test_names() -> list[str]:
    return [
        name
        for _, cls in inspect.getmembers(protection_tests, inspect.isclass)
        if cls.__module__ == protection_tests.__name__ and cls.__name__.startswith("Test")
        for name, _ in inspect.getmembers(cls, inspect.isfunction)
        if name.startswith("test_")
    ]


def test_no_protection_test_claims_an_absent_classifier_catches_attacks():
    claims = [
        name
        for name in _protection_test_names()
        if "catches" in name and any(term in name for term in UNSUPPORTED_DETECTION_TERMS)
    ]
    assert claims == []


@pytest.mark.parametrize(
    "name", ["test_stats_retain_injected_outlier", "test_stats_retain_slow_samples"]
)
def test_retention_tests_are_collected_under_their_behavior_names(name):
    statistical = protection_tests.TestStatisticalProtectionsCatchAttacks
    assert inspect.isfunction(getattr(statistical, name, None))


def test_compute_stats_keeps_outlier_and_slow_samples():
    harness, config = cpu_harness()
    contaminated = [1.0, 1.1, 0.9, 1.05, 0.95, 10.0]
    timing = harness._compute_stats(contaminated, config).timing
    assert timing.raw_times_ms == contaminated
    assert timing.iterations == len(contaminated)
    assert timing.max_ms == 10.0
    assert timing.mean_ms == pytest.approx(statistics.mean(contaminated))
    assert timing.std_ms == pytest.approx(statistics.stdev(contaminated))
