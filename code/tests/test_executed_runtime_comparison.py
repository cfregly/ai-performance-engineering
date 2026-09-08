"""Central runtime admission uses real benchmark-worker receipts."""

from __future__ import annotations

import math
import os

import pytest

from core.benchmark.models import BenchmarkRun
from core.benchmark.run_manifest import RunManifest
from core.benchmark.runtime_comparison import (
    ExecutedRuntimeComparisonError,
    compare_executed_runtime_provenance,
)
from tests.test_evaluation_harness_integration import _benchmark, _harness


@pytest.fixture(scope="module")
def real_executed_runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[BenchmarkRun, BenchmarkRun]:
    root = tmp_path_factory.mktemp("executed-runtime-comparison")
    reference = _harness().benchmark_with_manifest(
        _benchmark(root),
        run_id="runtime-reference-thread",
    )
    candidate = _harness(subprocess_mode=True).benchmark_with_manifest(
        _benchmark(root),
        run_id="runtime-candidate-subprocess",
    )
    assert not reference.result.errors, reference.result.errors
    assert not candidate.result.errors, candidate.result.errors
    assert reference.result.timing.iterations > 0
    assert candidate.result.timing.iterations > 0
    return reference, candidate


def _failure_codes(comparison, *, run=None) -> set[str]:
    return {
        failure.code
        for failure in comparison.integrity_failures
        if run is None or failure.run == run
    }


def test_manifest_runtime_capture_is_explicitly_deferrable_and_never_finalized_by_fallback() -> None:
    standalone = RunManifest.create(config={"execution_mode": "thread"})
    assert standalone.runtime_provenance is not None
    assert standalone.runtime_provenance.process_id == os.getpid()

    coordinator = RunManifest.create(
        config={"execution_mode": "thread"},
        capture_execution_runtime=False,
    )
    assert coordinator.runtime_provenance is None
    coordinator.finalize()
    assert coordinator.runtime_provenance is None


def test_real_thread_and_subprocess_receipts_are_admitted(real_executed_runs) -> None:
    reference, candidate = real_executed_runs

    assert reference.result.execution_process_ids == {0: os.getpid()}
    assert candidate.result.execution_process_ids[0] != os.getpid()
    assert (
        candidate.result.execution_process_ids[0]
        == candidate.result.runtime_provenance.process_id
    )

    comparison = compare_executed_runtime_provenance(reference, candidate)

    assert comparison.matches
    assert comparison.target == "cpu"
    assert comparison.integrity_failures == []
    assert comparison.runtime_parity is not None
    assert comparison.runtime_parity.matches


def test_two_sided_coordinator_substitution_is_rejected_by_independent_pid_receipt(
    real_executed_runs,
) -> None:
    reference, candidate = real_executed_runs
    substituted = candidate.model_copy(deep=True)
    coordinator_runtime = reference.result.runtime_provenance.model_copy(deep=True)
    substituted.result.runtime_provenance = coordinator_runtime.model_copy(deep=True)
    substituted.manifest.runtime_provenance = coordinator_runtime.model_copy(deep=True)

    comparison = compare_executed_runtime_provenance(reference, substituted)

    assert not comparison.matches
    assert comparison.target == "cpu"
    assert comparison.runtime_parity is not None and comparison.runtime_parity.matches
    assert "runtime_process_id_mismatch" in _failure_codes(
        comparison,
        run="candidate",
    )
    error = ExecutedRuntimeComparisonError(comparison)
    assert error.comparison is comparison
    assert "runtime_process_id_mismatch" in str(error)


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("manifest", "manifest_missing"),
        ("runtime", "result_runtime_provenance_missing"),
        ("execution_ids", "execution_process_ids_missing"),
        ("execution_pid", "runtime_process_id_mismatch"),
        ("manifest_result", "manifest_result_runtime_mismatch"),
    ],
)
def test_missing_or_invalid_receipts_return_structured_rejection(
    real_executed_runs,
    case,
    expected_code,
) -> None:
    reference, real_candidate = real_executed_runs
    candidate = real_candidate.model_copy(deep=True)

    if case == "manifest":
        candidate.manifest = None
    elif case == "runtime":
        candidate.result.runtime_provenance = None
        candidate.manifest.runtime_provenance = None
    elif case == "execution_ids":
        candidate.result.execution_process_ids = {}
    elif case == "execution_pid":
        candidate.result.execution_process_ids[0] += 1
    elif case == "manifest_result":
        candidate.result.runtime_provenance.torch_version += "-corrupt"
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(case)

    comparison = compare_executed_runtime_provenance(reference, candidate)

    assert not comparison.matches
    assert expected_code in _failure_codes(comparison, run="candidate")
    assert all(failure.detail for failure in comparison.integrity_failures)


@pytest.mark.parametrize(
    ("case", "expected_code"),
    [
        ("device_missing", "device_missing"),
        ("device_mismatch", "device_target_mismatch"),
        ("device_unsupported", "device_unsupported"),
        ("result_errors", "result_errors"),
        ("zero_timing", "timing_values_invalid"),
        ("nonfinite_timing", "timing_values_invalid"),
        ("zero_iterations", "timing_iterations_invalid"),
    ],
)
def test_invalid_executed_result_never_reaches_matching_admission(
    real_executed_runs,
    case,
    expected_code,
) -> None:
    reference, real_candidate = real_executed_runs
    candidate = real_candidate.model_copy(deep=True)

    if case == "device_missing":
        candidate.result.device = None
    elif case == "device_mismatch":
        candidate.result.device = "cuda:0"
    elif case == "device_unsupported":
        candidate.result.device = "mps"
    elif case == "result_errors":
        candidate.result.errors = ["worker reported an execution failure"]
    elif case == "zero_timing":
        candidate.result.timing.mean_ms = 0.0
    elif case == "nonfinite_timing":
        candidate.result.timing.mean_ms = math.inf
    elif case == "zero_iterations":
        candidate.result.timing.iterations = 0
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(case)

    comparison = compare_executed_runtime_provenance(reference, candidate)

    assert not comparison.matches
    assert expected_code in _failure_codes(comparison)
    if case in {"device_missing", "device_mismatch", "device_unsupported"}:
        assert comparison.target is None
