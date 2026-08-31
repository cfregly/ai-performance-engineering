#!/usr/bin/env python3
"""Behavior checks for protection detectors, with clean and violating controls.

CPU checks exercise real verification, diagnostic, and reporting entrypoints.
Only stream execution needs CUDA. Diagnostic timing/GPU-state inputs are test
fixtures, not measurements or hardware qualification. Statistical reporting
checks preserve all supplied samples; no outlier or cherry-picking classifier
exists, and these tests do not claim one. Test count is not protection coverage.
"""

import gc
import statistics
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from core.benchmark.models import TimingStats
from core.benchmark.verification import (
    InputSignature, PrecisionFlags, ToleranceSpec, compare_workload_metrics,
    detect_seed_mutation, set_deterministic_seeds,
)
from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness
from core.harness.validity_checks import (
    GPUState, check_gpu_state_consistency,
    check_graph_capture_integrity, check_input_output_aliasing,
    check_setup_precomputation, gc_disabled,
)
from tests.protection_test_utils import (
    TensorWork, audit_cuda_work, check_fresh_input, check_jitter, compare_tensors, cpu_harness,
    make_runner, preserve_rng_state,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA stream execution required")


@pytest.fixture
def runner(tmp_path):
    with preserve_rng_state():
        yield make_runner(tmp_path)


@pytest.fixture(autouse=True)
def restore_rng_state():
    with preserve_rng_state():
        yield


def _signature(shape=(4, 8), dtype="float32"):
    return InputSignature(
        shapes={"input": shape}, dtypes={"input": dtype}, batch_size=shape[0],
        parameter_count=0, precision_flags=PrecisionFlags(),
    )


def _assert_jitter_pair(runner):
    assert check_jitter(runner, "real") == (True, None)
    passed, reason = check_jitter(runner, "constant")
    assert not passed
    assert "output unchanged" in reason


def _assert_fresh_pair(runner):
    assert check_fresh_input(runner, "real") == (True, None)
    passed, reason = check_fresh_input(runner, "cached")
    assert not passed
    assert "identical under a different seed" in reason




class TestTimingProtectionsCatchAttacks:
    @requires_cuda
    def test_stream_auditor_catches_unsynced_work(self):
        assert audit_cuda_work(True) == (True, [])
        passed, reasons = audit_cuda_work(False)
        assert not passed
        assert any("STREAM SYNC WARNING" in reason for reason in reasons)

    def test_event_timing_catches_negative_elapsed(self):
        """Validate timing payload rejection, not the behavior of CUDA events."""
        valid = TimingStats(mean_ms=1, median_ms=1, std_ms=0, min_ms=1,
                            max_ms=1, iterations=3, warmup_iterations=1)
        BenchmarkHarness._validate_timing_summary(valid)
        invalid = valid.model_copy(update={"mean_ms": -1.0, "min_ms": -1.0})
        with pytest.raises(ValueError, match="nonnegative"):
            BenchmarkHarness._validate_timing_summary(invalid)

    def test_warmup_detection_catches_computation_bleed(self, runner):
        """The fresh-input detector rejects output retained from an earlier run."""
        _assert_fresh_pair(runner)


class TestOutputProtectionsCatchAttacks:
    def test_nan_check_catches_nan_in_output(self, runner):
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert compare_tensors(runner, expected, expected.clone()).passed
        actual = expected.clone()
        actual[1] = float("nan")
        assert not compare_tensors(runner, expected, actual).passed

    def test_inf_check_catches_inf_in_output(self, runner):
        expected = torch.tensor([1.0, 2.0, 3.0])
        assert compare_tensors(runner, expected, expected.clone()).passed
        actual = expected.clone()
        actual[1] = float("inf")
        assert not compare_tensors(runner, expected, actual).passed

    def test_constant_output_check_catches_hardcoded(self, runner):
        _assert_jitter_pair(runner)

    def test_shape_mismatch_catches_wrong_shape(self, runner):
        expected = torch.ones(4, 8)
        assert compare_tensors(runner, expected, expected.clone()).passed
        result = compare_tensors(runner, expected, expected.T)
        assert not result.passed
        assert result.max_diff == float("inf")

    def test_dtype_mismatch_catches_wrong_dtype(self, runner):
        """Runtime input validation must reject a falsely declared dtype."""
        signature = _signature()
        runner._validate_inputs_match_signature(signature, {"input": torch.ones(4, 8)})
        with pytest.raises(ValueError, match="dtype"):
            runner._validate_inputs_match_signature(signature, {"input": torch.ones(4, 8, dtype=torch.float16)})

    def test_tolerance_catches_large_difference(self, runner):
        expected = torch.ones(8)
        tolerance = ToleranceSpec(rtol=1e-5, atol=1e-8)
        assert compare_tensors(runner, expected, expected + 1e-6, tolerance).passed
        assert not compare_tensors(runner, expected, expected * 1.1, tolerance).passed


class TestWorkloadProtectionsCatchAttacks:
    def test_signature_catches_batch_shrinking(self):
        baseline = _signature((32, 128))
        assert baseline.matches(_signature((32, 128)))
        assert not baseline.matches(_signature((16, 128)))

    def test_signature_catches_sequence_truncation(self):
        baseline = _signature((32, 2048), "int64")
        assert baseline.matches(_signature((32, 2048), "int64"))
        assert not baseline.matches(_signature((32, 512), "int64"))

    def test_workload_metrics_catches_reduced_work(self):
        baseline = {"flops_per_iteration": 1e12}
        assert compare_workload_metrics(baseline, baseline)[0]
        passed, deltas = compare_workload_metrics(baseline, {"flops_per_iteration": 5e11})
        assert not passed
        assert deltas["flops_per_iteration"] > 0.01


class TestMemoryProtectionsCatchAttacks:
    def test_aliasing_catches_input_output_same_memory(self, runner):
        runner._run_with_seed(TensorWork(), 42)
        with pytest.raises(RuntimeError, match="OUTPUT ALIASING"):
            runner._run_with_seed(TensorWork("alias"), 42)

    def test_preallocated_output_catches_prefilled(self):
        """Call the setup detector; preallocation alone is legitimate."""
        output = torch.zeros(4)
        get_outputs = lambda: {"output": output}
        assert check_setup_precomputation(get_outputs, lambda: None) == (True, None)
        passed, reason = check_setup_precomputation(get_outputs, lambda: output.fill_(42))
        assert not passed
        assert "PRE-COMPUTATION" in reason

    def test_fresh_input_catches_cached_output(self, runner):
        _assert_fresh_pair(runner)


class TestCudaProtectionsCatchAttacks:
    @requires_cuda
    def test_sync_catches_async_work_not_timed(self):
        """Use the stream detector, not an elapsed-time nonnegativity assertion."""
        assert audit_cuda_work(True) == (True, [])
        passed, reasons = audit_cuda_work(False)
        assert not passed
        assert any("no synchronization" in reason for reason in reasons)

    def test_graph_capture_catches_work_in_capture(self):
        """Capture/replay diagnostic inputs; this does not qualify GPU graphs."""
        assert check_graph_capture_integrity(2.0, [1.0, 1.1, 0.9]) == (True, None)
        passed, reason = check_graph_capture_integrity(100.0, [1.0, 1.1, 0.9])
        assert not passed
        assert "Suspected work during capture" in reason
        passed, reason = check_graph_capture_integrity(0.0, [0.0])
        assert not passed
        assert "near-zero" in reason


class TestStatisticalProtectionsCatchAttacks:
    def test_outlier_detection_catches_injected_outlier(self):
        """Reporting retains the outlier; no outlier classifier exists."""
        harness, config = cpu_harness()
        clean = [1.0, 1.1, 0.9, 1.05, 0.95]
        contaminated = clean + [10.0]
        clean_result = harness._compute_stats(clean, config).timing
        result = harness._compute_stats(contaminated, config).timing
        assert clean_result.raw_times_ms == clean
        assert result.raw_times_ms == contaminated
        assert result.iterations == len(contaminated)
        assert result.max_ms == 10.0
        assert result.mean_ms == statistics.mean(contaminated)

    def test_variance_check_catches_cherry_picking(self):
        """The reporter must retain slow samples; it cannot detect omitted input."""
        harness, config = cpu_harness()
        samples = [1.0, 1.1, 8.0, 1.2, 7.0]
        result = harness._compute_stats(samples, config).timing
        assert result.raw_times_ms == samples
        assert result.iterations == 5
        assert result.std_ms == pytest.approx(statistics.stdev(samples))
        assert result.mean_ms > statistics.mean(sorted(samples)[:3])

    def test_sample_count_catches_insufficient_samples(self, runner):
        """Fairness rejects reduced declared samples, not statistical power."""
        baseline = SimpleNamespace(config=BenchmarkConfig(iterations=50, warmup=5))
        clean = SimpleNamespace(config=BenchmarkConfig(iterations=50, warmup=5))
        reduced = SimpleNamespace(config=BenchmarkConfig(iterations=3, warmup=5))
        assert runner._validate_timing_config(baseline, clean) == (True, None)
        passed, reason = runner._validate_timing_config(baseline, reduced)
        assert not passed
        assert "measurement_iterations" in reason

    def test_gc_interference_catches_gc_during_timing(self):
        """GC guard disables collection and restores state even on failure."""
        previous = gc.isenabled()
        try:
            gc.enable()
            with gc_disabled():
                assert not gc.isenabled()
            assert gc.isenabled()
            with pytest.raises(RuntimeError, match="injected work failure"):
                with gc_disabled():
                    assert not gc.isenabled()
                    raise RuntimeError("injected work failure")
            assert gc.isenabled()
            gc.disable()
            with gc_disabled():
                assert not gc.isenabled()
            assert not gc.isenabled()
        finally:
            gc.enable() if previous else gc.disable()


class TestSeedProtectionsCatchAttacks:
    def test_seed_mutation_catches_changed_seed(self, runner):
        seed_info = set_deterministic_seeds(42)
        assert not detect_seed_mutation(seed_info)
        torch.manual_seed(43)
        assert detect_seed_mutation(seed_info)
        runner._run_with_seed(TensorWork(), 42)
        with pytest.raises(RuntimeError, match="mutated RNG seeds"):
            runner._run_with_seed(TensorWork("mutate_seed"), 42)

    def test_determinism_catches_non_reproducible(self, runner):
        work = TensorWork()
        first, *_ = runner._run_with_seed(work, 42)
        repeated, *_ = runner._run_with_seed(work, 42)
        changed, *_ = runner._run_with_seed(work, 43)
        assert runner._compare_outputs(first, repeated).passed
        assert not runner._compare_outputs(first, changed).passed


class TestJitterProtectionsCatchAttacks:
    def test_jitter_catches_hardcoded_output(self, runner):
        _assert_jitter_pair(runner)

    def test_jitter_accepts_legitimate_sensitivity(self, runner):
        _assert_jitter_pair(runner)


class TestEnvironmentProtectionsCatchAttacks:
    def test_thermal_throttling_catches_temperature_change(self):
        """Test state diagnostics without pretending to heat or qualify a GPU."""
        before = GPUState(device_index=0, device_name="diagnostic fixture", temperature_c=50, clock_mhz=1500)
        assert check_gpu_state_consistency(before, replace(before)) == (True, [])
        after = replace(before, temperature_c=75, clock_mhz=1000, throttle_reason="HwThermalSlowdown")
        consistent, reasons = check_gpu_state_consistency(before, after)
        assert not consistent
        assert any("temperature increased" in reason for reason in reasons)
        assert any("clock dropped" in reason for reason in reasons)
        assert any("throttling detected" in reason for reason in reasons)


class TestMissingEdgeCases:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64])
    def test_empty_tensor_handling(self, runner, dtype):
        empty = torch.empty(0, 4, dtype=dtype)
        assert compare_tensors(runner, empty, empty.clone()).passed
        assert not compare_tensors(runner, empty, torch.ones(1, 4, dtype=dtype)).passed
        assert not compare_tensors(runner, empty, torch.empty(0, 8, dtype=dtype)).passed

    def test_empty_tensor_does_not_bypass_remaining_outputs(self, runner):
        expected = {"empty": torch.empty(0), "value": torch.ones(2)}
        assert runner._compare_outputs(expected, expected).passed
        assert not runner._compare_outputs(expected, {"empty": torch.empty(0), "value": torch.zeros(2)}).passed
        assert not runner._compare_outputs(expected, {"empty": torch.empty(0)}).passed

    def test_empty_tensor_preserves_custom_comparator(self, runner):
        empty = torch.empty(0)
        require_nonempty = ToleranceSpec(rtol=0, atol=0, comparator_fn=lambda expected, actual: expected.numel() > 0)
        assert compare_tensors(runner, torch.ones(1), torch.ones(1), require_nonempty).passed
        assert not compare_tensors(runner, empty, empty, require_nonempty).passed

    def test_negative_stride_handling(self, runner):
        # torch.flip materializes reversed values; PyTorch has no negative-stride
        # Tensor views. Compare the values, without claiming stride support.
        expected = torch.arange(10).flip(0)
        assert compare_tensors(runner, expected, expected.clone()).passed
        assert not compare_tensors(runner, expected, torch.arange(10)).passed

    def test_non_contiguous_tensor_handling(self, runner):
        expected = torch.arange(32.0).reshape(4, 8).T
        assert not expected.is_contiguous()
        assert compare_tensors(runner, expected, expected.contiguous()).passed
        assert not compare_tensors(runner, expected, expected + 1).passed

    def test_very_small_learning_rate_precision(self, runner):
        expected = torch.tensor([1.0 - 1e-8], dtype=torch.float64)
        lost_update = (torch.ones(1) - 1e-8).to(torch.float64)
        tolerance = ToleranceSpec(rtol=0, atol=1e-10)
        assert compare_tensors(runner, expected, expected.clone(), tolerance).passed
        assert not compare_tensors(runner, expected, lost_update, tolerance).passed

    def test_integer_overflow_handling(self, runner):
        expected = torch.tensor([2**31], dtype=torch.int64)
        overflow = (torch.tensor([2**31 - 1], dtype=torch.int32) + 1).to(torch.int64)
        assert compare_tensors(runner, expected, expected.clone()).passed
        assert not compare_tensors(runner, expected, overflow).passed

    def test_mixed_precision_comparison(self, runner):
        expected = torch.linspace(-1, 1, 100)
        quantized = expected.half().float()
        tolerance = ToleranceSpec(rtol=1e-3, atol=1e-3)
        assert compare_tensors(runner, expected, quantized, tolerance).passed
        assert not compare_tensors(runner, expected, quantized + 0.1, tolerance).passed

    def test_cuda_oom_recovery(self):
        """Injected OOM propagation; no giant allocation or GPU recovery claim."""
        harness, config = cpu_harness()
        work = TensorWork()
        work.setup()
        samples, _ = harness._benchmark_custom(work.benchmark_fn, config)
        assert len(samples) == config.iterations

        def allocation_failure():
            raise torch.OutOfMemoryError("injected allocation failure")

        with pytest.raises(torch.OutOfMemoryError, match="injected allocation failure"):
            harness._benchmark_custom(allocation_failure, config)
        # An independent real CPU computation still executes after the failure.
        samples, _ = harness._benchmark_custom(work.benchmark_fn, config)
        assert len(samples) == config.iterations
        torch.testing.assert_close(work.output, work.input * 2)

    def test_gradient_checkpointing_memory(self, runner):
        """Verify real checkpointed gradients; this is not a memory-saving claim."""
        from torch.utils.checkpoint import checkpoint
        original = torch.arange(8.0, requires_grad=True)
        candidate = original.detach().clone().requires_grad_(True)
        original.square().sum().backward()
        checkpoint(lambda value: value.square(), candidate, use_reentrant=False).sum().backward()
        assert compare_tensors(runner, original.grad, candidate.grad).passed
        assert not compare_tensors(runner, original.grad, candidate.grad + 1).passed

    def test_deterministic_algorithm_enforcement(self, runner):
        from core.harness.validity_checks import capture_precision_policy_state, check_precision_policy_consistency
        before = capture_precision_policy_state()
        assert check_precision_policy_consistency(before, before) == (True, [])
        torch.use_deterministic_algorithms(not torch.are_deterministic_algorithms_enabled())
        after = capture_precision_policy_state()
        passed, reasons = check_precision_policy_consistency(before, after)
        assert not passed
        assert any("deterministic_algorithms" in reason for reason in reasons)


class TestMetaTestValidity:
    def test_nan_detection_actually_works(self, runner):
        expected = torch.ones(3)
        assert compare_tensors(runner, expected, expected.clone()).passed
        assert not compare_tensors(runner, expected, torch.full_like(expected, float("nan"))).passed

    def test_tolerance_comparison_actually_works(self, runner):
        expected = torch.ones(3)
        tolerance = ToleranceSpec(rtol=1e-3, atol=1e-5)
        assert compare_tensors(runner, expected, expected + 1e-4, tolerance).passed
        assert not compare_tensors(runner, expected, expected + 0.1, tolerance).passed

    def test_shape_comparison_actually_works(self, runner):
        expected = torch.ones(4, 8)
        assert compare_tensors(runner, expected, expected.clone()).passed
        assert not compare_tensors(runner, expected, expected.T).passed

    def test_memory_pointer_comparison_works(self):
        value = torch.ones(8)
        assert check_input_output_aliasing({"in": value}, {"out": value.clone()}) == (True, None)
        passed, reason = check_input_output_aliasing({"in": value}, {"out": value})
        assert not passed
        assert "OUTPUT ALIASING" in reason
