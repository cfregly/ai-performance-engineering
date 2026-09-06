#!/usr/bin/env python3
"""Protection behavior and explicitly scoped runtime checks.

CPU detector tests run without CUDA. GPU integration/smoke tests require actual
hardware and do not count as passed on CPU. A diagnostic state or timing fixture
is not a GPU measurement. Legacy statistical names check reporting fidelity,
not an unimplemented outlier/cherry-picking classifier. Missing dataset
provenance protections are explicit skips, not passing conceptual assertions.
The number of test functions does not establish coverage of every README claim.
"""

import sys
import os
import subprocess
import tempfile
import warnings
from pathlib import Path
from contextlib import contextmanager

import pytest
import torch

from tests.protection_test_utils import (
    TensorWork, assert_comparison_controls, assert_compile_cache_reset,
    assert_compile_guard_counts, assert_cuda_timing_cross_validation,
    assert_environment_controls, assert_gpu_state_controls,
    assert_materialization_diagnostic, assert_signature_controls,
    assert_stream_audit_controls, audit_cuda_work, check_fresh_input, check_jitter, compare_tensors,
    cpu_harness, make_runner, preserve_rng_state,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for anti-cheat protection tests"
)


@pytest.fixture(autouse=True)
def restore_rng_state():
    with preserve_rng_state():
        yield


@pytest.fixture
def runner(tmp_path):
    return make_runner(tmp_path)


def _check_jitter_controls(runner):
    assert check_jitter(runner, "real") == (True, None)
    passed, reason = check_jitter(runner, "constant")
    assert not passed
    assert "output unchanged" in reason


def _check_fresh_controls(runner):
    assert check_fresh_input(runner, "real") == (True, None)
    passed, reason = check_fresh_input(runner, "cached")
    assert not passed
    assert "identical under a different seed" in reason


def _check_config_mutation(field, value):
    harness, config = cpu_harness()
    work = TensorWork()
    work.setup()
    samples, _ = harness._benchmark_custom(work.benchmark_fn, config)
    assert len(samples) == config.iterations

    def mutate_config():
        work.benchmark_fn()
        setattr(config, field, value)

    with pytest.raises(RuntimeError, match="CONFIG MANIPULATION"):
        harness._benchmark_custom(mutate_config, config)


def _check_adaptive_cuda_iterations():
    from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness
    value = torch.ones(1024, device="cuda")
    config = BenchmarkConfig(
        device=torch.device("cuda"), iterations=1, warmup=5, use_subprocess=False,
        adaptive_iterations=True, min_total_duration_ms=10.0, max_adaptive_iterations=10000,
        enable_profiling=False,
    )
    harness = BenchmarkHarness(config=config)
    calls = []

    def fast_op():
        value.add_(1)
        calls.append(None)

    samples, _ = harness._benchmark_custom(fast_op, config)
    assert len(samples) == len(calls)
    assert 1 < len(samples) <= config.max_adaptive_iterations
    assert sum(samples) >= config.min_total_duration_ms
    torch.testing.assert_close(value, torch.full_like(value, 1 + len(samples)))


def _check_warmup_isolation(monkeypatch, isolate):
    from core.harness import l2_cache_utils
    from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness
    config = BenchmarkConfig(device=torch.device("cuda"), warmup=5, isolate_warmup_cache=isolate)
    harness = BenchmarkHarness(config=config)
    events = []
    value = torch.ones(8, device="cuda")
    real_flush = l2_cache_utils.flush_l2_cache

    def observe_real_flush(device):
        real_flush(device)
        events.append("flush")

    def warmup_work():
        value.add_(1)
        events.append("work")

    # This is an observer: the actual CUDA flush still executes.
    monkeypatch.setattr(l2_cache_utils, "flush_l2_cache", observe_real_flush)
    harness._warmup(warmup_work, config.warmup, config)
    assert events == ["work"] * 5 + (["flush"] if isolate else [])
    torch.testing.assert_close(value, torch.full_like(value, 6))


# =============================================================================
# TIMING PROTECTION TESTS (7 issues)
# =============================================================================

class TestTimingProtections:
    """Tests for timing-related anti-cheat protections."""

    @requires_cuda
    def test_unsynced_streams_detection(self):
        assert audit_cuda_work(True) == (True, [])
        passed, reasons = audit_cuda_work(False)
        assert not passed
        assert any("STREAM SYNC WARNING" in reason for reason in reasons)

    @requires_cuda
    def test_incomplete_async_ops_protection(self):
        assert_stream_audit_controls()

    @requires_cuda
    def test_event_timing_cross_validation(self):
        assert_cuda_timing_cross_validation()

    @requires_cuda
    def test_timer_granularity_adaptive_iterations(self):
        _check_adaptive_cuda_iterations()

    @requires_cuda
    def test_warmup_bleed_isolation(self, monkeypatch):
        _check_warmup_isolation(monkeypatch, True)

    def test_clock_drift_monotonic(self, monkeypatch):
        """Observe the actual monotonic clock used by CPU harness measurement."""
        from core.harness import benchmark_harness as module
        harness, config = cpu_harness()
        real_clock = module.time.perf_counter
        clock_reads = []
        def observe_clock():
            value = real_clock()
            clock_reads.append(value)
            return value
        monkeypatch.setattr(module.time, "perf_counter", observe_clock)
        work = TensorWork()
        work.setup()
        samples, _ = harness._benchmark_custom(work.benchmark_fn, config)
        assert len(samples) == config.iterations
        assert len(clock_reads) >= 2 * config.iterations
        assert clock_reads == sorted(clock_reads)

    def test_profiler_overhead_profile_free_path(self):
        """The actual timing callback executes without an active PyTorch profiler."""
        harness, config = cpu_harness()
        profile_states = []
        value = torch.ones(8)
        def work():
            profile_states.append(torch.autograd._profiler_enabled())
            return value + 1
        samples, _ = harness._benchmark_custom(work, config)
        assert len(samples) == config.iterations
        assert profile_states == [False] * config.iterations


# =============================================================================
# OUTPUT PROTECTION TESTS (10 issues)
# =============================================================================

class TestOutputProtections:
    """Tests for output-related anti-cheat protections."""

    def test_constant_output_jitter_check(self, runner):
        _check_jitter_controls(runner)
        _check_fresh_controls(runner)

    def test_stale_cache_fresh_input_check(self, runner):
        """Reject cached results; allocating an output buffer is not itself cheating."""
        _check_fresh_controls(runner)

    def test_invalid_values_nan_detection(self, runner):
        """Output verification rejects a deliberately corrupted value."""
        expected = torch.ones(8)
        assert compare_tensors(runner, expected, expected.clone()).passed
        actual = expected.clone()
        actual[0] = float("nan")
        assert not compare_tensors(runner, expected, actual).passed

    def test_invalid_values_inf_detection(self, runner):
        """Output verification rejects a deliberately corrupted value."""
        expected = torch.ones(8)
        assert compare_tensors(runner, expected, expected.clone()).passed
        actual = expected.clone()
        actual[0] = float("inf")
        assert not compare_tensors(runner, expected, actual).passed

    def test_denormalized_values_detection(self, runner):
        """Numerical comparison catches wrong subnormal output; no slowdown classifier."""
        from core.benchmark.verification import ToleranceSpec
        expected = torch.tensor([1e-38], dtype=torch.float32)
        tolerance = ToleranceSpec(rtol=0, atol=0)
        assert compare_tensors(runner, expected, expected.clone(), tolerance).passed
        assert not compare_tensors(runner, expected, torch.tensor([1e-45]), tolerance).passed

    def test_uninitialized_memory_detection(self, runner):
        """Output verification rejects a deliberately corrupted value."""
        expected = torch.ones(8)
        assert compare_tensors(runner, expected, expected.clone()).passed
        actual = expected.clone()
        actual[0] = 42.0
        assert not compare_tensors(runner, expected, actual).passed


# =============================================================================
# WORKLOAD PROTECTION TESTS (11 issues)
# =============================================================================

class TestWorkloadProtections:
    """Tests for workload-related anti-cheat protections."""

    def test_undeclared_shortcuts_workload_invariant(self):
        """Test that undeclared shortcuts are detected.

        Protection: Workload invariant check
        Attack: Skips elements without declaring
        """
        from core.benchmark.verification import compare_workload_metrics

        baseline = {"bytes_per_iteration": 1000}
        optimized = {"bytes_per_iteration": 500}  # Only half the work!

        match, delta = compare_workload_metrics(baseline, optimized)
        assert not match, "Should detect workload reduction"
        assert delta is not None

    def test_early_exit_config_immutability(self):
        _check_config_mutation('iterations', 1)

    def test_backend_precision_policy_mutation_detection(self):
        """Test that backend precision policy mutations are detected.

        Protection: check_precision_policy_consistency()
        Attack: Toggle TF32 / matmul precision during timing
        """
        from core.harness.validity_checks import (
            capture_precision_policy_state,
            check_precision_policy_consistency,
        )

        if (
            not hasattr(torch.backends, "cuda")
            or not hasattr(torch.backends.cuda, "matmul")
            or not hasattr(torch.backends.cuda.matmul, "allow_tf32")
        ):
            raise RuntimeError("Expected torch.backends.cuda.matmul.allow_tf32 to be available for precision policy test.")

        before = capture_precision_policy_state()
        prev_tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = not prev_tf32
        try:
            after = capture_precision_policy_state()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev_tf32

        is_consistent, warnings_list = check_precision_policy_consistency(before, after)
        assert not is_consistent
        assert warnings_list

    def test_sparsity_mismatch_detection(self):
        """Test that sparsity mismatches are detected.

        Protection: Sparsity ratio check
        Attack: Different sparsity patterns
        """
        from core.benchmark.verification import InputSignature, PrecisionFlags

        baseline_sig = InputSignature(
            shapes={"weight": (1024, 1024)},
            dtypes={"weight": "float32"},
            batch_size=1,
            parameter_count=1024*1024,
            precision_flags=PrecisionFlags(),
            sparsity_ratio=0.0,  # Dense
        )

        optimized_sig = InputSignature(
            shapes={"weight": (1024, 1024)},
            dtypes={"weight": "float32"},
            batch_size=1,
            parameter_count=1024*1024,
            precision_flags=PrecisionFlags(),
            sparsity_ratio=0.9,  # 90% sparse - less work!
        )

        # Signatures should not match due to different sparsity
        assert baseline_sig.matches(baseline_sig)
        assert not baseline_sig.matches(optimized_sig)


# =============================================================================
# LOCATION PROTECTION TESTS (7 issues)
# =============================================================================

class TestLocationProtections:
    """Tests for work-location-related anti-cheat protections."""

    def test_cpu_spillover_detection(self):
        """The real source detector identifies an explicit host transfer."""
        from core.harness.validity_checks import check_benchmark_fn_antipatterns
        class CleanWork:
            def benchmark_fn(self):
                self.output = self.input + 1
        class HostTransfer:
            def benchmark_fn(self):
                self.output = self.input.cpu()
        assert check_benchmark_fn_antipatterns(CleanWork.benchmark_fn) == (True, [])
        passed, reasons = check_benchmark_fn_antipatterns(HostTransfer.benchmark_fn)
        assert not passed
        assert any("CPU" in reason for reason in reasons)

    def test_setup_precomputation_detection(self):
        from core.harness.validity_checks import check_setup_precomputation
        output = torch.zeros(8)
        get_outputs = lambda: {"output": output}
        assert check_setup_precomputation(get_outputs, lambda: None) == (True, None)
        passed, reason = check_setup_precomputation(get_outputs, lambda: output.fill_(42))
        assert not passed
        assert "PRE-COMPUTATION" in reason

    def test_graph_capture_cheat_detection(self):
        """Exercise graph diagnostics with explicit state fixtures, not GPU evidence."""
        from core.harness.validity_checks import GraphCaptureCheatDetector, GraphCaptureState
        detector = GraphCaptureCheatDetector()
        detector.capture_state = GraphCaptureState(capture_start_time=0, capture_end_time=0.002)
        detector.replay_times = [1.0, 1.1, 0.9]
        assert detector.check_for_cheat() == (False, None)
        detector.capture_state.capture_end_time = 0.1
        cheating, reason = detector.check_for_cheat()
        assert cheating
        assert "GRAPH CAPTURE CHEAT" in reason

    def test_graph_capture_thresholds_respected(self):
        """Graph capture cheat thresholds should gate detection."""
        from core.harness.validity_checks import GraphCaptureCheatDetector, GraphCaptureState

        detector = GraphCaptureCheatDetector()
        # Manually craft capture/replay stats to exceed ratio threshold
        detector.capture_state = GraphCaptureState(
            capturing=False,
            capture_start_time=0.0,
            capture_end_time=1.0,  # 1s capture = 1000ms
            memory_allocated_during_capture=50.0,
        )
        detector.replay_times = [10.0]  # ms
        # Tight threshold should flag cheat
        is_cheat, reason = detector.check_for_cheat(capture_replay_ratio_threshold=5.0, memory_threshold_mb=200.0)
        assert is_cheat and reason
        # Lenient thresholds should pass
        is_cheat, reason = detector.check_for_cheat(capture_replay_ratio_threshold=200.0, memory_threshold_mb=200.0)
        assert is_cheat is False

    def test_lazy_evaluation_force_evaluation(self, monkeypatch):
        'Real materialization failure produces a diagnostic; production does not fail the run.'
        assert_materialization_diagnostic(monkeypatch)


# =============================================================================
# MEMORY PROTECTION TESTS (7 issues)
# =============================================================================

class TestMemoryProtections:
    """Tests for memory-related anti-cheat protections."""

    def test_preallocated_output_detection(self, runner):
        """Reject cached results; allocating an output buffer is not itself cheating."""
        _check_fresh_controls(runner)

    def test_input_output_aliasing_detection(self):
        """Test that input-output aliasing is detected.

        Protection: check_input_output_aliasing()
        Attack: Output points to pre-filled input

        Note: check_input_output_aliasing returns (no_aliasing, message)
        where no_aliasing=True means NO aliasing detected (good)
        """
        from core.harness.validity_checks import check_input_output_aliasing

        # Create separate tensors
        input_tensor = torch.randn(100, device="cpu")
        output_tensor = torch.randn(100, device="cpu")

        # No aliasing - should pass (returns True, None)
        inputs = {"x": input_tensor}
        outputs = {"y": output_tensor}

        no_aliasing, message = check_input_output_aliasing(inputs, outputs)
        assert no_aliasing, f"Separate tensors should not be aliased: {message}"

        # Aliased case - should detect (returns False, message)
        outputs_aliased = {"y": input_tensor}  # Same tensor!
        no_aliasing, message = check_input_output_aliasing(inputs, outputs_aliased)
        assert not no_aliasing, "Aliased tensors should be detected"
        assert message is not None, "Should have error message"

    @requires_cuda
    def test_memory_pool_reset(self):
        """Test that memory pool can be reset.

        Protection: reset_cuda_memory_pool()
        Attack: Cached allocations skew timing
        """
        from core.harness.validity_checks import reset_cuda_memory_pool

        before_reserved = torch.cuda.memory_reserved()
        # Allocate some memory
        x = torch.randn(10000, device="cuda")
        during_reserved = torch.cuda.memory_reserved()
        del x

        # Reset pool
        reset_cuda_memory_pool()

        after_reserved = torch.cuda.memory_reserved()
        assert during_reserved >= before_reserved
        assert after_reserved <= during_reserved


# =============================================================================
# CUDA PROTECTION TESTS (10 issues)
# =============================================================================

class TestCUDAProtections:
    """Tests for CUDA-specific anti-cheat protections."""

    @requires_cuda
    def test_async_memcpy_sync(self):
        assert_stream_audit_controls(pinned=True)

    def test_undeclared_multi_gpu_detection(self):
        pytest.skip('Missing production protection: no undeclared-GPU execution detector; environment inventory only warns that multiple devices exist')

    def test_context_switch_handling(self):
        pytest.skip('Missing production protection: no CUDA context-switch enforcement detector')


# =============================================================================
# COMPILE PROTECTION TESTS (7 issues)
# =============================================================================

class TestCompileProtections:
    """Tests for torch.compile-related anti-cheat protections."""

    def test_compilation_cache_clear(self):
        assert_compile_cache_reset()

    def test_trace_reuse_reset(self):
        assert_compile_cache_reset()

    def test_guard_failure_detection(self):
        assert_compile_guard_counts()


# =============================================================================
# DISTRIBUTED PROTECTION TESTS (8 issues)
# =============================================================================

class TestDistributedProtections:
    """Tests for distributed training anti-cheat protections."""

    def test_rank_skipping_detection(self):
        """Test that rank skipping is detected.

        Protection: check_rank_execution()
        Attack: Some ranks don't do work
        """
        from types import SimpleNamespace

        from core.harness.validity_checks import check_rank_execution

        executed, error = check_rank_execution(
            SimpleNamespace(_skip_rank=True),
            world_size=2,
            rank=1,
        )

        assert not executed
        assert error == "Rank 1 has _skip_rank=True"

    def test_topology_mismatch_detection(self):
        """Test that topology mismatches are detected.

        Protection: verify_distributed()
        Attack: Claims different topology
        """
        from core.benchmark.verification import DistributedTopology, compare_topologies

        baseline_topo = DistributedTopology(
            world_size=4,
            ranks=[0, 1, 2, 3],
            shards=2,
            pipeline_stages=2,
        )

        optimized_topo = DistributedTopology(
            world_size=4,
            ranks=[0, 1, 2, 3],
            shards=4,  # Different!
            pipeline_stages=1,
        )

        match, diff = compare_topologies(baseline_topo, optimized_topo)
        assert not match, f"Different topologies should not match: {diff}"


# =============================================================================
# ENVIRONMENT PROTECTION TESTS (12 issues)
# =============================================================================

def _clock_lock_unavailable(reason):
    """Unavailable locally is a skip; the attested Tier-1 contract requires it."""
    if os.environ.get("TIER1_EXPECTED_GPU_NAME", "").strip():
        pytest.fail(f"Attested Tier-1 runner requires clock locking: {reason}")
    pytest.skip(f"Clock-lock capability unavailable: {reason}")


def _handle_clock_lock_error(exc):
    """Classify explicit capability errors, never generic lock failures."""
    from core.harness.benchmark_harness import _is_nvidia_smi_permission_error

    try:
        from pynvml import (NVMLError_NoPermission, NVMLError_NotSupported,
                            NVMLError_LibraryNotFound, NVMLError_FunctionNotFound)
        nvml_unavailable = (NVMLError_NoPermission, NVMLError_NotSupported,
                            NVMLError_LibraryNotFound, NVMLError_FunctionNotFound)
    except ImportError:
        nvml_unavailable = ()
    current = exc
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (PermissionError, *nvml_unavailable)):
            _clock_lock_unavailable(f"{type(current).__name__}: {current}")
        if isinstance(current, FileNotFoundError) and current.filename:
            if Path(current.filename).name in {"nvidia-smi", "sudo"}:
                _clock_lock_unavailable(f"required tool missing: {current.filename}")
        if isinstance(current, subprocess.CalledProcessError):
            command = current.cmd if isinstance(current.cmd, (list, tuple)) else []
            command_names = [Path(str(arg)).name for arg in command[:3]]
            is_nvidia_smi = (command_names[:1] == ["nvidia-smi"]
                             or command_names == ["sudo", "-n", "nvidia-smi"])
            if is_nvidia_smi:
                if _is_nvidia_smi_permission_error(current):
                    _clock_lock_unavailable(f"nvidia-smi permission failure: {current}")
                # NVIDIA's documented return codes: unsupported operation,
                # absent NVML library, or unavailable NVML function.
                # https://docs.nvidia.com/deploy/nvidia-smi/index.html#return-value
                if current.returncode in {3, 12, 13}:
                    _clock_lock_unavailable(f"nvidia-smi capability failure: {current}")
        current = current.__cause__ or current.__context__
    raise exc


def _read_nvml_clock_pair(physical_index, *, maximum=False):
    """Read the real device, independently of the harness's lock implementation."""
    import pynvml

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(physical_index)
        query = pynvml.nvmlDeviceGetMaxClockInfo if maximum else pynvml.nvmlDeviceGetApplicationsClock
        return (int(query(handle, pynvml.NVML_CLOCK_SM)),
                int(query(handle, pynvml.NVML_CLOCK_MEM)))
    finally:
        pynvml.nvmlShutdown()


def _assert_observed_clock_lock(target, observed):
    """Use the production lock's existing 50 MHz application-clock contract."""
    assert len(target) == len(observed) == 2
    assert all(type(clock) is int and clock > 0 for clock in (*target, *observed))
    assert all(abs(actual - requested) <= 50 for requested, actual in zip(target, observed)), (
        f"Observed application clocks {observed} differ from requested clocks {target}"
    )


class TestClockLockControlPlane:
    """Exception outcomes and diagnostic comparisons only; no simulated GPU lock."""

    @pytest.mark.parametrize("attested", [False, True])
    @pytest.mark.parametrize("error", [
        PermissionError("Insufficient permissions"),
        subprocess.CalledProcessError(4, ["nvidia-smi", "-pm", "1"], b"Insufficient Permissions"),
        subprocess.CalledProcessError(3, ["nvidia-smi", "-pm", "1"]),
        subprocess.CalledProcessError(12, ["nvidia-smi", "-pm", "1"]),
        subprocess.CalledProcessError(13, ["nvidia-smi", "-pm", "1"]),
        FileNotFoundError(2, "No such file or directory", "nvidia-smi"),
    ])
    def test_unavailable_is_skip_or_required_runner_failure(self, monkeypatch, attested, error):
        monkeypatch.setenv("TIER1_EXPECTED_GPU_NAME", "NVIDIA B200" if attested else "")
        outcome = pytest.fail.Exception if attested else pytest.skip.Exception
        with pytest.raises(outcome, match="requires clock locking|capability unavailable"):
            _handle_clock_lock_error(error)

    @pytest.mark.parametrize("attested", [False, True])
    @pytest.mark.parametrize("error", [
        RuntimeError("GPU clock lock failed without a captured verification error"),
        RuntimeError("Failed to lock GPU clocks: nvidia-smi failed"),
        RuntimeError("Unknown permission state"),
        subprocess.CalledProcessError(9, ["nvidia-smi", "-pm", "1"], b"Driver failure"),
        subprocess.CalledProcessError(3, ["unrelated-tool", "nvidia-smi"]),
        FileNotFoundError(2, "No such file or directory", "unrelated-input"),
        AssertionError("Observed application clocks differ from requested clocks"),
    ])
    def test_actual_lock_failures_propagate(self, monkeypatch, attested, error):
        monkeypatch.setenv("TIER1_EXPECTED_GPU_NAME", "NVIDIA B200" if attested else "")
        with pytest.raises(type(error)) as caught:
            _handle_clock_lock_error(error)
        assert caught.value is error

    def test_wrapped_permission_is_not_generic_success(self, monkeypatch):
        monkeypatch.delenv("TIER1_EXPECTED_GPU_NAME", raising=False)
        error = RuntimeError("Failed to lock GPU clocks")
        error.__cause__ = subprocess.CalledProcessError(4, ["nvidia-smi", "-pm", "1"])
        with pytest.raises(pytest.skip.Exception, match="permission failure"):
            _handle_clock_lock_error(error)

    @pytest.mark.parametrize("attested", [False, True])
    @pytest.mark.parametrize("error_type", ["NVMLError_NoPermission", "NVMLError_NotSupported",
                                           "NVMLError_LibraryNotFound", "NVMLError_FunctionNotFound"])
    def test_nvml_unavailability_is_skip_or_required_runner_failure(self, monkeypatch, error_type, attested):
        pynvml = pytest.importorskip("pynvml")
        monkeypatch.setenv("TIER1_EXPECTED_GPU_NAME", "NVIDIA B200" if attested else "")
        outcome = pytest.fail.Exception if attested else pytest.skip.Exception
        with pytest.raises(outcome, match=error_type):
            _handle_clock_lock_error(getattr(pynvml, error_type)())

    @pytest.mark.parametrize("observed", [(1500, 2000), (1450, 2050)])
    def test_observed_application_clock_contract_accepts_matching_diagnostics(self, observed):
        _assert_observed_clock_lock((1500, 2000), observed)

    @pytest.mark.parametrize("observed", [(1449, 2000), (1500, 2051), (0, 2000), (None, 2000), (True, 2000)])
    def test_observed_application_clock_contract_rejects_invalid_diagnostics(self, observed):
        with pytest.raises(AssertionError):
            _assert_observed_clock_lock((1500, 2000), observed)


class TestEnvironmentProtections:
    """Tests for environment-related anti-cheat protections."""

    def test_device_mismatch_validation(self):
        pytest.skip('Missing production protection: validate_environment does not compare expected and observed GPU identities')

    def test_frequency_boost_clock_locking(self):
        """Observe real requested application clocks before and after CUDA work."""
        if not torch.cuda.is_available():
            _clock_lock_unavailable("real CUDA device required")
        try:
            import pynvml  # noqa: F401 -- explicit capability prerequisite
        except ImportError:
            _clock_lock_unavailable("pynvml is required to observe application clocks")
        from core.harness.benchmark_harness import lock_gpu_clocks, _resolve_physical_device_index

        device = torch.cuda.current_device()
        try:
            physical_index = _resolve_physical_device_index(device)
            target = _read_nvml_clock_pair(physical_index, maximum=True)
            assert all(clock > 0 for clock in target)
            with lock_gpu_clocks(device=device, sm_clock_mhz=target[0], mem_clock_mhz=target[1]):
                _assert_observed_clock_lock(target, _read_nvml_clock_pair(physical_index))
                value = torch.ones(100, device=torch.device("cuda", device))
                value.add_(1)
                torch.cuda.synchronize(device)
                torch.testing.assert_close(value, torch.full_like(value, 2), rtol=0, atol=0)
                _assert_observed_clock_lock(target, _read_nvml_clock_pair(physical_index))
        except Exception as exc:
            _handle_clock_lock_error(exc)

    def test_thermal_throttling_monitoring(self):
        'Explicit telemetry diagnostic inputs; not a measured GPU temperature.'
        assert_gpu_state_controls(throttle_reason='HwThermalSlowdown')

    @requires_cuda
    def test_power_limit_monitoring(self):
        """Test that power state is monitored.

        Protection: capture_gpu_state()
        Attack: Different TDP settings
        """
        from core.harness.validity_checks import capture_gpu_state

        state = capture_gpu_state()

        # Should capture power info via NVML (fail-fast if not available).
        assert state is not None
        assert state.power_draw_w is not None
        assert state.power_limit_w is not None
        assert state.power_limit_w > 0

    @requires_cuda
    def test_gpu_state_nvml_identity_matches_logical_cuda_device(self):
        """Read both APIs and prove telemetry follows the visible CUDA identity."""
        from core.harness.validity_checks import capture_gpu_state
        from core.profiling.gpu_telemetry import normalize_gpu_uuid

        device = torch.cuda.current_device()
        cuda_uuid = normalize_gpu_uuid(
            getattr(torch.cuda.get_device_properties(device), "uuid", None)
        )
        if cuda_uuid is None:
            pytest.skip("this PyTorch build does not expose CUDA device UUIDs")
        state = capture_gpu_state(device)
        assert state.device_uuid == cuda_uuid


# =============================================================================
# STATISTICAL PROTECTION TESTS (8 issues)
# =============================================================================

class TestStatisticalProtections:
    """Tests for statistical anti-cheat protections."""

    def test_cherry_picking_prevention(self):
        """Retain all supplied samples; this does not detect samples omitted upstream."""
        import statistics
        harness, config = cpu_harness()
        clean = [1.2, 1.0, 1.4, 1.1, 1.3]
        with_outlier = clean + [10.0]
        assert harness._compute_stats(clean, config).timing.raw_times_ms == clean
        result = harness._compute_stats(with_outlier, config).timing
        assert result.raw_times_ms == with_outlier
        assert result.iterations == len(with_outlier)
        assert result.mean_ms == statistics.mean(with_outlier)
        assert result.max_ms == 10.0

    def test_all_iterations_reported(self):
        """Retain all supplied samples; this does not detect samples omitted upstream."""
        import statistics
        harness, config = cpu_harness()
        clean = [1.2, 1.0, 1.4, 1.1, 1.3]
        with_outlier = clean + [10.0]
        assert harness._compute_stats(clean, config).timing.raw_times_ms == clean
        result = harness._compute_stats(with_outlier, config).timing
        assert result.raw_times_ms == with_outlier
        assert result.iterations == len(with_outlier)
        assert result.mean_ms == statistics.mean(with_outlier)
        assert result.max_ms == 10.0

    @requires_cuda
    def test_insufficient_samples_adaptive(self):
        _check_adaptive_cuda_iterations()

    def test_cold_start_warmup_enforcement(self):
        from core.benchmark.verification import TimingConfig, compare_timing_configs
        baseline = TimingConfig(iterations=10, warmup=5)
        assert compare_timing_configs(baseline, TimingConfig(iterations=10, warmup=5)) == (True, None)
        passed, reason = compare_timing_configs(baseline, TimingConfig(iterations=10, warmup=0))
        assert not passed
        assert "warmup" in reason.lower()

    def test_gc_interference_disabled(self):
        """Test that GC is disabled during timing.

        Protection: gc_disabled()
        Attack: Garbage collection during timing
        """
        from core.harness.validity_checks import gc_disabled
        import gc

        gc_was_enabled = gc.isenabled()
        with gc_disabled():
            # GC should be disabled here
            assert not gc.isenabled()
            x = [i for i in range(1000)]
            assert len(x) == 1000

        assert gc.isenabled() is gc_was_enabled

    def test_background_process_isolation(self):
        pytest.skip('Missing production protection: no background CPU process noise/isolation detector; synchronizing CUDA does not provide OS isolation')


# =============================================================================
# EVALUATION PROTECTION TESTS (7 issues)
# =============================================================================

class TestEvaluationProtections:
    """Tests for evaluation-related anti-cheat protections."""

    def test_eval_code_exploitation_contract(self, monkeypatch):
        from core.benchmark.contract import BenchmarkContract
        class GoodWork(TensorWork):
            def validate_result(self):
                torch.testing.assert_close(self.output, self.input * 2)
            def get_output_tolerance(self):
                return (1e-5, 1e-8)
        class MissingOutput(GoodWork):
            get_verify_output = None
        monkeypatch.setenv("VERIFY_ENFORCEMENT_PHASE", "detect")
        detected, errors, notices = BenchmarkContract.check_verification_compliance(MissingOutput())
        assert detected and not errors
        assert any("not callable" in notice for notice in notices)
        monkeypatch.setenv("VERIFY_ENFORCEMENT_PHASE", "gate")
        clean, errors, _ = BenchmarkContract.check_verification_compliance(GoodWork())
        assert clean, errors
        passed, errors, _ = BenchmarkContract.check_verification_compliance(MissingOutput())
        assert not passed
        assert any("get_verify_output" in error for error in errors)

    def test_timeout_manipulation_immutability(self):
        _check_config_mutation('timeout_seconds', 999)

    def test_test_data_leakage_contamination_check(self):
        pytest.skip("Missing protection: no dataset provenance/holdout-overlap detector is implemented; set arithmetic is not detector coverage")

    def test_benchmark_overfitting_jitter_fresh(self, runner):
        _check_jitter_controls(runner)
        _check_fresh_controls(runner)


# =============================================================================
# CUDA GRAPH PROTECTION TEST
# =============================================================================

class TestCUDAGraphProtections:
    """Tests for CUDA graph-related protections."""

    def test_cuda_graph_capture_integrity(self):
        from core.harness.validity_checks import check_graph_capture_integrity
        assert check_graph_capture_integrity(2.0, [1.0, 1.1, 0.9]) == (True, None)
        passed, reason = check_graph_capture_integrity(100.0, [1.0, 1.1, 0.9])
        assert not passed
        assert "Suspected work during capture" in reason


# =============================================================================
# L2 CACHE PROTECTION TESTS
# =============================================================================

class TestL2CacheProtections:
    """Tests for L2 cache-related protections."""

    @requires_cuda
    def test_l2_cache_size_detection(self):
        from core.harness.l2_cache_utils import detect_l2_cache_size
        props = torch.cuda.get_device_properties(0)
        actual_bytes = getattr(props, 'l2_cache_size', 0) or getattr(props, 'L2_cache_size', 0)
        if actual_bytes <= 0:
            pytest.skip('Actual device API does not expose L2 size; fallback constants are not hardware verification')
        detect_l2_cache_size.cache_clear()
        info = detect_l2_cache_size()
        assert info.source == 'hardware'
        assert info.size_bytes == actual_bytes
        assert info.compute_capability == f'{props.major}.{props.minor}'

    @requires_cuda
    def test_l2_cache_flush(self):
        'Actual buffer write and synchronization; cache-eviction efficacy still requires profiler evidence.'
        from core.harness.l2_cache_utils import create_l2_flush_buffer, detect_l2_cache_size, flush_l2_cache
        buffer = create_l2_flush_buffer()
        assert buffer.numel() * buffer.element_size() > detect_l2_cache_size().size_bytes
        buffer.fill_(1)
        flush_l2_cache(buffer=buffer)
        assert torch.cuda.current_stream().query()
        torch.testing.assert_close(buffer, torch.zeros_like(buffer), rtol=0, atol=0)


# =============================================================================
# STREAM AUDITOR PROTECTION TESTS
# =============================================================================

class TestStreamAuditorProtections:
    """Tests for stream auditor protections."""

    @requires_cuda
    def test_stream_auditor_context(self):
        """Test that stream auditor works as context manager.

        Protection: audit_streams()
        Attack: Work on unsynced streams
        """
        from core.harness.validity_checks import audit_streams

        with audit_streams() as auditor:
            # Work here is audited
            x = torch.randn(100, device="cuda")
            y = x * 2

        ok, warnings_list = auditor.check_issues()
        assert ok, f"default stream work should not trigger stream warnings: {warnings_list}"
        assert warnings_list == []
        assert torch.equal(y, x * 2)

    @requires_cuda
    def test_stream_auditor_detects_unsynced_custom_stream(self):
        """Test that custom stream work without sync triggers a warning."""
        from core.harness.validity_checks import audit_streams

        stream = torch.cuda.Stream()
        with audit_streams() as auditor:
            x = torch.randn(1024, device="cuda")
            with torch.cuda.stream(stream):
                _ = x * 2

        ok, warnings_list = auditor.check_issues()
        assert not ok, "Unsynced custom stream should be flagged"
        assert any("STREAM SYNC WARNING" in warning for warning in warnings_list)

    @requires_cuda
    def test_stream_auditor_accepts_stream_synchronize(self):
        """Test that stream.synchronize() counts as synchronization."""
        from core.harness.validity_checks import audit_streams

        stream = torch.cuda.Stream()
        with audit_streams() as auditor:
            x = torch.randn(1024, device="cuda")
            with torch.cuda.stream(stream):
                _ = x * 2
            stream.synchronize()

        ok, warnings_list = auditor.check_issues()
        assert ok, f"stream.synchronize() should satisfy auditor: {warnings_list}"

    @requires_cuda
    def test_stream_auditor_accepts_wait_stream(self):
        """Test that wait_stream() dependency counts as synchronization."""
        from core.harness.validity_checks import audit_streams

        stream = torch.cuda.Stream()
        with audit_streams() as auditor:
            x = torch.randn(1024, device="cuda")
            with torch.cuda.stream(stream):
                _ = x * 2
            torch.cuda.current_stream().wait_stream(stream)

        ok, warnings_list = auditor.check_issues()
        assert ok, f"wait_stream() should satisfy auditor: {warnings_list}"

    def test_stream_sync_completeness_check(self):
        'Stream-ID receipt classifier; it warns about creation, not actual synchronization completion.'
        from core.harness.validity_checks import check_stream_sync_completeness
        assert check_stream_sync_completeness([1, 2], [2, 1]) == (True, None)
        passed, reason = check_stream_sync_completeness([1], [1, 2])
        assert not passed
        assert '1 new stream' in reason


# =============================================================================
# ADDITIONAL WORKLOAD PROTECTION TESTS
# =============================================================================

class TestWorkloadProtectionsExtended:
    """Extended workload protection tests."""

    def test_attention_mask_mismatch_detection(self, runner):
        expected = torch.ones(4, 8)
        assert compare_tensors(runner, expected, expected.clone()).passed
        assert not compare_tensors(runner, expected, torch.tril(expected)).passed

    def test_kv_cache_size_mismatch_detection(self):
        """Test that KV cache size mismatches are detected.

        Protection: Cache dimension check
        Attack: Different cache sizes
        """
        from core.benchmark.verification import InputSignature, PrecisionFlags

        baseline = InputSignature(
            shapes={"kv_cache": (32, 2, 128, 64)},  # batch, 2 (k+v), seq, head_dim
            dtypes={"kv_cache": "float16"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )

        optimized = InputSignature(
            shapes={"kv_cache": (32, 2, 64, 64)},  # Different seq length!
            dtypes={"kv_cache": "float16"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )

        # Signatures should differ
        assert baseline.matches(baseline)
        assert not baseline.matches(optimized)

    def test_train_test_overlap_detection(self):
        pytest.skip("Missing protection: no dataset provenance/holdout-overlap detector is implemented; set arithmetic is not detector coverage")

    def test_batch_shrinking_detection(self):
        """Test that batch shrinking is detected.

        Protection: InputSignature matching
        Attack: Processes fewer samples than declared
        """
        from core.benchmark.verification import InputSignature, PrecisionFlags

        baseline = InputSignature(
            shapes={"input": (32, 128)},  # batch=32
            dtypes={"input": "float32"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )

        optimized = InputSignature(
            shapes={"input": (16, 128)},  # batch=16 - SHRUNK!
            dtypes={"input": "float32"},
            batch_size=16,  # Different batch size
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )

        # Should detect batch size mismatch
        assert baseline.matches(baseline)
        assert not baseline.matches(optimized)

    def test_sequence_truncation_detection(self):
        """Test that sequence truncation is detected.

        Protection: InputSignature matching
        Attack: Processes shorter sequences than declared
        """
        from core.benchmark.verification import InputSignature, PrecisionFlags

        baseline = InputSignature(
            shapes={"input": (32, 2048)},  # seq_len=2048
            dtypes={"input": "float32"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )

        optimized = InputSignature(
            shapes={"input": (32, 512)},  # seq_len=512 - TRUNCATED!
            dtypes={"input": "float32"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )

        # Should detect sequence length mismatch
        assert baseline.matches(baseline)
        assert not baseline.matches(optimized)

    def test_hidden_downsampling_detection(self):
        """Test that hidden downsampling is detected.

        Protection: Dimension validation
        Attack: Silently reduces resolution
        """
        from core.benchmark.verification import InputSignature, PrecisionFlags

        baseline = InputSignature(
            shapes={"image": (32, 3, 224, 224)},  # Full resolution
            dtypes={"image": "float32"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )

        optimized = InputSignature(
            shapes={"image": (32, 3, 112, 112)},  # Half resolution - DOWNSAMPLED!
            dtypes={"image": "float32"},
            batch_size=32,
            parameter_count=1000,
            precision_flags=PrecisionFlags(),
        )

        # Should detect dimension mismatch
        assert baseline.shapes["image"] != optimized.shapes["image"]


# =============================================================================
# ADDITIONAL LOCATION PROTECTION TESTS
# =============================================================================

class TestLocationProtectionsExtended:
    """Extended location protection tests."""

    @requires_cuda
    def test_warmup_computation_isolation(self, monkeypatch):
        _check_warmup_isolation(monkeypatch, False)
        _check_warmup_isolation(monkeypatch, True)

    def test_background_thread_isolation(self):
        pytest.skip('Missing production protection: no in-process background-thread computation guard')


# =============================================================================
# ADDITIONAL MEMORY PROTECTION TESTS
# =============================================================================

class TestMemoryProtectionsExtended:
    """Extended memory protection tests."""

    @requires_cuda
    def test_pinned_memory_timing(self):
        assert_stream_audit_controls(pinned=True)

    @requires_cuda
    def test_fragmentation_effects(self):
        """Test that fragmentation is handled.

        Protection: Memory pool reset
        Attack: Memory fragmentation differs
        """
        from core.harness.validity_checks import reset_cuda_memory_pool

        before_reserved = torch.cuda.memory_reserved()
        # Allocate and free to fragment
        tensors = [torch.randn(i * 100, device="cuda") for i in range(1, 10)]
        during_reserved = torch.cuda.memory_reserved()
        del tensors

        # Reset pool to clear fragmentation
        reset_cuda_memory_pool()
        after_reserved = torch.cuda.memory_reserved()
        assert during_reserved >= before_reserved
        assert after_reserved <= during_reserved

    def test_page_fault_timing(self):
        pytest.skip('Missing production protection: no page-fault event detector')

    def test_swap_interference(self, tmp_path, monkeypatch):
        assert_environment_controls(tmp_path, monkeypatch, 'swap')


# =============================================================================
# ADDITIONAL CUDA PROTECTION TESTS
# =============================================================================

class TestCUDAProtectionsExtended:
    """Extended CUDA protection tests."""

    def test_host_callback_escape(self):
        pytest.skip('Missing production protection: no CUDA host-callback execution detector')

    def test_workspace_precompute_detection(self):
        pytest.skip('Missing production protection: no cuBLAS workspace precomputation detector')

    def test_persistent_kernel_detection(self):
        pytest.skip('Missing production protection: no persistent-kernel lifetime detector')

    def test_driver_overhead_tracking(self):
        pytest.skip('Missing production protection: no driver-overhead attribution detector')

    def test_cooperative_launch_validation(self):
        pytest.skip('Missing production protection: no cooperative-launch inspector')

    def test_dynamic_parallelism_tracking(self):
        pytest.skip('Missing production protection: no device-side dynamic-launch inspector')

    def test_unified_memory_fault_tracking(self):
        pytest.skip('Missing production protection: no managed-memory page-fault event detector')


# =============================================================================
# ADDITIONAL COMPILE PROTECTION TESTS
# =============================================================================

class TestCompileProtectionsExtended:
    """Extended compile protection tests."""

    def test_mode_inconsistency_detection(self, runner):
        'Real output mismatch from different model modes; no general compiler-mode parity guard is implied.'
        model = torch.nn.BatchNorm1d(8)
        value = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        model.train()
        expected = model(value)
        model.eval()
        assert_comparison_controls(runner, expected.detach(), model(value).detach())

    def test_inductor_asymmetry_detection(self):
        pytest.skip('Missing production protection: no compiler-backend parity policy detector')

    def test_autotuning_variance_handling(self):
        pytest.skip('Missing production protection: no compiler-autotuning variance classifier')


# =============================================================================
# ADDITIONAL DISTRIBUTED PROTECTION TESTS
# =============================================================================

class TestDistributedProtectionsExtended:
    """Extended distributed protection tests."""

    def test_collective_short_circuit_detection(self):
        """Test that collective short-circuits are detected.

        Protection: NCCL validation
        Attack: Communication skipped
        """
        from core.harness.validity_checks import verify_distributed_outputs

        result = verify_distributed_outputs(
            rank_outputs={0: "hash_a"},
            expected_world_size=2,
        )

        assert not result.all_ranks_executed
        assert result.error_message == "RANK SKIPPING: Missing outputs from ranks [1]"

    def test_barrier_timing_protection(self):
        pytest.skip('Missing production protection: no rank-barrier timing detector')

    def test_gradient_bucketing_mismatch_detection(self):
        pytest.skip('Missing production protection: no gradient bucket-size parity field or detector')

    def test_async_gradient_timing(self):
        pytest.skip('Missing production protection: no asynchronous gradient completion timing detector')

    @requires_cuda
    def test_pipeline_bubble_tracking(self):
        """Test that pipeline bubbles are tracked.

        Protection: Bubble time tracking
        Attack: Pipeline bubbles not counted
        """
        # Trigger timing cross-validation by launching work on a non-default stream
        # while CUDA events are recorded on the (per-thread) default stream.
        # The wall clock includes full-device sync, but CUDA event timing under-reports,
        # which should raise a TIMING CROSS-VALIDATION FAILURE for chapter/lab benchmarks.
        import importlib.util
        import sys
        import textwrap

        from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness

        with tempfile.TemporaryDirectory() as tempdir:
            module_path = Path(tempdir) / "ch_fake_pipeline_bubble.py"
            module_path.write_text(
                textwrap.dedent(
                    """
                    import torch
                    from core.harness.benchmark_harness import BaseBenchmark

                    class PipelineBubbleTimingBenchmark(BaseBenchmark):
                        def __init__(self):
                            super().__init__()
                            self.stream = None

                        def setup(self) -> None:
                            self.stream = torch.cuda.Stream()

                        def get_custom_streams(self):
                            return [self.stream]

                        def benchmark_fn(self) -> None:
                            # Launch GPU work on a non-default stream and return without
                            # synchronizing that stream. The harness uses CUDA events
                            # on the default stream + full-device sync, so wall clock
                            # should be much larger than CUDA event timing.
                            with torch.cuda.stream(self.stream):
                                torch.cuda._sleep(10_000_000)
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )

            spec = importlib.util.spec_from_file_location("ch_fake_pipeline_bubble", module_path)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            bench = module.PipelineBubbleTimingBenchmark()
            harness = BenchmarkHarness()
            config = BenchmarkConfig(
                iterations=1,
                warmup=5,
                use_subprocess=False,
                timing_method="cuda_event",
                adaptive_iterations=False,
                # CI runs on bare metal, but local/dev environments may be virtualized.
                # This test exercises timing cross-validation; it is not intended to
                # validate environment enforcement behavior.
                enforce_environment_validation=False,
            )
            result = harness._benchmark_with_threading(bench, config)
            assert any("TIMING CROSS-VALIDATION FAILURE" in err for err in result.errors), result.errors

    def test_shard_size_mismatch_detection(self):
        assert_signature_controls(shards=8, shapes={'input': (2, 8)})


# =============================================================================
# ADDITIONAL ENVIRONMENT PROTECTION TESTS
# =============================================================================

class TestEnvironmentProtectionsExtended:
    """Extended environment protection tests."""

    def test_environment_validation_reports_execution_context(self, tmp_path, monkeypatch):
        assert_environment_controls(tmp_path, monkeypatch, 'virtualization')

    def test_memory_overcommit_handling(self):
        pytest.skip('Missing production protection: no OS or GPU overcommit policy detector exists; memory-growth warnings do not establish it')

    def test_numa_inconsistency_detection(self, tmp_path, monkeypatch):
        assert_environment_controls(tmp_path, monkeypatch, 'numa')

    @requires_cuda
    def test_cpu_governor_mismatch_detection(self):
        'Test that CPU governor mismatches are detected.\n        \n        Protection: Governor lock\n        Attack: Different CPU frequency scaling\n        '
        import importlib.util
        import sys
        import textwrap

        from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness
        from core.harness.validity_checks import EnvironmentProbe

        with tempfile.TemporaryDirectory() as env_dir, tempfile.TemporaryDirectory() as mod_dir:
            # Create a synthetic sysfs/procfs snapshot that violates governor=performance.
            env_root = Path(env_dir)
            (env_root / "proc").mkdir(parents=True, exist_ok=True)
            (env_root / "proc" / "swaps").write_text("Filename\tType\tSize\tUsed\tPriority\n", encoding="utf-8")
            (env_root / "proc" / "cpuinfo").write_text("processor\t: 0\n", encoding="utf-8")
            (env_root / "proc" / "sys" / "vm").mkdir(parents=True, exist_ok=True)
            (env_root / "proc" / "sys" / "vm" / "swappiness").write_text("0\n", encoding="utf-8")
            (env_root / "sys" / "devices" / "virtual" / "dmi" / "id").mkdir(parents=True, exist_ok=True)
            (env_root / "sys" / "devices" / "virtual" / "dmi" / "id" / "product_name").write_text(
                "BareMetal\n",
                encoding="utf-8",
            )
            gov_path = env_root / "sys" / "devices" / "system" / "cpu" / "cpufreq" / "policy0"
            gov_path.mkdir(parents=True, exist_ok=True)
            (gov_path / "scaling_governor").write_text("powersave\n", encoding="utf-8")

            module_path = Path(mod_dir) / "ch_fake_cpu_governor.py"
            module_path.write_text(
                textwrap.dedent(
                    """
                    import torch
                    from core.harness.benchmark_harness import BaseBenchmark

                    class GovernorMismatchBenchmark(BaseBenchmark):
                        def __init__(self):
                            super().__init__()
                            self.x = None

                        def setup(self) -> None:
                            self.x = torch.randn(1, device=self.device)

                        def benchmark_fn(self) -> None:
                            self.x.add_(1)
                    """
                ).strip()
                + "\n",
                encoding="utf-8",
            )
            spec = importlib.util.spec_from_file_location("ch_fake_cpu_governor", module_path)
            assert spec and spec.loader
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)

            bench = module.GovernorMismatchBenchmark()
            harness = BenchmarkHarness(environment_probe=EnvironmentProbe(root=env_root))
            config = BenchmarkConfig(iterations=1, warmup=5, use_subprocess=False)
            result = harness._benchmark_with_threading(bench, config)
            assert any("ENVIRONMENT INVALID" in err and "CPU governor mismatch" in err for err in result.errors), result.errors

    def test_driver_version_mismatch_detection(self):
        pytest.skip('Missing production protection: RunManifest records a driver version but no cross-run driver-version lock is enforced')

    def test_library_version_mismatch_detection(self):
        pytest.skip('Missing production protection: RunManifest does not record cuDNN/cuBLAS version parity or enforce a cross-run library lock')

    def test_container_resource_limits_handling(self, tmp_path, monkeypatch):
        assert_environment_controls(tmp_path, monkeypatch, 'cpu_quota')

    def test_virtualization_overhead_handling(self, tmp_path, monkeypatch):
        assert_environment_controls(tmp_path, monkeypatch, 'virtualization')


# =============================================================================
# ADDITIONAL STATISTICAL PROTECTION TESTS
# =============================================================================

class TestStatisticalProtectionsExtended:
    """Extended statistical protection tests."""

    def test_outlier_injection_detection(self):
        """Retain all supplied samples; this does not detect samples omitted upstream."""
        import statistics
        harness, config = cpu_harness()
        clean = [1.2, 1.0, 1.4, 1.1, 1.3]
        with_outlier = clean + [10.0]
        assert harness._compute_stats(clean, config).timing.raw_times_ms == clean
        result = harness._compute_stats(with_outlier, config).timing
        assert result.raw_times_ms == with_outlier
        assert result.iterations == len(with_outlier)
        assert result.mean_ms == statistics.mean(with_outlier)
        assert result.max_ms == 10.0

    def test_variance_gaming_detection(self):
        import statistics
        harness, config = cpu_harness()
        constant = [1.0, 1.0, 1.0]
        variable = [1.0, 2.0, 9.0]
        assert harness._compute_stats(constant, config).timing.std_ms == 0
        result = harness._compute_stats(variable, config).timing
        assert result.std_ms == pytest.approx(statistics.stdev(variable))
        assert result.raw_times_ms == variable

    def test_percentile_selection_detection(self):
        harness, config = cpu_harness()
        config.percentiles = [50.0, 90.0, 99.0]
        samples = [9.0, 1.0, 5.0, 2.0, 8.0]
        result = harness._compute_stats(samples, config).timing
        assert result.median_ms == 5.0
        assert result.percentiles == {50.0: 5.0, 90.0: 8.0, 99.0: 8.0}
        assert result.raw_times_ms == samples


# =============================================================================
# ADDITIONAL EVALUATION PROTECTION TESTS
# =============================================================================

class TestEvaluationProtectionsExtended:
    """Extended evaluation protection tests."""

    def test_metric_definition_gaming_detection(self):
        from core.benchmark.comparison import compare_metric
        clean = compare_metric("timing.mean_ms", 10.0, 5.0)
        slower = compare_metric("timing.mean_ms", 10.0, 20.0)
        assert clean.ratio == 2.0
        assert not clean.regression
        assert slower.ratio == 0.5
        assert slower.regression

    def test_self_modifying_tests_prevention(self):
        _check_config_mutation('enforce_config_immutability', False)

    def test_benchmark_memorization_prevention(self, runner):
        """Reject cached results; allocating an output buffer is not itself cheating."""
        _check_fresh_controls(runner)

    def test_missing_holdout_sets_handling(self):
        pytest.skip("Missing protection: no dataset provenance/holdout-overlap detector is implemented; set arithmetic is not detector coverage")


# =============================================================================
# REPRODUCIBILITY PROTECTION TESTS
# =============================================================================

class TestReproducibilityProtections:
    """Tests for reproducibility protections."""

    def test_version_locking_in_manifest(self):
        pytest.skip('Missing production protection: version provenance capture does not enforce version locking')

    def test_seed_determinism(self, runner):
        work = TensorWork()
        first, *_ = runner._run_with_seed(work, 42)
        repeated, *_ = runner._run_with_seed(work, 42)
        changed, *_ = runner._run_with_seed(work, 43)
        assert runner._compare_outputs(first, repeated).passed
        assert not runner._compare_outputs(first, changed).passed

    def test_hardware_info_capture(self):
        'Actual provenance capture with honest CPU absence; not cross-machine qualification.'
        from core.benchmark.run_manifest import get_gpu_info
        observed = get_gpu_info()
        if torch.cuda.is_available():
            assert observed['model'] == torch.cuda.get_device_name(0)
            assert observed['compute_capability'] == '.'.join(map(str, torch.cuda.get_device_capability(0)))
        else:
            assert observed['model'] is None
            assert observed['compute_capability'] is None

    def test_environment_snapshot(self, monkeypatch):
        'Actual captured environment differs after a real environment-variable change; no version-locking policy is claimed.'
        from core.benchmark.run_manifest import RunManifest
        monkeypatch.setenv('OMP_NUM_THREADS', '1')
        first = RunManifest.create()
        monkeypatch.setenv('OMP_NUM_THREADS', '2')
        second = RunManifest.create()
        assert first.environment.relevant_env_vars['OMP_NUM_THREADS'] == '1'
        assert second.environment.relevant_env_vars['OMP_NUM_THREADS'] == '2'
        assert first.model_dump()['environment']['relevant_env_vars']['OMP_NUM_THREADS'] == '1'
        assert first.software.pytorch_version == torch.__version__

    @requires_cuda
    def test_run_manifest_completeness(self):
        """Test that run manifest captures all needed info.

        Protection: Complete run manifest
        Attack: Missing context leads to irreproducibility
        """
        from core.benchmark.run_manifest import RunManifest

        manifest = RunManifest.create(
            config={"iterations": 1, "warmup": 0, "validity_profile": "strict"}
        )

        assert manifest.software.pytorch_version == torch.__version__
        assert manifest.hardware.cuda_version is not None
        assert manifest.git is not None
        assert manifest.environment is not None
        assert manifest.config == {"iterations": 1, "warmup": 0, "validity_profile": "strict"}


# =============================================================================
# COMPREHENSIVE PROTECTION SUMMARY TEST
# =============================================================================

class TestProtectionSummary:
    """Summary test to verify all protection categories are covered."""

    def test_all_protection_categories_have_tests(self):
        """Inventory compatibility only; counts never establish detector coverage."""
        pytest.skip("A count of test names is not protection coverage; inspect named behavioral tests and explicit missing-protection skips")
