#!/usr/bin/env python3
"""Behavioral edge cases for implemented benchmark protections.

All original test IDs are retained. CPU tests exercise actual verification or
explicit diagnostic inputs, not GPU execution. CUDA tests require real hardware.
Missing protections are explicit skips; narrower reporting and diagnostic tests
state their limits rather than claim that a broader attack detector exists.
"""

import gc
import statistics
from dataclasses import replace

import pytest
import torch

from core.benchmark.verification import InputSignature, PrecisionFlags, ToleranceSpec
from core.benchmark.verify_runner import VerifyConfig
from tests.protection_test_utils import (
    TensorWork, assert_comparison_controls, assert_compile_cache_reset,
    assert_compile_guard_counts, assert_config_immutability, assert_environment_controls,
    assert_gpu_state_controls, assert_materialization_diagnostic,
    assert_memory_pattern_controls, assert_signature_controls,
    assert_stream_audit_controls, check_jitter, compare_tensors, cpu_harness,
    make_runner, preserve_rng_state,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA protection integration requires a device")


@pytest.fixture(autouse=True)
def restore_rng_state():
    with preserve_rng_state():
        yield


@pytest.fixture
def runner(tmp_path):
    return make_runner(tmp_path)



class TestTimingEdgeCases:
    @requires_cuda
    def test_unsynced_streams_multiple_streams(self):
        assert_stream_audit_controls(count=4)

    @requires_cuda
    def test_unsynced_streams_nested_streams(self):
        assert_stream_audit_controls(nested=True)

    @requires_cuda
    def test_unsynced_streams_priority_streams(self):
        assert_stream_audit_controls(priorities=True)

    @requires_cuda
    def test_async_ops_chained_operations(self):
        assert_stream_audit_controls(count=1)

    @requires_cuda
    def test_async_ops_mixed_cpu_gpu(self):
        assert_stream_audit_controls(pinned=True)

    def test_event_timing_zero_duration(self):
        'Timing-summary diagnostic boundary, not a CUDA event measurement.'
        from core.benchmark.models import TimingStats
        from core.harness.benchmark_harness import BenchmarkHarness
        clean = TimingStats(mean_ms=0, median_ms=0, std_ms=0, min_ms=0, max_ms=0,
                            iterations=1, warmup_iterations=0)
        BenchmarkHarness._validate_timing_summary(clean)
        with pytest.raises(ValueError, match='nonnegative'):
            BenchmarkHarness._validate_timing_summary(clean.model_copy(update={'mean_ms': -1e-9}))

    def test_event_timing_very_long_duration(self):
        'Large finite timing-summary boundary, not a long-running GPU allocation.'
        from core.benchmark.models import TimingStats
        from core.harness.benchmark_harness import BenchmarkHarness
        clean = TimingStats(mean_ms=1e10, median_ms=1e10, std_ms=0, min_ms=1e10, max_ms=1e10,
                            iterations=1, warmup_iterations=0)
        BenchmarkHarness._validate_timing_summary(clean)
        with pytest.raises(ValueError, match='finite'):
            BenchmarkHarness._validate_timing_summary(clean.model_copy(update={'mean_ms': float('inf')}))

    @requires_cuda
    def test_timer_granularity_sub_microsecond(self):
        'Real CUDA adaptive iteration accumulation; no assertion of sub-microsecond hardware resolution.'
        from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness
        value = torch.ones(8, device='cuda')
        for adaptive in [False, True]:
            config = BenchmarkConfig(device=torch.device('cuda'), iterations=1, warmup=5,
                                     use_subprocess=False, enable_profiling=False,
                                     adaptive_iterations=adaptive, min_total_duration_ms=5.0,
                                     max_adaptive_iterations=10000)
            calls = []
            def work(calls=calls):
                value.add_(1)
                calls.append(None)
            samples, _ = BenchmarkHarness(config=config)._benchmark_custom(work, config)
            assert len(samples) == len(calls)
            if adaptive:
                assert len(samples) > 1
                assert sum(samples) >= config.min_total_duration_ms
            else:
                assert len(samples) == 1

    def test_warmup_bleed_jit_compilation(self):
        'Real CPU JIT and harness phase separation; no first-call speed assumption.'
        harness, config = cpu_harness()
        compiled = torch.jit.trace(lambda value: value * 2 + 1, torch.ones(8))
        outputs = []
        def work():
            outputs.append(compiled(torch.ones(8)))
        harness._warmup(work, config.warmup, config)
        assert len(outputs) == config.warmup
        outputs.clear()
        samples, _ = harness._benchmark_custom(work, config)
        assert len(samples) == len(outputs) == config.iterations
        for output in outputs:
            torch.testing.assert_close(output, torch.full((8,), 3.0))

    def test_warmup_bleed_cudnn_autotuning(self):
        'Actual backend-policy mutation detection; no cuDNN timing or GPU claim.'
        from core.harness.validity_checks import capture_precision_policy_state, check_precision_policy_consistency
        before = capture_precision_policy_state()
        assert check_precision_policy_consistency(before, capture_precision_policy_state()) == (True, [])
        torch.backends.cudnn.benchmark = not before.cudnn_benchmark
        passed, reasons = check_precision_policy_consistency(before, capture_precision_policy_state())
        assert not passed
        assert any('cudnn_benchmark' in reason for reason in reasons)

    def test_clock_drift_long_benchmark(self):
        'Clock-drop diagnostic classifier, not a measured long GPU run.'
        assert_gpu_state_controls(clock_mhz=800)

    def test_profiler_overhead_nested_profilers(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no nested-profiler rejection guard is implemented; a single profiler context did not test nesting')



class TestOutputEdgeCases:
    def test_constant_output_single_value(self, runner):
        assert check_jitter(runner, 'real') == (True, None)
        passed, reason = check_jitter(runner, 'constant')
        assert not passed
        assert 'output unchanged' in reason

    def test_constant_output_near_constant(self, runner):
        assert check_jitter(runner, 'real') == (True, None)
        class NearConstant(TensorWork):
            def benchmark_fn(self):
                self.output = torch.ones_like(self.input) + self.input * 1e-10
        work = NearConstant()
        work.setup()
        work.benchmark_fn()
        passed, reason = runner._run_jitter_check(work, work.get_input_signature(), VerifyConfig())
        assert not passed
        assert 'output unchanged' in reason

    def test_stale_cache_after_input_change(self, runner):
        assert check_jitter(runner, 'real') == (True, None)
        passed, reason = check_jitter(runner, 'cached')
        assert not passed
        assert 'output unchanged' in reason

    def test_approximation_drift_accumulated_error(self, runner):
        assert_comparison_controls(runner, torch.ones(8), torch.ones(8) * (1.0001 ** 1000))

    def test_approximation_drift_fp16_accumulation(self, runner):
        expected = torch.full((8,), 1.1, dtype=torch.float16)
        actual = torch.ones(8, dtype=torch.float16)
        for _ in range(100):
            actual.add_(0.001)
        assert_comparison_controls(runner, expected, actual, ToleranceSpec(rtol=1e-4, atol=1e-4))

    def test_nan_from_division_by_zero(self, runner):
        assert_comparison_controls(runner, torch.ones(3), torch.zeros(3) / torch.zeros(3))

    def test_nan_from_sqrt_negative(self, runner):
        assert_comparison_controls(runner, torch.ones(3), torch.sqrt(torch.full((3,), -1.0)))

    def test_nan_propagation(self, runner):
        assert_comparison_controls(runner, torch.ones(3), torch.tensor([1.0, float('nan'), 2.0]) * 2)

    def test_inf_from_overflow(self, runner):
        assert_comparison_controls(runner, torch.ones(3), torch.full((3,), 1e38).square())

    def test_inf_from_division(self, runner):
        assert_comparison_controls(runner, torch.ones(3), torch.full((3,), 1e38) / 1e-38)

    def test_ground_truth_contains_nan(self, runner):
        expected = torch.tensor([1.0, 2.0])
        assert compare_tensors(runner, expected, expected.clone()).passed
        invalid = expected.clone()
        invalid[0] = float('nan')
        assert not compare_tensors(runner, invalid, expected).passed
        assert not compare_tensors(runner, invalid, invalid.clone()).passed

    def test_shape_mismatch_broadcast_ambiguity(self, runner):
        assert_comparison_controls(runner, torch.ones(4, 8), torch.ones(1, 8))

    def test_shape_mismatch_transposed(self, runner):
        assert_comparison_controls(runner, torch.ones(4, 8), torch.ones(8, 4))

    def test_dtype_mismatch_float_precisions(self, runner):
        'Runtime input dtype enforcement; no claim that output dtype coercion is a detector.'
        work = TensorWork()
        work.setup()
        signature = work.get_input_signature()
        runner._validate_inputs_match_signature(signature, work.get_verify_inputs())
        with pytest.raises(ValueError, match='dtype mismatch'):
            runner._validate_inputs_match_signature(signature, {'input': work.input.to(torch.float16)})

    def test_dtype_mismatch_int_vs_float(self, runner):
        'Runtime input dtype enforcement; no claim that output dtype coercion is a detector.'
        work = TensorWork()
        work.setup()
        signature = work.get_input_signature()
        runner._validate_inputs_match_signature(signature, work.get_verify_inputs())
        with pytest.raises(ValueError, match='dtype mismatch'):
            runner._validate_inputs_match_signature(signature, {'input': work.input.to(torch.int64)})

    def test_denormalized_values_detection(self, runner):
        'Exact comparator rejects flushing subnormal values to zero; default tolerance is unchanged.'
        expected = torch.tensor([1e-40], dtype=torch.float32)
        assert_comparison_controls(runner, expected, torch.zeros_like(expected), ToleranceSpec(rtol=0, atol=0))

    def test_uninitialized_memory_torch_empty(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no uninitialized-memory provenance detector; filling torch.empty before comparison erases the condition')



class TestWorkloadEdgeCases:
    def test_batch_shrinking_by_one(self):
        'Production signature equivalence rejects changed workload metadata, including zero batch versus nonzero.'
        assert_signature_controls(shapes={'input': (3, 8)}, batch_size=3)

    def test_batch_shrinking_zero_batch(self):
        'Production signature equivalence rejects changed workload metadata, including zero batch versus nonzero.'
        assert_signature_controls(shapes={'input': (0, 8)}, batch_size=0)

    def test_sequence_truncation_by_one(self):
        'Production signature equivalence rejects changed workload metadata, including zero batch versus nonzero.'
        assert_signature_controls(shapes={'input': (4, 7)})

    def test_hidden_downsampling_power_of_two(self):
        'Production signature equivalence rejects changed workload metadata, including zero batch versus nonzero.'
        assert_signature_controls(shapes={'input': (4, 4)})

    def test_precision_mismatch_tf32(self):
        'Production signature equivalence rejects changed workload metadata, including zero batch versus nonzero.'
        assert_signature_controls(precision_flags=PrecisionFlags(tf32=False))

    def test_precision_mismatch_fp8_variants(self):
        'FP8 variant identity is carried by actual signature dtype strings, not the shared fp8 flag.'
        baseline = InputSignature(shapes={'input': (4, 8)}, dtypes={'input': 'float8_e4m3fn'},
                                  batch_size=4, parameter_count=0, precision_flags=PrecisionFlags(fp8=True))
        assert baseline.matches(InputSignature.from_dict(baseline.to_dict()))
        assert not baseline.matches(replace(baseline, dtypes={'input': 'float8_e5m2'}))

    def test_undeclared_shortcuts_skip_layers(self, runner):
        full = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.Linear(8, 8))
        shortcut = torch.nn.Sequential(full[0])
        value = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        assert_comparison_controls(runner, full(value), shortcut(value))
        assert_signature_controls(parameter_count=sum(p.numel() for p in shortcut.parameters()))

    def test_early_exit_threshold_boundary(self, runner):
        value = torch.tensor([0.95, 0.96], dtype=torch.float64)
        expected = torch.where(value > 0.95, value * 2, value * 3)
        incorrect = torch.where(value >= 0.95, value * 2, value * 3)
        assert_comparison_controls(runner, expected, incorrect)

    def test_sparsity_mismatch_one_percent_difference(self):
        'Production signature equivalence rejects changed workload metadata, including zero batch versus nonzero.'
        assert_signature_controls(sparsity_ratio=0.51)

    def test_sparsity_mismatch_near_zero(self):
        'Production signature equivalence rejects changed workload metadata, including zero batch versus nonzero.'
        assert_signature_controls(sparsity_ratio=0.001)

    def test_attention_mask_causal_vs_full(self, runner):
        assert_comparison_controls(runner, torch.ones(4, 4), torch.tril(torch.ones(4, 4)))

    def test_kv_cache_size_off_by_one(self):
        'Production signature equivalence rejects changed workload metadata, including zero batch versus nonzero.'
        assert_signature_controls(shapes={'input': (4, 7)})

    def test_train_test_overlap_single_sample(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no dataset provenance, feature-label leakage or holdout-overlap detector is implemented')



class TestLocationEdgeCases:
    def test_cpu_spillover_single_op_on_cpu(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no per-operation CPU spillover detector is implemented; wall-time measurement alone does not identify execution placement')

    def test_cpu_spillover_data_dependent_branch(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no per-operation CPU spillover detector is implemented; wall-time measurement alone does not identify execution placement')

    def test_setup_precomputation_cached_result(self, runner):
        assert check_jitter(runner, 'real') == (True, None)
        passed, reason = check_jitter(runner, 'cached')
        assert not passed
        assert 'output unchanged' in reason

    def test_graph_capture_computation_in_capture(self):
        'Timing diagnostic classification, not a captured graph or measured GPU work.'
        from core.harness.validity_checks import check_graph_capture_integrity
        assert check_graph_capture_integrity(2.0, [1.0, 1.1, 0.9]) == (True, None)
        passed, reason = check_graph_capture_integrity(100.0, [1.0, 1.1, 0.9])
        assert not passed
        assert 'Suspected work during capture' in reason

    def test_warmup_computation_result_reused(self, runner):
        assert check_jitter(runner, 'real') == (True, None)
        passed, reason = check_jitter(runner, 'cached')
        assert not passed
        assert 'output unchanged' in reason

    def test_background_thread_computation(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no background-thread computation isolation guard; process isolation does not forbid threads inside a benchmark')

    def test_lazy_evaluation_never_materialized(self, monkeypatch):
        'The production materializer reports a capability warning for a real meta tensor; it does not reject the run.'
        assert_materialization_diagnostic(monkeypatch)

    def test_jit_compilation_first_call_slow(self):
        'Real CPU JIT and harness phase separation; no first-call speed assumption.'
        harness, config = cpu_harness()
        compiled = torch.jit.trace(lambda value: value * 2 + 1, torch.ones(8))
        outputs = []
        def work():
            outputs.append(compiled(torch.ones(8)))
        harness._warmup(work, config.warmup, config)
        assert len(outputs) == config.warmup
        outputs.clear()
        samples, _ = harness._benchmark_custom(work, config)
        assert len(samples) == len(outputs) == config.iterations
        for output in outputs:
            torch.testing.assert_close(output, torch.full((8,), 3.0))



class TestMemoryEdgeCases:
    def test_preallocated_output_exact_shape(self, runner):
        'Preallocated storage is allowed when recomputed; cached output is rejected.'
        class Preallocated(TensorWork):
            def setup(self):
                super().setup()
                self.output = torch.empty_like(self.input)
            def benchmark_fn(self):
                self.output.copy_(self.input * 2)
        work = Preallocated()
        work.setup()
        work.benchmark_fn()
        assert runner._run_jitter_check(work, work.get_input_signature(), VerifyConfig()) == (True, None)
        passed, reason = check_jitter(runner, 'cached')
        assert not passed
        assert 'output unchanged' in reason

    def test_preallocated_output_view(self, runner):
        'Preallocated storage is allowed when recomputed; cached output is rejected.'
        class Preallocated(TensorWork):
            def setup(self):
                super().setup()
                self.storage = torch.empty(8, 4)
                self.output = self.storage.T
            def benchmark_fn(self):
                self.output.copy_(self.input * 2)
        work = Preallocated()
        work.setup()
        work.benchmark_fn()
        assert runner._run_jitter_check(work, work.get_input_signature(), VerifyConfig()) == (True, None)
        passed, reason = check_jitter(runner, 'cached')
        assert not passed
        assert 'output unchanged' in reason

    def test_input_output_aliasing_inplace(self):
        from core.harness.validity_checks import check_input_output_aliasing
        value = torch.arange(8, dtype=torch.float32)
        assert check_input_output_aliasing({'input': value}, {'output': value.clone()}) == (True, None)
        passed, reason = check_input_output_aliasing({'input': value}, {'output': value})
        assert not passed
        assert 'OUTPUT ALIASING DETECTED' in reason

    def test_input_output_aliasing_view(self):
        from core.harness.validity_checks import check_input_output_aliasing
        value = torch.arange(8, dtype=torch.float32)
        assert check_input_output_aliasing({'input': value}, {'output': value.clone()}) == (True, None)
        passed, reason = check_input_output_aliasing({'input': value}, {'output': value.view(2, 4)})
        assert not passed
        assert 'OUTPUT ALIASING DETECTED' in reason

    @requires_cuda
    def test_pinned_memory_transfer_timing(self):
        assert_stream_audit_controls(pinned=True)

    @requires_cuda
    def test_memory_pool_cached_allocation(self):
        'Real CUDA allocator cleanup integration; does not claim cached address reuse is cheating.'
        from core.harness.validity_checks import reset_cuda_memory_pool
        value = torch.ones(1024, device='cuda')
        reset_cuda_memory_pool(torch.device('cuda'))
        # A live allocation must survive cache cleanup; freed cached blocks must not.
        torch.testing.assert_close(value, torch.ones_like(value))
        assert torch.cuda.memory_stats()['allocated_bytes.all.current'] >= value.numel() * value.element_size()
        assert torch.cuda.memory_stats()['active_bytes.all.current'] <= torch.cuda.memory_reserved()

    def test_fragmentation_many_small_allocs(self):
        'Production peak-memory diagnostic, not proof of fragmentation detection or an allocation stress test.'
        assert_memory_pattern_controls('peak')

    def test_page_fault_first_access(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no page-fault or managed-memory event detector is implemented')

    def test_swap_interference_large_allocation(self, tmp_path, monkeypatch):
        assert_environment_controls(tmp_path, monkeypatch, 'swap')



class TestCudaEdgeCases:
    def test_host_callback_with_stream(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no host callback execution detector is implemented')

    @requires_cuda
    def test_async_memcpy_stream_ordering(self):
        assert_stream_audit_controls(count=2, pinned=True)

    def test_workspace_precomputed_cublas(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no cuBLAS workspace precomputation detector is implemented')

    def test_persistent_kernel_occupancy(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no persistent kernel lifetime detector is implemented')

    def test_undeclared_multi_gpu_single_declared(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no undeclared GPU execution detector is implemented')

    def test_context_switch_device_change(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no CUDA context switch enforcement detector is implemented')

    def test_driver_overhead_many_small_kernels(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no driver-overhead attribution detector is implemented')

    def test_cooperative_launch_grid_sync(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no cooperative launch inspection detector is implemented')

    def test_dynamic_parallelism_nested_launch(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no device-side dynamic launch inspection detector is implemented')

    def test_unified_memory_managed_allocation(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no page-fault or managed-memory event detector is implemented')



class TestCompileEdgeCases:
    def test_cache_hit_shape_variation(self):
        'Real CPU Dynamo cache hit and reset behavior, with graph execution checked.'
        assert_compile_cache_reset(dynamic=False)

    def test_trace_reuse_dynamic_shapes(self):
        'Real CPU Dynamo cache hit and reset behavior, with graph execution checked.'
        assert_compile_cache_reset(dynamic=True)

    def test_mode_inconsistency_train_vs_eval(self, runner):
        'Output verification detects this real mode-dependent difference; no general mode-policy validator is implied.'
        model = torch.nn.BatchNorm1d(8)
        value = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        model.train()
        expected = model(value)
        model.eval()
        assert_comparison_controls(runner, expected, model(value))

    def test_inductor_asymmetry_different_backends(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no compiler-backend parity policy detector is implemented; eager versus compiled equivalence alone cannot enforce fair compiler modes')

    def test_guard_failure_control_flow(self):
        assert_compile_guard_counts()

    def test_autotuning_variance_repeated_runs(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no autotuning variance or compiler-mode parity detector is implemented')

    def test_symbolic_shape_specialization(self):
        'Production signature equivalence rejects changed workload metadata, including zero batch versus nonzero.'
        assert_signature_controls(shapes={'input': (4, 9)})



class TestDistributedEdgeCases:
    def test_rank_skipping_single_rank(self):
        'Rank receipt validation, not a distributed execution measurement.'
        from core.harness.validity_checks import verify_distributed_outputs
        clean = {rank: 'same-output-hash' for rank in range(4)}
        assert verify_distributed_outputs(clean, 4).all_ranks_executed
        violation = dict(clean)
        del violation[2]
        result = verify_distributed_outputs(violation, 4)
        assert not result.all_ranks_executed
        assert '2' in result.error_message

    def test_collective_shortcircuit_single_element(self, runner):
        'Verifies a missing reduction result; no multiprocess collective is claimed.'
        expected = torch.tensor([4.0])
        assert_comparison_controls(runner, expected, torch.tensor([1.0]))

    def test_topology_mismatch_ring_vs_tree(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: DistributedTopology does not encode ring versus tree algorithm, so it cannot enforce this claimed policy')

    def test_barrier_timing_straggler(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no rank barrier timing detector is implemented')

    def test_gradient_bucketing_different_sizes(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no gradient bucket-size parity field or detector is implemented')

    def test_async_gradient_overlap(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no async gradient completion timing detector is implemented')

    def test_pipeline_bubble_microbatch_count(self):
        'Declared per-rank workload comparison, not timing or detecting pipeline bubbles.'
        assert_signature_controls(per_rank_batch_size=1, pipeline_stages=2, pipeline_stage_boundaries=[(0, 1), (2, 3)])

    def test_shard_size_imbalanced(self):
        'Declared shard dimensions are compared; no multiprocess run is claimed.'
        assert_signature_controls(shards=4, shapes={'input': (3, 8)})



class TestEnvironmentEdgeCases:
    def test_device_mismatch_compute_capability(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no expected-versus-observed compute-capability comparison is performed by validate_environment')

    def test_frequency_boost_detection(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: GPU state consistency detects clock drops, not clock boosts or absolute clock locking')

    def test_priority_elevation_process_nice(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no process-priority parity detector is implemented')

    def test_memory_overcommit_reservation(self):
        'Detects excessive allocated-memory growth in diagnostic snapshots; no OS overcommit policy detector is implied.'
        assert_memory_pattern_controls('increase')

    def test_numa_gpu_affinity(self, tmp_path, monkeypatch):
        assert_environment_controls(tmp_path, monkeypatch, 'numa')

    def test_cpu_governor_consistency(self, tmp_path, monkeypatch):
        assert_environment_controls(tmp_path, monkeypatch, 'governor')

    def test_thermal_throttling_temperature_check(self):
        assert_gpu_state_controls(temperature_c=51)

    def test_power_limit_variation(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: GPUState has no power-limit field and check_gpu_state_consistency does not compare power limits')

    def test_driver_version_recorded(self):
        'Compare actual driver provenance with NVML, separately from the CUDA runtime version.'
        if not torch.cuda.is_available():
            pytest.skip('real CUDA/NVML driver provenance integration requires a device')
        import pynvml
        from core.benchmark.run_manifest import get_cuda_info
        pynvml.nvmlInit()
        try:
            actual_driver = pynvml.nvmlSystemGetDriverVersion()
        finally:
            pynvml.nvmlShutdown()
        if isinstance(actual_driver, bytes):
            actual_driver = actual_driver.decode()
        observed = get_cuda_info()
        assert observed['driver_version'] == actual_driver
        assert observed['version'] == torch.version.cuda

    def test_library_version_cudnn(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: RunManifest does not record cuDNN version or compare baseline/optimized library versions')

    def test_container_cgroup_limits(self, tmp_path, monkeypatch):
        assert_environment_controls(tmp_path, monkeypatch, 'memory_limit')

    def test_virtualization_detection(self, tmp_path, monkeypatch):
        assert_environment_controls(tmp_path, monkeypatch, 'virtualization')



class TestStatisticalEdgeCases:
    def test_outlier_injection_single_extreme(self):
        'Reporting fidelity only. No production statistical outlier/omission/cherry-pick classifier exists; that requirement stays open.'
        samples = [1.0] * 99 + [100.0]
        harness, config = cpu_harness()
        result = harness._compute_stats(samples, config).timing
        assert result.raw_times_ms == samples
        assert result.iterations == len(samples)
        assert result.mean_ms == pytest.approx(statistics.mean(samples))
        assert result.std_ms == pytest.approx(statistics.stdev(samples))
        assert result.max_ms == max(samples)

    def test_outlier_injection_median_robust(self):
        'Reporting fidelity only. No production statistical outlier/omission/cherry-pick classifier exists; that requirement stays open.'
        samples = [1.0] * 99 + [1000.0]
        harness, config = cpu_harness()
        result = harness._compute_stats(samples, config).timing
        assert result.raw_times_ms == samples
        assert result.iterations == len(samples)
        assert result.mean_ms == pytest.approx(statistics.mean(samples))
        assert result.std_ms == pytest.approx(statistics.stdev(samples))
        assert result.max_ms == max(samples)

    def test_variance_gaming_consistent_slow(self):
        'Reporting fidelity only. No production statistical outlier/omission/cherry-pick classifier exists; that requirement stays open.'
        samples = [0.5, 1.5, 0.6, 1.4, 0.7, 1.3]
        harness, config = cpu_harness()
        result = harness._compute_stats(samples, config).timing
        assert result.raw_times_ms == samples
        assert result.iterations == len(samples)
        assert result.mean_ms == pytest.approx(statistics.mean(samples))
        assert result.std_ms == pytest.approx(statistics.stdev(samples))
        assert result.max_ms == max(samples)

    def test_percentile_selection_p50_vs_p99(self):
        'Reporting fidelity only. No production statistical outlier/omission/cherry-pick classifier exists; that requirement stays open.'
        samples = [1.0] * 99 + [10.0]
        harness, config = cpu_harness()
        result = harness._compute_stats(samples, config).timing
        assert result.raw_times_ms == samples
        assert result.iterations == len(samples)
        assert result.mean_ms == pytest.approx(statistics.mean(samples))
        assert result.std_ms == pytest.approx(statistics.stdev(samples))
        assert result.max_ms == max(samples)

    def test_insufficient_samples_high_variance(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no statistical-power or minimum-samples-based-on-variance detector is implemented')

    def test_cold_start_first_iteration_slow(self):
        'Reporting fidelity only. No production statistical outlier/omission/cherry-pick classifier exists; that requirement stays open.'
        samples = [10.0, 1.0, 1.0, 1.0]
        harness, config = cpu_harness()
        result = harness._compute_stats(samples, config).timing
        assert result.raw_times_ms == samples
        assert result.iterations == len(samples)
        assert result.mean_ms == pytest.approx(statistics.mean(samples))
        assert result.std_ms == pytest.approx(statistics.stdev(samples))
        assert result.max_ms == max(samples)

    def test_gc_interference_collection_pause(self):
        from core.harness.validity_checks import gc_disabled
        before = gc.isenabled()
        with pytest.raises(RuntimeError, match='injected failure'):
            with gc_disabled():
                assert not gc.isenabled()
                raise RuntimeError('injected failure')
        assert gc.isenabled() == before

    def test_background_noise_cpu_bound(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no background CPU noise or process-priority detector is implemented')

    def test_cherry_picking_best_of_n(self):
        'Reporting fidelity only. No production statistical outlier/omission/cherry-pick classifier exists; that requirement stays open.'
        samples = [1.2, 1.0, 1.4, 1.1, 1.3, 1.5, 0.9, 1.25]
        harness, config = cpu_harness()
        result = harness._compute_stats(samples, config).timing
        assert result.raw_times_ms == samples
        assert result.iterations == len(samples)
        assert result.mean_ms == pytest.approx(statistics.mean(samples))
        assert result.std_ms == pytest.approx(statistics.stdev(samples))
        assert result.max_ms == max(samples)



class TestEvaluationEdgeCases:
    def test_eval_code_exploitation_hardcoded(self, runner):
        assert check_jitter(runner, 'real') == (True, None)
        passed, reason = check_jitter(runner, 'constant')
        assert not passed
        assert 'output unchanged' in reason

    def test_timeout_manipulation_extend(self):
        assert_config_immutability('timeout_seconds', 999)

    def test_metric_gaming_threshold_tuning(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no evaluation-threshold policy or dataset evaluator contract is implemented')

    def test_data_leakage_feature_from_label(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no dataset provenance, feature-label leakage or holdout-overlap detector is implemented')

    def test_overfitting_train_on_test(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no dataset provenance, feature-label leakage or holdout-overlap detector is implemented')

    def test_self_modifying_immutable_test(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no test-source immutability guard is implemented in the benchmark protection path')

    def test_memorization_hash_detection(self, runner):
        assert check_jitter(runner, 'real') == (True, None)
        passed, reason = check_jitter(runner, 'cached')
        assert not passed
        assert 'output unchanged' in reason

    def test_missing_holdout_temporal_split(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no dataset provenance, feature-label leakage or holdout-overlap detector is implemented')



class TestBoundaryConditions:
    def test_tolerance_exactly_at_boundary(self, runner):
        expected = torch.zeros(1, dtype=torch.float64)
        tolerance = ToleranceSpec(rtol=0, atol=0.125)
        assert compare_tensors(runner, expected, torch.tensor([0.125], dtype=torch.float64), tolerance).passed
        outside = torch.nextafter(torch.tensor([0.125], dtype=torch.float64), torch.tensor([float('inf')], dtype=torch.float64))
        assert not compare_tensors(runner, expected, outside, tolerance).passed

    def test_workload_mismatch_at_one_percent(self):
        from core.benchmark.verification import compare_workload_metrics
        assert compare_workload_metrics({'bytes': 1000000}, {'bytes': 990000})[0]
        assert not compare_workload_metrics({'bytes': 1000000}, {'bytes': 989999})[0]

    def test_sparsity_at_detection_threshold(self):
        'Signature sparsity policy is exact matching; no invented 1% tolerance.'
        baseline = InputSignature(shapes={'input': (4, 8)}, dtypes={'input': 'float32'},
                                  batch_size=4, parameter_count=0, precision_flags=PrecisionFlags(), sparsity_ratio=0.50)
        assert baseline.matches(replace(baseline))
        assert not baseline.matches(replace(baseline, sparsity_ratio=0.51))

    def test_timing_variance_coefficient(self):
        'Graph replay variance diagnostic, not a general timing acceptance policy.'
        from core.harness.validity_checks import GraphCaptureCheatDetector, GraphCaptureState
        detector = GraphCaptureCheatDetector()
        detector.capture_state = GraphCaptureState(capture_start_time=0.0, capture_end_time=0.001)
        detector.replay_times = [1.0, 1.0, 1.0]
        assert detector.check_for_cheat() == (False, None)
        detector.replay_times = [0.1, 1.0, 2.0]
        detected, reason = detector.check_for_cheat()
        assert detected
        assert 'VARIANCE' in reason



class TestExtremeValues:
    def test_very_large_tensor(self):
        'Real failure propagation; explicit injected OOM avoids allocating most of the device or claiming actual exhaustion.'
        harness, config = cpu_harness()
        work = TensorWork()
        work.setup()
        assert len(harness._benchmark_custom(work.benchmark_fn, config)[0]) == config.iterations
        def allocation_failure():
            raise torch.OutOfMemoryError('injected allocation failure')
        with pytest.raises(torch.OutOfMemoryError, match='injected allocation failure'):
            harness._benchmark_custom(allocation_failure, config)

    def test_very_small_values(self, runner):
        'Exact comparator rejects flushing subnormal values to zero; default tolerance is unchanged.'
        expected = torch.tensor([1e-40], dtype=torch.float32)
        assert_comparison_controls(runner, expected, torch.zeros_like(expected), ToleranceSpec(rtol=0, atol=0))

    def test_zero_dimensions(self, runner):
        expected = torch.empty(10, 0)
        assert compare_tensors(runner, expected, expected.clone()).passed
        assert not compare_tensors(runner, expected, torch.empty(0, 10)).passed
        assert not compare_tensors(runner, expected, torch.ones(10, 1)).passed

    def test_single_element(self, runner):
        assert_comparison_controls(runner, torch.tensor([2.0]), torch.tensor([1.0]))

    def test_maximum_dimensions(self, runner):
        expected = torch.ones(*([2] * 10))
        violation = expected.clone()
        violation[(1,) * 10] = 2
        assert_comparison_controls(runner, expected, violation)



class TestRaceConditions:
    @requires_cuda
    def test_concurrent_stream_operations(self):
        assert_stream_audit_controls(threaded=True)

    def test_concurrent_allocations(self):
        'Requirement remains open; this retained test ID is not passing coverage.'
        pytest.skip('Missing production protection: no concurrent-allocation race detector is implemented; successful allocations did not validate a protection')


class TestAdjacentVerificationRegressions:
    @pytest.mark.parametrize('cached', [False, True])
    def test_jitter_snapshots_nested_reused_output(self, runner, cached):
        class NestedWork(TensorWork):
            def setup(self):
                super().setup()
                self.storage = torch.empty(8, 4)
                self.view = self.storage.T
                self.output = {'result': [self.view, (self.view[:, :4],)], 'metadata': 'stable'}
                self.ran = False

            def benchmark_fn(self):
                if not cached or not self.ran:
                    self.view.copy_(self.input * 2)
                self.ran = True

        work = NestedWork()
        work.setup()
        work.benchmark_fn()
        input_before = work.input.clone()
        output_container = work.output
        output_pointer = work.view.data_ptr()
        passed, reason = runner._run_jitter_check(work, work.get_input_signature(), VerifyConfig())
        assert passed is not cached
        assert reason is None if not cached else 'output unchanged' in reason
        torch.testing.assert_close(work.input, input_before, rtol=0, atol=0)
        assert work.output is output_container
        assert work.output['result'][0].data_ptr() == output_pointer
        assert not work.view.is_contiguous()

    @pytest.mark.parametrize('failure_location', ['capture', 'output'])
    def test_jitter_restores_input_after_output_refresh_failure(self, runner, failure_location):
        class FailingRefresh(TensorWork):
            calls = 0

            def benchmark_fn(self):
                super().benchmark_fn()
                self.calls += 1

            def capture_verification_payload(self):
                if failure_location == 'capture':
                    raise RuntimeError('injected capture failure')

            def get_verify_output(self):
                if failure_location == 'output' and self.calls > 1:
                    raise RuntimeError('injected output failure')
                return super().get_verify_output()

        work = FailingRefresh()
        work.setup()
        work.benchmark_fn()
        input_before = work.input.clone()
        passed, reason = runner._run_jitter_check(work, work.get_input_signature(), VerifyConfig())
        # A failed refresh after perturbation cannot establish input dependence.
        assert not passed
        assert 'failed due to error' in reason
        assert f'injected {failure_location} failure' in reason
        torch.testing.assert_close(work.input, input_before, rtol=0, atol=0)

    def test_snapshot_does_not_hide_input_output_aliasing(self, runner):
        work = TensorWork('alias')
        with pytest.raises(RuntimeError, match='OUTPUT ALIASING DETECTED'):
            runner._run_with_seed(work, 42)

    @pytest.mark.parametrize('invalid_output', ['none', 'structure', 'leaf_type', 'shape', 'dtype', 'nan', 'inf'])
    def test_jitter_rejects_uncomparable_output(self, runner, invalid_output):
        class ChangingOutput(TensorWork):
            calls = 0

            def benchmark_fn(self):
                super().benchmark_fn()
                self.calls += 1

            def get_verify_output(self):
                if self.calls <= 1:
                    return {'result': self.output}
                if invalid_output == 'none':
                    return None
                if invalid_output == 'structure':
                    return [self.output]
                if invalid_output == 'leaf_type':
                    return {'result': 'not a tensor'}
                if invalid_output == 'shape':
                    return {'result': self.output.flatten()}
                if invalid_output == 'dtype':
                    return {'result': self.output.to(torch.float64)}
                return {'result': torch.full_like(self.output, float(invalid_output))}

        work = ChangingOutput()
        work.setup()
        work.benchmark_fn()
        original_input = work.input.clone()
        passed, reason = runner._run_jitter_check(work, work.get_input_signature(), VerifyConfig())
        assert not passed
        assert reason.startswith('Jitter check failed:')
        torch.testing.assert_close(work.input, original_input, rtol=0, atol=0)
