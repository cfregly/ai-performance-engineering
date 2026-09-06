"""Real CPU workloads and scoped verification utilities for protection tests.

These are correctness/control-plane tests, never GPU performance evidence.
No detector, benchmark result, CUDA capability, or measurement is mocked.
"""

import random
from contextlib import contextmanager
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from core.benchmark.quarantine import QuarantineManager
from core.benchmark.verification import InputSignature, PrecisionFlags
from core.benchmark.verify_runner import VerifyConfig, VerifyRunner
from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness


def make_runner(tmp_path):
    return VerifyRunner(
        cache_dir=tmp_path / "golden",
        quarantine_manager=QuarantineManager(cache_dir=tmp_path / "quarantine"),
    )


@contextmanager
def preserve_rng_state():
    """Restore state changed by real seed-verification calls, including failures."""
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    deterministic = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    cudnn_deterministic = torch.backends.cudnn.deterministic
    cudnn_benchmark = torch.backends.cudnn.benchmark
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
        torch.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        torch.backends.cudnn.deterministic = cudnn_deterministic
        torch.backends.cudnn.benchmark = cudnn_benchmark


class TensorWork:
    """Small real CPU computation with explicitly injected incorrect behaviors."""

    device = torch.device("cpu")

    def __init__(self, behavior="real"):
        self.behavior = behavior
        self.output = None
        self.cached_output = None

    def setup(self):
        self.input = torch.randn(4, 8)

    def benchmark_fn(self):
        if self.behavior == "constant":
            self.output = torch.ones_like(self.input)
        elif self.behavior == "cached":
            if self.cached_output is None:
                self.cached_output = self.input * 2
            self.output = self.cached_output
        elif self.behavior == "alias":
            self.output = self.input
        else:
            self.output = self.input * 2
        if self.behavior == "mutate_seed":
            torch.manual_seed(torch.initial_seed() + 1)

    def get_verify_inputs(self):
        return {"input": self.input}

    def get_verify_output(self):
        return self.output

    def get_input_signature(self):
        return InputSignature(
            shapes={"input": tuple(self.input.shape)},
            dtypes={"input": str(self.input.dtype)},
            batch_size=self.input.shape[0],
            parameter_count=0,
            precision_flags=PrecisionFlags(),
        )


def check_jitter(runner, behavior):
    work = TensorWork(behavior)
    work.setup()
    work.benchmark_fn()
    original_input = work.input.clone()
    result = runner._run_jitter_check(work, work.get_input_signature(), VerifyConfig())
    torch.testing.assert_close(work.input, original_input, rtol=0, atol=0)
    return result


def check_fresh_input(runner, behavior):
    work = TensorWork(behavior)
    config = VerifyConfig(seed=42)
    outputs, *_ = runner._run_with_seed(work, config.seed)
    return runner._run_fresh_input_check(work, outputs, config)


def cpu_harness(**kwargs):
    """CPU control-flow execution; environment diagnostics remain visible."""
    config = BenchmarkConfig(
        device=torch.device("cpu"), iterations=3, warmup=5,
        use_subprocess=False, enable_profiling=False, enable_memory_tracking=False,
        enforce_environment_validation=False, **kwargs,
    )
    return BenchmarkHarness(config=config), config


def compare_tensors(runner, expected, actual, tolerance=None):
    return runner._compare_outputs({"output": expected}, {"output": actual}, tolerance)


def audit_cuda_work(synchronize):
    # Allocation occurs before auditing so allocator synchronization cannot mask
    # the deliberately missing synchronization in the timed-work violation.
    stream = torch.cuda.Stream()
    value = torch.ones(1024, device="cuda")
    torch.cuda.synchronize()
    from core.harness.validity_checks import StreamAuditor

    auditor = StreamAuditor()
    auditor.start()
    try:
        with torch.cuda.stream(stream):
            value.add_(1)
        if synchronize:
            stream.synchronize()
    finally:
        auditor.stop()
        stream.synchronize()
    torch.testing.assert_close(value, torch.full_like(value, 2))
    return auditor.check_issues()


def assert_comparison_controls(runner, expected, violation, tolerance=None):
    """Run the real output comparator, including the unchanged-output control."""
    assert compare_tensors(runner, expected, expected.clone(), tolerance).passed
    result = compare_tensors(runner, expected, violation, tolerance)
    assert not result.passed, result


def assert_signature_controls(**changes):
    baseline = InputSignature(
        shapes={"input": (4, 8)}, dtypes={"input": "torch.float32"},
        batch_size=4, parameter_count=100, precision_flags=PrecisionFlags(),
    )
    assert not baseline.validate(strict=True)
    assert baseline.matches(InputSignature.from_dict(baseline.to_dict()))
    assert not baseline.matches(replace(baseline, **changes))


def assert_gpu_state_controls(**changes):
    """Exercise diagnostic classification, not live telemetry or GPU timing."""
    from core.harness.validity_checks import GPUState, check_gpu_state_consistency

    before = GPUState(0, "diagnostic fixture", temperature_c=40, clock_mhz=1000)
    assert check_gpu_state_consistency(before, replace(before)) == (True, [])
    passed, reasons = check_gpu_state_consistency(before, replace(before, **changes))
    assert not passed
    assert reasons


def assert_environment_controls(tmp_path, monkeypatch, violation):
    """Read real fixture files through EnvironmentProbe's documented test root.

    Only the platform selector is scoped to Linux; no CUDA availability or
    hardware response is changed. Results classify supplied host diagnostics.
    """
    from core.harness import validity_checks as module

    monkeypatch.setattr(module, "sys", SimpleNamespace(platform="linux"))

    def write(path, text):
        target = tmp_path / path.lstrip("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    write("/proc/swaps", "Filename Type Size Used Priority\n")
    write("/proc/cpuinfo", "processor: 0\n")
    write("/sys/devices/virtual/dmi/id/product_name", "BareMetal\n")
    write("/sys/devices/system/node/node0/cpulist", "0-1\n")
    write("/sys/devices/system/node/node1/cpulist", "2-3\n")
    write("/sys/devices/system/cpu/cpufreq/policy0/scaling_governor", "performance\n")
    write("/proc/self/cgroup", "0::/tests\n")
    write("/sys/fs/cgroup/tests/cpu.max", "max 100000\n")
    write("/sys/fs/cgroup/tests/memory.max", "max\n")
    write("/sys/fs/cgroup/tests/cpuset.cpus.effective", "0-1\n")
    write("/sys/fs/cgroup/tests/cpuset.mems.effective", "0\n")
    probe = module.EnvironmentProbe(root=tmp_path, env={}, cpu_affinity={0, 1})
    clean = module.validate_environment(device=torch.device("cpu"), probe=probe)
    assert clean.is_valid and not clean.errors, clean
    assert clean.details["execution_environment"] == "bare_metal"
    if violation == "governor":
        write("/sys/devices/system/cpu/cpufreq/policy0/scaling_governor", "powersave\n")
        expected = "CPU governor mismatch"
    elif violation == "swap":
        write("/proc/swaps", "Filename Type Size Used Priority\n/swap file 100 1 -2\n")
        expected = "Swap is enabled"
    elif violation in {"cpu_quota", "memory_limit"}:
        filename = "cpu.max" if violation == "cpu_quota" else "memory.max"
        write(f"/sys/fs/cgroup/tests/{filename}", "100000 100000\n" if violation == "cpu_quota" else "1024\n")
        expected = filename
    elif violation == "numa":
        probe.cpu_affinity = {0, 2}
        result = module.validate_environment(device=torch.device("cpu"), probe=probe)
        assert result.is_valid  # NUMA spread is advisory in the actual policy.
        assert result.details["numa_nodes_in_affinity"] == [0, 1]
        assert any("CPU affinity spans multiple NUMA nodes" in x for x in result.warnings)
        return
    elif violation == "virtualization":
        write("/proc/cpuinfo", "processor: 0\nflags: hypervisor\n")
        write("/sys/devices/virtual/dmi/id/product_name", "KVM\n")
        result = module.validate_environment(device=torch.device("cpu"), probe=probe)
        assert result.is_valid  # A notice, not rejection, is the strict policy.
        assert result.details["virtualized"] is True
        assert any("non-canonical" in x for x in result.notices)
        assert not clean.notices
        return
    else:
        raise AssertionError(f"Unknown environment violation: {violation}")
    result = module.validate_environment(device=torch.device("cpu"), probe=probe)
    assert not result.is_valid
    assert any(expected in error for error in result.errors), result


def assert_memory_pattern_controls(violation):
    """Classifier inputs are explicit diagnostic snapshots, not allocations."""
    from core.harness.validity_checks import MemoryAllocationSnapshot, MemoryAllocationTracker

    tracker = MemoryAllocationTracker()
    tracker.start_snapshot = MemoryAllocationSnapshot(10, 10, 10, 1, 0)
    tracker.end_snapshot = MemoryAllocationSnapshot(11, 11, 11, 2, 0)
    assert tracker.check_patterns() == (True, [])
    if violation == "peak":
        tracker.end_snapshot = replace(tracker.end_snapshot, max_allocated_mb=30)
        reason = "MEMORY PEAK SPIKE"
    elif violation == "preallocation":
        tracker.start_snapshot = replace(tracker.start_snapshot, allocated_mb=200)
        tracker.end_snapshot = replace(tracker.end_snapshot, allocated_mb=201, max_allocated_mb=201)
        reason = "POTENTIAL PRE-ALLOCATION"
    else:
        tracker.end_snapshot = replace(tracker.end_snapshot, allocated_mb=111, max_allocated_mb=111)
        reason = "MEMORY INCREASE"
    passed, reasons = tracker.check_patterns()
    assert not passed
    assert any(reason in item for item in reasons)


def assert_compile_cache_reset(*, dynamic=False):
    """Real CPU Dynamo tracing with an observing backend that executes the graph."""
    from core.harness.validity_checks import clear_compile_cache

    compiled_graphs = []

    def backend(graph, example_inputs):
        compiled_graphs.append(graph)
        return graph.forward

    def work(value):
        return value * 2 + 1

    assert clear_compile_cache()
    try:
        compiled = torch.compile(work, backend=backend, dynamic=dynamic)
        value = torch.arange(8, dtype=torch.float32)
        torch.testing.assert_close(compiled(value), work(value))
        first_count = len(compiled_graphs)
        assert first_count > 0
        torch.testing.assert_close(compiled(value + 1), work(value + 1))
        assert len(compiled_graphs) == first_count  # Same-shape cache hit.
        assert clear_compile_cache()
        torch.testing.assert_close(compiled(value), work(value))
        assert len(compiled_graphs) > first_count  # The real cache was invalidated.
    finally:
        assert clear_compile_cache()


def assert_compile_guard_counts():
    from core.harness.validity_checks import clear_compile_cache, get_compile_state

    assert clear_compile_cache()
    try:
        before = get_compile_state()["compile_count"]

        def work(value, multiply):
            return value * 2 if multiply else value + 1

        compiled = torch.compile(work, backend="eager", dynamic=False)
        value = torch.arange(8, dtype=torch.float32)
        torch.testing.assert_close(compiled(value, True), work(value, True))
        observed = get_compile_state()
        first = observed["compile_count"]
        assert first > before
        assert observed["compile_count_available"] is True
        assert observed["compile_count_source"] in {"stats.unique_graphs", "compile.calls"}
        assert observed["cache_entries_available"] is False
        torch.testing.assert_close(compiled(value + 1, True), work(value + 1, True))
        assert get_compile_state()["compile_count"] == first
        torch.testing.assert_close(compiled(value, False), work(value, False))
        assert get_compile_state()["compile_count"] > first
    finally:
        assert clear_compile_cache()


def assert_materialization_diagnostic(monkeypatch):
    from core.harness import validity_checks as module

    # Isolate only the once-per-process diagnostic registry; the real tensor
    # materialization operation and its failure are not replaced.
    monkeypatch.setattr(module, "_EMITTED_VALIDITY_LIMITATIONS", set())
    monkeypatch.setattr(module, "_VALIDITY_LIMITATION_RECORDS", {})
    value = torch.arange(8, dtype=torch.float32)
    module.force_tensor_evaluation({"output": value * 2})
    assert not module._VALIDITY_LIMITATION_RECORDS
    with pytest.warns(RuntimeWarning, match="could not materialize"):
        module.force_tensor_evaluation({"output": torch.empty(8, device="meta")})
    assert "force_tensor_evaluation_item_failure" in module._VALIDITY_LIMITATION_RECORDS


def assert_config_immutability(field, value):
    harness, config = cpu_harness()
    work = TensorWork()
    work.setup()
    samples, _ = harness._benchmark_custom(work.benchmark_fn, config)
    assert len(samples) == config.iterations

    def violation():
        work.benchmark_fn()
        setattr(config, field, value)

    with pytest.raises(RuntimeError, match="CONFIG MANIPULATION"):
        harness._benchmark_custom(violation, config)


def assert_stream_audit_controls(*, count=2, nested=False, priorities=False, pinned=False, threaded=False):
    """Actual CUDA operations: synchronized control and missing-sync violation."""
    from core.harness.validity_checks import StreamAuditor
    import threading

    def run(synchronize):
        streams = [torch.cuda.Stream(priority=-1 if priorities and i == 0 else 0) for i in range(count)]
        inputs = [torch.ones(1024, device="cuda") for _ in streams]
        outputs = [torch.empty(1024, pin_memory=True) for _ in streams] if pinned else inputs
        torch.cuda.synchronize()
        auditor = StreamAuditor()
        errors = []

        def work(index):
            try:
                with torch.cuda.stream(streams[index]):
                    if pinned:
                        outputs[index].copy_(inputs[index], non_blocking=True)
                    else:
                        for _ in range(3):
                            outputs[index].add_(1)
            except BaseException as exc:
                errors.append(exc)

        auditor.start()
        try:
            if threaded:
                threads = [threading.Thread(target=work, args=(i,)) for i in range(count)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
            elif nested:
                with torch.cuda.stream(streams[0]):
                    for i in range(count):
                        work(i)
            else:
                for i in range(count):
                    work(i)
            if synchronize:
                for stream in streams:
                    stream.synchronize()
        finally:
            auditor.stop()
            for stream in streams:
                stream.synchronize()
        assert not errors, errors
        for output in outputs:
            torch.testing.assert_close(output, torch.full_like(output, 1 if pinned else 4))
        return auditor.check_issues()

    assert run(True) == (True, [])
    passed, reasons = run(False)
    assert not passed
    assert any("STREAM SYNC WARNING" in reason for reason in reasons)


def assert_cuda_timing_cross_validation():
    """Real GPU delay on the measured versus an unmeasured stream.

    Full device synchronization remains enabled in both cases. It completes
    both workloads, but events on the default stream miss side-stream work.
    Neither timer nor measurement is replaced; the production threshold stays
    unchanged. This requires actual CUDA and is not exercised on CPU hosts.
    """
    import warnings

    side_stream = torch.cuda.Stream()
    value = torch.ones(8, device="cuda")
    config = BenchmarkConfig(
        device=torch.device("cuda"), iterations=3, warmup=5,
        use_subprocess=False, enable_profiling=False, full_device_sync=True,
        force_synchronize=False, adaptive_iterations=False,
        # Exercise the clock cross-check independently of the stream-registration
        # guard, which has its own real-stream coverage.
        audit_stream_sync=False,
    )
    harness = BenchmarkHarness(config=config)

    def work():
        torch.cuda._sleep(50_000_000)
        value.add_(1)

    def violation():
        with torch.cuda.stream(side_stream):
            work()

    try:
        work()
        torch.cuda.synchronize()
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            samples, _ = harness._benchmark_custom(work, config)
        assert len(samples) == config.iterations
        assert not any("TIMING CROSS-VALIDATION FAILURE" in str(x.message) for x in observed)
        with pytest.warns(RuntimeWarning, match="TIMING CROSS-VALIDATION FAILURE"):
            harness._benchmark_custom(violation, config)
        torch.testing.assert_close(value, torch.full_like(value, 2 + 2 * config.iterations))
    finally:
        torch.cuda.synchronize()
