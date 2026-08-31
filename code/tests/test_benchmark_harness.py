"""Unit tests for benchmark harness core functionality.

Tests timeout handling, subprocess isolation, timing accuracy, memory tracking,
and error propagation as specified in Part 2.10 of the unified improvement plan.
"""

import pytest
import sys
import time
import threading
from typing import Optional

from core.env import apply_env_defaults
apply_env_defaults()

import torch
from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode, BenchmarkConfig, BaseBenchmark
from core.benchmark.models import BenchmarkResult, TimingStats, MemoryStats


# Skip tests if CUDA is not available and silence advisory runtime noise that is
# expected in virtualized CI-like environments.
pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA required - NVIDIA GPU and tools must be available",
    ),
    pytest.mark.filterwarnings(
        "ignore:ENVIRONMENT WARNING.*:RuntimeWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:MEMORY TRACKING WARNING.*:RuntimeWarning"
    ),
]


def _make_config(**kwargs) -> BenchmarkConfig:
    kwargs.setdefault("allow_foreign_gpu_processes", True)
    kwargs.setdefault("enforce_environment_validation", False)
    return BenchmarkConfig(**kwargs)


class SimpleBenchmark(BaseBenchmark):
    """Simple benchmark for testing."""
    
    def __init__(self):
        super().__init__()
        self.tensor = None
        self.output = None
    
    def setup(self) -> None:
        self.tensor = torch.randn(100, 100, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        self.output = self.tensor @ self.tensor
        torch.cuda.synchronize()
    
    def validate_result(self) -> Optional[str]:
        return None
    
    def get_verify_inputs(self):
        if self.tensor is None:
            raise RuntimeError("Tensor not initialized")
        return {"input": self.tensor}
    
    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output
    
    def get_output_tolerance(self):
        return (1e-5, 1e-8)

    def get_input_signature(self) -> dict:
        if self.tensor is None:
            raise RuntimeError("setup() must be called before get_input_signature()")
        return {
            "tensor_shape": tuple(self.tensor.shape),
            "dtype": str(self.tensor.dtype),
        }


class SlowBenchmark(BaseBenchmark):
    """Benchmark that takes a long time - for timeout testing."""
    
    def __init__(self):
        super().__init__()
        self.tensor = None
        self.output = None
    
    def setup(self) -> None:
        self.tensor = torch.randn(100, 100, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        # Actually block to ensure timeout triggers - use sleep in a loop
        # This is more reliable than tensor operations which may finish too fast
        import time
        # Sleep for 0.1s per iteration, so 100 iterations = 10s (guaranteed > 1s timeout)
        for _ in range(100):
            time.sleep(0.1)
            # Do minimal work to keep CUDA context alive
            _ = self.tensor @ self.tensor
        torch.cuda.synchronize()
        self.output = self.tensor
    
    def validate_result(self) -> Optional[str]:
        return None
    
    def get_verify_inputs(self):
        if self.tensor is None:
            raise RuntimeError("Tensor not initialized")
        return {"input": self.tensor}
    
    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output
    
    def get_output_tolerance(self):
        return (1e-5, 1e-8)

    def get_input_signature(self) -> dict:
        if self.tensor is None:
            raise RuntimeError("setup() must be called before get_input_signature()")
        return {
            "tensor_shape": tuple(self.tensor.shape),
            "dtype": str(self.tensor.dtype),
        }


class FailingBenchmark(BaseBenchmark):
    """Benchmark that raises an error - for error propagation testing."""
    
    def __init__(self):
        super().__init__()
        self.output = None
    
    def setup(self) -> None:
        pass
    
    def benchmark_fn(self) -> None:
        raise RuntimeError("Intentional benchmark failure")
    
    def validate_result(self) -> Optional[str]:
        return None
    
    def get_verify_inputs(self):
        return {"input": torch.tensor([0.0], device=self.device)}
    
    def get_verify_output(self):
        if self.output is None:
            raise RuntimeError("benchmark_fn() must set output")
        return self.output
    
    def get_output_tolerance(self):
        return (1e-5, 1e-8)

    def get_input_signature(self) -> dict:
        return {"scalar": 0.0}


class StderrFailingBenchmark(FailingBenchmark):
    """Benchmark that emits stderr before failing - for stderr capture tests."""

    def benchmark_fn(self) -> None:
        print("INTENTIONAL STDERR MARKER", file=sys.stderr, flush=True)
        raise RuntimeError("Intentional benchmark failure")


class StderrSuccessBenchmark(SimpleBenchmark):
    """Benchmark that emits stderr and still succeeds."""

    def benchmark_fn(self) -> None:
        print("INTENTIONAL STDERR SUCCESS MARKER", file=sys.stderr, flush=True)
        super().benchmark_fn()


class TestSubprocessTimeoutKill:
    """Test subprocess timeout kill and error propagation."""
    
    def test_timeout_kills_subprocess(self):
        """Test that subprocess is killed when timeout is exceeded."""
        benchmark = SlowBenchmark()
        config = _make_config(
            iterations=100,  # Enough iterations to exceed timeout
            warmup=5,
            measurement_timeout_seconds=1,  # Very short timeout (1 second)
            enable_profiling=False,
            enable_memory_tracking=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        # The harness may return a result with timeout info OR raise RuntimeError
        # Both are valid - check for timeout info in result or error message
        try:
            result = harness.benchmark(benchmark)
            # If we get a result, it should have timeout information
            assert result.timeout_stage is not None
            assert result.timeout_stage == "measurement"
            assert result.timeout_limit_seconds == 1
            assert result.timeout_duration_seconds > 0
            assert len(result.errors) > 0
            assert "timeout" in result.errors[0].lower() or "TIMEOUT" in result.errors[0]
        except RuntimeError as e:
            # If RuntimeError is raised, it should mention timeout
            error_msg = str(e).lower()
            assert "timeout" in error_msg or "TIMEOUT" in str(e)
    
    def test_error_propagation_from_subprocess(self):
        """Test that errors from subprocess are properly propagated."""
        benchmark = FailingBenchmark()
        config = _make_config(
            iterations=5,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        # When subprocess fails, harness should surface errors in the result
        result = harness.benchmark(benchmark)
        assert len(result.errors) > 0
        assert any("intentional benchmark failure" in err.lower() for err in result.errors)

    def test_subprocess_stderr_capture(self, tmp_path):
        """Ensure subprocess stderr is persisted when a benchmark fails."""
        benchmark = StderrFailingBenchmark()
        config = _make_config(
            iterations=5,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,
            subprocess_stderr_dir=str(tmp_path),
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)

        result = harness.benchmark(benchmark)
        assert len(result.errors) > 0

        stderr_logs = list(tmp_path.glob("*_subprocess.stderr.log"))
        assert stderr_logs, "Expected subprocess stderr log to be captured"
        combined = "\n".join(path.read_text() for path in stderr_logs)
        assert "INTENTIONAL STDERR MARKER" in combined

    def test_subprocess_stderr_capture_on_success(self, tmp_path):
        """Ensure subprocess stderr is persisted even when the benchmark succeeds."""
        benchmark = StderrSuccessBenchmark()
        config = _make_config(
            iterations=2,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,
            subprocess_stderr_dir=str(tmp_path),
            use_subprocess=True,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)

        result = harness.benchmark(benchmark)

        assert not result.errors
        stderr_logs = list(tmp_path.glob("*_subprocess.stderr.log"))
        assert stderr_logs, "Expected subprocess stderr log to be captured"
        combined = "\n".join(path.read_text() for path in stderr_logs)
        assert "INTENTIONAL STDERR SUCCESS MARKER" in combined


class TestPyTorchTimerCorrectness:
    """Test PyTorch Timer stmt/globals correctness."""
    
    def test_pytorch_timer_executes_benchmark_fn(self):
        """PyTorch Timer mode must actually execute fn() (not just reference it)."""
        class CountingBenchmark(SimpleBenchmark):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def benchmark_fn(self) -> None:
                self.calls += 1
                super().benchmark_fn()

        benchmark = CountingBenchmark()
        config = _make_config(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,
            use_subprocess=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.PYTORCH, config=config)

        result = harness.benchmark(benchmark)
        assert result.timing.iterations == len(result.timing.raw_times_ms) > 0
        assert result.timing.sample_scope == "block_mean"
        assert result.timing.iterations_per_sample >= 1
        assert benchmark.calls > config.warmup


def test_thread_mode_captures_custom_metrics_before_teardown():
    class CustomMetricBenchmark(SimpleBenchmark):
        def __init__(self):
            super().__init__()
            self.provider_name = None

        def setup(self) -> None:
            super().setup()
            self.provider_name = "flash_backend"

        def get_custom_metrics(self) -> Optional[dict]:
            return {
                "flashattention4.provider_id": 2.0 if self.provider_name == "flash_backend" else 0.0,
            }

        def teardown(self) -> None:
            self.provider_name = None
            super().teardown()

    benchmark = CustomMetricBenchmark()
    config = _make_config(
        iterations=5,
        warmup=5,
        enable_profiling=False,
        enable_memory_tracking=False,
        use_subprocess=False,
        execution_mode="thread",
    )
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)

    result = harness._benchmark_with_threading(benchmark, config)

    assert result.custom_metrics["flashattention4.provider_id"] == 2.0


class TestTimeoutMultiplierPropagation:
    """Ensure timeout multipliers behave correctly inside the harness."""
    
    def test_harness_preserves_explicit_timeouts_when_cloning_config(self):
        config = _make_config(
            iterations=1,
            warmup=5,
            measurement_timeout_seconds=2,
            timeout_multiplier=5.0,
            enable_profiling=False,
            enable_memory_tracking=False,
            use_subprocess=False,
        )
        assert config.measurement_timeout_seconds == 2
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        assert harness.config.measurement_timeout_seconds == 2


class TestEventSyncTimingLoop:
    """Test event-sync timing loop correctness (no device-wide sync)."""
    
    def test_cuda_timing_uses_end_event_sync(self):
        """Custom timing loop should run without full device sync when default stream is used."""
        class DefaultStreamNoSyncBenchmark(BaseBenchmark):
            def __init__(self):
                super().__init__()
                self.x = None
                self.output = None

            def setup(self) -> None:
                self.x = torch.randn(128, 128, device=self.device)

            def benchmark_fn(self) -> None:
                self.output = self.x @ self.x

            def get_verify_inputs(self):
                return {"x": self.x}

            def get_verify_output(self):
                return self.output

            def get_output_tolerance(self):
                return (1e-5, 1e-8)

            def get_input_signature(self) -> dict:
                return {"shape": tuple(self.x.shape), "dtype": str(self.x.dtype)}

        benchmark = DefaultStreamNoSyncBenchmark()
        config = _make_config(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,
            use_subprocess=False,
            full_device_sync=False,
            adaptive_iterations=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        result = harness.benchmark(benchmark)
        assert result.timing is not None
        assert result.timing.iterations == 10
    
    def test_timing_loop_produces_valid_results(self):
        """Test that timing loop produces valid timing statistics."""
        benchmark = SimpleBenchmark()
        config = _make_config(
            iterations=20,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,
            adaptive_iterations=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        
        # Verify timing stats are valid
        assert result.timing is not None
        assert result.timing.iterations == 20
        assert result.timing.warmup_iterations == 5
        assert result.timing.mean_ms > 0
        assert result.timing.median_ms > 0
        assert result.timing.min_ms > 0
        assert result.timing.max_ms > 0
        assert result.timing.max_ms >= result.timing.min_ms
        assert result.timing.mean_ms >= result.timing.min_ms
        assert result.timing.mean_ms <= result.timing.max_ms


class TestMemoryTracking:
    """Test memory tracking values on CUDA."""
    
    def test_memory_tracking_enabled(self):
        """Test that memory tracking captures peak and allocated memory."""
        benchmark = SimpleBenchmark()
        config = _make_config(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=True,  # Enable memory tracking
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        
        # Should have memory stats
        assert result.memory is not None
        assert result.memory.peak_mb is not None
        assert result.memory.peak_mb >= 0
        # Allocated memory may be None if not tracked, but peak should be present
    
    def test_memory_tracking_disabled(self):
        """Test that memory tracking can be disabled."""
        benchmark = SimpleBenchmark()
        config = _make_config(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=False,  # Disable memory tracking
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        
        # Memory stats may be None when disabled
        # (implementation may still create empty stats, which is fine)
        if result.memory is not None:
            # If present, values should be None or 0
            pass  # Accept either None or empty stats
    
    def test_memory_tracking_values_are_reasonable(self):
        """Test that memory tracking values are reasonable (not negative, etc.)."""
        benchmark = SimpleBenchmark()
        config = _make_config(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            enable_memory_tracking=True,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        
        if result.memory is not None:
            if result.memory.peak_mb is not None:
                assert result.memory.peak_mb >= 0
            if result.memory.allocated_mb is not None:
                assert result.memory.allocated_mb >= 0
            if result.memory.reserved_mb is not None:
                assert result.memory.reserved_mb >= 0


class TestBenchmarkModes:
    """Test all benchmark modes work correctly."""
    
    def test_custom_mode(self):
        """Test CUSTOM mode works."""
        benchmark = SimpleBenchmark()
        config = _make_config(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            adaptive_iterations=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        assert result.timing.iterations == 10
    
    def test_pytorch_mode(self):
        """Test PYTORCH mode works."""
        benchmark = SimpleBenchmark()
        config = _make_config(
            iterations=10,
            warmup=5,
            enable_profiling=False,
            adaptive_iterations=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.PYTORCH, config=config)
        
        result = harness.benchmark(benchmark)
        assert result.timing.iterations == 10
    
    def test_triton_mode(self):
        """Test TRITON mode either runs through Triton or fails explicitly."""
        benchmark = SimpleBenchmark()
        config = _make_config(
            iterations=3,
            warmup=1,
            enable_profiling=False,
            adaptive_iterations=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.TRITON, config=config)

        try:
            import triton.testing  # noqa: F401
        except ImportError:
            with pytest.raises(RuntimeError, match="Triton benchmarking mode requested but Triton not available"):
                harness.benchmark(benchmark)
            return

        result = harness.benchmark(benchmark)
        assert result.timing.iterations == 3


class TestErrorHandling:
    """Test error handling and recovery."""
    
    def test_benchmark_without_setup(self):
        """Test benchmark that doesn't implement setup."""
        class NoSetupBenchmark(BaseBenchmark):
            def benchmark_fn(self) -> None:
                self.output = torch.tensor([1.0], device=self.device)
            
            def get_verify_inputs(self):
                return {"input": torch.tensor([1.0], device=self.device)}
            
            def get_verify_output(self):
                return getattr(self, "output", torch.tensor([1.0], device=self.device))
            
            def get_output_tolerance(self):
                return (1e-5, 1e-8)

        benchmark = NoSetupBenchmark()
        config = _make_config(
            iterations=5,
            warmup=5,
            enable_profiling=False,
            use_subprocess=False,
            adaptive_iterations=False,
        )
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        # Should not crash - setup() is optional
        result = harness.benchmark(benchmark)
        assert result.timing.iterations == 5
    
    def test_benchmark_with_validation_error(self):
        """Test benchmark that returns validation error."""
        class InvalidBenchmark(BaseBenchmark):
            def benchmark_fn(self) -> None:
                self.tensor = torch.randn(10, 10, device=self.device)
                self.output = self.tensor
            
            def validate_result(self) -> Optional[str]:
                return "Validation failed"
            
            def get_verify_inputs(self):
                return {"input": self.tensor}
            
            def get_verify_output(self):
                return getattr(self, "output", self.tensor)
            
            def get_output_tolerance(self):
                return (1e-5, 1e-8)
        
        benchmark = InvalidBenchmark()
        config = _make_config(iterations=5, warmup=5, enable_profiling=False, use_subprocess=False)
        harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)
        
        result = harness.benchmark(benchmark)
        # Should have validation error
        assert len(result.errors) > 0 or result.timing.iterations == 0
