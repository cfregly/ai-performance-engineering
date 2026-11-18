"""Harness-level negative tests for anti-cheat protections."""

import json
import time
from typing import Optional

import pytest
import torch

from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkConfig, BaseBenchmark

# Skip if CUDA is unavailable (harness requires CUDA devices)
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required - NVIDIA GPU and tools must be available",
)


def _make_config(**kwargs) -> BenchmarkConfig:
    kwargs.setdefault("allow_foreign_gpu_processes", True)
    kwargs.setdefault("enforce_environment_validation", False)
    return BenchmarkConfig(**kwargs)


class _CheatingSetupBenchmark(BaseBenchmark):
    """Benchmark that pre-computes output in setup (should be caught)."""

    def __init__(self):
        super().__init__()
        self.output = None

    def setup(self) -> None:
        self.output = torch.randn(8, device=self.device)

    def benchmark_fn(self) -> None:
        # No-op timed body
        return None

    def get_verify_output(self) -> torch.Tensor:
        if self.output is None:
            raise RuntimeError("output missing")
        return self.output
    
    def get_verify_inputs(self):
        if self.output is None:
            raise RuntimeError("output missing")
        return {"x": self.output}

    def get_input_signature(self):
        return {"shapes": {"x": (1, 8)}, "dtypes": {"x": "float32"}, "batch_size": 1, "parameter_count": 0, "precision_flags": {}}

    def get_output_tolerance(self):
        return (1e-3, 1e-3)


class _GraphCheatBenchmark(BaseBenchmark):
    """Benchmark that triggers graph capture cheat detection."""

    def __init__(self):
        super().__init__()
        self.x = None

    def setup(self) -> None:
        self.x = torch.ones(4, device=self.device)

    def benchmark_fn(self) -> torch.Tensor:
        time.sleep(0.2)
        self.x.add_(1)
        return self.x

    def get_verify_output(self) -> torch.Tensor:
        return self.x
    
    def get_verify_inputs(self):
        if self.x is None:
            raise RuntimeError("input missing")
        return {"x": self.x}

    def get_input_signature(self):
        return {"shapes": {"x": (4,)}, "dtypes": {"x": "float32"}, "batch_size": 1, "parameter_count": 0, "precision_flags": {}}

    def get_output_tolerance(self):
        return (1e-3, 1e-3)


class _StreamCheatBenchmark(BaseBenchmark):
    """Benchmark used to test stream audit failures."""

    def __init__(self):
        super().__init__()
        self.x = None
        self._stream = None
        self._call_count = 0

    def setup(self) -> None:
        self.x = torch.ones(4, device=self.device)
        self._stream = torch.cuda.Stream(device=self.device)

    def benchmark_fn(self) -> torch.Tensor:
        self._call_count += 1
        # BenchmarkHarness enforces a minimum warmup (currently 5). Switch streams
        # after warmup so the stream appears during the measured section.
        if self._call_count == 6:
            torch.cuda.set_stream(self._stream)
        self.x.add_(1)
        return self.x

    def get_verify_output(self) -> torch.Tensor:
        return self.x
    
    def get_verify_inputs(self):
        if self.x is None:
            raise RuntimeError("input missing")
        return {"x": self.x}

    def get_input_signature(self):
        return {"shapes": {"x": (4,)}, "dtypes": {"x": "float32"}, "batch_size": 1, "parameter_count": 0, "precision_flags": {}}

    def get_output_tolerance(self):
        return (1e-3, 1e-3)


def _run_harness(benchmark: BaseBenchmark, config: BenchmarkConfig):
    """Helper to run harness in thread mode and return result."""
    harness = BenchmarkHarness()
    # Disable setup precomputation detection for tests that aren't validating it
    if not isinstance(benchmark, _CheatingSetupBenchmark):
        config.detect_setup_precomputation = False
    result = harness._benchmark_with_threading(benchmark, config)
    return result


def test_setup_precompute_failures():
    """Harness should block benchmarks that compute outputs during setup."""
    bench = _CheatingSetupBenchmark()
    config = _make_config(iterations=1, warmup=0)
    result = _run_harness(bench, config)
    assert result.timeout_stage is None
    assert result.timeout_duration_seconds is None
    assert result.timeout_limit_seconds is None
    assert json.loads(result.validation_message)["setup"]["status"] == "error"
    assert any("pre-computation" in err.lower() for err in result.errors)


def test_graph_cheat_detection_forces_failure():
    """Harness should raise when graph capture is suspiciously slow vs replay."""
    bench = _GraphCheatBenchmark()
    config = _make_config(iterations=1, warmup=0, enable_cuda_graph=True)

    result = _run_harness(bench, config)
    assert any("graph capture cheat" in err.lower() for err in result.errors)


def test_stream_audit_failure():
    """Harness should fail when streams change mid-benchmark."""
    bench = _StreamCheatBenchmark()
    config = _make_config(iterations=1, warmup=0)
    try:
        result = _run_harness(bench, config)
        assert any("stream timing violation" in err.lower() for err in result.errors)
    finally:
        torch.cuda.set_stream(torch.cuda.default_stream())
