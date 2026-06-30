"""baseline_stream_ordered.py - cudaMalloc baseline for stream-ordered allocators.

This benchmark measures the cost of repeated device allocations inside a
multi-stream workload using the default (synchronizing) allocator path.
"""

from __future__ import annotations

from types import ModuleType
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.profiling.nvtx_helper import canonicalize_nvtx_name
from core.profiling.stream_ordered import load_stream_ordered_module


class BaselineStreamOrderedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: cudaMalloc / cudaFree inside a multi-stream loop."""

    def __init__(self) -> None:
        super().__init__()
        # Keep buffers small so allocator overhead dominates.
        self.elements = 1 << 12  # 4,096 floats (~16KB) per stream buffer
        # Many alloc/free cycles to amplify stream-ordered allocator benefit.
        # Increased from 200 to 500 to better demonstrate the speedup from
        # avoiding global synchronization in cudaMalloc/cudaFree.
        self.inner_iterations = 500
        self.profile_inner_iterations = 8
        self.num_streams = 8
        self.output: Optional[torch.Tensor] = None
        self._last_inner_iterations = self.inner_iterations
        self._active_inner_iterations_count = self.inner_iterations
        self._elements_tensor = torch.tensor([self.elements], dtype=torch.int64)
        self._inner_iterations_tensor = torch.empty((1,), dtype=torch.int64)
        self._inner_iterations_tensor[0] = self.inner_iterations
        self._module: Optional[ModuleType] = None
        # Application replay is not stable for this allocator-heavy profile on NCU.
        self.preferred_ncu_replay_mode = "kernel"
        self.preferred_ncu_metric_set = "minimal"

        bytes_per_buffer = float(self.elements * 4)
        # Each inner-iteration: one H2D + D2H per stream.
        self.register_workload_metadata(
            bytes_per_iteration=float(self.num_streams) * float(self.inner_iterations) * bytes_per_buffer * 2.0,
            requests_per_iteration=float(self.inner_iterations),
        )

    def _active_inner_iterations(self) -> int:
        config = getattr(self, "_config", None)
        if config is not None and bool(getattr(config, "enable_profiling", False)):
            return self.profile_inner_iterations
        return self.inner_iterations

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self._active_inner_iterations_count = self._active_inner_iterations()
        # Compile + warm the extension outside the timed region.
        self._module = load_stream_ordered_module()
        _ = self._module.run_standard_allocator_capture(1024, 1)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self._module is None:
            raise RuntimeError("Stream-ordered allocator module not initialized")
        inner_iterations = self._active_inner_iterations_count
        with torch.inference_mode(), self._nvtx_range("stream_ordered_baseline"):
            self.output = self._module.run_standard_allocator_capture(self.elements, inner_iterations)
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")
        self._last_inner_iterations = inner_iterations
        self._inner_iterations_tensor[0] = inner_iterations

    def capture_verification_payload(self) -> None:
        if self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={
                "elements": self._elements_tensor,
                "inner_iterations": self._inner_iterations_tensor,
            },
            output=self.output,
            batch_size=int(self._last_inner_iterations),
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.0, 0.0),
        )

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            ncu_replay_mode="application",
            ncu_metric_set="minimal",
            nsys_nvtx_include=[canonicalize_nvtx_name("stream_ordered_baseline")],
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None

    def teardown(self) -> None:
        self._module = None
        super().teardown()


def get_benchmark() -> BaseBenchmark:
    return BaselineStreamOrderedBenchmark()
