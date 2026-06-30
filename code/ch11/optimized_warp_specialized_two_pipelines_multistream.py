"""Optimized dual-pipeline warp specialization benchmark (Chapter 11)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig  # noqa: E402
from core.profiling.nvtx_helper import canonicalize_nvtx_name
from core.utils.extension_loader_template import load_cuda_extension_v2
from core.harness.hardware_capabilities import ensure_dsmem_supported  # noqa: E402

_EXT_NAME = "optimized_warp_specialized_two_pipelines_ext"


@lru_cache(maxsize=1)
def _load_optimized_extension():
    return load_cuda_extension_v2(
        name=_EXT_NAME,
        sources=[Path(__file__).with_name("warp_specialized_two_pipelines_optimized_extension.cu")],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--allow-unsupported-compiler"],
    )


class OptimizedDualPipelineBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Improved dual-pipeline warp specialization using CUDA::pipeline and additional compute warps."""

    def __init__(self) -> None:
        super().__init__()
        # Use a small, explicit stream count to reduce per-call stream overhead
        # while still demonstrating overlap.
        self.num_streams = 4
        # Skip on hardware without DSMEM/cluster support to avoid launch failures.
        ensure_dsmem_supported(description="warp-specialized cluster pipelines")
        self.ext = None
        self.input_a: torch.Tensor | None = None
        self.input_b: torch.Tensor | None = None
        self.output: torch.Tensor | None = None
        self._verify_output_buffer: torch.Tensor | None = None

        # Match constants from baseline for fair comparison
        self.tile_elems = 1024
        # Increase tile count so stream-management overhead is amortized.
        self.tiles = 8192
        self.baseline_total_elements = self.tiles * self.tile_elems
        # Warp specialization benchmark - fixed dimensions for pipeline analysis
        total_elems = self.tiles * self.tile_elems
        self.register_workload_metadata(
            tokens_per_iteration=float(total_elems * 2),
            requests_per_iteration=1.0,
        )

    def setup(self) -> None:
        torch.manual_seed(42)  # Match baseline seed
        self.ext = _load_optimized_extension()
        total_elems = self.tiles * self.tile_elems
        self.input_a = torch.randn(total_elems, device=self.device, dtype=torch.float32)
        self.input_b = torch.randn(total_elems, device=self.device, dtype=torch.float32)
        self._verify_output_buffer = torch.empty(total_elems, device=self.device, dtype=torch.float32)
        self.output = None
        self._synchronize()
        tokens = float(total_elems * 2)
        self.register_workload_metadata(
            tokens_per_iteration=tokens,
            requests_per_iteration=1.0,
        )

    def benchmark_fn(self) -> None:
        assert self.input_a is not None and self.input_b is not None and self.ext is not None
        with torch.inference_mode(), self._nvtx_range("optimized_dual_pipeline_multistream"):
            result = self.ext.warp_specialized_multistream_forward(
                self.input_a,
                self.input_b,
                self.num_streams,
            )
        self.output = result
        if self.input_a is None or self.input_b is None or self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input_a": self.input_a, "input_b": self.input_b},
            output=self._verify_output_buffer,
            batch_size=int(self.tiles),
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(1e-4, 1e-4),
        )

    def teardown(self) -> None:
        self.ext = None
        self.input_a = None
        self.input_b = None
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            # The one-time cold JIT compile of this warp-specialized dual-pipeline
            # multistream kernel takes ~121s on GB300 (the kernel itself then runs
            # in well under a millisecond). 120s was just too tight for the cold
            # build, so the first inventory run timed out in setup. 300s covers the
            # cold compile with headroom; warm-cache runs are instant.
            measurement_timeout_seconds=300,
            setup_timeout_seconds=300,
            ncu_replay_mode="application",
            ncu_metric_set="minimal",
            nsys_nvtx_include=[canonicalize_nvtx_name("optimized_dual_pipeline_multistream")],
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_stream_metrics
        return compute_stream_metrics(
            sequential_time_ms=None,
            overlapped_time_ms=getattr(self, '_last_elapsed_ms', None),
            num_streams=getattr(self, 'num_streams', 4),
            num_operations=getattr(self, 'num_operations', 4),
        )

    def validate_result(self) -> str | None:
        if self.output is None:
            return "Output tensor not initialized"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> OptimizedDualPipelineBenchmark:
    return OptimizedDualPipelineBenchmark()
