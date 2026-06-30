#!/usr/bin/env python3
"""Baseline symmetric-memory perf microbench (single GPU).

Measures copy latency/bandwidth using per-iteration buffer allocation.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch

from ch04.symmetric_memory_perf_common import build_square_verification_probe
from core.benchmark.cuda_event_timing import elapsed_ms
from core.benchmark.metrics import compute_memory_transfer_metrics
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineSymmetricMemoryPerfBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline device copy benchmark with per-iteration allocation."""

    allowed_benchmark_fn_antipatterns = ("allocation",)
    story_metadata = {
        "pair_role": "canonical",
        "chapter_alignment": "native",
        "chapter_native_exemplar": True,
        "comparison_axis": "allocation_blocking_copy_vs_preallocated_async_copy",
        "optimization_mechanism": "per_iteration_allocation_plus_blocking_copy",
        "compound_optimization": False,
    }

    def __init__(self, size_mb: float = 0.0625):
        super().__init__()
        self.size_mb = size_mb
        self.numel = int((size_mb * 1024 * 1024) / 4)  # float32
        self.tensor: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._last_avg_ms = 0.0
        self._bytes_transferred = 0.0
        self._timing_pair: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._pending_timing_pair: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._verify_numel = 0

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: requires CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.tensor = torch.randn(self.numel, device=self.device, dtype=torch.float32)
        self._verify_input, self._verify_numel = build_square_verification_probe(self.tensor)
        self._verify_output_buffer = torch.empty_like(self._verify_input, dtype=torch.float32)
        self._timing_pair = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        torch.cuda.synchronize(self.device)

    def _get_timing_pair(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        if self._timing_pair is None:
            self._timing_pair = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        return self._timing_pair

    def benchmark_fn(self) -> Optional[Dict[str, float]]:
        if self.tensor is None:
            raise RuntimeError("Tensor not initialized")

        timing_pair = self._get_timing_pair()
        start, end = timing_pair

        current_stream = torch.cuda.current_stream(self.device)
        start.record(current_stream)
        output = torch.empty_like(self.tensor)
        output.copy_(self.tensor, non_blocking=False)
        end.record(current_stream)
        self._pending_timing_pair = timing_pair
        self.output = output
        return None

    def finalize_iteration_metrics(self) -> Optional[Dict[str, float]]:
        if self._pending_timing_pair is None or self.tensor is None:
            return None
        elapsed_ms_value = elapsed_ms(self._pending_timing_pair)
        self._pending_timing_pair = None
        self._last_avg_ms = elapsed_ms_value
        self._bytes_transferred = float(self.tensor.numel() * self.tensor.element_size())
        return None

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        if self._verify_input is None or self._verify_output_buffer is None:
            raise RuntimeError("Verification buffers not initialized")
        if self.output is None:
            probe = self._verify_input
        else:
            probe = self.output[: self._verify_numel].view_as(self._verify_input).detach()
        self._verify_output_buffer.copy_(probe)

        self._set_verification_payload(
            inputs={"tensor": probe},
            output=self._verify_output_buffer,
            batch_size=int(probe.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-5, 1e-5),
            signature_overrides={"world_size": 1},
        )

    def teardown(self) -> None:
        self.tensor = None
        self.output = None
        self._timing_pair = None
        self._pending_timing_pair = None
        self._verify_output_buffer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=10)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        self.finalize_iteration_metrics()
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred,
            elapsed_ms=self._last_avg_ms,
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        self.finalize_iteration_metrics()
        if self.output is None:
            return "No output captured"
        if self._last_avg_ms <= 0:
            return "No timing recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineSymmetricMemoryPerfBenchmark()
