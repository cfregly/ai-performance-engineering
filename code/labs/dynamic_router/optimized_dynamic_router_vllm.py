"""Optimized vLLM-backed routing benchmark (feedback-driven placement)."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router import vllm_runner
from labs.dynamic_router.topology import detect_topology
from labs.dynamic_router.verification import metric_row_buffer, numeric_metric_values, scalar_int_buffer
from labs.dynamic_router.vllm_runner import run_vllm_routing_with_topology


class OptimizedDynamicRouterVllmBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs vLLM with the feedback-based router (prefill/decode scoring)."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}
        self.output: Optional[torch.Tensor] = None
        self._metric_values: Optional[list[float]] = None
        self._metric_output_buffer: Optional[torch.Tensor] = None
        self._mode_input: Optional[torch.Tensor] = None
        self._topology = None
        self._summary_ready = False
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self._topology = detect_topology(max_gpus=torch.cuda.device_count())

    def benchmark_fn(self) -> None:
        self._summary = run_vllm_routing_with_topology(
            "optimized",
            topology_snapshot=self._topology,
            cli_args=vllm_runner._CLI_ARGS,
        )
        self._summary_ready = True

    def capture_verification_payload(self) -> None:
        if not self._summary_ready:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        metric_values = numeric_metric_values(self._summary, self._metric_values)
        self._metric_values = metric_values
        self.output = metric_row_buffer(self, metric_values)
        self._set_verification_payload(
            inputs={
                "mode": scalar_int_buffer(self, "_mode_input", 2),
            },  # optimized
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "tf32": False},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.output = None
        self._metric_values = None
        self._metric_output_buffer = None
        self._mode_input = None
        self._topology = None
        self._summary_ready = False
        super().teardown()

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=5)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._summary or None


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedDynamicRouterVllmBenchmark()
