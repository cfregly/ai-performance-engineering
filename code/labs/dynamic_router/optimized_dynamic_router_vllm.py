"""Optimized vLLM-backed routing benchmark (feedback-driven placement)."""

from __future__ import annotations

import json
from numbers import Number
from typing import Dict, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router.topology import detect_topology
from labs.dynamic_router.vllm_runner import run_vllm_routing_with_topology


class OptimizedDynamicRouterVllmBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs vLLM with the feedback-based router (prefill/decode scoring)."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}
        self.output: Optional[torch.Tensor] = None
        self._metric_values: Optional[list[float]] = None
        self._topology = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self._topology = detect_topology(max_gpus=torch.cuda.device_count())

    def benchmark_fn(self) -> None:
        from labs.dynamic_router import vllm_runner

        self._summary = run_vllm_routing_with_topology(
            "optimized",
            topology_snapshot=self._topology,
            cli_args=vllm_runner._CLI_ARGS,
        )
        metric_values = [float(v) for v in self._summary.values() if isinstance(v, Number)]
        if not metric_values:
            metric_values = [0.0]
        self._metric_values = metric_values

    def capture_verification_payload(self) -> None:
        if self._metric_values is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self.output = torch.tensor(self._metric_values, dtype=torch.float32).unsqueeze(0)
        self._set_verification_payload(
            inputs={
                "mode": torch.tensor([2], dtype=torch.int64),
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
        self._topology = None
        super().teardown()

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=5)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._summary or None


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedDynamicRouterVllmBenchmark()

