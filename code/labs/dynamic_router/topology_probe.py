"""Benchmark/utility that records GPU↔NUMA topology to artifacts/topology/."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router.topology import detect_topology, write_topology
from labs.dynamic_router.verification import metric_row_buffer, numeric_metric_values, scalar_int_buffer


class TopologyProbeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Capture a snapshot of GPU↔NUMA mapping for downstream routing demos."""

    # This benchmark's explicit purpose is to probe and materialize topology files.
    allowed_benchmark_fn_antipatterns = ("io",)

    def __init__(self) -> None:
        super().__init__()
        self.snapshot = None
        self.output_path: Optional[Path] = None
        self.output: Optional[torch.Tensor] = None
        self._metric_values: Optional[list[float]] = None
        self._metric_output_buffer: Optional[torch.Tensor] = None
        self._num_gpus_input: Optional[torch.Tensor] = None

    def setup(self) -> None:
        # Nothing to initialize besides ensuring artifacts dir exists (handled by write_topology).
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

    def benchmark_fn(self) -> None:
        topo = detect_topology()
        self.output_path = write_topology(topo)
        self.snapshot = topo

    def capture_verification_payload(self) -> None:
        if self.snapshot is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        metrics_dict = self.get_custom_metrics() or {}
        metric_values = numeric_metric_values(metrics_dict, self._metric_values)
        self._metric_values = metric_values
        self.output = metric_row_buffer(self, metric_values)
        self._set_verification_payload(
            inputs={
                "num_gpus": scalar_int_buffer(
                    self,
                    "_num_gpus_input",
                    len(self.snapshot.gpu_numa) if self.snapshot else 0,
                ),
            },
            output=self.output,
            batch_size=1,
            parameter_count=0,
            precision_flags={"fp16": False, "bf16": False, "tf32": False},
            output_tolerance=(0.1, 1.0),
        )

    def get_config(self) -> Optional[BenchmarkConfig]:
        # Single-shot capture
        return BenchmarkConfig(iterations=1, warmup=5)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        if self.snapshot is None:
            return None
        gpu_numa = {f"gpu{idx}_numa": float(node) if node is not None else -1.0 for idx, node in self.snapshot.gpu_numa.items()}
        gpus_with_known_numa = sum(1 for node in self.snapshot.gpu_numa.values() if node is not None)
        gpu_numa["num_gpus_detected"] = float(len(self.snapshot.gpu_numa))
        gpu_numa["gpus_with_known_numa"] = float(gpus_with_known_numa)
        gpu_numa["host_numa_nodes_detected"] = float(len(self.snapshot.distance))
        for status in ("unknown", "partial", "complete"):
            gpu_numa[f"gpu_numa_status_{status}"] = 1.0 if self.snapshot.gpu_numa_status == status else 0.0
        return gpu_numa

    def teardown(self) -> None:
        self.output = None
        self._metric_values = None
        self._metric_output_buffer = None
        self._num_gpus_input = None
        self.snapshot = None
        self.output_path = None
        super().teardown()



def get_benchmark() -> BaseBenchmark:
    return TopologyProbeBenchmark()
