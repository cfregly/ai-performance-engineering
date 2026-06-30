"""Baseline adaptive-parallelism benchmark using per-request Python routing."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch19.adaptive_parallelism_benchmark_common import (
    AdaptiveParallelismBenchmarkConfig,
    build_workload,
    classify_baseline,
    materialize_baseline_feature_rows,
)


class BaselineAdaptiveParallelismBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self, cfg: Optional[AdaptiveParallelismBenchmarkConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or AdaptiveParallelismBenchmarkConfig()
        self.workload: Optional[Dict[str, torch.Tensor]] = None
        self.output: Optional[torch.Tensor] = None
        self._result_buffer: Optional[torch.Tensor] = None
        self._feature_rows: Optional[torch.Tensor] = None
        self._feature_rows_cpu: Optional[torch.Tensor] = None
        self._strategy_ids_cpu: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.cfg.num_requests),
            tokens_per_iteration=float(self.cfg.num_requests),
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: adaptive_parallelism requires CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.workload = build_workload(self.cfg, self.device)
        self._result_buffer = torch.empty(
            self.cfg.num_requests,
            device=self.device,
            dtype=torch.int64,
        )
        self._feature_rows = torch.empty(
            self.cfg.num_requests,
            6,
            device=self.device,
            dtype=torch.float64,
        )
        self._feature_rows_cpu = torch.empty(
            self.cfg.num_requests,
            6,
            dtype=torch.float64,
            pin_memory=True,
        )
        self._strategy_ids_cpu = torch.empty(
            self.cfg.num_requests,
            dtype=torch.int64,
            pin_memory=True,
        )
        materialize_baseline_feature_rows(
            self.workload,
            feature_rows=self._feature_rows,
            feature_rows_cpu=self._feature_rows_cpu,
        )

    def benchmark_fn(self) -> None:
        if (
            self.workload is None
            or self._result_buffer is None
            or self._feature_rows is None
            or self._feature_rows_cpu is None
            or self._strategy_ids_cpu is None
        ):
            raise RuntimeError("adaptive_parallelism workload not initialized")
        self.output = classify_baseline(
            self.workload,
            device=self.device,
            feature_rows=self._feature_rows,
            feature_rows_cpu=self._feature_rows_cpu,
            refresh_feature_rows=False,
            strategy_ids_cpu=self._strategy_ids_cpu,
            result=self._result_buffer,
        )

    def capture_verification_payload(self) -> None:
        if self.workload is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={name: tensor.detach().cpu() for name, tensor in self.workload.items()},
            output=self.output.detach().cpu(),
            batch_size=self.cfg.num_requests,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.0, 0.0),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)


def get_benchmark() -> BaseBenchmark:
    return BaselineAdaptiveParallelismBenchmark()
