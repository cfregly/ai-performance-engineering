"""Baseline AllReduce + RMSNorm (separate kernels)."""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from ch15.allreduce_rmsnorm_common import (
    AllReduceRMSNormConfig,
    build_shards,
    naive_allreduce,
    rms_norm,
)


class BaselineAllReduceRMSNormBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: sequential all-reduce then RMSNorm."""

    def __init__(self, cfg: Optional[AllReduceRMSNormConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or AllReduceRMSNormConfig()
        self.shards: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.cfg.batch_size),
            tokens_per_iteration=float(self.cfg.tokens_per_iter),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.cfg.batch_size),
            tokens_per_iteration=float(self.cfg.tokens_per_iter),
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for AR+RMSNorm fusion")
        self.shards = build_shards(self.device, self.cfg)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.shards is None:
            raise RuntimeError("Benchmark not initialized")
        with torch.inference_mode():
            reduced = naive_allreduce(self.shards)
            self.output = rms_norm(reduced, self.cfg.eps)
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.shards is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"shards": self.shards.detach()},
            output=self.output.detach(),
            batch_size=self.cfg.batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": self.cfg.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-2, 1e-1),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=10)


def get_benchmark() -> BaseBenchmark:
    return BaselineAllReduceRMSNormBenchmark()
