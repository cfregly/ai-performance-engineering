"""Baseline DEP2 analogue: sequential DP replicas + naive expert loops."""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from ch15.dep2_parallel_common import Dep2Config, Dep2Workload


class BaselineDep2ParallelBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: process DP replicas sequentially with naive MoE."""

    def __init__(self, cfg: Optional[Dep2Config] = None) -> None:
        super().__init__()
        self.cfg = cfg or Dep2Config()
        self.workload: Optional[Dep2Workload] = None
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
            raise RuntimeError("SKIPPED: CUDA required for DEP2 parallelism")
        self.workload = Dep2Workload(self.cfg, self.device)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.workload is None:
            raise RuntimeError("Benchmark not initialized")
        with torch.inference_mode():
            self.output = self.workload.forward_naive()
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.workload is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"x": self.workload.x.detach()},
            output=self.output.detach(),
            batch_size=self.cfg.batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": self.cfg.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            # torch.compile in the optimized path can change reduction order
            # enough to shift outputs by a few units even when semantics match.
            output_tolerance=(5e-2, 1e1),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)


def get_benchmark() -> BaseBenchmark:
    return BaselineDep2ParallelBenchmark()
