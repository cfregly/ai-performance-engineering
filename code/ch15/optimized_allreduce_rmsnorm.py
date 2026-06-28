"""Optimized AllReduce + RMSNorm (single eager fused reduction)."""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from ch15.allreduce_rmsnorm_common import (
    AllReduceRMSNormConfig,
    build_shards,
    fused_allreduce_rmsnorm_out,
)


class OptimizedAllReduceRMSNormBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: eager fused all-reduce + RMSNorm with lower launch overhead."""

    def __init__(self, cfg: Optional[AllReduceRMSNormConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or AllReduceRMSNormConfig()
        self.shards: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._reduced_buffer: Optional[torch.Tensor] = None
        self._squares_buffer: Optional[torch.Tensor] = None
        self._variance_buffer: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
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
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.shards = build_shards(self.device, self.cfg)
        self._reduced_buffer = torch.empty(
            self.cfg.batch_size,
            self.cfg.hidden_size,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self._squares_buffer = torch.empty_like(self._reduced_buffer)
        self._variance_buffer = torch.empty(
            self.cfg.batch_size,
            1,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self._output_buffer = torch.empty_like(self._reduced_buffer)
        # Warm the eager kernels once so the timed loop doesn't pay first-use overhead.
        with torch.inference_mode():
            fused_allreduce_rmsnorm_out(
                self.shards,
                self.cfg.eps,
                reduced=self._reduced_buffer,
                squares=self._squares_buffer,
                variance=self._variance_buffer,
                out=self._output_buffer,
            )
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if (
            self.shards is None
            or self._reduced_buffer is None
            or self._squares_buffer is None
            or self._variance_buffer is None
            or self._output_buffer is None
        ):
            raise RuntimeError("Benchmark not initialized")
        with torch.inference_mode():
            self.output = fused_allreduce_rmsnorm_out(
                self.shards,
                self.cfg.eps,
                reduced=self._reduced_buffer,
                squares=self._squares_buffer,
                variance=self._variance_buffer,
                out=self._output_buffer,
            )
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
    return OptimizedAllReduceRMSNormBenchmark()
