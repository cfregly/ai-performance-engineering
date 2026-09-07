"""Baseline MoE backend (naive expert loops)."""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from labs.moe_cuda.moe_backend_common import MoEBackendConfig, MoEBackendWorkload


class BaselineMoEBackendSelectionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: naive per-expert loops (no backend selection)."""

    def __init__(self, cfg: Optional[MoEBackendConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or MoEBackendConfig()
        self.workload: Optional[MoEBackendWorkload] = None
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
            raise RuntimeError("SKIPPED: CUDA required for MoE backend selection")
        self.workload = MoEBackendWorkload(self.cfg, self.device)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.workload is None:
            raise RuntimeError("Benchmark not initialized")
        with torch.inference_mode():
            self.output = self.workload.forward_naive(self.workload.x)
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
            output_tolerance=(5e-2, 5e-1),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=6, warmup=6)


def get_benchmark() -> BaseBenchmark:
    return BaselineMoEBackendSelectionBenchmark()
