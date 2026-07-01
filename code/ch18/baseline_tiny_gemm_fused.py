"""Baseline tiny GEMMs for QKV + router projections (separate kernels)."""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from ch18.tiny_gemm_common import TinyGemmConfig, build_tiny_gemm_inputs


class BaselineTinyGemmBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: four separate small GEMMs (Q, K, V, router)."""

    def __init__(self, cfg: Optional[TinyGemmConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or TinyGemmConfig()
        self.x: Optional[torch.Tensor] = None
        self.w_q: Optional[torch.Tensor] = None
        self.w_k: Optional[torch.Tensor] = None
        self.w_v: Optional[torch.Tensor] = None
        self.w_router: Optional[torch.Tensor] = None
        self.w_fused: Optional[torch.Tensor] = None
        self._q_buffer: Optional[torch.Tensor] = None
        self._k_buffer: Optional[torch.Tensor] = None
        self._v_buffer: Optional[torch.Tensor] = None
        self._router_buffer: Optional[torch.Tensor] = None
        self._sum_buffer: Optional[torch.Tensor] = None
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
            raise RuntimeError("SKIPPED: CUDA required for tiny GEMM benchmark")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        (
            self.x,
            self.w_q,
            self.w_k,
            self.w_v,
            self.w_router,
            self.w_fused,
        ) = build_tiny_gemm_inputs(self.device, self.cfg)
        output_shape = (self.cfg.tokens, self.cfg.hidden_size)
        self._q_buffer = torch.empty(output_shape, device=self.device, dtype=self.cfg.dtype)
        self._k_buffer = torch.empty_like(self._q_buffer)
        self._v_buffer = torch.empty_like(self._q_buffer)
        self._router_buffer = torch.empty_like(self._q_buffer)
        self._sum_buffer = torch.empty_like(self._q_buffer)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if (
            self.x is None
            or self.w_q is None
            or self.w_k is None
            or self.w_v is None
            or self.w_router is None
            or self._q_buffer is None
            or self._k_buffer is None
            or self._v_buffer is None
            or self._router_buffer is None
            or self._sum_buffer is None
        ):
            raise RuntimeError("Benchmark not initialized")
        with torch.inference_mode():
            torch.mm(self.x, self.w_q, out=self._q_buffer)
            torch.mm(self.x, self.w_k, out=self._k_buffer)
            torch.mm(self.x, self.w_v, out=self._v_buffer)
            torch.mm(self.x, self.w_router, out=self._router_buffer)
            torch.add(self._q_buffer, self._k_buffer, out=self._sum_buffer)
            self._sum_buffer.add_(self._v_buffer)
            self._sum_buffer.add_(self._router_buffer)
            self.output = self._sum_buffer
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.x is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"x": self.x.detach()},
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
        return BenchmarkConfig(
            iterations=6,
            warmup=6,
            timing_method="wall_clock",
            full_device_sync=True,
        )


def get_benchmark() -> BaseBenchmark:
    return BaselineTinyGemmBenchmark()
