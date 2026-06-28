"""Optimized tiny GEMM (fused QKV + router projection)."""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from ch18.tiny_gemm_common import TinyGemmConfig, build_tiny_gemm_inputs


class OptimizedTinyGemmBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: single fused GEMM for QKV + router."""

    def __init__(self, cfg: Optional[TinyGemmConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or TinyGemmConfig()
        self.x: Optional[torch.Tensor] = None
        self.w_q: Optional[torch.Tensor] = None
        self.w_k: Optional[torch.Tensor] = None
        self.w_v: Optional[torch.Tensor] = None
        self.w_router: Optional[torch.Tensor] = None
        self.w_fused: Optional[torch.Tensor] = None
        self._proj_buffer: Optional[torch.Tensor] = None
        self._q_view: Optional[torch.Tensor] = None
        self._k_view: Optional[torch.Tensor] = None
        self._v_view: Optional[torch.Tensor] = None
        self._router_view: Optional[torch.Tensor] = None
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
        self._proj_buffer = torch.empty(
            self.cfg.tokens,
            self.cfg.hidden_size * 4,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        hidden = self.cfg.hidden_size
        self._q_view = self._proj_buffer.narrow(1, 0, hidden)
        self._k_view = self._proj_buffer.narrow(1, hidden, hidden)
        self._v_view = self._proj_buffer.narrow(1, hidden * 2, hidden)
        self._router_view = self._proj_buffer.narrow(1, hidden * 3, hidden)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if (
            self.x is None
            or self.w_fused is None
            or self._proj_buffer is None
            or self._q_view is None
            or self._k_view is None
            or self._v_view is None
            or self._router_view is None
        ):
            raise RuntimeError("Benchmark not initialized")
        with torch.inference_mode():
            torch.mm(self.x, self.w_fused, out=self._proj_buffer)
            q = self._q_view
            k = self._k_view
            v = self._v_view
            router = self._router_view
            q.add_(k)
            q.add_(v)
            q.add_(router)
            self.output = q
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
    return OptimizedTinyGemmBenchmark()
