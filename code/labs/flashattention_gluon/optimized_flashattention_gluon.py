"""Optimized FlashAttention lab: Gluon/Triton warp-specialized kernel (fallback to flash-attn)."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.flashattention_gluon.flashattention_gluon_common import (
    FlashAttentionInputs,
    FlashAttentionKernel,
    build_flashattention_inputs,
    resolve_gluon_flash_attention,
)


class OptimizedFlashAttentionGluonBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized attention: fused kernel (prefer Gluon warp specialization; fallback flash-attn)."""

    def __init__(self) -> None:
        super().__init__()
        self.batch = 2
        self.batch_size = self.batch
        self.seq_len = 1024
        self.heads = 8
        self.head_dim = 64
        self.hidden_dim = self.heads * self.head_dim
        self.dtype = torch.float16
        self.inputs: Optional[FlashAttentionInputs] = None
        self.kernel: Optional[FlashAttentionKernel] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_q: Optional[torch.Tensor] = None
        self._payload_k: Optional[torch.Tensor] = None
        self._payload_v: Optional[torch.Tensor] = None
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.kernel = resolve_gluon_flash_attention()
        self.inputs = build_flashattention_inputs(
            batch=self.batch,
            seq_len=self.seq_len,
            heads=self.heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self._output_buffer = torch.empty_like(self.inputs.q)
        self._verify_output_buffer = torch.empty_like(self.inputs.q, dtype=torch.float32)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None or self.kernel is None or self._output_buffer is None:
            raise RuntimeError("FlashAttention inputs/kernel not initialized")

        with torch.inference_mode():
            with self._nvtx_range("flashattention_optimized_fused"):
                with self._nvtx_range(f"flashattention_optimized_{self.kernel.provider}"):
                    q = self.inputs.q
                    k = self.inputs.k
                    v = self.inputs.v
                    result = self.kernel.fn(q, k, v, out=self._output_buffer)
                    self.output = result
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")
        self._payload_k = k
        self._payload_q = q
        self._payload_v = v

    def capture_verification_payload(self) -> None:
        k = self._payload_k
        q = self._payload_q
        v = self._payload_v
        if q is None or k is None or v is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"q": q.detach(), "k": k.detach(), "v": v.detach()},
            output=self._verify_output_buffer,
            batch_size=self.batch,
            parameter_count=0,
            precision_flags={"fp16": True, "bf16": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.inputs = None
        self.kernel = None
        self.output = None
        self._output_buffer = None
        self._verify_output_buffer = None
        self._payload_q = None
        self._payload_k = None
        self._payload_v = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return {"provider": self.kernel.provider if self.kernel else "unset"}

    def validate_result(self) -> Optional[str]:
        if self.inputs is None or self.kernel is None:
            return "FlashAttention inputs/kernel not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
    return OptimizedFlashAttentionGluonBenchmark()
