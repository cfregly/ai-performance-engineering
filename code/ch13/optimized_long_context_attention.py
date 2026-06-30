"""Optimized long-context attention using Flash SDP."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from core.benchmark.verification import PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

_SDPA_KERNEL = getattr(torch.nn.attention, "sdpa_kernel", None)
_FLASH_SDP_BACKEND = getattr(torch.nn.attention.SDPBackend, "FLASH_ATTENTION", None)


class OptimizedLongContextAttentionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Long-context attention optimized with Flash SDP."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_size = 1
        self.seq_len = 12288
        self.hidden_dim = 512
        self.num_heads = 4
        self.head_dim = self.hidden_dim // self.num_heads
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.q: Optional[torch.Tensor] = None
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        dtype = torch.bfloat16
        self.q = torch.randn(
            self.batch_size,
            self.num_heads,
            self.seq_len,
            self.head_dim,
            device=self.device,
            dtype=dtype,
        )
        self.k = torch.randn_like(self.q)
        self.v = torch.randn_like(self.q)
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            self.num_heads,
            min(128, self.seq_len),
            self.head_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.q is None or self.k is None or self.v is None:
            raise RuntimeError("Benchmark not configured")
        with torch.inference_mode():
            if _SDPA_KERNEL is None or _FLASH_SDP_BACKEND is None:
                raise RuntimeError("torch.nn.attention.sdpa_kernel is required for flash attention")
            if not torch.backends.cuda.flash_sdp_enabled():
                raise RuntimeError("Flash SDP backend is not available on this build")
            with _SDPA_KERNEL(_FLASH_SDP_BACKEND):
                self.output = F.scaled_dot_product_attention(
                    self.q,
                    self.k,
                    self.v,
                    dropout_p=0.0,
                    is_causal=True,
                )
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if (
            self.q is None
            or self.k is None
            or self.v is None
            or self.output is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            : self._verify_output_buffer.shape[2],
            : self._verify_output_buffer.shape[3],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"q": self.q, "k": self.k, "v": self.v},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            precision_flags=PrecisionFlags(bf16=True, tf32=False),
            output_tolerance=(0.2, 2.0),
        )

    def teardown(self) -> None:
        self.q = None
        self.k = None
        self.v = None
        self.output = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.q is None or self.k is None or self.v is None:
            return "Inputs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedLongContextAttentionBenchmark()
