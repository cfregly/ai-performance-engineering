"""Baseline long-context attention using explicit softmax + matmul."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification import PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineLongContextAttentionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Long-context attention baseline (explicit matmul + softmax)."""

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
        self._k_t: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self._mask: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._scale = 0.0
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
        self._k_t = self.k.transpose(-2, -1)
        self._scale = 1.0 / (self.head_dim ** 0.5)
        pos = torch.arange(self.seq_len, device=self.device)
        mask = pos.unsqueeze(0) > pos.unsqueeze(1)
        self._mask = mask.view(1, 1, self.seq_len, self.seq_len)
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
        if self.q is None or self.k is None or self._k_t is None or self.v is None or self._mask is None:
            raise RuntimeError("Benchmark not configured")
        with torch.inference_mode():
            scores = torch.matmul(self.q, self._k_t)
            scores.mul_(self._scale)
            scores.masked_fill_(self._mask, float("-inf"))
            attn = torch.softmax(scores, dim=-1)
            self.output = torch.matmul(attn, self.v)
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
        self._k_t = None
        self.v = None
        self._mask = None
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
    return BaselineLongContextAttentionBenchmark()
