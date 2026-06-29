"""Optimized block-sparse attention using FlashInfer + output projection."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

try:
    import flashinfer
except Exception as exc:  # pragma: no cover - optional dependency
    flashinfer = None
    _FLASHINFER_IMPORT_ERROR = exc
else:
    _FLASHINFER_IMPORT_ERROR = None

from ch16.block_sparse_attention_utils import (
    build_block_sparse_pattern,
    build_bsr_from_block_mask,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedFlashInferAttentionLab(VerificationPayloadMixin, BaseBenchmark):
    """FlashInfer block-sparse attention optimized path."""

    def __init__(self) -> None:
        super().__init__()
        self.seq_len = 2048
        self.head_dim = 64
        self.heads = 45
        self.hidden_size = self.heads * self.head_dim
        self.block_size = 16
        self.window_blocks = 2
        self.q: Optional[torch.Tensor] = None
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.out_proj: Optional[nn.Linear] = None
        self._proj_weight_t: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.wrapper: Optional[flashinfer.BlockSparseAttentionWrapper] = None
        self.sparsity_ratio = 0.0
        self._payload_parameter_count = 0
        tokens = float(self.seq_len)
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )

    def setup(self) -> None:
        if flashinfer is None:
            raise RuntimeError(f"SKIPPED: flashinfer not available ({_FLASHINFER_IMPORT_ERROR})")
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.q = torch.randn(self.seq_len, self.heads, self.head_dim, device=self.device, dtype=torch.float16)
        self.k = torch.randn(self.seq_len, self.heads, self.head_dim, device=self.device, dtype=torch.float16)
        self.v = torch.randn(self.seq_len, self.heads, self.head_dim, device=self.device, dtype=torch.float16)
        block_mask = build_block_sparse_pattern(
            seq_len=self.seq_len,
            block_size=self.block_size,
            window_blocks=self.window_blocks,
        )
        indptr, indices, self.sparsity_ratio = build_bsr_from_block_mask(block_mask, device=self.device)
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        self.wrapper = flashinfer.BlockSparseAttentionWrapper(workspace)
        sm_scale = 1.0 / math.sqrt(self.head_dim)
        self.wrapper.plan(
            indptr,
            indices,
            self.seq_len,
            self.seq_len,
            self.block_size,
            self.block_size,
            self.heads,
            self.heads,
            self.head_dim,
            sm_scale=sm_scale,
        )
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device, dtype=torch.float16)
        self._proj_weight_t = self.out_proj.weight.t()
        self._output_buffer = torch.empty(
            self.seq_len,
            self.hidden_size,
            device=self.device,
            dtype=torch.float16,
        )
        self._verify_output_buffer = torch.empty(
            min(128, self.seq_len),
            min(128, self.hidden_size),
            device=self.device,
            dtype=torch.float16,
        )
        self._payload_parameter_count = self.out_proj.weight.numel()
        self._synchronize()

    def benchmark_fn(self) -> None:
        if (
            self.q is None
            or self.k is None
            or self.v is None
            or self.wrapper is None
            or self.out_proj is None
            or self._proj_weight_t is None
            or self._output_buffer is None
        ):
            raise RuntimeError("Benchmark not initialized")
        with self._nvtx_range("optimized_flashinfer_attention"):
            with torch.inference_mode():
                attn_out = self.wrapper.run(self.q, self.k, self.v)
                proj_in = attn_out.reshape(self.seq_len, self.hidden_size)
                self.output = torch.matmul(proj_in, self._proj_weight_t, out=self._output_buffer)
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
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        verify_output = self._verify_output_buffer
        verify_output.copy_(self.output[: verify_output.shape[0], : verify_output.shape[1]])
        self._set_verification_payload(
            inputs={"q": self.q, "k": self.k, "v": self.v},
            output=verify_output,
            batch_size=self.seq_len,
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-2, 1e-2),
            signature_overrides={
                "sparsity_ratio": self.sparsity_ratio,
            },
        )

    def teardown(self) -> None:
        self.q = None
        self.k = None
        self.v = None
        self.out_proj = None
        self._proj_weight_t = None
        self.output = None
        self._output_buffer = None
        self._verify_output_buffer = None
        self.wrapper = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            ncu_replay_mode="kernel",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedFlashInferAttentionLab()
