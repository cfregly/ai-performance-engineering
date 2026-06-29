"""Baseline block-sparse attention using dense SDP with an explicit mask."""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn.functional as F

from ch16.block_sparse_attention_utils import (
    build_block_sparse_pattern,
    build_bsr_from_block_mask,
    build_dense_attention_mask,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineFlashInferBlockSparseBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline block-sparse attention via dense SDP + mask."""

    def __init__(self) -> None:
        super().__init__()
        self.seq_len = 2048
        self.heads = 45
        self.head_dim = 64
        self.block_size = 16
        self.window_blocks = 2
        self.q: Optional[torch.Tensor] = None
        self.k: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self._q_sdp: Optional[torch.Tensor] = None
        self._k_sdp: Optional[torch.Tensor] = None
        self._v_sdp: Optional[torch.Tensor] = None
        self.attn_mask: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.sparsity_ratio = 0.0
        tokens = self.seq_len * self.heads
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.heads),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.heads),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.q = torch.randn(self.seq_len, self.heads, self.head_dim, device=self.device, dtype=torch.float16)
        self.k = torch.randn(self.seq_len, self.heads, self.head_dim, device=self.device, dtype=torch.float16)
        self.v = torch.randn(self.seq_len, self.heads, self.head_dim, device=self.device, dtype=torch.float16)
        self._q_sdp = self.q.transpose(0, 1).unsqueeze(0)
        self._k_sdp = self.k.transpose(0, 1).unsqueeze(0)
        self._v_sdp = self.v.transpose(0, 1).unsqueeze(0)
        self._verify_output_buffer = torch.empty(
            self.seq_len,
            self.heads,
            self.head_dim,
            device=self.device,
            dtype=torch.float16,
        )
        block_mask = build_block_sparse_pattern(
            seq_len=self.seq_len,
            block_size=self.block_size,
            window_blocks=self.window_blocks,
        )
        self.attn_mask = build_dense_attention_mask(
            block_mask,
            block_size=self.block_size,
            device=self.device,
            dtype=torch.float16,
        )
        _, _, self.sparsity_ratio = build_bsr_from_block_mask(block_mask, device=self.device)

    def benchmark_fn(self) -> None:
        if (
            self.q is None
            or self.k is None
            or self.v is None
            or self._q_sdp is None
            or self._k_sdp is None
            or self._v_sdp is None
            or self.attn_mask is None
        ):
            raise RuntimeError("Benchmark not initialized")
        with self._nvtx_range("baseline_flashinfer_block_sparse"):
            with torch.inference_mode():
                out = F.scaled_dot_product_attention(
                    self._q_sdp,
                    self._k_sdp,
                    self._v_sdp,
                    attn_mask=self.attn_mask,
                    is_causal=False,
                )
                self.output = out.squeeze(0).transpose(0, 1)
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
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"q": self.q, "k": self.k, "v": self.v},
            output=self._verify_output_buffer,
            batch_size=self.seq_len,
            parameter_count=0,
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
        self._q_sdp = None
        self._k_sdp = None
        self._v_sdp = None
        self.attn_mask = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.q is None or self.k is None or self.v is None:
            return "Inputs not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineFlashInferBlockSparseBenchmark()
