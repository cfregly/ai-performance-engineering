"""Baseline block-sparse attention (dense SDP + mask) with output projection."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ch16.block_sparse_attention_utils import (
    build_block_sparse_pattern,
    build_bsr_from_block_mask,
    build_dense_attention_mask,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineFlashInferAttentionLab(VerificationPayloadMixin, BaseBenchmark):
    """Baseline block-sparse attention via dense SDP + mask."""

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
        self.attn_mask: Optional[torch.Tensor] = None
        self.out_proj: Optional[nn.Linear] = None
        self.output: Optional[torch.Tensor] = None
        self.sparsity_ratio = 0.0
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
        self.attn_mask = build_dense_attention_mask(
            block_mask,
            block_size=self.block_size,
            device=self.device,
            dtype=torch.float16,
        )
        _, _, self.sparsity_ratio = build_bsr_from_block_mask(block_mask, device=self.device)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False).to(self.device, dtype=torch.float16)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.q is None or self.k is None or self.v is None or self.attn_mask is None or self.out_proj is None:
            raise RuntimeError("Benchmark not initialized")
        with self._nvtx_range("baseline_flashinfer_attention"):
            with torch.inference_mode():
                q = self.q.transpose(0, 1).unsqueeze(0)
                k = self.k.transpose(0, 1).unsqueeze(0)
                v = self.v.transpose(0, 1).unsqueeze(0)
                out = F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=self.attn_mask,
                    is_causal=False,
                )
                attn_out = out.squeeze(0).transpose(0, 1)
                proj_in = attn_out.reshape(self.seq_len, self.hidden_size)
                self.output = self.out_proj(proj_in)
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.q is None or self.k is None or self.v is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        verify_output = self.output[:128, :128]
        parameter_count = self.out_proj.weight.numel() if self.out_proj is not None else 0
        self._set_verification_payload(
            inputs={"q": self.q, "k": self.k, "v": self.v},
            output=verify_output.detach().clone(),
            batch_size=self.seq_len,
            parameter_count=parameter_count,
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
        self.attn_mask = None
        self.out_proj = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            ncu_replay_mode="application",
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineFlashInferAttentionLab()

