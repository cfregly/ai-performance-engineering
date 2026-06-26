"""Informational compound variant for the KV-cache chapter.

This target preserves the older blockwise decode plus FlashAttention-backed
story. It is intentionally noncanonical because it changes both cache layout
and the amount/backend of attention work.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older PyTorch fallback
    SDPBackend = None  # type: ignore[assignment]
    sdpa_kernel = None  # type: ignore[assignment]

from ch13.kv_cache_workload import get_workload
from ch13.optimized_kv_cache_naive import PagedKVCache
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

WORKLOAD = get_workload()


def _flash_sdp_context():
    if sdpa_kernel is None or SDPBackend is None or not hasattr(SDPBackend, "FLASH_ATTENTION"):
        return nullcontext()
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION])


class FlashBlockwiseAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, dtype=dtype)
        self.proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: PagedKVCache,
        request_id: str,
        layer_idx: int,
        cache_pos: int,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_block = k.permute(2, 0, 1, 3).contiguous()
        v_block = v.permute(2, 0, 1, 3).contiguous()
        kv_cache.append_block(request_id, layer_idx, k_block, v_block, cache_pos)

        if cache_pos > 0:
            cached_k, cached_v = kv_cache.get(request_id, layer_idx, 0, cache_pos)
            cached_k = cached_k.permute(1, 2, 0, 3)
            cached_v = cached_v.permute(1, 2, 0, 3)
            k = torch.cat([cached_k, k], dim=2)
            v = torch.cat([cached_v, v], dim=2)

        with _flash_sdp_context():
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.proj(attn_out)


class OptimizedKVCacheNaiveFlashBlockwiseBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Informational compound target: paged cache plus blockwise FlashAttention."""

    def __init__(self):
        super().__init__()
        self.layers = None
        self.kv_cache = None
        self.inputs = None
        self.workload = WORKLOAD
        self.page_size = self.workload.page_size
        self.num_layers = self.workload.num_layers
        self.num_heads = self.workload.num_heads
        self.head_dim = self.workload.head_dim
        self.hidden_dim = self.workload.hidden_dim
        self.batch_size = self.workload.batch_size
        self.sequence_lengths = list(self.workload.lengths())
        self.block_size = self.workload.block_size
        total_tokens = self.batch_size * sum(self.sequence_lengths)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(len(self.sequence_lengths)),
            tokens_per_iteration=float(total_tokens),
        )
        self.output = None
        self._verify_input = None
        self.parameter_count = 0
        self.register_workload_metadata(
            requests_per_iteration=float(len(self.sequence_lengths)),
            tokens_per_iteration=float(total_tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        self.layers = nn.ModuleList(
            [
                FlashBlockwiseAttentionLayer(self.hidden_dim, self.num_heads, self.head_dim, dtype=self.workload.dtype)
                for _ in range(self.num_layers)
            ]
        ).to(self.device).eval()
        self.parameter_count = sum(p.numel() for p in self.layers.parameters())

        self.kv_cache = PagedKVCache(
            page_size=self.page_size,
            batch_size=self.batch_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=self.workload.dtype,
            device=self.device,
        )

        self.inputs = []
        for seq_len in self.sequence_lengths:
            x = torch.randn(self.batch_size, seq_len, self.hidden_dim, device=self.device, dtype=self.workload.dtype)
            self.inputs.append(x)
        if self.inputs:
            self._verify_input = self.inputs[0].detach().clone()
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.layers is None or self.kv_cache is None or self.inputs is None:
            raise RuntimeError("Benchmark not configured")

        with self._nvtx_range("kv_cache_naive_flash_blockwise"):
            for seq_idx, x in enumerate(self.inputs):
                request_id = f"req_{seq_idx}"
                seq_len = x.size(1)
                self.kv_cache.allocate(request_id, seq_len)

                for pos in range(0, seq_len, self.block_size):
                    hidden = x[:, pos:pos + self.block_size, :]
                    for layer_idx, layer in enumerate(self.layers):
                        hidden = layer(hidden, self.kv_cache, request_id, layer_idx, pos)

                self.kv_cache.free(request_id)
            self.output = hidden[:, -1:, :].detach()
        if self._verify_input is None:
            raise RuntimeError("Verification input not initialized")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self.output.float(),
            batch_size=self._verify_input.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": self.workload.dtype == torch.float16,
                "bf16": self.workload.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(5e-2, 5e-1),
        )

    def teardown(self) -> None:
        self.layers = None
        self.kv_cache = None
        self.inputs = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
            measurement_timeout_seconds=300,
            warmup_timeout_seconds=120,
            setup_timeout_seconds=120,
            timeout_multiplier=1.0,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.layers is None:
            return "Model layers not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedKVCacheNaiveFlashBlockwiseBenchmark()
