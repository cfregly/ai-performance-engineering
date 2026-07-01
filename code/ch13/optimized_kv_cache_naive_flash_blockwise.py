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
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, dtype=dtype)
        self.proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self._qkv_buffer: Optional[torch.Tensor] = None
        self._qkv_weight_t: Optional[torch.Tensor] = None
        self._workspace_k: torch.Tensor | None = None
        self._workspace_v: torch.Tensor | None = None
        self._attn_merge_buffer: Optional[torch.Tensor] = None
        self._workspace_batch_size = 0
        self._workspace_seq_capacity = 0

    def prepare_inference(self) -> None:
        self._qkv_weight_t = self.qkv.weight.t()

    def _qkv_buffer_for(self, x: torch.Tensor) -> torch.Tensor:
        rows = int(x.size(0) * x.size(1))
        width = self.hidden_dim * 3
        if (
            self._qkv_buffer is None
            or self._qkv_buffer.device != x.device
            or self._qkv_buffer.dtype != x.dtype
            or self._qkv_buffer.size(0) < rows
        ):
            self._qkv_buffer = torch.empty(rows, width, device=x.device, dtype=x.dtype)
        return self._qkv_buffer[:rows]

    def _project_qkv(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            return self.qkv(x)
        if self._qkv_weight_t is None:
            self.prepare_inference()
        x_2d = x.reshape(x.size(0) * x.size(1), self.hidden_dim)
        qkv_2d = self._qkv_buffer_for(x)
        qkv = torch.matmul(x_2d, self._qkv_weight_t, out=qkv_2d)
        if self.qkv.bias is not None:
            qkv.add_(self.qkv.bias)
        return qkv.view(x.size(0), x.size(1), self.hidden_dim * 3)

    def configure_kv_workspace(
        self,
        max_seq_len: int,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        self._workspace_batch_size = batch_size
        self._workspace_seq_capacity = max_seq_len
        numel = batch_size * self.num_heads * max_seq_len * self.head_dim
        self._workspace_k = torch.empty(numel, dtype=dtype, device=device)
        self._workspace_v = torch.empty_like(self._workspace_k)

    def _get_kv_workspace(
        self,
        batch_size: int,
        total_len: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        workspace_missing = (
            self._workspace_k is None
            or self._workspace_v is None
            or self._workspace_batch_size < batch_size
            or self._workspace_seq_capacity < total_len
            or self._workspace_k.dtype != dtype
            or self._workspace_k.device != device
        )
        if workspace_missing:
            self.configure_kv_workspace(
                max(total_len, self._workspace_seq_capacity),
                max(batch_size, self._workspace_batch_size),
                dtype,
                device,
            )

        shape = (batch_size, self.num_heads, total_len, self.head_dim)
        stride = (
            self.num_heads * total_len * self.head_dim,
            total_len * self.head_dim,
            self.head_dim,
            1,
        )
        return (
            self._workspace_k.as_strided(shape, stride),
            self._workspace_v.as_strided(shape, stride),
        )

    def _attention_merge_buffer_for(
        self,
        x: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> torch.Tensor:
        rows = int(batch_size * seq_len)
        if (
            self._attn_merge_buffer is None
            or self._attn_merge_buffer.device != x.device
            or self._attn_merge_buffer.dtype != x.dtype
            or self._attn_merge_buffer.size(0) < rows
        ):
            self._attn_merge_buffer = torch.empty(rows, self.hidden_dim, device=x.device, dtype=x.dtype)
        return self._attn_merge_buffer[:rows].view(batch_size, seq_len, self.num_heads, self.head_dim)

    def _cached_attention_inputs(
        self,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: PagedKVCache,
        request_id: str,
        layer_idx: int,
        cache_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached_k, cached_v = kv_cache.get(request_id, layer_idx, 0, cache_pos)
        prefix_len = cached_k.size(0)
        total_len = prefix_len + k.size(2)
        key_buf, value_buf = self._get_kv_workspace(k.size(0), total_len, k.dtype, k.device)
        key_buf[:, :, :prefix_len, :].copy_(cached_k.permute(1, 2, 0, 3))
        value_buf[:, :, :prefix_len, :].copy_(cached_v.permute(1, 2, 0, 3))
        key_buf[:, :, prefix_len:total_len, :].copy_(k)
        value_buf[:, :, prefix_len:total_len, :].copy_(v)
        return key_buf, value_buf

    def forward(
        self,
        x: torch.Tensor,
        kv_cache: PagedKVCache,
        request_id: str,
        layer_idx: int,
        cache_pos: int,
    ) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self._project_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_block = k.permute(2, 0, 1, 3)
        v_block = v.permute(2, 0, 1, 3)
        kv_cache.append_block(request_id, layer_idx, k_block, v_block, cache_pos)

        if cache_pos > 0:
            k, v = self._cached_attention_inputs(k, v, kv_cache, request_id, layer_idx, cache_pos)

        with _flash_sdp_context():
            attn_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        if torch.is_grad_enabled():
            attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        else:
            attn_merge_buffer = self._attention_merge_buffer_for(x, batch_size, seq_len)
            attn_merge_buffer.copy_(attn_out.transpose(1, 2))
            attn_out = attn_merge_buffer.view(batch_size, seq_len, hidden_dim)
        return self.proj(attn_out)


class OptimizedKVCacheNaiveFlashBlockwiseBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Informational compound target: paged cache plus blockwise FlashAttention."""

    def __init__(self):
        super().__init__()
        self.layers = None
        self.kv_cache = None
        self.inputs = None
        self._input_count = 0
        self._request_ids: list[str] = []
        self._input_block_views: list[tuple[int, list[tuple[int, torch.Tensor]]]] = []
        self._request_block_groups: list[tuple[str, int, list[tuple[int, torch.Tensor]]]] = []
        self._request_group_counts: tuple[int, int, int] = (0, 0, 0)
        self._expected_request_group_counts: tuple[int, int, int] = (0, 0, 0)
        self._layer_groups: list[tuple[int, nn.Module]] = []
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
        self._verify_output_buffer: Optional[torch.Tensor] = None
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
        self._layer_groups = list(enumerate(self.layers))
        for layer in self.layers:
            layer.prepare_inference()
            layer.configure_kv_workspace(
                self.workload.max_seq_len,
                self.batch_size,
                self.workload.dtype,
                self.device,
            )
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
        self._input_count = len(self.inputs)
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            1,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float32,
        )
        self._request_ids = [f"req_{seq_idx}" for seq_idx in range(len(self.inputs))]
        self._input_block_views = [
            (
                x.size(1),
                [
                    (block_idx * self.block_size, block_view)
                    for block_idx, block_view in enumerate(x.split(self.block_size, dim=1))
                ],
            )
            for x in self.inputs
        ]
        self._request_block_groups = [
            (request_id, seq_len, block_views)
            for request_id, (seq_len, block_views) in zip(
                self._request_ids,
                self._input_block_views,
                strict=True,
            )
        ]
        self._request_group_counts = (
            len(self._request_ids),
            len(self._input_block_views),
            len(self._request_block_groups),
        )
        self._expected_request_group_counts = (
            self._input_count,
            self._input_count,
            self._input_count,
        )
        if self.inputs:
            self._verify_input = self.inputs[0].detach().clone()
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.layers is None or self.kv_cache is None or self.inputs is None:
            raise RuntimeError("Benchmark not configured")
        if self._request_group_counts != self._expected_request_group_counts:
            raise RuntimeError("Request block groups not initialized")
        if not self._layer_groups:
            raise RuntimeError("Layer groups not initialized")

        with torch.inference_mode(), self._nvtx_range("kv_cache_naive_flash_blockwise"):
            for request_id, seq_len, block_views in self._request_block_groups:
                self.kv_cache.allocate(request_id, seq_len)

                for pos, block_view in block_views:
                    hidden = block_view
                    for layer_idx, layer in self._layer_groups:
                        hidden = layer(hidden, self.kv_cache, request_id, layer_idx, pos)

                self.kv_cache.free(request_id)
            self.output = hidden[:, -1:, :]
        if self._verify_input is None:
            raise RuntimeError("Verification input not initialized")

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("Verification input/output not initialized")
        with torch.inference_mode():
            self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
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
        self._input_count = 0
        self._request_ids = []
        self._input_block_views = []
        self._request_block_groups = []
        self._request_group_counts = (0, 0, 0)
        self._expected_request_group_counts = (0, 0, 0)
        self._layer_groups = []
        self.output = None
        self._verify_input = None
        self._verify_output_buffer = None
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
