"""optimized_kv_cache_naive.py - Token-by-token paged KV cache implementation.

The canonical optimized pair keeps the same token-by-token decode loop and SDPA
math as the baseline while replacing repeated `torch.cat` growth with a paged
cache layout.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ch13.kv_cache_workload import get_workload
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

WORKLOAD = get_workload()


@dataclass(slots=True)
class _PagedLayerEntry:
    pages: int
    buffer: tuple[torch.Tensor, torch.Tensor]
    length: int = 0


class PagedKVCache:
    """KV cache that reuses right-sized contiguous slabs."""
    
    def __init__(
        self,
        page_size: int,
        batch_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.page_size = page_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.buffer_pool: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        self.allocations: dict[str, list[_PagedLayerEntry]] = {}
        self._empty = torch.empty(
            0,
            self.batch_size,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
    
    def _acquire_buffer(self, pages: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.buffer_pool[pages]:
            return self.buffer_pool[pages].pop()
        length = pages * self.page_size
        k_buf = torch.empty(
            length,
            self.batch_size,
            self.num_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        v_buf = torch.empty_like(k_buf)
        return k_buf, v_buf
    
    def _release_buffer(self, pages: int, buffer: tuple[torch.Tensor, torch.Tensor]) -> None:
        # Entry lengths define the valid prefix; pooled slabs do not need a
        # full clear before reuse.
        self.buffer_pool[pages].append(buffer)
    
    def allocate(self, request_id: str, seq_len: int) -> None:
        if request_id in self.allocations:
            return
        pages = max(1, math.ceil(seq_len / self.page_size))
        per_layer: list[_PagedLayerEntry] = []
        for _ in range(self.num_layers):
            per_layer.append(_PagedLayerEntry(pages=pages, buffer=self._acquire_buffer(pages)))
        self.allocations[request_id] = per_layer
    
    def _ensure_capacity(self, entry: _PagedLayerEntry, target_pos: int) -> None:
        pages = entry.pages
        capacity = pages * self.page_size
        if target_pos < capacity:
            return
        new_pages = max(pages * 2, math.ceil((target_pos + 1) / self.page_size))
        new_buffer = self._acquire_buffer(new_pages)
        old_buffer = entry.buffer
        valid = min(entry.length, capacity)
        new_buffer[0][:valid].copy_(old_buffer[0][:valid])
        new_buffer[1][:valid].copy_(old_buffer[1][:valid])
        self._release_buffer(pages, old_buffer)
        entry.buffer = new_buffer
        entry.pages = new_pages
    
    def append(self, request_id: str, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos: int) -> None:
        if request_id not in self.allocations:
            self.allocate(request_id, pos + self.page_size)
        entry = self.allocations[request_id][layer_idx]
        self._ensure_capacity(entry, pos)
        buffer_k, buffer_v = entry.buffer
        buffer_k[pos].copy_(k)
        buffer_v[pos].copy_(v)
        entry.length = max(entry.length, pos + 1)
    
    def append_block(
        self,
        request_id: str,
        layer_idx: int,
        k_block: torch.Tensor,
        v_block: torch.Tensor,
        start_pos: int,
    ) -> None:
        block = int(k_block.size(0))
        if request_id not in self.allocations:
            self.allocate(request_id, start_pos + block)
        entry = self.allocations[request_id][layer_idx]
        self._ensure_capacity(entry, start_pos + block - 1)
        buffer_k, buffer_v = entry.buffer
        buffer_k[start_pos:start_pos + block].copy_(k_block)
        buffer_v[start_pos:start_pos + block].copy_(v_block)
        entry.length = max(entry.length, start_pos + block)
    
    def get(self, request_id: str, layer_idx: int, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        if request_id not in self.allocations:
            return self._empty, self._empty
        entry = self.allocations[request_id][layer_idx]
        valid_end = min(end, entry.length)
        if start >= valid_end:
            return self._empty, self._empty
        buffer_k, buffer_v = entry.buffer
        return buffer_k[start:valid_end], buffer_v[start:valid_end]
    
    def free(self, request_id: str) -> None:
        if request_id not in self.allocations:
            return
        for entry in self.allocations.pop(request_id):
            self._release_buffer(entry.pages, entry.buffer)


class PagedAttentionLayer(nn.Module):
    """Attention layer that reuses paged cache storage with baseline-equivalent math."""
    
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, dtype=dtype)
        self.proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self._qkv_buffer: Optional[torch.Tensor] = None
        self._qkv_weight_t: Optional[torch.Tensor] = None

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
        
        kv_cache.append(request_id, layer_idx, k[:, :, 0, :], v[:, :, 0, :], cache_pos)
        cached_k, cached_v = kv_cache.get(request_id, layer_idx, 0, cache_pos + 1)
        cached_k = cached_k.permute(1, 2, 0, 3)
        cached_v = cached_v.permute(1, 2, 0, 3)

        attn_out = F.scaled_dot_product_attention(q, cached_k, cached_v, dropout_p=0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2)
        if seq_len == 1:
            attn_out = attn_out.view(batch_size, seq_len, hidden_dim)
        else:
            attn_out = attn_out.contiguous().view(batch_size, seq_len, hidden_dim)
        return self.proj(attn_out)


class OptimizedKVCachePagedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Paged KV cache optimization with token-by-token decode semantics."""
    
    def __init__(self):
        super().__init__()
        self.layers = None
        self.kv_cache = None
        self.inputs = None
        self._input_count = 0
        self._request_ids: list[str] = []
        self._input_token_views: list[list[torch.Tensor]] = []
        self._input_token_steps: list[list[tuple[int, torch.Tensor]]] = []
        self._request_token_groups: list[tuple[str, int, list[tuple[int, torch.Tensor]]]] = []
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
        """Setup: Initialize model and paged KV cache."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        self.layers = nn.ModuleList(
            [
                PagedAttentionLayer(self.hidden_dim, self.num_heads, self.head_dim, dtype=self.workload.dtype)
                for _ in range(self.num_layers)
            ]
        ).to(self.device).eval()
        for layer in self.layers:
            layer.prepare_inference()
        self._layer_groups = list(enumerate(self.layers))
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
        self._input_token_views = [
            list(x.split(1, dim=1))
            for x in self.inputs
        ]
        self._input_token_steps = [
            list(enumerate(token_views))
            for token_views in self._input_token_views
        ]
        self._request_token_groups = [
            (request_id, len(token_views), token_steps)
            for request_id, token_views, token_steps in zip(
                self._request_ids,
                self._input_token_views,
                self._input_token_steps,
                strict=True,
            )
        ]
        self._request_group_counts = (
            len(self._request_ids),
            len(self._input_token_views),
            len(self._request_token_groups),
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
        """Benchmark token-by-token decode with paged cache allocation."""
        if self.layers is None or self.kv_cache is None or self.inputs is None:
            raise RuntimeError("Benchmark not configured")
        if self._request_group_counts != self._expected_request_group_counts:
            raise RuntimeError("Request token groups not initialized")
        if not self._layer_groups:
            raise RuntimeError("Layer groups not initialized")

        with torch.inference_mode(), self._nvtx_range("kv_cache_naive"):
            for request_id, seq_len, token_steps in self._request_token_groups:
                self.kv_cache.allocate(request_id, seq_len)

                for pos, token_view in token_steps:
                    hidden = token_view
                    for layer_idx, layer in self._layer_groups:
                        hidden = layer(hidden, self.kv_cache, request_id, layer_idx, pos)

                self.kv_cache.free(request_id)
            self.output = hidden
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
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(5e-2, 5e-1),
        )

    
    def teardown(self) -> None:
        """Cleanup."""
        self.layers = None
        self.kv_cache = None
        self.inputs = None
        self._input_count = 0
        self._request_ids = []
        self._input_token_views = []
        self._input_token_steps = []
        self._request_token_groups = []
        self._request_group_counts = (0, 0, 0)
        self._expected_request_group_counts = (0, 0, 0)
        self._layer_groups = []
        self.output = None
        self._verify_input = None
        self._verify_output_buffer = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
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
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=None,
            reduced_precision_time_ms=getattr(self, '_last_elapsed_ms', None),
            precision_type="fp16",
        )

    def get_optimization_goal(self) -> str:
        """Paged allocation is kept as a memory benchmark, not a speed benchmark."""
        return "memory"

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.layers is None:
            return "Model layers not initialized"
        return None

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (5e-2, 5e-1)


def get_benchmark() -> OptimizedKVCachePagedBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedKVCachePagedBenchmark()
