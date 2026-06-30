"""optimized_integrated_kv_cache.py - Optimized integrated KV cache (optimized).

Optimized KV cache integration with paged memory management.
Efficient memory reuse and reduced fragmentation.

Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import math
from collections import defaultdict
from contextlib import nullcontext
from functools import partial

import torch
import torch.nn as nn
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older PyTorch fallback
    SDPBackend = None  # type: ignore[assignment]
    sdpa_kernel = None  # type: ignore[assignment]

from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

def _flash_sdp_context():
    """Prefer the new sdpa_kernel API; fall back to no-op if unavailable."""
    if sdpa_kernel is None or SDPBackend is None or not hasattr(SDPBackend, "FLASH_ATTENTION"):
        return nullcontext()
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION])

resolve_device = partial(require_cuda_device, "CUDA required for ch20")

CacheEntry = dict[str, object]

class PagedKVCache:
    """KV cache that reuses contiguous slabs sized by sequence length."""
    
    def __init__(self, page_size: int, num_layers: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        self.page_size = page_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.buffer_pool: dict[int, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)
        self.allocations: dict[str, list[CacheEntry]] = {}
        self._empty = torch.empty(0, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
    
    def _acquire_buffer(self, pages: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.buffer_pool[pages]:
            return self.buffer_pool[pages].pop()
        length = pages * self.page_size
        k_buf = torch.empty(length, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
        v_buf = torch.empty_like(k_buf)
        return k_buf, v_buf
    
    def _release_buffer(self, pages: int, buffer: tuple[torch.Tensor, torch.Tensor]) -> None:
        # Valid ranges are tracked per allocation, so releasing does not need to
        # clear the whole slab before it re-enters the pool.
        self.buffer_pool[pages].append(buffer)
    
    def allocate(self, request_id: str, estimated_len: int) -> list[CacheEntry]:
        existing = self.allocations.get(request_id)
        if existing is not None:
            return existing
        pages = max(1, math.ceil(estimated_len / self.page_size))
        per_layer: list[CacheEntry] = []
        for _ in range(self.num_layers):
            per_layer.append(
                {
                    "pages": pages,
                    "buffer": self._acquire_buffer(pages),
                    "length": 0,
                }
            )
        self.allocations[request_id] = per_layer
        return per_layer
    
    def _ensure_capacity(self, entry: CacheEntry, target_pos: int) -> None:
        pages = int(entry["pages"])
        capacity = pages * self.page_size
        if target_pos < capacity:
            return
        new_pages = max(pages * 2, math.ceil((target_pos + 1) / self.page_size))
        new_buffer = self._acquire_buffer(new_pages)
        old_buffer = entry["buffer"]  # type: ignore[index]
        valid = min(int(entry["length"]), capacity)
        new_buffer[0][:valid].copy_(old_buffer[0][:valid])
        new_buffer[1][:valid].copy_(old_buffer[1][:valid])
        self._release_buffer(pages, old_buffer)  # type: ignore[arg-type]
        entry["buffer"] = new_buffer
        entry["pages"] = new_pages
    
    def append(self, request_id: str, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos: int) -> None:
        entries = self.allocations.get(request_id)
        if entries is None:
            entries = self.allocate(request_id, pos + self.page_size)
        entry = entries[layer_idx]
        self._ensure_capacity(entry, pos)
        buffer_k, buffer_v = entry["buffer"]  # type: ignore[assignment]
        buffer_k[pos].copy_(k)
        buffer_v[pos].copy_(v)
        entry["length"] = max(int(entry["length"]), pos + 1)
    
    def append_block(
        self,
        request_id: str,
        layer_idx: int,
        k_block: torch.Tensor,
        v_block: torch.Tensor,
        start_pos: int,
    ) -> None:
        entries = self.allocations.get(request_id)
        if entries is None:
            entries = self.allocate(request_id, start_pos + int(k_block.size(0)))
        self.append_block_entry(entries[layer_idx], k_block, v_block, start_pos)

    def append_block_entry(
        self,
        entry: CacheEntry,
        k_block: torch.Tensor,
        v_block: torch.Tensor,
        start_pos: int,
    ) -> None:
        block = int(k_block.size(0))
        self._ensure_capacity(entry, start_pos + block - 1)
        buffer_k, buffer_v = entry["buffer"]  # type: ignore[assignment]
        buffer_k[start_pos:start_pos + block].copy_(k_block)
        buffer_v[start_pos:start_pos + block].copy_(v_block)
        entry["length"] = max(int(entry["length"]), start_pos + block)

    def project_kv_block_entry(
        self,
        entry: CacheEntry,
        x_block: torch.Tensor,
        k_weight_t: torch.Tensor,
        k_bias: Optional[torch.Tensor],
        v_weight_t: torch.Tensor,
        v_bias: Optional[torch.Tensor],
        start_pos: int,
    ) -> None:
        block = int(x_block.size(0))
        self._ensure_capacity(entry, start_pos + block - 1)
        buffer_k, buffer_v = entry["buffer"]  # type: ignore[assignment]
        k_out = buffer_k[start_pos:start_pos + block].flatten(1)
        v_out = buffer_v[start_pos:start_pos + block].flatten(1)
        torch.matmul(x_block, k_weight_t, out=k_out)
        torch.matmul(x_block, v_weight_t, out=v_out)
        if k_bias is not None:
            k_out.add_(k_bias)
        if v_bias is not None:
            v_out.add_(v_bias)
        entry["length"] = max(int(entry["length"]), start_pos + block)
    
    def get(self, request_id: str, layer_idx: int, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        entries = self.allocations.get(request_id)
        if entries is None:
            return self._empty, self._empty
        return self.get_entry(entries[layer_idx], start, end)

    def get_entry(self, entry: CacheEntry, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        valid_end = min(end, int(entry["length"]))
        if start >= valid_end:
            return self._empty, self._empty
        buffer_k, buffer_v = entry["buffer"]  # type: ignore[assignment]
        return buffer_k[start:valid_end], buffer_v[start:valid_end]
    
    def free(self, request_id: str) -> None:
        if request_id not in self.allocations:
            return
        for entry in self.allocations.pop(request_id):
            self._release_buffer(entry["pages"], entry["buffer"])  # type: ignore[arg-type]

class AttentionLayer(nn.Module):
    """Attention layer with paged KV cache writes.

    The baseline/optimized pair focuses on KV cache integration cost (append/read),
    not attention math. Keep the forward path comparable to baseline so post-timing
    verification can assert output equivalence.
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, dtype=dtype)
        self.proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.register_buffer("_cache_touch", torch.empty((), dtype=dtype), persistent=False)

    def _project_single_batch_kv(
        self,
        x: torch.Tensor,
        kv_cache: PagedKVCache,
        cache_entry: CacheEntry,
        cache_pos: int,
    ) -> None:
        hidden = self.hidden_dim
        weight = self.qkv.weight
        bias = self.qkv.bias
        kv_cache.project_kv_block_entry(
            cache_entry,
            x[0],
            weight[hidden : hidden * 2].t(),
            None if bias is None else bias[hidden : hidden * 2],
            weight[hidden * 2 : hidden * 3].t(),
            None if bias is None else bias[hidden * 2 : hidden * 3],
            cache_pos,
        )
    
    def forward(self, x: torch.Tensor, kv_cache: PagedKVCache, cache_entry: CacheEntry, cache_pos: int) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        if batch_size == 1:
            self._project_single_batch_kv(x, kv_cache, cache_entry, cache_pos)
        else:
            qkv = self.qkv(x)
            _, k, v = qkv.chunk(3, dim=-1)
            k = k.view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.view(batch_size, seq_len, self.num_heads, self.head_dim)
            append_block_entry = kv_cache.append_block_entry
            for batch_idx in range(batch_size):
                append_block_entry(
                    cache_entry,
                    k[batch_idx],
                    v[batch_idx],
                    cache_pos,
                )
        
        if cache_pos > 0:
            cached_k, cached_v = kv_cache.get_entry(cache_entry, 0, cache_pos)
            torch.sum(cached_k, dim=(0, 1, 2), out=self._cache_touch)
            torch.sum(cached_v, dim=(0, 1, 2), out=self._cache_touch)

        return self.proj(x)

class OptimizedIntegratedKVCacheBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Integrated KV cache in full inference pipeline."""
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.layers = None
        # Optimization: Compile model for kernel fusion and optimization

        # Optimization: Compile model for kernel fusion and optimization

        self.kv_cache = None
        self.inputs = None
        self.request_ids: list[str] = []
        self._input_block_views: list[tuple[int, list[tuple[int, torch.Tensor]]]] = []
        self._request_block_groups: list[tuple[str, int, list[tuple[int, torch.Tensor]]]] = []
        self._request_group_counts: tuple[int, int, int] = (0, 0, 0)
        self._expected_request_group_counts: tuple[int, int, int] = (0, 0, 0)
        self._layer_groups: list[tuple[int, nn.Module]] = []
        self.page_size = 128
        self.num_layers = 2
        self.num_heads = 2
        self.head_dim = 32
        self.hidden_dim = self.num_heads * self.head_dim
        self.batch_size = 1
        self.sequence_lengths = [512, 1024, 2048]
        self.block_size = 8
        self.register_workload_metadata(requests_per_iteration=1.0)
        self.output: Optional[torch.Tensor] = None
        self._verification_payload = None
        self._payload_parameter_count = 0
        self._enable_nvtx = False
    
    def setup(self) -> None:
        """Setup: Initialize model with integrated KV cache."""

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.layers = nn.ModuleList(
            [
                AttentionLayer(self.hidden_dim, self.num_heads, self.head_dim, dtype=torch.float16)
                for _ in range(self.num_layers)
            ]
        ).to(self.device).eval()
        self._layer_groups = list(enumerate(self.layers))
        self._payload_parameter_count = sum(p.numel() for p in self.layers.parameters())
        
        self.kv_cache = PagedKVCache(
            page_size=self.page_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=torch.float16,
            device=self.device
        )
        
        self.inputs = []
        for seq_len in self.sequence_lengths:
            x = torch.randn(self.batch_size, seq_len, self.hidden_dim, device=self.device, dtype=torch.float16)
            self.inputs.append(x)
        self.request_ids = [f"req_{seq_idx}" for seq_idx in range(len(self.inputs))]
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
                self.request_ids,
                self._input_block_views,
                strict=True,
            )
        ]
        input_count = len(self.inputs)
        self._request_group_counts = (
            len(self.request_ids),
            len(self._input_block_views),
            len(self._request_block_groups),
        )
        self._expected_request_group_counts = (
            input_count,
            input_count,
            input_count,
        )
        self._verify_input = self.inputs[-1] if self.inputs else None
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - integrated KV cache pipeline."""
        with torch.inference_mode():
            with nvtx_range("integrated_kv_cache", enable=self._enable_nvtx):
                if self._request_group_counts != self._expected_request_group_counts:
                    raise RuntimeError("Request block groups not initialized")
                if not self._layer_groups:
                    raise RuntimeError("Layer groups not initialized")
                kv_cache = self.kv_cache
                if kv_cache is None:
                    raise RuntimeError("KV cache not initialized")
                for request_id, seq_len, block_views in self._request_block_groups:
                    request_entries = kv_cache.allocate(request_id, seq_len)

                    for pos, block_view in block_views:
                        hidden = block_view
                        for layer_idx, layer in self._layer_groups:
                            hidden = layer(hidden, kv_cache, request_entries[layer_idx], pos)

                    kv_cache.free(request_id)
        self.output = hidden[:, -1:, :] if hidden is not None else None

    def capture_verification_payload(self) -> None:
        if self.layers is None or self._verify_input is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=self._payload_parameter_count,
            precision_flags={"fp16": True, "bf16": False, "fp8": False, "tf32": False},
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        """Cleanup."""
        self._input_block_views = []
        self._request_block_groups = []
        self._request_group_counts = (0, 0, 0)
        self._expected_request_group_counts = (0, 0, 0)
        self._layer_groups = []
        del self.layers, self.kv_cache, self.inputs
        self.request_ids = []
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_ai_optimization_metrics
        return compute_ai_optimization_metrics(
            original_time_ms=None,
            ai_optimized_time_ms=getattr(self, '_last_elapsed_ms', None),
            suggestions_applied=None,
            suggestions_total=None,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.layers is None:
            return "Model layers not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedIntegratedKVCacheBenchmark()
