"""optimized_kv_cache_naive_pool.py - KV cache pooling (optimized).

Pairs with: baseline_kv_cache_naive.py

Optimization: reuse preallocated KV buffers across requests to avoid per-request
allocations and reduce allocator/fragmentation overhead while keeping the model
math identical.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ch13.kv_cache_workload import get_workload
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

WORKLOAD = get_workload()


class OptimizedKVCache:
    """Optimized KV cache - efficient memory management."""
    
    def __init__(
        self,
        max_seq_len: int,
        batch_size: int,
        num_layers: int,
        num_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.cache_pool = []
        self.allocated_caches = {}
        # The benchmark processes requests sequentially (single in-flight request),
        # so a small pool demonstrates reuse without inflating the allocator's
        # reserved footprint.
        pool_size = 2
        for _ in range(pool_size):
            cache_entry = []
            for _ in range(self.num_layers):
                # Store in SDPA layout: [batch, heads, seq, dim] to avoid permutes.
                k = torch.empty(
                    self.batch_size,
                    self.num_heads,
                    self.max_seq_len,
                    self.head_dim,
                    dtype=self.dtype,
                    device=self.device,
                )
                v = torch.empty_like(k)
                cache_entry.append((k, v))
            self.cache_pool.append(cache_entry)
        self.free_indices = list(range(pool_size))

    def allocate(self, request_id: str) -> None:
        if request_id in self.allocated_caches:
            return
        if self.free_indices:
            cache_idx = self.free_indices.pop()
            self.allocated_caches[request_id] = cache_idx
        else:
            raise RuntimeError("OptimizedKVCache pool exhausted; increase pool_size in setup instead of allocating in benchmark_fn()")

    def append(self, request_id: str, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos: int) -> None:
        if request_id not in self.allocated_caches:
            self.allocate(request_id)
        cache_idx = self.allocated_caches[request_id]
        cache_k, cache_v = self.cache_pool[cache_idx][layer_idx]
        cache_k[:, :, pos, :].copy_(k)
        cache_v[:, :, pos, :].copy_(v)

    def get(self, request_id: str, layer_idx: int, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache_idx = self.allocated_caches[request_id]
        cache_k, cache_v = self.cache_pool[cache_idx][layer_idx]
        return cache_k[:, :, start:end, :], cache_v[:, :, start:end, :]

    def free(self, request_id: str) -> None:
        if request_id in self.allocated_caches:
            cache_idx = self.allocated_caches[request_id]
            # Reuse buffers without clearing: decode only reads the prefix that
            # was written for the current request (0..cache_pos), so stale tail
            # values are never observed. Clearing here is pure overhead.
            self.free_indices.append(cache_idx)
            del self.allocated_caches[request_id]


class SimpleAttentionLayer(nn.Module):
    """Simple attention layer for KV cache demo."""
    
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.output = None
        self._verify_input = None
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
    
    def forward(self, x: torch.Tensor, kv_cache: OptimizedKVCache, request_id: str, layer_idx: int, cache_pos: int) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self._project_qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Store the current token KV and attend over the cached prefix (decode-style).
        kv_cache.append(request_id, layer_idx, k[:, :, 0, :], v[:, :, 0, :], cache_pos)

        cached_k, cached_v = kv_cache.get(request_id, layer_idx, 0, cache_pos + 1)

        out = torch.nn.functional.scaled_dot_product_attention(q, cached_k, cached_v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2)
        if seq_len == 1:
            out = out.view(batch_size, seq_len, hidden_dim)
        else:
            out = out.contiguous().view(batch_size, seq_len, hidden_dim)
        return self.proj(out)


class OptimizedKVCacheNaivePoolBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized KV cache: pooled preallocation + reuse."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.kv_cache: Optional[OptimizedKVCache] = None
        self.inputs: Optional[list[torch.Tensor]] = None
        self._input_count = 0
        self._request_ids: list[str] = []
        self._input_token_views: list[list[torch.Tensor]] = []
        self._input_token_steps: list[list[tuple[int, torch.Tensor]]] = []
        self._request_token_groups: list[tuple[str, list[tuple[int, torch.Tensor]]]] = []
        self._request_group_counts: tuple[int, int, int] = (0, 0, 0)
        self._expected_request_group_counts: tuple[int, int, int] = (0, 0, 0)
        self._layer_groups: list[tuple[int, nn.Module]] = []
        self.output: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.workload = WORKLOAD
        self.num_layers = self.workload.num_layers
        self.num_heads = self.workload.num_heads
        self.head_dim = self.workload.head_dim
        self.hidden_dim = self.workload.hidden_dim
        self.batch_size = self.workload.batch_size
        self.max_seq_len = self.workload.max_seq_len
        self.sequence_lengths = list(self.workload.lengths())
        total_tokens = self.batch_size * sum(self.sequence_lengths)
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(len(self.sequence_lengths)),
            tokens_per_iteration=float(total_tokens),
        )
        self.parameter_count: int = 0
        self.register_workload_metadata(
            requests_per_iteration=float(len(self.sequence_lengths)),
            tokens_per_iteration=float(total_tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        self.model = nn.ModuleList(
            [
                SimpleAttentionLayer(self.hidden_dim, self.num_heads, self.head_dim, dtype=self.workload.dtype)
                for _ in range(self.num_layers)
            ]
        ).to(self.device).eval()
        for layer in self.model:
            layer.prepare_inference()
        self._layer_groups = list(enumerate(self.model))
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.kv_cache = OptimizedKVCache(
            max_seq_len=self.max_seq_len,
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
        self._request_token_groups = list(zip(self._request_ids, self._input_token_steps, strict=True))
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
        if self.model is None or self.kv_cache is None or self.inputs is None:
            raise RuntimeError("Benchmark not configured")
        if self._request_group_counts != self._expected_request_group_counts:
            raise RuntimeError("Request token groups not initialized")
        if not self._layer_groups:
            raise RuntimeError("Layer groups not initialized")

        with torch.inference_mode(), self._nvtx_range("kv_cache_naive_pool"):
            for request_id, token_steps in self._request_token_groups:
                self.kv_cache.allocate(request_id)

                for pos, token_view in token_steps:
                    hidden = token_view
                    for layer_idx, layer in self._layer_groups:
                        hidden = layer(hidden, self.kv_cache, request_id, layer_idx, pos)

                self.kv_cache.free(request_id)
        # Capture output from the final forward for verification
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
                "fp16": self.workload.dtype == torch.float16,
                "bf16": self.workload.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(5e-2, 5e-1),
        )

    def teardown(self) -> None:
        self.model = None
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
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
            measurement_timeout_seconds=300,
            warmup_timeout_seconds=120,
            setup_timeout_seconds=120,
            timeout_multiplier=1.0,
            ncu_replay_mode="application",
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

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        if self.output is None:
            return "Output not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedKVCacheNaivePoolBenchmark()
