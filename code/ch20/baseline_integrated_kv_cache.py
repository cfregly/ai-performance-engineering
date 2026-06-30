"""baseline_integrated_kv_cache.py - Baseline integrated KV cache (baseline).

Baseline KV cache integration in inference pipeline without optimizations.
Uses naive cache allocation and access patterns.

Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

from functools import partial
import torch
import torch.nn as nn


from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

resolve_device = partial(require_cuda_device, "CUDA required for ch20")


class NaiveKVCache:
    """Naive KV cache - simple allocation."""
    
    def __init__(self, max_seq_len: int, num_layers: int, num_heads: int, head_dim: int, dtype: torch.dtype, device: torch.device):
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device
        self.cache: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    
    def allocate(self, request_id: str) -> None:
        if request_id not in self.cache:
            self.cache[request_id] = []
            for _ in range(self.num_layers):
                k = torch.empty(self.max_seq_len, self.num_heads, self.head_dim, dtype=self.dtype, device=self.device)
                v = torch.empty_like(k)
                self.cache[request_id].append((k, v))
    
    def append(self, request_id: str, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos: int) -> None:
        if request_id not in self.cache:
            self.allocate(request_id)
        cache_k, cache_v = self.cache[request_id][layer_idx]
        cache_k[pos].copy_(k)
        cache_v[pos].copy_(v)
    
    def get(self, request_id: str, layer_idx: int, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        cache_k, cache_v = self.cache[request_id][layer_idx]
        return cache_k[start:end], cache_v[start:end]
    
    def free(self, request_id: str) -> None:
        if request_id in self.cache:
            del self.cache[request_id]


class AttentionLayer(nn.Module):
    """Attention layer with KV cache."""
    
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.output = None
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, dtype=dtype)
        self.proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
        self.register_buffer("_cache_touch", torch.empty((), dtype=dtype), persistent=False)
    
    def forward(self, x: torch.Tensor, kv_cache: NaiveKVCache, request_id: str, layer_idx: int, cache_pos: int) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        k_to_cache = k[:, :, 0, :].transpose(0, 1)
        v_to_cache = v[:, :, 0, :].transpose(0, 1)
        k_single = k_to_cache[:, 0, :]
        v_single = v_to_cache[:, 0, :]
        kv_cache.append(request_id, layer_idx, k_single, v_single, cache_pos)
        
        if cache_pos > 0:
            cached_k, cached_v = kv_cache.get(request_id, layer_idx, 0, cache_pos)
            # Touch cached tensors to simulate reads without rebuilding large attention matrices
            torch.sum(cached_k, dim=(0, 1, 2), out=self._cache_touch)
            torch.sum(cached_v, dim=(0, 1, 2), out=self._cache_touch)
        
        return self.proj(x)


class BaselineIntegratedKVCacheBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline integrated KV cache - naive implementation."""
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.model = None
        self.kv_cache = None
        self.inputs = None
        self.request_ids: list[str] = []
        self._request_token_groups: list[tuple[str, list[tuple[int, torch.Tensor]]]] = []
        self._layer_groups: list[tuple[int, nn.Module]] = []
        self.output: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self.max_seq_len = 8192
        self.num_layers = 2
        self.num_heads = 2
        self.head_dim = 32
        self.hidden_dim = self.num_heads * self.head_dim
        self.batch_size = 1
        self.sequence_lengths = [512, 1024, 2048]
        self.register_workload_metadata(requests_per_iteration=1.0)
        self._payload_parameter_count = 0
        self._enable_nvtx = False
    
    def setup(self) -> None:
        """Setup: Initialize baseline model with naive KV cache."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        dtype = torch.float16
        
        layers = []
        for _ in range(self.num_layers):
            layers.append(AttentionLayer(self.hidden_dim, self.num_heads, self.head_dim, dtype=dtype))
        self.model = nn.Sequential(*layers).to(self.device).eval()
        self._layer_groups = list(enumerate(self.model))
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.kv_cache = NaiveKVCache(
            max_seq_len=self.max_seq_len,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dtype=dtype,
            device=self.device
        )
        
        self.inputs = []
        self.request_ids = []
        for seq_len in self.sequence_lengths:
            x = torch.randn(self.batch_size, seq_len, self.hidden_dim, device=self.device, dtype=dtype)
            self.inputs.append(x)
            self.request_ids.append(f"req_{len(self.request_ids)}")
        for request_id in self.request_ids:
            self.kv_cache.allocate(request_id)
        self._request_token_groups = [
            (
                request_id,
                [
                    (pos, x[:, pos : pos + 1, :])
                    for pos in range(x.size(1))
                ],
            )
            for request_id, x in zip(self.request_ids, self.inputs, strict=True)
        ]
        self._verify_input = self.inputs[-1] if self.inputs else None
        self.output = None
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - baseline integrated KV cache."""
        with torch.inference_mode():
            with nvtx_range("baseline_integrated_kv_cache", enable=self._enable_nvtx):
                if not self._request_token_groups or not self._layer_groups:
                    raise RuntimeError("setup() must precompute request and layer groups")
                for request_id, token_views in self._request_token_groups:
                    for pos, token in token_views:
                        hidden = token
                        for layer_idx, layer in self._layer_groups:
                            hidden = layer(hidden, self.kv_cache, request_id, layer_idx, pos)
        # Capture last hidden state for verification (no cloning in the hot path).
        self.output = hidden if hidden is not None else None

    def capture_verification_payload(self) -> None:
        if self.model is None or self._verify_input is None or self.output is None:
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
        del self.model, self.kv_cache, self.inputs
        self.request_ids = []
        self._request_token_groups = []
        self._layer_groups = []
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
            original_time_ms=getattr(self, '_last_elapsed_ms', None),
            ai_optimized_time_ms=None,
            suggestions_applied=None,
            suggestions_total=None,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        return super().get_verify_output()

    def get_input_signature(self) -> dict:
        return super().get_input_signature()


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineIntegratedKVCacheBenchmark()
