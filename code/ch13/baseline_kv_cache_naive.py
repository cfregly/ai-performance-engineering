"""baseline_kv_cache_naive.py - Naive KV cache baseline (baseline)."""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ch13.kv_cache_workload import get_workload
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

WORKLOAD = get_workload()


class NaiveKVCache:
    """Naive KV cache - simple dictionary-based storage."""
    
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
        self.cache = {}  # request_id -> list of (k, v) tensors per layer
    
    def allocate(self, request_id: str) -> None:
        """Allocate empty per-layer caches for a request.

        The naive baseline grows KV one token at a time via repeated `torch.cat`,
        which triggers many reallocations/copies and allocator fragmentation.
        """
        self.cache[request_id] = []
        for _ in range(self.num_layers):
            k = torch.empty(
                0,
                self.batch_size,
                self.num_heads,
                self.head_dim,
                dtype=self.dtype,
                device=self.device,
            )
            v = torch.empty_like(k)
            self.cache[request_id].append((k, v))
    
    def append(self, request_id: str, layer_idx: int, k: torch.Tensor, v: torch.Tensor, pos: int) -> None:
        """Append keys/values to cache."""
        if request_id not in self.cache:
            self.allocate(request_id)
        cache_k, cache_v = self.cache[request_id][layer_idx]
        if cache_k.size(0) != pos:
            raise RuntimeError(f"NaiveKVCache append position mismatch: expected {cache_k.size(0)}, got {pos}")
        cache_k = torch.cat([cache_k, k.unsqueeze(0)], dim=0)
        cache_v = torch.cat([cache_v, v.unsqueeze(0)], dim=0)
        self.cache[request_id][layer_idx] = (cache_k, cache_v)
    
    def get(self, request_id: str, layer_idx: int, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get cached keys/values."""
        cache_k, cache_v = self.cache[request_id][layer_idx]
        return cache_k[start:end], cache_v[start:end]
    
    def free(self, request_id: str) -> None:
        """Free cache for a sequence."""
        if request_id in self.cache:
            del self.cache[request_id]


class SimpleAttentionLayer(nn.Module):
    """Simple attention layer for KV cache demo."""
    
    def __init__(self, hidden_dim: int, num_heads: int, head_dim: int, dtype: torch.dtype = torch.float16):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, dtype=dtype)
        self.proj = nn.Linear(hidden_dim, hidden_dim, dtype=dtype)
    
    def forward(self, x: torch.Tensor, kv_cache: NaiveKVCache, request_id: str, layer_idx: int, cache_pos: int) -> torch.Tensor:
        batch_size, seq_len, hidden_dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Store the current token KV and attend over the cached prefix (decode-style).
        kv_cache.append(request_id, layer_idx, k[:, :, 0, :], v[:, :, 0, :], cache_pos)

        cached_k, cached_v = kv_cache.get(request_id, layer_idx, 0, cache_pos + 1)
        # Cache layout: [seq, batch, heads, dim] -> SDPA expects [batch, heads, seq, dim]
        cached_k = cached_k.permute(1, 2, 0, 3)
        cached_v = cached_v.permute(1, 2, 0, 3)

        out = F.scaled_dot_product_attention(q, cached_k, cached_v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_dim)
        return self.proj(out)


class BaselineKVCacheNaiveBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Naive KV cache baseline."""

    allowed_benchmark_fn_antipatterns = ("allocation",)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.kv_cache = None
        self.inputs: Optional[List[torch.Tensor]] = None
        self._input_token_views: list[list[torch.Tensor]] = []
        self._request_ids: list[str] = []
        self._input_token_steps: list[list[tuple[int, torch.Tensor]]] = []
        self._request_token_groups: list[tuple[str, list[tuple[int, torch.Tensor]]]] = []
        self._layer_groups: list[tuple[int, nn.Module]] = []
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
        self.output = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
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
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self._layer_groups = list(enumerate(self.model))
        
        self.kv_cache = NaiveKVCache(
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
        self._request_ids = [f"req_{seq_idx}" for seq_idx in range(len(self.inputs))]
        self._input_token_views = [
            [x[:, pos : pos + 1, :] for pos in range(x.size(1))]
            for x in self.inputs
        ]
        self._input_token_steps = [
            list(enumerate(token_views))
            for token_views in self._input_token_views
        ]
        self._request_token_groups = list(
            zip(self._request_ids, self._input_token_steps, strict=True)
        )
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            1,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float32,
        )
        if self.inputs:
            self._verify_input = self.inputs[0].detach().clone()
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        if self.model is None or self.kv_cache is None or self.inputs is None:
            raise RuntimeError("Benchmark not configured")
        if not self._request_token_groups or not self._layer_groups:
            raise RuntimeError("setup() must precompute request and layer groups")

        with torch.inference_mode(), self._nvtx_range("kv_cache_baseline_naive"):
            for request_id, token_steps in self._request_token_groups:
                self.kv_cache.allocate(request_id)

                for pos, token in token_steps:
                    for layer_idx, layer in self._layer_groups:
                        token = layer(token, self.kv_cache, request_id, layer_idx, pos)

                self.kv_cache.free(request_id)
            # Store last output for verification
            self.output = token
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
        self.model = None
        self.kv_cache = None
        self.inputs = None
        self._input_token_views = []
        self._input_token_steps = []
        self._request_ids = []
        self._request_token_groups = []
        self._layer_groups = []
        self.output = None
        self._verify_input = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()
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
            fp32_time_ms=getattr(self, '_last_elapsed_ms', None),
            reduced_precision_time_ms=None,
            precision_type="fp16",
        )

    def get_optimization_goal(self) -> str:
        """Naive-vs-paged KV cache is tracked as a memory benchmark."""
        return "memory"

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Model not initialized"
        return None

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (5e-2, 5e-1)


def get_benchmark() -> BaselineKVCacheNaiveBenchmark:
    """Factory function for harness discovery."""
    return BaselineKVCacheNaiveBenchmark()
