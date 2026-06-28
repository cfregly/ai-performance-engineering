"""kv_cache_management_math.py - Math-only KV cache management.

This is a chapter utility variant (NOT a baseline/optimized comparable benchmark).
It forces the math SDP backend for environments where flash/mem-efficient kernels
are unavailable, so it is expected to be slower.
"""

from __future__ import annotations

from typing import Optional
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older PyTorch fallback
    SDPBackend = None  # type: ignore[assignment]
    sdpa_kernel = None  # type: ignore[assignment]

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin


def _math_sdp_context():
    """Prefer the new sdpa_kernel API; fall back to no-op if unavailable."""
    if sdpa_kernel is None or SDPBackend is None:
        return nullcontext()
    backend = getattr(SDPBackend, "MATH", None)
    if backend is None:
        return nullcontext()
    return sdpa_kernel([backend])


class KVCacheManagementMathBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Math-only SDP variant to avoid flash-attn kernel requirements."""
    
    def __init__(self):
        super().__init__()
        self.q_proj: Optional[nn.Linear] = None
        self.k_proj: Optional[nn.Linear] = None
        self.v_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None
        self.inputs: Optional[list[torch.Tensor]] = None
        self._sequence_inputs: Optional[torch.Tensor] = None
        self.cache_buffer: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.batch_size = 8
        self.hidden_dim = 256
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
        self.steps = 32
        tokens = self.batch_size * self.steps
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self._verify_input: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0
    
    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            module.eval()
        self._payload_parameter_count = sum(
            p.numel()
            for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj)
            for p in layer.parameters()
        )
        
        self.cache_buffer = torch.empty(self.batch_size, self.steps, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
        self.inputs = [
            torch.randn(self.batch_size, 1, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
            for _ in range(self.steps)
        ]
        self._sequence_inputs = torch.empty_like(self.cache_buffer)
        torch.cat(self.inputs, dim=1, out=self._sequence_inputs)
        self._synchronize()
        self._verify_input = self.inputs[0].detach()
    
    def benchmark_fn(self) -> None:
        assert self.q_proj is not None and self.k_proj is not None and self.v_proj is not None and self.out_proj is not None
        assert self.inputs is not None and self.cache_buffer is not None and self._sequence_inputs is not None
        with self._nvtx_range("kv_cache_management_math"):
            with torch.inference_mode():
                queries = self._sequence_inputs
                k_cache = self._sequence_inputs
                
                q = self.q_proj(queries)
                k = self.k_proj(k_cache)
                v = self.v_proj(k_cache)
                
                q = q.view(self.batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                k = k.view(self.batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                v = v.view(self.batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
                
                with _math_sdp_context():
                    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                attn = attn.transpose(1, 2).contiguous().view(self.batch_size, -1, self.hidden_dim)
                self.output = self.out_proj(attn)
                
                # Update cache with the newest token block without reallocation.
                self.cache_buffer.copy_(k_cache)
                _ = self.output[:, -1, :].sum()

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_input is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        if any(layer is None for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj)):
            raise RuntimeError("Projection layers not initialized")
        self._set_verification_payload(
            inputs={"tokens": self._verify_input},
            output=self.output,
            batch_size=int(self.batch_size),
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-3),
        )
    
    def teardown(self) -> None:
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.out_proj = None
        self.inputs = None
        self._sequence_inputs = None
        self.cache_buffer = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=None,
            tpot_ms=None,
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        if any(layer is None for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj)):
            return "Projection layers not initialized"
        if self.inputs is None or self.cache_buffer is None:
            return "Inputs/cache not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return KVCacheManagementMathBenchmark()
