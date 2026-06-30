"""optimized_kv_cache_management.py - KV cache decode with reuse (optimized)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin


class OptimizedKVCacheManagementBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: reuse projected K/V across decode steps (KV cache)."""
    
    def __init__(self):
        super().__init__()
        self.q_proj: Optional[nn.Linear] = None
        self.k_proj: Optional[nn.Linear] = None
        self.v_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None
        self.tokens: Optional[torch.Tensor] = None
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None
        self._q_proj_weight_t: Optional[torch.Tensor] = None
        self._k_proj_weight_t: Optional[torch.Tensor] = None
        self._v_proj_weight_t: Optional[torch.Tensor] = None
        self._out_proj_weight_t: Optional[torch.Tensor] = None
        # Match baseline batch_size for fair comparison
        self.batch_size = 64
        # Use a moderately large hidden dim so K/V projection reuse is measurable.
        self.hidden_dim = 1024
        self.num_heads = 8
        self.head_dim = self.hidden_dim // self.num_heads
        self.steps = 256
        tokens = self.batch_size * self.steps
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._q_buffer: Optional[torch.Tensor] = None
        self._tokens_2d: Optional[torch.Tensor] = None
        self._k_cache_2d: Optional[torch.Tensor] = None
        self._v_cache_2d: Optional[torch.Tensor] = None
        self._q_attn_view: Optional[torch.Tensor] = None
        self._token_step_views: list[torch.Tensor] = []
        self._k_prefix_views: list[torch.Tensor] = []
        self._v_prefix_views: list[torch.Tensor] = []
        self._k_attn_views: list[torch.Tensor] = []
        self._v_attn_views: list[torch.Tensor] = []
        self._output_step_views: list[torch.Tensor] = []
        self._decode_step_groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._view_counts: tuple[int, ...] = ()
        self._expected_view_counts: tuple[int, ...] = ()
        self._verify_input: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.q_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False).to(self.device, dtype=torch.bfloat16)
        for module in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            module.eval()
        self._q_proj_weight_t = self.q_proj.weight.t()
        self._k_proj_weight_t = self.k_proj.weight.t()
        self._v_proj_weight_t = self.v_proj.weight.t()
        self._out_proj_weight_t = self.out_proj.weight.t()
        self._payload_parameter_count = sum(
            p.numel()
            for layer in (self.q_proj, self.k_proj, self.v_proj, self.out_proj)
            for p in layer.parameters()
        )

        self.tokens = torch.randn(
            self.batch_size,
            self.steps,
            self.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._tokens_2d = self.tokens.reshape(self.batch_size * self.steps, self.hidden_dim)
        self.k_cache = torch.empty(
            self.batch_size,
            self.steps,
            self.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.v_cache = torch.empty(
            self.batch_size,
            self.steps,
            self.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._k_cache_2d = self.k_cache.reshape(self.batch_size * self.steps, self.hidden_dim)
        self._v_cache_2d = self.v_cache.reshape(self.batch_size * self.steps, self.hidden_dim)
        self._output_buffer = torch.empty(
            self.batch_size,
            self.steps,
            self.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._q_buffer = torch.empty(
            self.batch_size,
            self.hidden_dim,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self._q_attn_view = self._q_buffer.view(
            self.batch_size,
            1,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        self._token_step_views = [self.tokens[:, t, :] for t in range(self.steps)]
        self._k_prefix_views = [self.k_cache[:, : t + 1, :] for t in range(self.steps)]
        self._v_prefix_views = [self.v_cache[:, : t + 1, :] for t in range(self.steps)]
        self._k_attn_views = [
            prefix.view(self.batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            for prefix in self._k_prefix_views
        ]
        self._v_attn_views = [
            prefix.view(self.batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
            for prefix in self._v_prefix_views
        ]
        self._output_step_views = [self._output_buffer[:, t, :] for t in range(self.steps)]
        self._decode_step_groups = list(
            zip(
                self._token_step_views,
                self._k_attn_views,
                self._v_attn_views,
                self._output_step_views,
                strict=True,
            )
        )
        self._view_counts = (
            len(self._token_step_views),
            len(self._k_prefix_views),
            len(self._v_prefix_views),
            len(self._k_attn_views),
            len(self._v_attn_views),
            len(self._output_step_views),
            len(self._decode_step_groups),
        )
        self._expected_view_counts = (
            self.steps,
            self.steps,
            self.steps,
            self.steps,
            self.steps,
            self.steps,
            self.steps,
        )
        self._synchronize()
        self._verify_input = self.tokens.detach()
    
    def benchmark_fn(self) -> None:
        assert self.q_proj is not None and self.k_proj is not None and self.v_proj is not None and self.out_proj is not None
        assert self.tokens is not None and self.k_cache is not None and self.v_cache is not None
        assert self._output_buffer is not None
        assert self._q_buffer is not None and self._q_attn_view is not None and self._tokens_2d is not None
        assert self._k_cache_2d is not None and self._v_cache_2d is not None
        assert self._q_proj_weight_t is not None
        assert self._k_proj_weight_t is not None
        assert self._v_proj_weight_t is not None
        assert self._out_proj_weight_t is not None
        assert self._view_counts == self._expected_view_counts
        with self._nvtx_range("optimized_kv_cache_management"):
            with torch.inference_mode():
                # Model "prefill-produced" KV cache: project the full token buffer once,
                # then reuse those projected tensors across the decode loop.
                torch.mm(self._tokens_2d, self._k_proj_weight_t, out=self._k_cache_2d)
                torch.mm(self._tokens_2d, self._v_proj_weight_t, out=self._v_cache_2d)

                outputs = self._output_buffer
                q = self._q_attn_view
                for query, k, v, output_step in self._decode_step_groups:
                    torch.mm(query, self._q_proj_weight_t, out=self._q_buffer)

                    # q_len=1 and k/v contain only the prefix (no future tokens),
                    # so a causal mask is unnecessary here; is_causal=True would
                    # incorrectly mask all but the first key.
                    attn = F.scaled_dot_product_attention(q, k, v, is_causal=False)
                    attn = attn[:, :, 0, :].reshape(self.batch_size, self.hidden_dim)
                    torch.mm(attn, self._out_proj_weight_t, out=output_step)

                self.output = outputs

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
        self.tokens = None
        self.k_cache = None
        self.v_cache = None
        self._q_proj_weight_t = None
        self._k_proj_weight_t = None
        self._v_proj_weight_t = None
        self._out_proj_weight_t = None
        self._output_buffer = None
        self._q_buffer = None
        self._tokens_2d = None
        self._k_cache_2d = None
        self._v_cache_2d = None
        self._q_attn_view = None
        self._token_step_views = []
        self._k_prefix_views = []
        self._v_prefix_views = []
        self._k_attn_views = []
        self._v_attn_views = []
        self._output_step_views = []
        self._decode_step_groups = []
        self._view_counts = ()
        self._expected_view_counts = ()
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
        if self.tokens is None or self.k_cache is None or self.v_cache is None:
            return "Tokens/cache not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedKVCacheManagementBenchmark()
