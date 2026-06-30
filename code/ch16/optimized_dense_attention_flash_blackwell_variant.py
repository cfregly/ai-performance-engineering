"""Dense attention Flash SDPA variant tagged for Blackwell follow-up.

This uses scaled_dot_product_attention which leverages Flash Attention for:
- O(n) memory instead of O(n²)
- Fused kernel (no intermediate materialization)
- Hardware-optimized attention computation

Compare with baseline_dense_attention_flash.py which uses naive O(n²) attention.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range
from core.utils.logger import get_logger

logger = get_logger(__name__)


class DenseAttentionFlashBlackwellVariantBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Flash SDPA variant kept as an explicit non-canonical hardware variant."""

    story_metadata = {
        "pair_role": "variant",
        "chapter_alignment": "canonical",
        "chapter_native_exemplar": True,
        "variant_of": "dense_attention_flash",
        "variant_reason": "Hardware-specific dense-attention SDPA variant exposed separately from the canonical pair.",
    }
    
    def __init__(self):
        super().__init__()
        self.qkv_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None
        self.inputs: Optional[torch.Tensor] = None
        self._qkv_buffer: Optional[torch.Tensor] = None
        self._attn_merge_buffer: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._qkv_weight_t: Optional[torch.Tensor] = None
        self._out_proj_weight_t: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.batch_size = 4
        self.seq_length = 4096
        self.seq_len = self.seq_length
        self.hidden_dim = 1024
        self.num_heads = 16
        self.head_dim = self.hidden_dim // self.num_heads
        self.dtype = torch.float16
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._enable_nvtx = False
        self._payload_parameter_count = 0
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.seq_length),
        )
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)
    
    def setup(self) -> None:
        """Initialize Flash Attention model."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        
        device = torch.device("cuda")
        
        self.qkv_proj = nn.Linear(
            self.hidden_dim,
            self.hidden_dim * 3,
            bias=False,
            device=device,
            dtype=self.dtype,
        )
        self.out_proj = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
            device=device,
            dtype=self.dtype,
        )
        self._qkv_weight_t = self.qkv_proj.weight.t()
        self._out_proj_weight_t = self.out_proj.weight.t()
        self.inputs = torch.randn(
            self.batch_size,
            self.seq_length,
            self.hidden_dim,
            device=device,
            dtype=self.dtype,
        )
        self._verify_input = self.inputs.detach().clone()
        self._qkv_buffer = torch.empty(
            self.batch_size,
            self.seq_length,
            self.hidden_dim * 3,
            device=device,
            dtype=self.dtype,
        )
        self._attn_merge_buffer = torch.empty(
            self.batch_size,
            self.seq_length,
            self.hidden_dim,
            device=device,
            dtype=self.dtype,
        )
        self._output_buffer = torch.empty(
            self.batch_size,
            self.seq_length,
            self.hidden_dim,
            device=device,
            dtype=self.dtype,
        )
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            self.seq_length,
            self.hidden_dim,
            device=device,
            dtype=self.dtype,
        )
        self._payload_parameter_count = sum(p.numel() for p in self.qkv_proj.parameters()) + sum(
            p.numel() for p in self.out_proj.parameters()
        )
        
        # Proper warmup
        for _ in range(5):
            with torch.inference_mode():
                self._forward_flash()
    
    def _forward_flash(self):
        """Flash Attention via SDPA."""
        if (
            self.inputs is None
            or self.qkv_proj is None
            or self.out_proj is None
            or self._qkv_buffer is None
            or self._attn_merge_buffer is None
            or self._output_buffer is None
            or self._qkv_weight_t is None
            or self._out_proj_weight_t is None
        ):
            raise RuntimeError("Benchmark not configured")

        qkv = torch.matmul(self.inputs, self._qkv_weight_t, out=self._qkv_buffer)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        B, S, _ = q.shape
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Flash Attention
        output = F.scaled_dot_product_attention(
            q, k, v, 
            is_causal=True,
            dropout_p=0.0,
        )
        
        self._attn_merge_buffer.copy_(output.transpose(1, 2))
        return torch.matmul(self._attn_merge_buffer, self._out_proj_weight_t, out=self._output_buffer)
    
    def benchmark_fn(self) -> None:
        """Benchmark the Flash SDPA forward path for the Blackwell variant."""
        with nvtx_range("optimized_dense_attention_flash_blackwell_variant", enable=self._enable_nvtx):
            with torch.inference_mode():
                self.output = self._forward_flash()
        if self._verify_input is None:
            raise RuntimeError("Verification input missing")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_input is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        """Cleanup resources."""
        self.qkv_proj = None
        self.out_proj = None
        self.inputs = None
        self._qkv_buffer = None
        self._attn_merge_buffer = None
        self._output_buffer = None
        self._qkv_weight_t = None
        self._out_proj_weight_t = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_custom_metrics(self) -> Optional[dict]:
        return {
            "story.variant_pair": 1.0,
            "story.chapter_native_exemplar": 1.0,
            "dense_attention_flash.blackwell_variant": 1.0,
            "dense_attention_flash.seq_len": float(self.seq_len),
            "dense_attention_flash.num_heads": float(self.num_heads),
            "dense_attention_flash.head_dim": float(self.head_dim),
        }


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark harness discovery."""
    return DenseAttentionFlashBlackwellVariantBenchmark()
