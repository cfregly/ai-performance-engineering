"""Optimized dense attention - Flash Attention via SDPA.

This optimized version uses scaled_dot_product_attention which leverages
Flash Attention for:
- O(n) memory instead of O(n²)
- Fused kernel (no intermediate materialization)
- Hardware-optimized attention computation

Compare with baseline_dense_attention_flash.py which uses naive explicit attention.
Expected speedup: 5-20x depending on sequence length.
"""

from __future__ import annotations

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.common.device_utils import require_cuda_device
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

resolve_device = partial(require_cuda_device, "CUDA required for ch16")


class OptimizedDenseAttentionFlashBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Flash Attention via SDPA for dense causal attention."""
    
    def __init__(self):
        super().__init__()
        self.device = resolve_device()
        self.qkv_proj: Optional[nn.Linear] = None
        self.out_proj: Optional[nn.Linear] = None
        self.inputs: Optional[torch.Tensor] = None
        self._qkv_buffer: Optional[torch.Tensor] = None
        self._attn_merge_buffer: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._qkv_weight_t: Optional[torch.Tensor] = None
        self._out_proj_weight_t: Optional[torch.Tensor] = None
        self.batch_size = 4
        self.max_seq_len = 4096  # Same as baseline
        self.hidden_dim = 1024
        self.num_heads = 16
        self.head_dim = self.hidden_dim // self.num_heads
        self.dtype = torch.float16
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._enable_nvtx = False
        self._payload_parameter_count = 0
        
        tokens = self.batch_size * self.max_seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
    
    def setup(self) -> None:
        """Setup: Initialize Flash Attention model."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        
        self.qkv_proj = nn.Linear(
            self.hidden_dim,
            self.hidden_dim * 3,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )
        self.out_proj = nn.Linear(
            self.hidden_dim,
            self.hidden_dim,
            bias=False,
            device=self.device,
            dtype=self.dtype,
        )
        self._qkv_weight_t = self.qkv_proj.weight.t()
        self._out_proj_weight_t = self.out_proj.weight.t()
        self.inputs = torch.randn(
            self.batch_size,
            self.max_seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self._verify_input = self.inputs.detach().clone()
        self._qkv_buffer = torch.empty(
            self.batch_size,
            self.max_seq_len,
            self.hidden_dim * 3,
            device=self.device,
            dtype=self.dtype,
        )
        self._attn_merge_buffer = torch.empty(
            self.batch_size,
            self.max_seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self._output_buffer = torch.empty(
            self.batch_size,
            self.max_seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self._verify_output_buffer = torch.empty(
            self.batch_size,
            self.max_seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=self.dtype,
        )
        self._payload_parameter_count = sum(p.numel() for p in self.qkv_proj.parameters()) + sum(
            p.numel() for p in self.out_proj.parameters()
        )
        
        # Proper warmup
        for _ in range(5):
            with torch.inference_mode():
                self._forward_flash()
        self.register_workload_metadata(
            tokens_per_iteration=float(self.batch_size * self.max_seq_len),
            requests_per_iteration=float(self.batch_size),
        )

    def _forward_flash(self):
        """Flash Attention via SDPA - O(n) memory, fused kernel."""
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
        
        # Reshape for attention
        B, S, _ = q.shape
        q = q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Flash Attention via SDPA - O(n) memory, no S×S matrix!
        output = F.scaled_dot_product_attention(
            q, k, v,
            is_causal=True,
            dropout_p=0.0
        )
        
        # Output projection
        self._attn_merge_buffer.copy_(output.transpose(1, 2))
        return torch.matmul(self._attn_merge_buffer, self._out_proj_weight_t, out=self._output_buffer)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Flash Attention."""
        with nvtx_range("optimized_dense_attention_flash", enable=self._enable_nvtx):
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
        """Teardown: Clean up resources."""
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
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=5,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return None

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.qkv_proj is None or self.inputs is None:
            return "Model not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedDenseAttentionFlashBenchmark()
