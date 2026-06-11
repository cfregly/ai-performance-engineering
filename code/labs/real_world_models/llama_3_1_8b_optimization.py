#!/usr/bin/env python3
"""Real-world case study: Llama 3.1 8B optimization for Blackwell.

Demonstrates end-to-end optimization of Llama 3.1 8B:
- torch.compile with Blackwell-friendly settings
- Preferred SDPA backends (TE/Flash) via sdpa_kernel
- Avoids naive materialized attention when optimized
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional

from core.harness.arch_config import prefer_sdpa_backends
from core.utils.compile_utils import compile_model
from core.utils.logger import get_logger

logger = get_logger(__name__)


class Llama31_8B_Optimization:
    """Llama 3.1 8B optimization benchmark."""
    
    # Model specifications
    HIDDEN_SIZE = 4096
    NUM_HEADS = 32
    NUM_LAYERS = 32
    VOCAB_SIZE = 128256
    INTERMEDIATE_SIZE = 14336
    
    def __init__(
        self,
        batch_size: int = 1,
        seq_length: int = 2048,
        use_compile: bool = True,
        use_fp8: bool = False,
        use_flex_attention: bool = True,
        prefer_sdpa: bool = True,
    ):
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_compile = use_compile
        self.use_fp8 = use_fp8
        self.use_flex_attention = use_flex_attention
        self.prefer_sdpa = prefer_sdpa
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output: Optional[torch.Tensor] = None
        self.model: Optional[nn.Module] = None
        
        logger.info("Llama 3.1 8B Optimization")
        logger.info("  Compile: %s", use_compile)
        logger.info("  FP8: %s", use_fp8)
        logger.info("  Fast attention (SDPA): %s", use_flex_attention)
        logger.info("  Prefer SDPA backends: %s", prefer_sdpa)
    
    def _create_attention_layer(self):
        """Create attention layer (simplified for benchmark)."""
        use_fast_attention = bool(self.use_flex_attention)
        prefer_sdpa = bool(self.prefer_sdpa)

        class SimplifiedAttention(nn.Module):
            def __init__(self, hidden_size: int, num_heads: int, seq_len: int):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = hidden_size // num_heads
                self._scale = 1.0 / (self.head_dim ** 0.5)
                
                self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

                # For the naive baseline attention path, precompute a causal mask once
                # to avoid measuring mask materialization overhead.
                causal = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
                self.register_buffer("_causal_mask", causal, persistent=False)
            
            def forward(self, x):
                B, T, C = x.shape
                if T != self._causal_mask.shape[0]:
                    raise RuntimeError(
                        f"Unexpected sequence length: got T={T}, expected {self._causal_mask.shape[0]}"
                    )
                
                q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                
                if use_fast_attention:
                    cm = prefer_sdpa_backends() if prefer_sdpa else None
                    if cm is None:
                        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    else:
                        with cm:
                            attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                else:
                    # NAIVE baseline attention: materializes [B, H, T, T] and uses explicit softmax.
                    q_fp32 = q.float()
                    k_fp32 = k.float()
                    v_fp32 = v.float()
                    scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) * self._scale
                    scores = scores.masked_fill(self._causal_mask, float("-inf"))
                    probs = torch.softmax(scores, dim=-1)
                    attn = torch.matmul(probs, v_fp32).to(dtype=q.dtype)
                
                attn = attn.transpose(1, 2).contiguous().view(B, T, C)
                return self.o_proj(attn)
        
        return SimplifiedAttention(self.HIDDEN_SIZE, self.NUM_HEADS, self.seq_length)
    
    def _create_mlp_layer(self):
        """Create MLP layer (SwiGLU)."""
        class SimplifiedMLP(nn.Module):
            def __init__(self, hidden_size, intermediate_size):
                super().__init__()
                self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            
            def forward(self, x):
                return self.down_proj(nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))
        
        return SimplifiedMLP(self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE)
    
    def setup(self):
        """Initialize Llama 3.1 8B model (simplified)."""
        if self.use_fp8:
            raise RuntimeError("SKIPPED: use_fp8 is not implemented for this benchmark yet.")

        class SimplifiedLlamaLayer(nn.Module):
            def __init__(self, attention, mlp):
                super().__init__()
                self.attention = attention
                self.mlp = mlp
                self.input_layernorm = nn.RMSNorm(Llama31_8B_Optimization.HIDDEN_SIZE)
                self.post_attention_layernorm = nn.RMSNorm(Llama31_8B_Optimization.HIDDEN_SIZE)
            
            def forward(self, x):
                # Attention with residual
                h = x + self.attention(self.input_layernorm(x))
                # MLP with residual
                out = h + self.mlp(self.post_attention_layernorm(h))
                return out

        class LayerStack(nn.Module):
            def __init__(self, layers: nn.ModuleList):
                super().__init__()
                self.layers = layers

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.layers:
                    x = layer(x)
                return x
        
        # Create simplified model (just a few layers for benchmarking)
        self.layers = nn.ModuleList([
            SimplifiedLlamaLayer(
                self._create_attention_layer(),
                self._create_mlp_layer()
            )
            for _ in range(4)  # Use 4 layers for faster benchmarking
        ]).to(self.device).to(torch.bfloat16).eval()
        
        self.model = LayerStack(self.layers)
        self.model.eval()

        # Apply torch.compile if requested. (The old sm_103 fallback to "default"
        # is retired: the tcgen05.wait.st abort it dodged was caused by the
        # sm_103a de-suffix in core/benchmark/triton_compat.py, fixed 2026-06-11;
        # max-autotune re-verified clean on GB300 / Triton 3.7.)
        if self.use_compile:
            self.model = compile_model(
                self.model,
                mode="max-autotune",
                fullgraph=False,
                dynamic=False,
            )
        
        # Create input
        self.input = torch.randn(
            self.batch_size,
            self.seq_length,
            self.HIDDEN_SIZE,
            device=self.device,
            dtype=torch.bfloat16
        )
        self.output = None
        
        logger.info(f"Model setup complete: {self.seq_length} tokens")
    
    def run(self) -> None:
        """Execute forward pass (timed by the harness)."""
        if self.model is None:
            raise RuntimeError("Model not initialized (call setup() first)")
        with torch.inference_mode():
            self.output = self.model(self.input)
    
    def cleanup(self):
        """Clean up resources."""
        self.model = None
        self.layers = None
        self.input = None
        torch.cuda.empty_cache()

    # Harness adapter
    def teardown(self):
        """Alias required by harness callers."""
        self.cleanup()


def run_benchmark(
    batch_size: int = 1,
    seq_length: int = 2048,
    use_compile: bool = True,
    use_fp8: bool = False,
    use_flex_attention: bool = True,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run Llama 3.1 8B optimization benchmark."""
    
    benchmark = Llama31_8B_Optimization(
        batch_size=batch_size,
        seq_length=seq_length,
        use_compile=use_compile,
        use_fp8=use_fp8,
        use_flex_attention=use_flex_attention,
    )
    benchmark.setup()
    benchmark.run()
    benchmark.cleanup()
    return {"seq_length": seq_length, "optimizations": {"compile": use_compile, "fp8": use_fp8, "flex_attention": use_flex_attention}}
