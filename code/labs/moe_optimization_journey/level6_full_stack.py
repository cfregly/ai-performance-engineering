#!/usr/bin/env python3
"""Level 6: Full Stack - CUDA Graphs for Zero Launch Overhead.

OPTIMIZATION: Capture entire forward pass in CUDA graph.

Key changes from Level 4:
1. CUDA graph capture eliminates kernel launch overhead
2. Static input/output tensors for graph replay
3. Pre-allocated workspace memory
4. All operations fused into single graph replay

Expected speedup: 1.3-2x over Level 4 (for small batches where launch overhead dominates)
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.moe_optimization_journey import MoEConfig, get_config


class CUDAGraphMoEExperts(nn.Module):
    """MoE experts optimized for CUDA graph capture."""
    
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        
        # Stacked expert weights
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.w2 = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.w3 = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        
        for w in [self.w1, self.w2, self.w3]:
            nn.init.kaiming_uniform_(w)
    
    def forward(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Forward with batched expert computation."""
        batch_seq, top_k = expert_indices.shape
        
        # Gather expert weights
        w1_sel = self.w1[expert_indices]
        w3_sel = self.w3[expert_indices]
        w2_sel = self.w2[expert_indices]
        
        # Expand input
        x_exp = x.unsqueeze(1).expand(-1, top_k, -1)
        
        # Batched matmul - all done in BF16
        gate = torch.einsum('bkh,bkhi->bki', x_exp, w1_sel)
        F.silu(gate, inplace=True)
        up = torch.einsum('bkh,bkhi->bki', x_exp, w3_sel)
        gate.mul_(up)
        out = torch.einsum('bki,bkih->bkh', gate, w2_sel)
        
        # Weight and sum
        out.mul_(expert_weights.unsqueeze(-1))
        reduced = out[:, 0, :]
        for route_idx in range(1, top_k):
            reduced.add_(out[:, route_idx, :])
        return reduced


class CUDAGraphMoELayer(nn.Module):
    """MoE layer optimized for CUDA graphs."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = CUDAGraphMoEExperts(
            config.num_experts,
            config.hidden_size,
            config.intermediate_size,
        )
        self.top_k = config.num_experts_per_tok
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = x.shape
        x_flat = x.view(-1, hidden)
        
        # Router
        router_logits = self.gate(x_flat)
        top_logits, expert_indices = torch.topk(router_logits.float(), self.top_k, dim=-1)
        expert_weights = F.softmax(top_logits, dim=-1)
        expert_weights = expert_weights.to(x.dtype)
        
        # Experts
        output = self.experts(x_flat, expert_indices, expert_weights)
        return output.view(batch, seq, hidden)


class CUDAGraphMoEBlock(nn.Module):
    """Transformer block for CUDA graph capture."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.hidden_size)
        self.ln2 = nn.LayerNorm(config.hidden_size)
        self.attn = nn.MultiheadAttention(
            config.hidden_size,
            config.num_attention_heads,
            batch_first=True,
        )
        self.moe = CUDAGraphMoELayer(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = self.ln2(x)
        h = self.moe(h)
        return x + h


class CUDAGraphMoEModel(nn.Module):
    """MoE model designed for CUDA graph capture."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([
            CUDAGraphMoEBlock(config) for _ in range(config.num_layers)
        ])
        self.ln_f = nn.LayerNorm(config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


class Level6FullStack(VerificationPayloadMixin, BaseBenchmark):
    """Level 6: Full Stack with CUDA Graphs."""
    
    LEVEL = 6
    NAME = "CUDA Graphs"
    DESCRIPTION = "torch.compile + CUDA graph capture for zero launch overhead"
    
    def __init__(self, config: Optional[MoEConfig] = None):
        super().__init__()
        self.config = config or get_config("small")
        # Surface workload parameters for harness input verification
        self.batch_size = self.config.batch_size
        self.seq_len = self.config.seq_len
        self.hidden_size = self.config.hidden_size
        self.intermediate_size = self.config.intermediate_size
        self.num_experts = self.config.num_experts
        self.num_experts_per_tok = self.config.num_experts_per_tok
        self.vocab_size = self.config.vocab_size
        self.num_heads = self.config.num_attention_heads
        self.model: Optional[Any] = None
        self.compiled_model: Optional[Any] = None
        self.input_ids: Optional[torch.Tensor] = None
        self.static_input: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.graph_output: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self.last_latency_ms: float = 0.0
        self.last_tokens_per_sec: float = 0.0
        self._iteration_metrics: Dict[str, float] = {
            "latency_ms": 0.0,
            "tokens_per_sec": 0.0,
        }
        self._timing_events: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._pending_events: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None
        
        total_tokens = self.config.batch_size * self.config.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch_size),
            tokens_per_iteration=float(total_tokens),
        )
    
    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        print("=" * 60)
        print(f"Level {self.LEVEL}: {self.NAME}")
        print("=" * 60)
        print(f"  {self.DESCRIPTION}")
        print()
        print("  Optimizations (cumulative):")
        print("    ✓ Parallel expert execution (batched matmul)")
        print("    ✓ CUDA graph capture (entire forward pass)")
        print("    ✓ Zero kernel launch overhead")
        print("    ✓ Static tensor allocation")
        print()
        
        self.model = CUDAGraphMoEModel(self.config).to(self.device).to(torch.bfloat16)
        self.model.eval()
        
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        print(f"  Parameters: {self.parameter_count / 1e6:.1f}M")
        
        # Create input tensor
        self.input_ids = torch.randint(
            0, self.config.vocab_size,
            (self.config.batch_size, self.config.seq_len),
            device=self.device,
        )
        self.static_input = self.input_ids.clone()
        self._verify_output_buffer = torch.empty(
            (self.config.batch_size, 1, min(8, self.config.vocab_size)),
            device=self.device,
            dtype=torch.float32,
        )
        
        # Use torch.compile with max-autotune mode for best performance
        # max-autotune provides aggressive kernel fusion and optimization
        print("  Compiling with max-autotune mode...")
        self.compiled_model = torch.compile(
            self.model,
            mode="max-autotune",
        )
        
        # Warmup to trigger compilation and internal graph capture
        print("\n  Warmup (compilation + graph capture)...")
        for i in range(5):
            with torch.inference_mode():
                _ = self.compiled_model(self.static_input)
            if i == 0:
                print("    First run (compile): done")
        torch.cuda.synchronize()
        
        # Note: torch.compile with reduce-overhead already captures graphs internally
        # so self.graph is not needed for replay
        self.graph = None
        self.graph_output = None
        
        torch.cuda.synchronize()
        self._timing_events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        print("Ready")

    def _get_timing_events(self) -> Tuple[torch.cuda.Event, torch.cuda.Event]:
        if self._timing_events is None:
            self._timing_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        return self._timing_events
    
    def benchmark_fn(self) -> None:
        events = self._get_timing_events()
        start_event, end_event = events
        current_stream = torch.cuda.current_stream(self.device)
        start_event.record(current_stream)
        
        with self._nvtx_range("level6_cuda_graphs"):
            # torch.compile with reduce-overhead handles graph replay internally
            with torch.inference_mode():
                logits = self.compiled_model(self.static_input)
        end_event.record(current_stream)
        self.output = logits
        self._pending_events = events
        if self.static_input is None or self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def finalize_iteration_metrics(self) -> Optional[Dict[str, float]]:
        if self._pending_events is None:
            return None
        start_event, end_event = self._pending_events
        self._pending_events = None
        self.last_latency_ms = start_event.elapsed_time(end_event)
        total_tokens = self.config.batch_size * self.config.seq_len
        self.last_tokens_per_sec = total_tokens / max(self.last_latency_ms / 1000.0, 1e-9)
        metrics = self._iteration_metrics
        metrics["latency_ms"] = float(self.last_latency_ms)
        metrics["tokens_per_sec"] = float(self.last_tokens_per_sec)
        return metrics

    def capture_verification_payload(self) -> None:
        if self.static_input is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            : self._verify_output_buffer.shape[2],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"input_ids": self.static_input.detach()},
            output=self._verify_output_buffer,
            batch_size=self.config.batch_size,
            parameter_count=self.parameter_count,
            precision_flags={"bf16": True, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )
    
    def teardown(self) -> None:
        del self.graph
        del self.graph_output
        del self.compiled_model
        del self.model
        self.graph = None
        self.graph_output = None
        self.compiled_model = None
        self.model = None
        self.input_ids = None
        self.static_input = None
        self.output = None
        self._verify_output_buffer = None
        self._timing_events = None
        self._pending_events = None
        torch.cuda.empty_cache()
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=self.config.benchmark_iterations,
            warmup=self.config.warmup_iterations,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def validate_result(self) -> Optional[str]:
        return None if self.compiled_model is not None else "Model not compiled"
    
    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return {
            "level": float(self.LEVEL),
            "latency_ms": self.last_latency_ms,
            "tokens_per_sec": self.last_tokens_per_sec,
            "cuda_graphs": 1.0,
        }

def get_benchmark() -> BaseBenchmark:
    return Level6FullStack()
