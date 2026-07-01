#!/usr/bin/env python3
"""Configurable MoE Model - 7-Level Optimization Journey.

This demonstrates the SAME optimizations that torch.compile does,
but implemented manually so you can understand each technique.

Level 0: Naive        - Python loops over experts
Level 1: Batched      - Einsum parallelizes all tokens  
Level 2: Fused        - Triton kernel fuses SiLU * up
Level 3: MemEfficient - Eliminate intermediate tensors
Level 4: Grouped      - Sort tokens + per-expert GEMM
Level 5: BMM Fused    - Vectorized scatter + single BMM
Level 6: CUDAGraphs   - Capture and replay the fused expert path
Level 7: Compiled     - torch.compile on top of the graph-friendly model
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from ch19.mxfp8_moe_common import (
    bucket_by_expert as bucket_grouped_tokens,
    restore_bucketed as restore_grouped_tokens,
)

# Try to import Triton kernels
try:
    from labs.moe_optimization_journey.triton_kernels import fused_silu_mul
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    def fused_silu_mul(gate, up):
        return F.silu(gate) * up


def _flat_topk_token_ids(num_tokens: int, top_k: int, device: torch.device) -> torch.Tensor:
    token_ids = torch.arange(num_tokens * top_k, device=device, dtype=torch.int64)
    if top_k > 1:
        token_ids.div_(top_k, rounding_mode="floor")
    return token_ids


def _weight_routes_in_place_if_safe(out: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and out.requires_grad:
        return out * weights
    out.mul_(weights)
    return out


def _sum_routes_in_place_if_safe(out: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and out.requires_grad:
        return out.sum(dim=1)
    reduced = out[:, 0, :]
    for route_idx in range(1, out.shape[1]):
        reduced.add_(out[:, route_idx, :])
    return reduced


def _silu_mul_in_place_if_safe(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    if torch.is_grad_enabled() and gate.requires_grad:
        return F.silu(gate) * up
    F.silu(gate, inplace=True)
    gate.mul_(up)
    return gate


@dataclass
class MoEOptimizations:
    """Optimization flags for MoE model."""
    use_batched: bool = False       # Level 1: Batched einsum
    use_fused: bool = False         # Level 2: Triton fused SiLU*up
    use_mem_efficient: bool = False # Level 3: Memory efficient (in-place)
    use_grouped: bool = False       # Level 4: Sorted + per-expert GEMM
    use_bmm_fused: bool = False     # Level 5: BMM fusion (vectorized scatter)
    use_cuda_graphs: bool = False   # Level 6: CUDA graph capture  
    use_compile: bool = False       # Level 7: torch.compile


class MoEExperts(nn.Module):
    """Expert module supporting 7 levels of optimization."""
    
    def __init__(self, num_experts: int, hidden_size: int, intermediate_size: int, opts: MoEOptimizations):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.opts = opts
        
        # Individual expert weights (for naive mode)
        self.experts = nn.ModuleList([
            nn.ModuleDict({
                'w1': nn.Linear(hidden_size, intermediate_size, bias=False),
                'w2': nn.Linear(intermediate_size, hidden_size, bias=False),
                'w3': nn.Linear(hidden_size, intermediate_size, bias=False),
            })
            for _ in range(num_experts)
        ])
        
        # Stacked weights for optimized modes
        self.w1_stacked = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))
        self.w2_stacked = nn.Parameter(torch.empty(num_experts, intermediate_size, hidden_size))
        self.w3_stacked = nn.Parameter(torch.empty(num_experts, hidden_size, intermediate_size))

        with torch.inference_mode():
            for idx, expert in enumerate(self.experts):
                self.w1_stacked[idx].copy_(expert["w1"].weight.t())
                self.w2_stacked[idx].copy_(expert["w2"].weight.t())
                self.w3_stacked[idx].copy_(expert["w3"].weight.t())
        
        # Pre-allocated buffers for memory-efficient mode
        self._naive_output: Optional[torch.Tensor] = None
        self._gate_buffer: Optional[torch.Tensor] = None
        self._up_buffer: Optional[torch.Tensor] = None
        self._mem_out_buffer: Optional[torch.Tensor] = None
        self._cuda_graph = None
        self._cuda_graph_stream: Optional[torch.cuda.Stream] = None
        self._cuda_graph_signature: Optional[Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], str, str, str, int]] = None
        self._cuda_graph_static_x: Optional[torch.Tensor] = None
        self._cuda_graph_static_expert_indices: Optional[torch.Tensor] = None
        self._cuda_graph_static_expert_weights: Optional[torch.Tensor] = None
        self._cuda_graph_output: Optional[torch.Tensor] = None
        self._cuda_graph_attempted = False
        self._cuda_graph_captured = False
        self._cuda_graph_replays = 0
        self._cuda_graph_capture_failures = 0
        self._cuda_graph_last_error: Optional[str] = None
        self._bmm_padded_tokens: Optional[torch.Tensor] = None
        self._bmm_padded_weights: Optional[torch.Tensor] = None
        self._bmm_valid_out: Optional[torch.Tensor] = None
        self._bmm_restored: Optional[torch.Tensor] = None
        self._graph_padded_tokens: Optional[torch.Tensor] = None
        self._grouped_output: Optional[torch.Tensor] = None
        self._grouped_restored: Optional[torch.Tensor] = None
        self._bmm_flat_token_ids_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}
        self._bmm_position_ids_cache: Dict[Tuple[int, torch.device], torch.Tensor] = {}
        self._bmm_padded_token_index_view_cache: Dict[Tuple[int, int, int, torch.device], torch.Tensor] = {}
        self._bmm_padded_column_index_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}
        self._bmm_sorted_weight_column_cache: Dict[Tuple[int, int, torch.device], torch.Tensor] = {}

    @staticmethod
    def _is_torch_compiling() -> bool:
        compiler = getattr(torch, "compiler", None)
        if compiler is not None and hasattr(compiler, "is_compiling"):
            return bool(compiler.is_compiling())
        dynamo = getattr(torch, "_dynamo", None)
        if dynamo is not None and hasattr(dynamo, "is_compiling"):
            return bool(dynamo.is_compiling())
        return False

    @staticmethod
    def _graph_signature(
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], str, str, str, int]:
        device_index = x.device.index if x.device.index is not None else -1
        return (
            tuple(x.shape),
            tuple(expert_indices.shape),
            tuple(expert_weights.shape),
            str(x.dtype),
            str(expert_indices.dtype),
            str(expert_weights.dtype),
            device_index,
        )

    def _bmm_workspace(
        self,
        name: str,
        shape: Tuple[int, ...],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        shape = tuple(int(dim) for dim in shape)
        numel = 1
        for dim in shape:
            numel *= dim
        cached = getattr(self, name, None)
        if (
            not isinstance(cached, torch.Tensor)
            or cached.device != device
            or cached.dtype != dtype
            or cached.numel() < numel
        ):
            cached = torch.empty(numel, device=device, dtype=dtype)
            setattr(self, name, cached)
        return cached[:numel].view(shape)

    def _naive_output_like(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and x.requires_grad:
            return torch.empty_like(x)
        shape = tuple(int(dim) for dim in x.shape)
        numel = int(x.numel())
        if (
            self._naive_output is None
            or self._naive_output.device != x.device
            or self._naive_output.dtype != x.dtype
            or self._naive_output.numel() < numel
        ):
            self._naive_output = torch.empty(numel, device=x.device, dtype=x.dtype)
        return self._naive_output[:numel].view(shape)

    def _flat_topk_token_ids_for(self, num_tokens: int, top_k: int, device: torch.device) -> torch.Tensor:
        key = (num_tokens, top_k, device)
        cached = self._bmm_flat_token_ids_cache.get(key)
        if cached is None:
            cached = _flat_topk_token_ids(num_tokens, top_k, device)
            self._bmm_flat_token_ids_cache[key] = cached
        return cached

    def _position_ids_for(self, length: int, device: torch.device) -> torch.Tensor:
        key = (length, device)
        cached = self._bmm_position_ids_cache.get(key)
        if cached is None:
            cached = torch.arange(length, device=device, dtype=torch.int64)
            self._bmm_position_ids_cache[key] = cached
        return cached

    def _padded_token_index_view(self, padded_indices: torch.Tensor, width: int) -> torch.Tensor:
        key = (int(padded_indices.data_ptr()), int(padded_indices.numel()), int(width), padded_indices.device)
        cached = self._bmm_padded_token_index_view_cache.get(key)
        expected_shape = (int(padded_indices.numel()), int(width))
        if cached is None or cached.device != padded_indices.device or tuple(cached.shape) != expected_shape:
            cached = padded_indices.unsqueeze(1).expand(-1, width)
            self._bmm_padded_token_index_view_cache[key] = cached
        return cached

    def _padded_column_index(self, padded_indices: torch.Tensor) -> torch.Tensor:
        key = (int(padded_indices.data_ptr()), int(padded_indices.numel()), padded_indices.device)
        cached = self._bmm_padded_column_index_cache.get(key)
        expected_shape = (int(padded_indices.numel()), 1)
        if cached is None or cached.device != padded_indices.device or tuple(cached.shape) != expected_shape:
            cached = padded_indices.unsqueeze(1)
            self._bmm_padded_column_index_cache[key] = cached
        return cached

    def _sorted_weight_column(self, sorted_weights: torch.Tensor) -> torch.Tensor:
        key = (int(sorted_weights.data_ptr()), int(sorted_weights.numel()), sorted_weights.device)
        cached = self._bmm_sorted_weight_column_cache.get(key)
        expected_shape = (int(sorted_weights.numel()), 1)
        if cached is None or cached.device != sorted_weights.device or tuple(cached.shape) != expected_shape:
            cached = sorted_weights.unsqueeze(1)
            self._bmm_sorted_weight_column_cache[key] = cached
        return cached

    def _reset_cuda_graph_cache(self) -> None:
        self._cuda_graph = None
        self._cuda_graph_stream = None
        self._cuda_graph_signature = None
        self._cuda_graph_static_x = None
        self._cuda_graph_static_expert_indices = None
        self._cuda_graph_static_expert_weights = None
        self._cuda_graph_output = None

    def get_cuda_graph_metrics(self) -> Dict[str, float]:
        return {
            "cuda_graph_attempted": float(self._cuda_graph_attempted),
            "cuda_graph_captured": float(self._cuda_graph_captured),
            "cuda_graph_fallback": float(self._cuda_graph_last_error is not None),
            "cuda_graph_capture_failures": float(self._cuda_graph_capture_failures),
            "cuda_graph_replays": float(self._cuda_graph_replays),
        }

    def _capture_cuda_graph(
        self,
        x: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor,
        signature: Tuple[Tuple[int, ...], Tuple[int, ...], Tuple[int, ...], str, str, str, int],
    ) -> None:
        self._reset_cuda_graph_cache()
        self._cuda_graph_attempted = True
        self._cuda_graph_last_error = None
        self._cuda_graph_signature = signature
        self._cuda_graph_static_x = x.detach().clone()
        self._cuda_graph_static_expert_indices = expert_indices.detach().clone()
        self._cuda_graph_static_expert_weights = expert_weights.detach().clone()

        capture_stream = torch.cuda.Stream(device=x.device)
        self._cuda_graph_stream = capture_stream
        current_stream = torch.cuda.current_stream(device=x.device)
        capture_stream.wait_stream(current_stream)

        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                self._forward_bmm_fused_graphable(
                    self._cuda_graph_static_x,
                    self._cuda_graph_static_expert_indices,
                    self._cuda_graph_static_expert_weights,
                )
        capture_stream.synchronize()
        current_stream.wait_stream(capture_stream)

        self._cuda_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._cuda_graph, stream=capture_stream):
            self._cuda_graph_output = self._forward_bmm_fused_graphable(
                self._cuda_graph_static_x,
                self._cuda_graph_static_expert_indices,
                self._cuda_graph_static_expert_weights,
            )
        capture_stream.synchronize()
        current_stream.wait_stream(capture_stream)
        self._cuda_graph_captured = True
    
    def forward_naive(
        self, x: torch.Tensor, expert_indices: torch.Tensor, 
        expert_weights: torch.Tensor, num_experts_per_tok: int,
    ) -> torch.Tensor:
        """Level 0: NAIVE - Python loops over experts.
        
        This is how you might write MoE naively:
        - Loop over each top-K selection
        - Loop over each expert
        - Compute expert output
        - Accumulate weighted results
        
        Problems: Python loop overhead, no parallelism, memory inefficient
        """
        output = self._naive_output_like(x)
        
        for k in range(num_experts_per_tok):
            for expert_idx in range(self.num_experts):
                token_ids = (expert_indices[:, k] == expert_idx).nonzero(as_tuple=True)[0]
                if token_ids.numel() == 0:
                    continue
                expert_input = x[token_ids]
                expert = self.experts[expert_idx]
                gate = expert['w1'](expert_input)
                up = expert['w3'](expert_input)
                hidden = _silu_mul_in_place_if_safe(gate, up)
                expert_output = expert['w2'](hidden)
                weights = expert_weights[token_ids, k].unsqueeze(-1)
                weighted_output = _weight_routes_in_place_if_safe(expert_output, weights)
                if k == 0:
                    output[token_ids] = weighted_output
                else:
                    output[token_ids] += weighted_output
        return output
    
    def forward_batched(
        self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Level 1: BATCHED - Batched GEMMs parallelize all tokens.
        
        Instead of looping, we:
        1. Gather weights for selected experts: [batch, top_k, ...]
        2. Use batched matmul for parallel expert compute
        3. Sum weighted results
        
        Speedup: ~12x (eliminates Python loops)
        """
        batch_seq, top_k = expert_indices.shape
        
        w1_sel = self.w1_stacked[expert_indices]
        w3_sel = self.w3_stacked[expert_indices]
        w2_sel = self.w2_stacked[expert_indices]
        
        x_exp = x.unsqueeze(1).expand(-1, top_k, -1)

        # Prefer explicit batched GEMMs over einsum; for these shapes einsum can pick
        # slower kernels and lose to the naive Python dispatch (Level 0).
        total_tokens = batch_seq * top_k
        x_flat = x_exp.reshape(total_tokens, self.hidden_size)
        w1_flat = w1_sel.reshape(total_tokens, self.hidden_size, self.intermediate_size)
        w3_flat = w3_sel.reshape(total_tokens, self.hidden_size, self.intermediate_size)
        w2_flat = w2_sel.reshape(total_tokens, self.intermediate_size, self.hidden_size)

        gate = torch.bmm(x_flat.unsqueeze(1), w1_flat).squeeze(1)
        up = torch.bmm(x_flat.unsqueeze(1), w3_flat).squeeze(1)
        hidden = _silu_mul_in_place_if_safe(gate, up)
        out = torch.bmm(hidden.unsqueeze(1), w2_flat).squeeze(1)
        out = out.view(batch_seq, top_k, self.hidden_size)

        out = _weight_routes_in_place_if_safe(out, expert_weights.unsqueeze(-1))
        return _sum_routes_in_place_if_safe(out)
    
    def forward_fused(
        self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Level 2: FUSED - Triton kernel fuses SiLU * up.
        
        The SiLU activation and elementwise multiply are fused into
        one Triton kernel, eliminating a memory round-trip.
        
        Before: gate→memory→SiLU→memory→multiply→memory
        After:  gate→memory→fused_silu_mul→memory
        
        Speedup: Additional ~1.2x on top of batched
        """
        batch_seq, top_k = expert_indices.shape
        
        w1_sel = self.w1_stacked[expert_indices]
        w3_sel = self.w3_stacked[expert_indices]
        w2_sel = self.w2_stacked[expert_indices]
        
        x_exp = x.unsqueeze(1).expand(-1, top_k, -1)
        
        gate = torch.einsum('bkh,bkhi->bki', x_exp, w1_sel)
        up = torch.einsum('bkh,bkhi->bki', x_exp, w3_sel)
        
        # FUSED: SiLU(gate) * up in one kernel
        hidden = fused_silu_mul(gate, up)
        
        out = torch.einsum('bki,bkih->bkh', hidden, w2_sel)
        out = _weight_routes_in_place_if_safe(out, expert_weights.unsqueeze(-1))
        return _sum_routes_in_place_if_safe(out)
    
    def forward_mem_efficient(
        self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Level 3: MEMORY EFFICIENT - Eliminate intermediate tensors.
        
        Reuse buffers instead of allocating new tensors.
        Reduces memory pressure and allocation overhead.
        
        Speedup: Additional ~1.1x (less allocation overhead)
        """
        batch_seq, top_k = expert_indices.shape
        total_tokens = batch_seq * top_k
        
        gate_buffer = self._bmm_workspace(
            "_gate_buffer",
            (total_tokens, self.intermediate_size),
            device=x.device,
            dtype=x.dtype,
        )
        up_buffer = self._bmm_workspace(
            "_up_buffer",
            (total_tokens, self.intermediate_size),
            device=x.device,
            dtype=x.dtype,
        )
        
        w1_sel = self.w1_stacked[expert_indices].view(total_tokens, self.hidden_size, -1)
        w3_sel = self.w3_stacked[expert_indices].view(total_tokens, self.hidden_size, -1)
        w2_sel = self.w2_stacked[expert_indices].view(total_tokens, self.intermediate_size, -1)
        
        x_flat = x.unsqueeze(1).expand(-1, top_k, -1).reshape(total_tokens, self.hidden_size)
        
        # Compute into pre-allocated buffers
        torch.bmm(x_flat.unsqueeze(1), w1_sel, out=gate_buffer.unsqueeze(1))
        torch.bmm(x_flat.unsqueeze(1), w3_sel, out=up_buffer.unsqueeze(1))
        
        # Reuse the gate buffer as the activated hidden state.
        hidden = _silu_mul_in_place_if_safe(gate_buffer, up_buffer)
        
        # Final projection
        out_flat = self._bmm_workspace(
            "_mem_out_buffer",
            (total_tokens, self.hidden_size),
            device=x.device,
            dtype=x.dtype,
        )
        torch.bmm(hidden.unsqueeze(1), w2_sel, out=out_flat.unsqueeze(1))
        out = out_flat.view(batch_seq, top_k, self.hidden_size)
        
        out = _weight_routes_in_place_if_safe(out, expert_weights.unsqueeze(-1))
        return _sum_routes_in_place_if_safe(out)
    
    def forward_grouped(
        self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Level 4: GROUPED - Sort tokens + per-expert GEMM.
        
        This is how production MoE works (vLLM, SGLang):
        1. Sort tokens by expert (bucket_by_expert from ch19)
        2. Run contiguous GEMM per expert
        3. Restore original order
        
        Benefits:
        - Contiguous memory access per expert
        - Better cache utilization  
        - Enables CUTLASS grouped GEMM
        
        Speedup: ~21x total (best manual optimization!)
        """
        batch_seq, top_k = expert_indices.shape
        
        # Reuse the shared ch19 bucketing helpers instead of the older global-sort
        # path. That keeps the lab aligned with the production-style grouped-routing
        # utilities it already documents, and avoids the current sort-heavy slowdown.
        batch_seq, top_k = expert_indices.shape
        flat_token_ids = self._flat_topk_token_ids_for(batch_seq, top_k, x.device)
        flat_expert_ids = expert_indices.view(-1)
        (
            sorted_tokens,
            counts,
            bucket_indices,
            _expert_order,
            _bucket_token_ids,
            expert_order_host,
        ) = bucket_grouped_tokens(
            x,
            flat_expert_ids,
            self.num_experts,
            token_ids=flat_token_ids,
            return_expert_order_list=True,
        )
        sorted_weights = expert_weights.view(-1).index_select(0, bucket_indices)

        # Per-expert GEMM on contiguous tokens
        output = self._bmm_workspace(
            "_grouped_output",
            (sorted_tokens.shape[0], self.hidden_size),
            device=x.device,
            dtype=x.dtype,
        )

        offset = 0
        for expert_id, count in zip(expert_order_host, counts):
            tokens_e = sorted_tokens[offset:offset+count]
            weights_e = sorted_weights[offset:offset+count].unsqueeze(-1)
            
            # Contiguous GEMM for this expert
            gate = tokens_e @ self.w1_stacked[expert_id]
            up = tokens_e @ self.w3_stacked[expert_id]
            hidden = _silu_mul_in_place_if_safe(gate, up)
            expert_out = hidden @ self.w2_stacked[expert_id]
            
            weighted_out = _weight_routes_in_place_if_safe(expert_out, weights_e)
            output[offset:offset+count] = weighted_out
            offset += count
        
        # Restore order
        restored = self._bmm_workspace(
            "_grouped_restored",
            (batch_seq * top_k, self.hidden_size),
            device=x.device,
            dtype=x.dtype,
        )
        restore_grouped_tokens(output, bucket_indices, batch_seq * top_k, out=restored)
        restored = restored.view(batch_seq, top_k, self.hidden_size)
        return _sum_routes_in_place_if_safe(restored)
    
    def forward_bmm_fused(
        self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Level 5: BMM FUSION - Vectorized scatter + single BMM.
        
        Instead of 8 separate cuBLAS calls (one per expert), we:
        1. Scatter tokens into padded tensor using vectorized indexing
        2. Run ONE batched matmul (BMM) for ALL experts
        3. Gather results back
        
        This achieves MUCH higher GPU utilization because:
        - Single kernel launch (vs 8)
        - Better SM utilization 
        - Larger effective matrix = closer to peak TFLOPS
        
        Speedup: ~5-6x over grouped (at Llama-7B dimensions!)
        """
        batch_seq, top_k = expert_indices.shape
        device = x.device
        
        # Sort by expert
        flat_idx = expert_indices.view(-1)
        sorted_order = torch.argsort(flat_idx, stable=True)
        flat_token_ids = self._flat_topk_token_ids_for(batch_seq, top_k, device)
        assignments = flat_idx.numel()
        sorted_token_ids = self._bmm_workspace(
            "_bmm_sorted_token_ids",
            (assignments,),
            device=device,
            dtype=torch.int64,
        )
        torch.index_select(flat_token_ids, 0, sorted_order, out=sorted_token_ids)
        sorted_tokens = self._bmm_workspace(
            "_bmm_sorted_tokens",
            (assignments, self.hidden_size),
            device=device,
            dtype=x.dtype,
        )
        torch.index_select(x, 0, sorted_token_ids, out=sorted_tokens)
        sorted_weights = self._bmm_workspace(
            "_bmm_sorted_weights",
            (assignments,),
            device=device,
            dtype=expert_weights.dtype,
        )
        torch.index_select(expert_weights.view(-1), 0, sorted_order, out=sorted_weights)
        sorted_expert_ids = self._bmm_workspace(
            "_bmm_sorted_expert_ids",
            (assignments,),
            device=device,
            dtype=torch.int64,
        )
        torch.index_select(flat_idx, 0, sorted_order, out=sorted_expert_ids)
        counts = torch.bincount(sorted_expert_ids, minlength=self.num_experts)
        max_count = int(counts.max().item())
        
        # Vectorized scatter indices
        cumsum = self._bmm_workspace(
            "_bmm_expert_cumsum",
            (self.num_experts,),
            device=device,
            dtype=counts.dtype,
        )
        torch.cumsum(counts, dim=0, out=cumsum)
        starts = self._bmm_workspace(
            "_bmm_expert_starts",
            (self.num_experts,),
            device=device,
            dtype=counts.dtype,
        )
        starts[0] = 0
        starts[1:].copy_(cumsum[:-1])
        expert_offsets = self._bmm_workspace(
            "_bmm_expert_offsets",
            (assignments,),
            device=device,
            dtype=torch.int64,
        )
        torch.index_select(starts, 0, sorted_expert_ids, out=expert_offsets)
        position_ids = self._position_ids_for(sorted_expert_ids.numel(), device)
        positions = self._bmm_workspace(
            "_bmm_positions",
            (assignments,),
            device=device,
            dtype=torch.int64,
        )
        torch.sub(position_ids, expert_offsets, out=positions)
        padded_indices = self._bmm_workspace(
            "_bmm_padded_indices",
            (assignments,),
            device=device,
            dtype=torch.int64,
        )
        torch.mul(sorted_expert_ids, max_count, out=padded_indices)
        padded_indices.add_(positions)
        
        # Scatter tokens into padded tensor (vectorized!)
        padded_tokens = self._bmm_workspace(
            "_bmm_padded_tokens",
            (self.num_experts * int(max_count), self.hidden_size),
            device=device,
            dtype=x.dtype,
        )
        # Only rows addressed by padded_indices are gathered after the BMM.
        padded_token_index = self._padded_token_index_view(padded_indices, self.hidden_size)
        padded_tokens.scatter_(0, padded_token_index, sorted_tokens)
        padded_tokens = padded_tokens.view(self.num_experts, max_count, self.hidden_size)
        
        # SINGLE BMM for all experts!
        gate = torch.bmm(padded_tokens, self.w1_stacked)   # [E, max_count, I]
        up = torch.bmm(padded_tokens, self.w3_stacked)     # [E, max_count, I]
        hidden = _silu_mul_in_place_if_safe(gate, up)
        out = torch.bmm(hidden, self.w2_stacked)           # [E, max_count, H]
        
        # Scatter weights and apply
        padded_weights = self._bmm_workspace(
            "_bmm_padded_weights",
            (self.num_experts * int(max_count), 1),
            device=device,
            dtype=x.dtype,
        )
        # Padding rows are ignored by the gather, so clearing the workspace is unnecessary.
        padded_weight_index = self._padded_column_index(padded_indices)
        sorted_weight_column = self._sorted_weight_column(sorted_weights)
        padded_weights.scatter_(0, padded_weight_index, sorted_weight_column)
        padded_weights = padded_weights.view(self.num_experts, max_count, 1)
        out.mul_(padded_weights)
        
        # Gather back using same indices
        flat_out = out.view(-1, self.hidden_size)
        if torch.is_grad_enabled() and flat_out.requires_grad:
            valid_out = flat_out.index_select(0, padded_indices)
        else:
            valid_out = self._bmm_workspace(
                "_bmm_valid_out",
                (assignments, self.hidden_size),
                device=device,
                dtype=out.dtype,
            )
            torch.index_select(flat_out, 0, padded_indices, out=valid_out)
        
        # Restore order
        unsort = self._bmm_workspace(
            "_bmm_unsort",
            (assignments,),
            device=device,
            dtype=torch.int64,
        )
        unsort[sorted_order] = position_ids
        if torch.is_grad_enabled() and valid_out.requires_grad:
            restored = valid_out.index_select(0, unsort)
        else:
            restored = self._bmm_workspace(
                "_bmm_restored",
                (assignments, self.hidden_size),
                device=device,
                dtype=valid_out.dtype,
            )
            torch.index_select(valid_out, 0, unsort, out=restored)
        restored = restored.view(batch_seq, top_k, self.hidden_size)
        return _sum_routes_in_place_if_safe(restored)

    def _forward_bmm_fused_graphable(
        self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fixed-shape BMM path for CUDA graph capture.

        The level-5 kernel path sizes the expert buckets with `counts.max().item()`
        and rebuilds routing buffers from sorted indices. That host synchronization
        invalidates CUDA graph capture. For level 6 we switch to a fixed-capacity
        dense dispatch view: every expert sees a static `[batch_seq * top_k, hidden]`
        slot tensor, with inactive slots masked to zero. The stacked-weight expert
        compute stays vectorized, but all tensor shapes become capture-safe.
        """
        batch_seq, top_k = expert_indices.shape
        device = x.device
        flat_idx = expert_indices.reshape(-1)
        flat_weights = expert_weights.reshape(1, -1, 1).to(dtype=x.dtype)
        expanded_x = x[:, None, :].expand(batch_seq, top_k, self.hidden_size).reshape(
            batch_seq * top_k,
            self.hidden_size,
        )

        expert_mask = F.one_hot(flat_idx, num_classes=self.num_experts).transpose(0, 1)
        expert_mask = expert_mask.to(dtype=x.dtype)
        expert_mask_column = expert_mask.unsqueeze(-1)
        expanded_x_broadcast = expanded_x.unsqueeze(0)
        if torch.is_grad_enabled() and expanded_x.requires_grad:
            padded_tokens = expert_mask_column * expanded_x_broadcast
        else:
            padded_tokens = self._bmm_workspace(
                "_graph_padded_tokens",
                (self.num_experts, batch_seq * top_k, self.hidden_size),
                device=device,
                dtype=x.dtype,
            )
            torch.mul(expert_mask_column, expanded_x_broadcast, out=padded_tokens)

        gate = torch.bmm(padded_tokens, self.w1_stacked)
        up = torch.bmm(padded_tokens, self.w3_stacked)
        hidden = _silu_mul_in_place_if_safe(gate, up)
        out = torch.bmm(hidden, self.w2_stacked)
        out = _weight_routes_in_place_if_safe(out, expert_mask_column)
        out = _weight_routes_in_place_if_safe(out, flat_weights)

        combined = out.sum(dim=0)
        restored = combined.view(batch_seq, top_k, self.hidden_size)
        return _sum_routes_in_place_if_safe(restored)
    
    def forward_cuda_graphs(
        self, x: torch.Tensor, expert_indices: torch.Tensor, expert_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Level 6: CUDA GRAPHS - Capture kernel sequence.
        
        CUDA graphs capture the sequence of kernel launches and replay
        them with minimal CPU overhead. This eliminates:
        - Kernel launch latency
        - CPU-GPU synchronization
        - Python overhead
        
        Note: Graph capture itself happens inside this level on top of the
        level-5 fused BMM path. Static shapes are required.

        Speedup: Additional ~1.1x on top of BMM fusion
        """
        if x.device.type != "cuda":
            self._cuda_graph_captured = False
            self._cuda_graph_last_error = "cuda_graphs_require_cuda_inputs"
            return self.forward_bmm_fused(x, expert_indices, expert_weights)

        if self._is_torch_compiling():
            self._cuda_graph_captured = False
            self._cuda_graph_last_error = "torch_compile_handles_graph_capture"
            return self.forward_bmm_fused(x, expert_indices, expert_weights)

        signature = self._graph_signature(x, expert_indices, expert_weights)
        if self._cuda_graph is None or self._cuda_graph_signature != signature:
            try:
                self._capture_cuda_graph(x, expert_indices, expert_weights, signature)
            except Exception as exc:
                self._cuda_graph_capture_failures += 1
                self._cuda_graph_captured = False
                self._reset_cuda_graph_cache()
                self._cuda_graph_last_error = f"{type(exc).__name__}: {exc}"
                print(f"[moe_cuda_graphs] fallback to bmm_fused: {self._cuda_graph_last_error}")
                return self.forward_bmm_fused(x, expert_indices, expert_weights)

        if (
            self._cuda_graph is None
            or self._cuda_graph_static_x is None
            or self._cuda_graph_static_expert_indices is None
            or self._cuda_graph_static_expert_weights is None
            or self._cuda_graph_output is None
        ):
            self._cuda_graph_captured = False
            self._cuda_graph_last_error = "cuda_graph_cache_unavailable"
            return self.forward_bmm_fused(x, expert_indices, expert_weights)

        self._cuda_graph_static_x.copy_(x)
        self._cuda_graph_static_expert_indices.copy_(expert_indices)
        self._cuda_graph_static_expert_weights.copy_(expert_weights)
        current_stream = torch.cuda.current_stream(device=x.device)
        if self._cuda_graph_stream is not None:
            self._cuda_graph_stream.wait_stream(current_stream)
            with torch.cuda.stream(self._cuda_graph_stream):
                self._cuda_graph.replay()
            current_stream.wait_stream(self._cuda_graph_stream)
        else:
            self._cuda_graph.replay()
        self._cuda_graph_replays += 1
        self._cuda_graph_last_error = None
        return self._cuda_graph_output

    def forward(
        self, x: torch.Tensor, expert_indices: torch.Tensor,
        expert_weights: torch.Tensor, num_experts_per_tok: int,
    ) -> torch.Tensor:
        """Dispatch to appropriate implementation based on optimization level."""
        # Priority: highest optimization level that's enabled
        if self.opts.use_cuda_graphs and not self._is_torch_compiling():
            return self.forward_cuda_graphs(x, expert_indices, expert_weights)
        elif self.opts.use_compile and self._is_torch_compiling() and self.opts.use_bmm_fused:
            return self._forward_bmm_fused_graphable(x, expert_indices, expert_weights)
        elif self.opts.use_bmm_fused:
            return self.forward_bmm_fused(x, expert_indices, expert_weights)
        elif self.opts.use_grouped:
            return self.forward_grouped(x, expert_indices, expert_weights)
        elif self.opts.use_mem_efficient:
            return self.forward_mem_efficient(x, expert_indices, expert_weights)
        elif self.opts.use_fused:
            return self.forward_fused(x, expert_indices, expert_weights)
        elif self.opts.use_batched:
            return self.forward_batched(x, expert_indices, expert_weights)
        else:
            return self.forward_naive(x, expert_indices, expert_weights, num_experts_per_tok)


class MoELayer(nn.Module):
    """MoE layer with configurable optimizations."""
    
    def __init__(self, hidden_size: int, intermediate_size: int, 
                 num_experts: int, num_experts_per_tok: int,
                 opts: MoEOptimizations):
        super().__init__()
        self.opts = opts
        self.num_experts_per_tok = num_experts_per_tok
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = MoEExperts(num_experts, hidden_size, intermediate_size, opts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = x.shape
        x_flat = x.view(-1, hidden)
        
        router_logits = self.gate(x_flat)
        top_logits, expert_indices = torch.topk(router_logits.float(), self.num_experts_per_tok, dim=-1)
        expert_weights = F.softmax(top_logits, dim=-1)
        expert_weights = expert_weights.to(x.dtype)
        
        output = self.experts(x_flat, expert_indices, expert_weights, self.num_experts_per_tok)
        
        return output.view(batch, seq, hidden)


class MoEBlock(nn.Module):
    """Transformer block with MoE."""
    
    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_heads: int, num_experts: int, num_experts_per_tok: int,
                 opts: MoEOptimizations):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.moe = MoELayer(hidden_size, intermediate_size, num_experts, num_experts_per_tok, opts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        h = self.ln2(x)
        h = self.moe(h)
        return x + h


class ConfigurableMoEModel(nn.Module):
    """MoE model with configurable optimization levels."""
    
    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 512,
        intermediate_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        num_experts: int = 8,
        num_experts_per_tok: int = 2,
        opts: Optional[MoEOptimizations] = None,
    ):
        super().__init__()
        self.opts = opts or MoEOptimizations()
        
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.blocks = nn.ModuleList([
            MoEBlock(hidden_size, intermediate_size, num_heads,
                    num_experts, num_experts_per_tok, self.opts)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def get_cuda_graph_metrics(self) -> Dict[str, float]:
        attempted = 0.0
        captured = 0.0
        fallback = 0.0
        failures = 0.0
        replays = 0.0
        for block in self.blocks:
            metrics = block.moe.experts.get_cuda_graph_metrics()
            attempted += metrics["cuda_graph_attempted"]
            captured += metrics["cuda_graph_captured"]
            fallback += metrics["cuda_graph_fallback"]
            failures += metrics["cuda_graph_capture_failures"]
            replays += metrics["cuda_graph_replays"]
        return {
            "cuda_graph_attempted": attempted,
            "cuda_graph_captured": captured,
            "cuda_graph_fallback": fallback,
            "cuda_graph_capture_failures": failures,
            "cuda_graph_replays": replays,
        }


def create_model(level: int, **kwargs) -> Tuple[ConfigurableMoEModel, MoEOptimizations]:
    """Create model with optimizations enabled up to the given level.
    
    Level 0: Naive (Python loops)
    Level 1: + Batched (einsum parallelizes)
    Level 2: + Fused (Triton fuses SiLU*up)
    Level 3: + MemEfficient (reuse buffers)
    Level 4: + Grouped (sort + per-expert GEMM)
    Level 5: + BMM Fusion (vectorized scatter + single BMM)
    Level 6: + CUDAGraphs (capture and replay the fused expert path)
    Level 7: + Compiled (torch.compile on top of the graph-friendly model)
    """
    opts = MoEOptimizations(
        use_batched=(level >= 1),
        use_fused=(level >= 2),
        use_mem_efficient=(level >= 3),
        use_grouped=(level >= 4),
        use_bmm_fused=(level >= 5),
        use_cuda_graphs=(level >= 6),
        use_compile=(level >= 7),
    )
    
    model = ConfigurableMoEModel(opts=opts, **kwargs)
    return model, opts
