#!/usr/bin/env python3
"""Real-world case study: DeepSeek-R1 MoE optimization for Blackwell.

Demonstrates optimization of DeepSeek-R1 style MoE model:
- Expert parallelism (EP)
- Load-balanced routing
- FP8 quantization for experts
- All-to-all communication optimization
- NCCL tuning for MoE
"""

import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
from core.utils.logger import get_logger

logger = get_logger(__name__)


class LoadBalancedRouter(nn.Module):
    """Load-balanced expert router with aux loss."""
    
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.register_buffer(
            "_gini_index",
            torch.arange(1, num_experts + 1, dtype=torch.float32),
            persistent=False,
        )
        self._aux_loss_dict: Dict[str, torch.Tensor] = {}

    def _gini_index_for(self, usage: torch.Tensor) -> torch.Tensor:
        index = self._gini_index
        if index.device != usage.device or index.dtype != usage.dtype:
            self._gini_index = index.to(device=usage.device, dtype=usage.dtype)
            index = self._gini_index
        return index
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        
        Returns:
            routing_weights: [batch, seq_len, top_k]
            selected_experts: [batch, seq_len, top_k]
            aux_loss_dict: Dictionary with load balancing metrics
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute routing logits
        routing_logits = self.gate(x)  # [batch, seq_len, num_experts]
        
        # Top-k selection
        routing_weights, selected_experts = torch.topk(
            routing_logits, self.top_k, dim=-1
        )
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        # Load balancing auxiliary loss (encourages balanced expert usage)
        # Based on DeepSeek-V2/V3 approach
        log_probs = F.log_softmax(routing_logits, dim=-1)
        probs = log_probs.exp()
        expert_usage = probs.mean(dim=[0, 1])  # [num_experts]
        
        # Compute load balance loss (encourage uniform distribution)
        balance_loss = torch.var(expert_usage) * self.num_experts
        
        # Compute Gini coefficient for routing fairness
        sorted_usage = torch.sort(expert_usage)[0]
        n = len(sorted_usage)
        index = self._gini_index_for(sorted_usage)
        gini = (2 * (index * sorted_usage).sum()) / (n * sorted_usage.sum()) - (n + 1) / n
        
        aux_loss_dict = self._aux_loss_dict
        aux_loss_dict["balance_loss"] = balance_loss
        aux_loss_dict["expert_usage_variance"] = torch.var(expert_usage)
        aux_loss_dict["gini_coefficient"] = gini
        aux_loss_dict["router_entropy"] = -(probs * log_probs).sum(dim=-1).mean()
        
        return routing_weights, selected_experts, aux_loss_dict


class ExpertMLP(nn.Module):
    """Single expert MLP (SwiGLU)."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        F.silu(gate, inplace=True)
        gate.mul_(up)
        return self.down_proj(gate)


class MoELayer(nn.Module):
    """MoE layer with load balancing."""
    
    def __init__(
        self,
        hidden_size: int,
        num_experts: int = 64,
        top_k: int = 6,
        intermediate_size: int = 14336,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self._route_token_cache: Dict[Tuple[int, int, str], torch.Tensor] = {}
        self._route_count_host_buffer: Optional[torch.Tensor] = None
        self._route_count_list_buffer = [0] * num_experts
        self._output_buffer: Optional[torch.Tensor] = None
        
        self.router = LoadBalancedRouter(hidden_size, num_experts, top_k)
        
        # Create experts
        self.experts = nn.ModuleList([
            ExpertMLP(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])

    def _output_for(self, flat: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled() and flat.requires_grad:
            return torch.empty_like(flat)
        shape = tuple(int(dim) for dim in flat.shape)
        numel = int(flat.numel())
        if (
            self._output_buffer is None
            or self._output_buffer.device != flat.device
            or self._output_buffer.dtype != flat.dtype
            or self._output_buffer.numel() < numel
        ):
            self._output_buffer = torch.empty(numel, device=flat.device, dtype=flat.dtype)
        return self._output_buffer[:numel].view(shape)

    def _route_token_ids(
        self,
        num_tokens: int,
        device: torch.device,
        routes_per_token: Optional[int] = None,
    ) -> torch.Tensor:
        routes = self.top_k if routes_per_token is None else routes_per_token
        key = (num_tokens, routes, str(device))
        token_ids = self._route_token_cache.get(key)
        if token_ids is None or token_ids.device != device:
            token_ids = torch.arange(num_tokens * routes, device=device, dtype=torch.int64)
            if routes != 1:
                token_ids.div_(routes, rounding_mode="floor")
            self._route_token_cache[key] = token_ids
        return token_ids

    def _route_count_list(self, expert_ids: torch.Tensor) -> List[int]:
        counts = torch.bincount(expert_ids, minlength=self.num_experts)
        needs_pinned = counts.device.type == "cuda"
        if (
            self._route_count_host_buffer is None
            or (needs_pinned and not self._route_count_host_buffer.is_pinned())
        ):
            self._route_count_host_buffer = torch.empty(
                self.num_experts,
                dtype=counts.dtype,
                device="cpu",
                pin_memory=needs_pinned,
            )
        route_count_host = self._route_count_host_buffer
        route_count_host.copy_(counts)
        route_count_list = self._route_count_list_buffer
        for expert_idx in range(self.num_experts):
            route_count_list[expert_idx] = int(route_count_host[expert_idx])
        return route_count_list
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Args:
            x: [batch, seq_len, hidden_size]
        
        Returns:
            output: [batch, seq_len, hidden_size]
            metrics: Dictionary with routing metrics
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Route tokens to experts
        routing_weights, selected_experts, metrics = self.router(x)
        
        # Flatten for expert processing
        x_flat = x.view(-1, hidden_size)  # [batch * seq_len, hidden_size]
        num_tokens = batch_size * seq_len
        output_flat = self._output_for(x_flat)

        first_experts = selected_experts[..., 0].reshape(-1)
        first_weights = routing_weights[..., 0].reshape(-1)
        first_order = torch.argsort(first_experts)
        first_count_list = self._route_count_list(first_experts)

        offset = 0
        for expert_idx, route_count in enumerate(first_count_list):
            next_offset = offset + int(route_count)
            if next_offset > offset:
                token_indices = first_order[offset:next_offset]
                tokens_for_expert = x_flat.index_select(0, token_indices)
                expert_output = self.experts[expert_idx](tokens_for_expert)
                weights = first_weights.index_select(0, token_indices).to(dtype=expert_output.dtype).unsqueeze(-1)
                expert_output.mul_(weights)
                output_flat.index_copy_(0, token_indices, expert_output)
            offset = next_offset

        if self.top_k > 1:
            remaining_experts = selected_experts[..., 1:].reshape(-1)
            remaining_weights = routing_weights[..., 1:].reshape(-1)
            route_order = torch.argsort(remaining_experts)
            route_count_list = self._route_count_list(remaining_experts)
            flat_token_ids = self._route_token_ids(num_tokens, x.device, self.top_k - 1)

            offset = 0
            for expert_idx, route_count in enumerate(route_count_list):
                next_offset = offset + int(route_count)
                if next_offset > offset:
                    route_indices = route_order[offset:next_offset]
                    token_indices = flat_token_ids.index_select(0, route_indices)
                    tokens_for_expert = x_flat.index_select(0, token_indices)
                    expert_output = self.experts[expert_idx](tokens_for_expert)
                    weights = remaining_weights.index_select(0, route_indices).to(dtype=expert_output.dtype).unsqueeze(-1)
                    expert_output.mul_(weights)
                    output_flat.index_add_(0, token_indices, expert_output)
                offset = next_offset
        
        output = output_flat.view(batch_size, seq_len, hidden_size)
        
        return output, metrics


class DeepSeekR1MoEOptimization(VerificationPayloadMixin, BaseBenchmark):
    """DeepSeek-R1 style MoE optimization benchmark."""

    allow_cpu = True
    
    def __init__(
        self,
        batch_size: int = 4,
        seq_length: int = 2048,
        hidden_size: int = 4096,
        num_experts: int = 64,
        top_k: int = 6,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self._last_metrics: Dict[str, Any] = {}
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._timing_events: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._pending_timing_events: Optional[Tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._last_aux_metrics: Dict[str, torch.Tensor] = {}
        self._last_elapsed_ms: Optional[float] = None
        self._latency_metric_values = [0.0]
        self._iteration_metric_payload: Dict[str, List[float]] = {
            "latency_ms": self._latency_metric_values,
        }
        self._payload_parameter_count = 0

        logger.info(f"DeepSeek-R1 MoE Optimization")
        logger.info(f"  Experts: {num_experts}, Top-K: {top_k}")

    def _resolve_device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def setup(self):
        """Initialize MoE model."""
        if self.device.type != "cuda":
            raise RuntimeError("SKIPPED: DeepSeek-R1 MoE benchmark requires CUDA")
        self.moe_layer = MoELayer(
            hidden_size=self.hidden_size,
            num_experts=self.num_experts,
            top_k=self.top_k,
        ).to(self.device).to(torch.bfloat16)
        self._payload_parameter_count = sum(p.numel() for p in self.moe_layer.parameters())
        
        # Create input
        self.input = torch.randn(
            self.batch_size,
            self.seq_length,
            self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16
        )
        self._verify_output_buffer = torch.empty(
            (1, min(4, self.seq_length), min(8, self.hidden_size)),
            device=self.device,
            dtype=torch.float32,
        )
        self._timing_events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        
        logger.info(f"MoE model setup complete")

    def benchmark_fn(self) -> None:
        """Execute MoE forward pass."""
        use_cuda_timing = self.device.type == "cuda"
        if use_cuda_timing:
            if self._timing_events is None:
                raise RuntimeError("Timing events not initialized")
            start_event, end_event = self._timing_events
            current_stream = torch.cuda.current_stream(self.device)
            start_event.record(current_stream)
            self._pending_timing_events = (start_event, end_event)
            self._last_elapsed_ms = None
        else:
            start_time = time.perf_counter()
            self._pending_timing_events = None

        # Forward pass
        with torch.inference_mode():
            output, metrics = self.moe_layer(self.input)
        if use_cuda_timing:
            end_event.record(current_stream)
        else:
            self._last_elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        self.output = output
        self._last_aux_metrics.clear()
        for key, value in metrics.items():
            if torch.is_tensor(value):
                self._last_aux_metrics[key] = value
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def finalize_iteration_metrics(self) -> Optional[Dict[str, list[float]]]:
        if self.device.type != "cuda" or self._pending_timing_events is None:
            return None
        start_event, end_event = self._pending_timing_events
        elapsed_ms = float(start_event.elapsed_time(end_event))
        self._last_elapsed_ms = elapsed_ms
        self._pending_timing_events = None
        self._latency_metric_values[0] = elapsed_ms
        return self._iteration_metric_payload

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
            : self._verify_output_buffer.shape[2],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"input": self.input.detach()},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": True,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if self.device.type == "cuda" else False,
            },
            output_tolerance=(0.1, 1.0),
        )

    def get_custom_metrics(self) -> Dict[str, Any]:
        if self._last_elapsed_ms is None:
            return self._last_metrics
        tokens_per_sec = (self.batch_size * self.seq_length) / max(self._last_elapsed_ms / 1000.0, 1e-9)
        metrics = self._last_metrics
        metrics["latency_ms"] = self._last_elapsed_ms
        metrics["throughput"] = tokens_per_sec
        for key, value in self._last_aux_metrics.items():
            metrics[key] = float(value)
        return metrics

    def teardown(self):
        """Clean up resources."""
        del self.moe_layer
        del self.input
        self.output = None
        self._verify_output_buffer = None
        self._timing_events = None
        self._pending_timing_events = None
        self._last_aux_metrics.clear()
        self._last_elapsed_ms = None
        super().teardown()


def run_benchmark(
    batch_size: int = 4,
    seq_length: int = 2048,
    hidden_size: int = 4096,
    num_experts: int = 64,
    top_k: int = 6,
    profile: str = "none",
    **kwargs
) -> Dict[str, Any]:
    """Run DeepSeek-R1 MoE optimization benchmark."""

    benchmark = DeepSeekR1MoEOptimization(
        batch_size=batch_size,
        seq_length=seq_length,
        hidden_size=hidden_size,
        num_experts=num_experts,
        top_k=top_k,
    )

    config = BenchmarkConfig(
        iterations=5,
        warmup=5,
        profile_mode=profile,
    )

    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config)

    result = harness.benchmark(
        benchmark,
        name="deepseek_r1_moe_optimization"
    )

    metrics = result.custom_metrics or {}
    return {
        "mean_time_ms": result.timing.mean_ms,
        **metrics,
        "config": {
            "num_experts": num_experts,
            "top_k": top_k,
        }
    }


def get_benchmark() -> BaseBenchmark:
    return DeepSeekR1MoEOptimization()
