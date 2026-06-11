"""labs.moe_cuda/optimized_router_vectorized.py - Grouped expert dispatch step.

This step groups token assignments by expert to run larger GEMMs, then wraps the
forward pass in a CUDA graph to remove Python dispatch overhead. Baseline uses a
per-token weight gather path; this variant is the standard MoE grouped dispatch.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


class VectorizedTopKMoE(nn.Module):
    """Top-k router with batched expert MLPs and scatter accumulation."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, expansion: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expanded = hidden_size * expansion
        self.router = nn.Linear(hidden_size, num_experts)

        # Pack expert weights for vectorized matmuls: [E, H, H*exp] and [E, H*exp, H].
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_size, self.expanded))
        self.b1 = nn.Parameter(torch.zeros(num_experts, self.expanded))
        self.w2 = nn.Parameter(torch.empty(num_experts, self.expanded, hidden_size))
        self.b2 = nn.Parameter(torch.zeros(num_experts, hidden_size))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover - benchmarked
        logits = self.router(tokens)
        top_scores, expert_ids = torch.topk(logits, self.top_k, dim=-1)
        probs = torch.softmax(top_scores, dim=-1, dtype=tokens.dtype)

        flat_tokens = tokens.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, self.hidden_size)
        flat_expert_ids = expert_ids.reshape(-1)
        flat_probs = probs.reshape(-1, 1).to(tokens.dtype)

        w1 = self.w1[flat_expert_ids]
        b1 = self.b1[flat_expert_ids]
        # Avoid baddbmm meta-shape expand issues by separating matmul + bias add
        hidden = torch.bmm(flat_tokens.unsqueeze(1), w1).squeeze(1) + b1
        hidden = F.gelu(hidden)

        w2 = self.w2[flat_expert_ids]
        b2 = self.b2[flat_expert_ids]
        expert_out = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2
        weighted = expert_out * flat_probs

        output = torch.zeros_like(tokens, dtype=tokens.dtype)
        token_indices = torch.arange(tokens.shape[0], device=tokens.device).repeat_interleave(self.top_k)
        output.index_add_(0, token_indices, weighted)
        return output


class GroupedTopKMoE(VectorizedTopKMoE):
    """Top-k router that groups assignments by expert (larger matmuls).

    Dispatch is a sortless fixed-capacity dense layout so every op in the
    forward has a static shape: rank-within-expert via one-hot cumsum,
    index_copy_ into ``[num_experts * capacity, hidden]`` slots, batched GEMMs
    over all experts at once, then gather back by the same slot indices (which
    restores original token order for free). No ``nonzero``/boolean masking
    and no per-expert Python loop, so the whole forward is CUDA-graph
    capturable as a single graph.
    """

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, expansion: int = 2) -> None:
        super().__init__(hidden_size, num_experts, top_k, expansion)
        self.capacity: Optional[int] = None

    @torch.no_grad()
    def calibrate_capacity(self, tokens: torch.Tensor) -> int:
        """Pick the static per-expert slot capacity from observed routing.

        Must run eagerly (it is the ONLY host sync of the optimized path);
        call it once in ``setup()`` BEFORE graph capture. Capacity is 2x the
        observed max tokens/expert, rounded up to a multiple of 64, clamped to
        the routed-token count.
        """
        logits = self.router(tokens)
        _, expert_ids = torch.topk(logits, self.top_k, dim=-1)
        counts = torch.bincount(expert_ids.reshape(-1), minlength=self.num_experts)
        max_load = int(counts.max().item())
        total_assignments = tokens.shape[0] * self.top_k
        capacity = min(((max_load * 2 + 63) // 64) * 64, total_assignments)
        self.capacity = max(capacity, 64)
        return self.capacity

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover - benchmarked
        if self.capacity is None:
            self.calibrate_capacity(tokens)
        capacity = self.capacity
        assert capacity is not None

        logits = self.router(tokens)
        top_scores, expert_ids = torch.topk(logits, self.top_k, dim=-1)
        probs = torch.softmax(top_scores, dim=-1, dtype=tokens.dtype)

        batch = tokens.shape[0]
        flat_tokens = tokens.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, self.hidden_size)
        flat_expert_ids = expert_ids.reshape(-1)
        flat_probs = probs.reshape(-1, 1).to(tokens.dtype)
        token_indices = torch.arange(batch, device=tokens.device).repeat_interleave(self.top_k)

        # Rank of each assignment within its expert: one-hot cumsum (sortless,
        # static shapes, no nonzero). Layout is [E, N] so the scan runs along
        # the contiguous inner dim (the [N, E] outer-dim scan has only E
        # parallel lanes and is ~2 ms; this form is ~10 us).
        expert_range = torch.arange(self.num_experts, device=tokens.device).unsqueeze(1)
        one_hot_t = (flat_expert_ids.unsqueeze(0) == expert_range).long()
        ranks = one_hot_t.cumsum(1).gather(0, flat_expert_ids.unsqueeze(0)).squeeze(0) - 1

        # Unique slot per assignment; overflow beyond capacity (never happens
        # with a calibrated capacity) is routed to a trash row so shapes stay
        # static and writes stay in-bounds.
        num_slots = self.num_experts * capacity
        valid = ranks < capacity
        slots = flat_expert_ids * capacity + ranks.clamp_max(capacity - 1)
        slots = torch.where(valid, slots, torch.full_like(slots, num_slots))

        x_dense = flat_tokens.new_zeros(num_slots + 1, self.hidden_size)
        x_dense.index_copy_(0, slots, flat_tokens)
        x_grouped = x_dense[:num_slots].view(self.num_experts, capacity, self.hidden_size)

        # Batched expert MLPs over all experts at once; zero rows are inert.
        # baddbmm folds the bias add into the GEMM epilogue (the standalone
        # broadcast adds cost ~150 us each at these shapes).
        hidden = torch.baddbmm(self.b1.unsqueeze(1), x_grouped, self.w1)
        hidden = F.gelu(hidden)
        expert_out = torch.baddbmm(self.b2.unsqueeze(1), hidden, self.w2)

        # Gather back by the same slot indices (restores token order) and
        # zero out any overflow rows before the weighted scatter-accumulate.
        out_flat = expert_out.reshape(num_slots, self.hidden_size)
        gathered = out_flat[slots.clamp_max(num_slots - 1)]
        weighted = gathered * (flat_probs * valid.unsqueeze(1).to(tokens.dtype))

        output = torch.zeros_like(tokens, dtype=tokens.dtype)
        output.index_add_(0, token_indices, weighted)
        return output


class VectorizedRouterBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark for grouped top-k router with CUDA graphs."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024
        self.num_experts = 32
        self.top_k = 2
        # Match baseline batch_size for fair comparison
        self.batch_size = 4096
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_output: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        tokens = self.batch_size * self.top_k
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        import gc

        # CRITICAL: Clean up CUDA state from previous benchmarks
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        try:
            if hasattr(torch.cuda, 'graph_pool_trim'):
                torch.cuda.graph_pool_trim()
        except Exception:
            pass

        # Reset CUDA RNG state
        try:
            device_idx = torch.cuda.current_device()
            gen = torch.cuda.default_generators[device_idx]
            gen.set_offset(0)
            gen.manual_seed(42)
        except Exception:
            pass

        try:
            torch._dynamo.reset()
        except Exception:
            pass

        try:
            torch._inductor.cudagraph_trees.reset_cudagraph_trees()
        except Exception:
            pass

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        model = GroupedTopKMoE(self.hidden_size, self.num_experts, self.top_k, expansion=2)
        model = model.to(self.device, dtype=torch.bfloat16)
        model.eval()
        self.model = model

        # Use CPU randn + to(device) to avoid CUDA RNG graph capture issues
        self.inputs = torch.randn(
            self.batch_size,
            self.hidden_size,
            dtype=torch.bfloat16,
        ).to(self.device)

        self.output = None

        # Calibrate the static per-expert slot capacity and warm up kernels
        # (cuBLAS workspaces, one-hot/cumsum) eagerly BEFORE graph capture:
        # the only host sync of the optimized path lives in calibrate_capacity().
        model.calibrate_capacity(self.inputs)
        with torch.no_grad():
            for _ in range(3):
                model(self.inputs)
        torch.cuda.synchronize(self.device)

        # Capture the forward pass into a CUDA graph to hide Python dispatch overhead.
        self.graph = None
        self.static_output = None
        try:
            torch.cuda.synchronize(self.device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                assert self.model is not None and self.inputs is not None
                self.static_output = self.model(self.inputs)
            torch.cuda.synchronize(self.device)
        except Exception:
            self.graph = None
            self.static_output = None

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Model not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_cuda_router_vectorized", enable=enable_nvtx):
            if self.graph is not None:
                self.graph.replay()
                if self.static_output is None:
                    raise RuntimeError("CUDA graph replay missing static output buffer")
                self.output = self.static_output
            else:
                with torch.inference_mode():
                    self.output = self.model(self.inputs)
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self.output is None or self.model is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input": self.inputs.detach()},
            output=self.output.detach().float().clone(),
            batch_size=self.batch_size,
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={"bf16": True, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.graph = None
        self.static_output = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=10)  # torch.compile needs warmup

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline analysis metrics."""
        # Estimate problem size for roofline analysis
        n = getattr(self, 'N', 0) or getattr(self, 'hidden_dim', 0) or 4096
        batch = getattr(self, 'batch_size', 1) or getattr(self, 'batch', 1)
        # Simple FLOP estimate for linear layers
        flops = 2.0 * batch * n * n  # Rough estimate
        bytes_moved = batch * n * 4.0  # Input/output bytes
        arithmetic_intensity = flops / max(bytes_moved, 1.0)
        return {
            "router_vectorized.estimated_flops": flops,
            "router_vectorized.estimated_bytes": bytes_moved,
            "router_vectorized.arithmetic_intensity": arithmetic_intensity,
        }

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Vectorized router missing"
        if self.inputs is None:
            return "Inputs missing"
        return None

def get_benchmark() -> BaseBenchmark:
    return VectorizedRouterBenchmark()
