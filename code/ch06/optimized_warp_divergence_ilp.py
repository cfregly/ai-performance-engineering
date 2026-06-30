"""optimized_warp_divergence_ilp.py - Optimized ILP avoiding warp divergence.

Chapter 6: Occupancy and Instruction-Level Parallelism

Demonstrates how to avoid warp divergence using branchless operations.
The baseline (baseline_warp_divergence_ilp.py) uses conditional indexing
which causes warp divergence. This optimized version uses torch.where
for branchless selection, compiled once with torch.compile on the full tensor.

Key optimizations vs baseline:
- torch.where instead of boolean indexing (no warp divergence)
- Single compiled kernel on full tensor (no chunking overhead)
- Reused branch/scratch buffers (no per-iteration branch or roll allocations)
"""

from __future__ import annotations

import os
from typing import Callable, Optional, Tuple

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.utils.compile_utils import compile_callable
from core.optimization.inductor_guard import (
    InductorCudagraphState,
    disable_inductor_cudagraph_features,
    restore_inductor_cudagraph_features,
)
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch06.workload_config import WORKLOAD


def _fused_branchless_kernel(
    input_tensor: torch.Tensor,
    mask_input: torch.Tensor,
    iterations: int,
    result: torch.Tensor,
    mask_source: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    scratch: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fully fused branchless transform - no warp divergence.
    
    Uses torch.where for predicated selection instead of boolean indexing.
    All threads compute both branches; the result is selected via predicate.
    """
    result.copy_(input_tensor)
    mask_source.copy_(mask_input)
    for iteration in range(iterations):
        # Compute mask as float for branchless blending
        torch.sigmoid(mask_source, out=positive)
        torch.gt(positive, 0.5, out=mask)
        
        # Compute BOTH branches for all elements (branchless)
        torch.mul(result, 1.11, out=positive)
        positive.add_(0.25)
        torch.tanh(positive, out=positive)
        torch.mul(positive, positive, out=scratch)
        positive.mul_(1.003)
        positive.add_(scratch, alpha=0.0005)
        
        torch.mul(result, 0.77, out=negative)
        negative.add_(-0.35)
        torch.sin(negative, out=negative)
        torch.mul(negative, negative, out=scratch)
        negative.mul_(0.997)
        negative.add_(scratch, alpha=-0.0004)
        
        # Select result via predicate (no divergence - all threads do same work)
        torch.where(mask, positive, negative, out=result)
        shift = (iteration + 1) % int(result.numel())
        if shift == 0:
            scratch.copy_(result)
        else:
            scratch[:shift].copy_(result[-shift:])
            scratch[shift:].copy_(result[:-shift])
        mask_source.mul_(0.92)
        mask_source.add_(scratch, alpha=0.08)
    
    return result, mask_source


class OptimizedWarpDivergenceILPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: High ILP by avoiding warp divergence with fused branchless kernel."""

    def __init__(self):
        super().__init__()
        self.workload = WORKLOAD
        self.N = self.workload.warp_elements
        self.branch_iterations = self.workload.warp_branch_iterations
        self.input: Optional[torch.Tensor] = None
        self.routing_logits: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._compiled_fn: Optional[Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None
        self._inductor_state: Optional[InductorCudagraphState] = None
        self._result_buffer: Optional[torch.Tensor] = None
        self._mask_source_buffer: Optional[torch.Tensor] = None
        self._positive_buffer: Optional[torch.Tensor] = None
        self._negative_buffer: Optional[torch.Tensor] = None
        self._scratch_buffer: Optional[torch.Tensor] = None
        self._mask_buffer: Optional[torch.Tensor] = None
        token_count = self.N * self.branch_iterations
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.branch_iterations),
            tokens_per_iteration=float(token_count),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.routing_logits = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = None  # Will be set by benchmark_fn
        self._result_buffer = torch.empty_like(self.input)
        self._mask_source_buffer = torch.empty_like(self.routing_logits)
        self._positive_buffer = torch.empty_like(self.input)
        self._negative_buffer = torch.empty_like(self.input)
        self._scratch_buffer = torch.empty_like(self.input)
        self._mask_buffer = torch.empty(self.N, device=self.device, dtype=torch.bool)
        
        # Capture iterations in closure for compilation
        branch_iters = self.branch_iterations
        result_buffer = self._result_buffer
        mask_source_buffer = self._mask_source_buffer
        positive_buffer = self._positive_buffer
        negative_buffer = self._negative_buffer
        scratch_buffer = self._scratch_buffer
        mask_buffer = self._mask_buffer
        if any(
            buffer is None
            for buffer in (
                result_buffer,
                mask_source_buffer,
                positive_buffer,
                negative_buffer,
                scratch_buffer,
                mask_buffer,
            )
        ):
            raise RuntimeError("Scratch buffers not initialized")
        
        def fused_fn(data: torch.Tensor, logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            return _fused_branchless_kernel(
                data,
                logits,
                branch_iters,
                result_buffer,
                mask_source_buffer,
                positive_buffer,
                negative_buffer,
                scratch_buffer,
                mask_buffer,
            )
        
        # Keep eager branchless execution as default because compile can alter
        # transcendental numerics enough to fail strict output verification.
        use_compile = os.environ.get("AISP_CH06_ILP_USE_TORCH_COMPILE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if use_compile:
            if self._inductor_state is None:
                self._inductor_state = disable_inductor_cudagraph_features()
            self._compiled_fn = compile_callable(
                fused_fn,
                fullgraph=True,
                mode="reduce-overhead",
            )
        else:
            self._compiled_fn = fused_fn

        # Warmup the execution function
        _, _ = self._compiled_fn(self.input, self.routing_logits)
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.input is not None and self.routing_logits is not None
        with torch.inference_mode(), self._nvtx_range("optimized_warp_divergence_ilp"):
            # Single compiled call on full tensor - no chunking, no concat
            assert self._compiled_fn is not None
            self.output, self.routing_logits = self._compiled_fn(self.input, self.routing_logits)

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self.input, "routing_logits": self.routing_logits},
            output=self.output.detach(),
            batch_size=self.N,
            parameter_count=0,
            output_tolerance=(1e-5, 1e-5),
        )

    def teardown(self) -> None:
        self.input = None
        self.output = None
        self.routing_logits = None
        self._result_buffer = None
        self._mask_source_buffer = None
        self._positive_buffer = None
        self._negative_buffer = None
        self._scratch_buffer = None
        self._mask_buffer = None
        restore_inductor_cudagraph_features(self._inductor_state)
        self._inductor_state = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=self.workload.ilp_iterations,
            warmup=self.workload.ilp_warmup,
            adaptive_iterations=False,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_kernel_fundamentals_metrics
        return compute_kernel_fundamentals_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            num_iterations=1,
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output tensor not initialized"
        return None



def get_benchmark() -> BaseBenchmark:
    return OptimizedWarpDivergenceILPBenchmark()
