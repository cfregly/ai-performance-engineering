"""Benchmark wrapper for the optimized CUDA decode kernel."""

from __future__ import annotations

from typing import Optional
from types import ModuleType

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.harness.cuda_capabilities import tma_support_status
from core.harness.hardware_capabilities import detect_capabilities
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

from labs.moe_cuda.decode_kernels import (
    optimized_kernel_supported,
    load_optimized_kernel_module,
    is_optimized_available,
    get_optimized_error,
)


class OptimizedDecodeKernelBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs the TMA double-buffered CUDA decode kernel."""

    def __init__(self) -> None:
        super().__init__()
        if not torch.cuda.is_available():
            raise RuntimeError("labs.moe_cuda decode kernels require CUDA")
        self.rows = 4096
        self.cols = 1024
        self.input: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._module: Optional[ModuleType] = None
        tokens = self.rows * self.cols
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self._enable_nvtx = False

    def setup(self) -> None:
        import gc
        
        # CRITICAL: Comprehensive CUDA cleanup before TMA kernel setup
        # TMA tensor map encoding is very sensitive to CUDA memory/graph state
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        # Reset CUDA graph pool
        try:
            if hasattr(torch.cuda, 'graph_pool_trim'):
                torch.cuda.graph_pool_trim()
        except Exception:
            pass
        
        # Reset CUDA RNG state - this is CRITICAL
        try:
            device_idx = torch.cuda.current_device()
            gen = torch.cuda.default_generators[device_idx]
            gen.set_offset(0)
            gen.manual_seed(42)
        except Exception:
            pass
        
        # Reset dynamo/inductor state
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
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        
        # Allocate contiguous tensors with explicit memory layout
        # TMA requires contiguous tensors with proper alignment
        # Use CPU randn + to(device) to avoid CUDA RNG graph capture issues
        self.input = torch.randn(
            self.rows,
            self.cols,
            dtype=torch.float32,
        ).to(self.device).contiguous()  # Explicitly ensure contiguity

        self._module = load_optimized_kernel_module()
        self._output_buffer = torch.empty(
            self.rows,
            self.cols,
            dtype=torch.float32,
            device=self.device,
        ).contiguous()
        self._verify_output_buffer = torch.empty_like(self._output_buffer)
        self.output = None
        
        torch.cuda.synchronize(self.device)
        
        # Verify tensors are properly allocated before benchmark
        assert self.input.is_contiguous(), "Input tensor must be contiguous for TMA"
        assert self._output_buffer.is_contiguous(), "Output tensor must be contiguous for TMA"

    def benchmark_fn(self) -> None:
        if self.input is None or self._output_buffer is None:
            raise RuntimeError("Decode tensors not initialized")
        if self._module is None:
            raise RuntimeError("Optimized decode kernel module not initialized")

        with torch.inference_mode(), nvtx_range("moe_cuda_decode_kernel_optimized", enable=self._enable_nvtx):
            self._module.run_optimized(self.input, self._output_buffer)
        self.output = self._output_buffer

    def capture_verification_payload(self) -> None:
        if self.input is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.input.detach()},
            output=self._verify_output_buffer,
            batch_size=1,
            parameter_count=0,
            precision_flags={"tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(1e-3, 1e-3),
        )

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.input = None
        self._output_buffer = None
        self._verify_output_buffer = None
        self.output = None
        self._module = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5, measurement_timeout_seconds=300, setup_timeout_seconds=300)  # Min warmup for CUDA

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
            "decode_kernel.estimated_flops": flops,
            "decode_kernel.estimated_bytes": bytes_moved,
            "decode_kernel.arithmetic_intensity": arithmetic_intensity,
        }

    def validate_result(self) -> Optional[str]:
        if self.input is None or self.output is None:
            return "Decode tensors missing"
        return None

def get_benchmark() -> BaseBenchmark:
    """Return the optimized TMA decode kernel benchmark.
    
    TMA is required on Blackwell B200 - no fallbacks.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("SKIPPED: CUDA required for TMA decode kernel")
    
    supported, reason = tma_support_status()
    if not supported:
        raise RuntimeError(f"SKIPPED: TMA decode kernel unavailable: {reason}")
    
    cap = detect_capabilities()
    if cap is None:
        raise RuntimeError("SKIPPED: TMA decode kernel requires detected hardware capabilities")
    
    cap_desc = f"{cap.device_name} ({cap.compute_capability})"
    
    # Check if optimized kernel is available
    if not is_optimized_available():
        error = get_optimized_error() or "Unknown error"
        raise RuntimeError(f"SKIPPED: TMA optimized kernel not available: {error}")
    
    candidate = OptimizedDecodeKernelBenchmark()
    
    # Verify TMA support for this shape
    if not optimized_kernel_supported(candidate.rows, candidate.cols):
        raise RuntimeError(
            f"SKIPPED: TMA decode kernel not supported for shape ({candidate.rows}, {candidate.cols}) "
            f"on {cap_desc}. Check CUDA driver/runtime."
        )
    
    return candidate
