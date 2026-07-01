"""Benchmark wrapper for the baseline CUDA decode kernel."""

from __future__ import annotations

from typing import Optional
from types import ModuleType

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

from labs.moe_cuda.decode_kernels import load_baseline_kernel_module


class BaselineDecodeKernelBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Runs the naive global-load CUDA decode kernel."""

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
        
        # Clean up CUDA graph state from previous benchmarks
        # to prevent "Offset increment outside graph capture" errors
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        
        try:
            if hasattr(torch.cuda, 'graph_pool_trim'):
                torch.cuda.graph_pool_trim()
        except Exception:
            pass
        
        # Reset CUDA RNG state to prevent graph capture errors
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
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        # Use CPU randn + to(device) to avoid CUDA RNG graph capture issues
        self.input = torch.randn(
            self.rows,
            self.cols,
            dtype=torch.float32,
        ).to(self.device)
        self._module = load_baseline_kernel_module()
        self._output_buffer = torch.empty_like(self.input)
        self._verify_output_buffer = torch.empty_like(self._output_buffer)
        self.output = None
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.input is None or self._output_buffer is None:
            raise RuntimeError("Decode tensors not initialized")
        if self._module is None:
            raise RuntimeError("Baseline decode kernel module not initialized")

        with torch.inference_mode(), nvtx_range("moe_cuda_decode_kernel_baseline", enable=self._enable_nvtx):
            self._module.run_baseline(self.input, self._output_buffer)
        self.output = self._output_buffer
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

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
        # Use shorter runs to keep verification fast on slow builds/GPUs.
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
    return BaselineDecodeKernelBenchmark()
