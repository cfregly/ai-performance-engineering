"""optimized_cpu_reduction.py - Best practice: GPU-only reduction.

Key optimizations vs baseline_cpu_reduction.py:
- All tensor operations stay on GPU (no .cpu() calls)
- Uses in-place reduction to avoid allocations
- Pre-allocated output buffer reused across iterations
- Avoids PCIe round-trips that dominate baseline timing

This demonstrates why keeping data on GPU is critical for performance.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from ch04.reduction_common import ReusableReductionMlp
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


class OptimizedGpuReductionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Best practice: GPU-only reduction - no CPU round-trips."""

    def __init__(self):
        super().__init__()
        # Larger workload to make CPU aggregation in baseline visibly expensive
        self.batch_size = 1024
        self.hidden_dim = 4096
        self.inner_dim = 8192
        self.num_shards = 8  # More shards = more CPU overhead in baseline
        
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._model_shard_view: Optional[torch.Tensor] = None
        self._reduced_rows = 0
        self._enable_nvtx = False
        self._payload_parameter_count = 0
        
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        # Reduction benchmark: fixed dimensions

    def setup(self) -> None:
        # Use same seed as baseline for deterministic verification
        torch.manual_seed(42)
        
        # Build model with same architecture as baseline for fair comparison.
        self.model = ReusableReductionMlp(self.hidden_dim, self.inner_dim).to(self.device).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        
        self.input = torch.randn(self.batch_size, self.hidden_dim, device=self.device)
        if self.batch_size % self.num_shards != 0:
            raise RuntimeError("Batch size must be divisible by num_shards")
        self._reduced_rows = self.batch_size // self.num_shards
        self.output = None
        self._output_buffer = torch.empty((self._reduced_rows, self.hidden_dim), device=self.device)
        self.model.prepare_forward_buffers(self.input)
        with torch.inference_mode():
            model_out = self.model.forward_prepared(self.input)
        self._model_shard_view = model_out.view(self.num_shards, self._reduced_rows, self.hidden_dim)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        assert self.input is not None and self.model is not None
        
        with nvtx_range("optimized_gpu_reduction", enable=self._enable_nvtx):
            # Forward pass
            with torch.inference_mode():
                self.model.forward_prepared(self.input)
            
            # All-GPU reduction over a strided shard view, no CPU or Python shard loop.
            if self._output_buffer is None:
                raise RuntimeError("Output buffer not initialized")
            if self._model_shard_view is None:
                raise RuntimeError("Model shard view not initialized")
            torch.sum(self._model_shard_view, dim=0, out=self._output_buffer)
            
            # Average in place; the sum already landed in the reusable output buffer.
            self._output_buffer.div_(self.num_shards)
            self.output = self._output_buffer
        

    def capture_verification_payload(self) -> None:
        if self.input is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output,
            batch_size=int(self.batch_size),
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-4, 1e-4),
        )

    def teardown(self) -> None:
        self.model = None
        self.input = None
        self.output = None
        self._output_buffer = None
        self._model_shard_view = None
        self._reduced_rows = 0
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if self.input is None:
            return "Input not initialized"
        if self.output is None:
            return "Output not available"
        return None

    def get_input_signature(self) -> dict:
        """Return workload signature for input verification."""
        return super().get_input_signature()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return super().get_verify_output()

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison.
        
        CPU and GPU reductions may have slight numerical differences due to
        order of operations (floating point is not associative).
        """
        return (1e-4, 1e-4)



def get_benchmark() -> BaseBenchmark:
    return OptimizedGpuReductionBenchmark()
