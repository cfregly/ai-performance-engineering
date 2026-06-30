"""baseline_nccl.py - Baseline: CPU-based reduction (inefficient).

This baseline copies each shard to CPU for aggregation, then back to GPU.
This is visibly slower than the optimized path which stays on GPU.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from typing import Optional

from ch04.reduction_common import ReusableReductionMlp
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin


class BaselineNcclBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: CPU-based reduction - copies each shard to CPU for aggregation."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)
    
    def __init__(self):
        super().__init__()
        # Match optimized workload size
        self.batch_size = 1024
        self.hidden_dim = 4096
        self.inner_dim = 8192
        self.num_shards = 8  # More shards = more CPU round-trips
        
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._cpu_shard_buffers: list[torch.Tensor] = []
        self._cpu_shard_copy_pairs: list[tuple[int, torch.Tensor]] = []
        self._cpu_shard_count = 0
        self._reduced_rows = 0
        self._bytes_transferred: float = 0.0
        self._payload_parameter_count = 0
        
        tokens = self.batch_size * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        # Reduction benchmark: fixed dimensions
    
    def setup(self) -> None:
        """Setup: Initialize model."""
        torch.manual_seed(42)
        
        self.model = ReusableReductionMlp(self.hidden_dim, self.inner_dim).to(self.device).eval()
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        
        self.input = torch.randn(self.batch_size, self.hidden_dim, device=self.device)
        if self.batch_size % self.num_shards != 0:
            raise RuntimeError("Batch size must be divisible by num_shards")
        self._reduced_rows = self.batch_size // self.num_shards
        self._output_buffer = torch.empty((self._reduced_rows, self.hidden_dim), device=self.device)
        use_pinned_host = self.input.device.type == "cuda" and torch.cuda.is_available()
        self._cpu_shard_buffers = [
            torch.empty(
                (self._reduced_rows, self.hidden_dim),
                dtype=self.input.dtype,
                device="cpu",
                pin_memory=use_pinned_host,
            )
            for _ in range(self.num_shards)
        ]
        self._cpu_shard_copy_pairs = list(enumerate(self._cpu_shard_buffers))
        self._cpu_shard_count = len(self._cpu_shard_buffers)
        element_size = self.input.element_size()
        self._bytes_transferred = float(
            (self.batch_size * self.hidden_dim + self._reduced_rows * self.hidden_dim)
            * element_size
        )
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: CPU-based reduction (inefficient)."""
        assert self.input is not None and self.model is not None
        assert self._output_buffer is not None
        assert self._cpu_shard_count == self.num_shards

        with self._nvtx_range("baseline_nccl"):
            with torch.inference_mode():
                output = self.model(self.input)
                
                # Naively copy each shard to CPU to aggregate (slow!)
                shards = output.view(self.num_shards, self._reduced_rows, self.hidden_dim)
                # This is the bottleneck: multiple GPU->CPU copies + CPU addition
                for idx, shard in self._cpu_shard_copy_pairs:
                    shard.copy_(shards[idx], non_blocking=False)
                reduced = self._cpu_shard_buffers[0]
                for shard in self._cpu_shard_buffers[1:]:
                    reduced.add_(shard)
                reduced.div_(float(self.num_shards))
                # Copy result back to GPU without allocating a fresh destination.
                self._output_buffer.copy_(reduced, non_blocking=False)
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
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self.output = None
        self._output_buffer = None
        self._cpu_shard_buffers = []
        self._cpu_shard_copy_pairs = []
        self._cpu_shard_count = 0
        self._reduced_rows = 0
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
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
            bytes_transferred=self._bytes_transferred,
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            transfer_type="pcie",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
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
    """Factory function for harness discovery."""
    return BaselineNcclBenchmark()
