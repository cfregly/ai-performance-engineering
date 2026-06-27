"""optimized_compute_bound.py - Optimized compute-bound kernel.

Optimization strategy: capture the repeated MLP chain with CUDA graphs to
eliminate Python dispatch and per-op launch overhead while keeping the math,
shapes, and dtypes identical to the baseline.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class OptimizedComputeBoundBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Compute-bound kernel - uses CUDA graphs to cut launch overhead."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_output: Optional[torch.Tensor] = None
        self.repeats = 16
        self.N = 4096
        tokens = self.N * self.repeats
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: initialize model, inputs, and capture a CUDA graph replay."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = nn.Sequential(
            nn.Linear(self.N, self.N * 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.N * 2, self.N),
        ).to(self.device, dtype=torch.float16).eval()
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float16)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for compute-bound CUDA graph capture")

        # Warm up to initialize cuBLAS handles and any lazy kernels.
        with torch.inference_mode():
            out = self.input
            for _ in range(2):
                out = self.model(out)

        # Capture the full repeated chain into a CUDA graph.
        graph = torch.cuda.CUDAGraph()
        static_output: Optional[torch.Tensor] = None
        with torch.cuda.graph(graph):
            out = self.input
            for _ in range(self.repeats):
                out = self.model(out)
            static_output = out

        if static_output is None:
            raise RuntimeError("CUDA graph capture failed to produce output")
        self._graph = graph
        self._static_output = static_output
    
    def benchmark_fn(self) -> None:
        """Benchmark: replay captured CUDA graph."""
        if self._graph is None or self._static_output is None:
            raise RuntimeError("CUDA graph not initialized")
        with self._nvtx_range("optimized_compute_bound"):
            self._graph.replay()
            self.output = self._static_output
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output.detach().clone(),
            batch_size=self.input.shape[0],
            parameter_count=sum(p.numel() for p in self.model.parameters()),
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-1, 1e-1),
        )
    
    def teardown(self) -> None:
        self.model = None
        self.input = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=5,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
            nsys_timeout_seconds=1200,
            nsys_preset_override="light",
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return compute-bound analysis metrics using the centralized helper."""
        from core.benchmark.metrics import compute_roofline_metrics
        # Same FLOP/byte estimates as baseline model.
        layer1_flops = 2 * self.N * (self.N * 2) * self.N
        layer2_flops = 2 * self.N * self.N * (self.N * 2)
        total_flops = (layer1_flops + layer2_flops) * self.repeats
        element_size = 2  # FP16
        total_bytes = (self.N + self.N) * element_size * self.repeats
        return compute_roofline_metrics(
            total_flops=total_flops,
            total_bytes=total_bytes,
            elapsed_ms=getattr(self, "_last_elapsed_ms", None),
            precision="fp16",
        )
    
    def validate_result(self) -> Optional[str]:
        if self.input is None or self.model is None:
            return "Model/input not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedComputeBoundBenchmark()

