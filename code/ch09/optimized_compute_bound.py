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


class BufferedVectorMlp(nn.Module):
    """Vector MLP with reusable inference buffers for CUDA graph capture."""

    def __init__(self, hidden_dim: int, inner_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self._fc1_buffer: Optional[torch.Tensor] = None
        self._fc2_buffer: Optional[torch.Tensor] = None

    def _ensure_forward_buffers(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            self._fc1_buffer is None
            or self._fc1_buffer.shape != (self.fc1.out_features,)
            or self._fc1_buffer.device != x.device
            or self._fc1_buffer.dtype != x.dtype
        ):
            self._fc1_buffer = torch.empty(self.fc1.out_features, device=x.device, dtype=x.dtype)
        if (
            self._fc2_buffer is None
            or self._fc2_buffer.shape != (self.fc2.out_features,)
            or self._fc2_buffer.device != x.device
            or self._fc2_buffer.dtype != x.dtype
        ):
            self._fc2_buffer = torch.empty(self.fc2.out_features, device=x.device, dtype=x.dtype)
        return self._fc1_buffer, self._fc2_buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_grad_enabled():
            x = self.relu(self.fc1(x))
            return self.fc2(x)

        fc1_out, fc2_out = self._ensure_forward_buffers(x)
        torch.mv(self.fc1.weight, x, out=fc1_out)
        if self.fc1.bias is not None:
            fc1_out.add_(self.fc1.bias)
        self.relu(fc1_out)
        torch.mv(self.fc2.weight, fc1_out, out=fc2_out)
        if self.fc2.bias is not None:
            fc2_out.add_(self.fc2.bias)
        return fc2_out


class OptimizedComputeBoundBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Compute-bound kernel - uses CUDA graphs to cut launch overhead."""
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._graph: Optional[torch.cuda.CUDAGraph] = None
        self._static_output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.repeats = 16
        self.N = 4096
        self._repeat_range = range(self.repeats)
        self._payload_parameter_count = 0
        tokens = self.N * self.repeats
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: initialize model, inputs, and capture a CUDA graph replay."""
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.model = BufferedVectorMlp(self.N, self.N * 2).to(self.device, dtype=torch.float16).eval()
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float16)
        self._verify_output_buffer = torch.empty_like(self.input)
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())

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
            for _ in self._repeat_range:
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
        with torch.inference_mode(), self._nvtx_range("optimized_compute_bound"):
            self._graph.replay()
            self.output = self._static_output
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self._verify_output_buffer,
            batch_size=self.input.shape[0],
            parameter_count=self._payload_parameter_count,
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
        self.output = None
        self._graph = None
        self._static_output = None
        self._verify_output_buffer = None
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
