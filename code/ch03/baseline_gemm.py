"""Baseline comparison GEMM that serializes many small launches.

This is a Chapter 3 host/runtime comparison workload rather than a NUMA-specific
kernel study. It keeps the math fixed while fragmenting the launch pattern so
host-side scheduling overhead is measurable.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineGemmBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Comparison workload with many small GEMM launches."""

    story_metadata = {
        "pair_role": "comparison",
        "variant_role": "baseline",
        "chapter_alignment": "supplementary",
        "chapter_native_exemplar": False,
        "comparison_reason": (
            "Quantifies Chapter 3 host/runtime launch overhead without claiming a "
            "NUMA-local math-kernel optimization."
        ),
        "comparison_axis": "fragmented_vs_amortized_launches",
        "execution_pattern": "fragmented_gemm_launches",
        "chapter_native_targets": ["pageable_copy", "rack_prep", "pinned_prefetch_mlp", "double_buffered_batch_provisioning"],
    }

    def __init__(self):
        super().__init__()
        # Matrix dimensions (must match optimized for verification)
        self.m = 2048
        self.n = 2048
        self.k = 2048
        # Micro-batch size for blocked computation
        self.block_size = 256
        self.num_blocks = self.k // self.block_size
        self._block_range = range(self.num_blocks)
        
        self.left: Optional[torch.Tensor] = None
        self.right: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Create input matrices - same as optimized version
        self.left = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.right = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        self._output_buffer = torch.empty(self.m, self.n, device=self.device, dtype=torch.float32)
        self._verify_output_buffer = torch.empty_like(self._output_buffer)
        self._block_range = range(self.num_blocks)
        self.output = None
        self._synchronize()

    def benchmark_fn(self) -> None:
        """Compute C = A @ B using blocked micro-batches.

        The intent is to expose host/runtime launch overhead around a fixed
        GEMM, not to demonstrate a Chapter 3-specific math-kernel trick.
        """
        # Accumulate result from blocked matmul operations
        # C = A @ B = sum over blocks of (A[:, block_i] @ B[block_i, :])
        result = self._output_buffer
        if result is None:
            raise RuntimeError("Output buffer not initialized")
        result.zero_()
        
        with torch.inference_mode(), self._nvtx_range("baseline_gemm"):
            for i in self._block_range:
                start = i * self.block_size
                end = start + self.block_size
                # Extract block slices
                left_block = self.left[:, start:end]  # (m, block_size)
                right_block = self.right[start:end, :]  # (block_size, n)
                # Accumulate partial result
                result += torch.matmul(left_block, right_block)
        
        self.output = result
        if self.left is None or self.right is None:
            raise RuntimeError("benchmark_fn() must be called after setup initializes inputs")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"left": self.left, "right": self.right},
            output=self._verify_output_buffer,
            batch_size=self.left.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": True,
            },
            output_tolerance=(1e-4, 1e-3),
        )

    def teardown(self) -> None:
        self.left = None
        self.right = None
        self.output = None
        self._output_buffer = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics
        metrics = compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 0),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )
        metrics.update(
            {
                "story.comparison_pair": 1.0,
                "story.chapter_native_exemplar": 0.0,
                "launch.gemm_calls_per_iteration": float(self.num_blocks),
                "launch.block_k": float(self.block_size),
            }
        )
        return metrics

    def get_optimization_goal(self) -> str:
        """Keep the host/runtime GEMM pair as a supplementary comparison workload."""
        return "comparison"

    def validate_result(self) -> Optional[str]:
        if self.left is None or self.right is None:
            return "Input matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineGemmBenchmark()
