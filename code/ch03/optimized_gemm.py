"""Optimized comparison GEMM that removes launch fragmentation with `torch.compile`.

This remains a Chapter 3 host/runtime comparison workload. The optimization is
that `torch.compile(mode="reduce-overhead")` lets the runtime amortize the
fragmented launch pattern into one compiled GEMM path without changing the math.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
)


class OptimizedGemmBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Comparison workload with one compiled GEMM call."""

    story_metadata = {
        "pair_role": "comparison",
        "variant_role": "optimized",
        "chapter_alignment": "supplementary",
        "chapter_native_exemplar": False,
        "comparison_reason": (
            "Quantifies Chapter 3 host/runtime launch overhead without claiming a "
            "NUMA-local math-kernel optimization."
        ),
        "comparison_axis": "fragmented_vs_amortized_launches",
        "execution_pattern": "single_compiled_gemm_launch",
        "optimization_mechanism": (
            'use torch.compile(mode="reduce-overhead") to amortize launch '
            "fragmentation while keeping math fixed"
        ),
        "chapter_native_targets": ["pageable_copy", "rack_prep", "pinned_prefetch_mlp", "double_buffered_batch_provisioning"],
    }

    def __init__(self):
        super().__init__()
        # Matrix dimensions (must match baseline for verification)
        self.m = 2048
        self.n = 2048
        self.k = 2048
        
        self.left: Optional[torch.Tensor] = None
        self.right: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.fn = None
        
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Create input matrices - same as baseline version
        self.left = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.right = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        self._verify_output_buffer = torch.empty(self.m, self.n, device=self.device, dtype=torch.float32)
        self.output = None

        def matmul_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b)

        compile_fn = getattr(torch, "compile", None)
        if compile_fn is not None:
            self.fn = compile_fn(matmul_fn, mode="reduce-overhead")
        else:
            raise RuntimeError("torch.compile is required for this benchmark")

        # Warmup compiled function
        for _ in range(3):
            _ = self.fn(self.left, self.right)
        self._synchronize()

    def benchmark_fn(self) -> None:
        """Compute C = A @ B using a single compiled matmul.

        This keeps the GEMM identical to the baseline while using
        `torch.compile(mode="reduce-overhead")` to remove the fragmented launch
        pattern seen by the runtime.
        """
        assert self.left is not None and self.right is not None
        op = self.fn
        
        with torch.inference_mode(), self._nvtx_range("optimized_gemm"):
            result = op(self.left, self.right)
        
        self.output = result

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
        self._verify_output_buffer = None
        self.fn = None
        super().teardown()

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
                "launch.gemm_calls_per_iteration": 1.0,
                "launch.block_k": float(self.k),
            }
        )
        return metrics

    def get_optimization_goal(self) -> str:
        """Keep the host/runtime GEMM pair as a supplementary comparison workload."""
        return "comparison"

    def validate_result(self) -> Optional[str]:
        if self.fn is None:
            return "Compiled function not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedGemmBenchmark()
