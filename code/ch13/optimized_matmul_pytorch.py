"""optimized_matmul_pytorch.py - PyTorch matmul with fused eager epilogue.

Uses the same FP16 compute dtype as the eager baseline, but replaces the
`matmul -> add -> relu -> add -> mul` chain with an `addmm`-based epilogue
that stays on the fast cuBLAS path on current B200 stacks.
"""

from __future__ import annotations

import torch
from typing import Optional

from ch13.matmul_epilogue_extension import load_matmul_epilogue_extension
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


def optimized_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor,
    residual: torch.Tensor,
    out: torch.Tensor,
    scale: float,
    extension,
) -> torch.Tensor:
    """Keep GEMM on the fastest cuBLAS path and fuse the elementwise epilogue."""
    torch.mm(A, B, out=out)
    extension.matmul_epilogue_(out, bias, residual, scale)
    return out


class OptimizedMatmulPyTorchBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """PyTorch matmul optimization using a fused eager epilogue."""

    signature_equivalence_group = "ch13_matmul_pytorch_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.C = None
        self._verify_output_buffer = None
        self.bias = None
        self.residual = None
        self.extension = None
        self.scale = 0.125
        # Match baseline dimensions for fair comparison
        self.m = 4096
        self.n = 4096
        self.k = 4096
        tokens = self.m * self.n
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        # Register workload metadata at init time for compliance check
        bytes_per_iter = (self.m * self.k + self.k * self.n + self.m * self.n * 3) * 2
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
            bytes_per_iteration=float(bytes_per_iter),
        )
    
    def setup(self) -> None:
        """Setup: Initialize matrices."""
        
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.extension = load_matmul_epilogue_extension()

        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        self.bias = torch.randn(self.m, self.n, device=self.device, dtype=torch.float16)
        self.residual = torch.randn(self.m, self.n, device=self.device, dtype=torch.float16)
        self.C = torch.empty(self.m, self.n, device=self.device, dtype=torch.float16)
        self._verify_output_buffer = torch.empty_like(self.C)
        for _ in range(10):
            _ = optimized_matmul(
                self.A,
                self.B,
                self.bias,
                self.residual,
                self.C,
                self.scale,
                self.extension,
            )
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - compiled matmul."""
        assert (
            self.A is not None
            and self.B is not None
            and self.bias is not None
            and self.residual is not None
            and self.C is not None
            and self.extension is not None
        )
        with torch.inference_mode(), self._nvtx_range("matmul_pytorch"):
            self.C = optimized_matmul(
                self.A,
                self.B,
                self.bias,
                self.residual,
                self.C,
                self.scale,
                self.extension,
            )

    def capture_verification_payload(self) -> None:
        if self.C is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.C)
        self._set_verification_payload(
            inputs={
                "A": self.A,
                "B": self.B,
                "bias": self.bias,
                "residual": self.residual,
            },
            output=self._verify_output_buffer,
            batch_size=self.m,
            parameter_count=0,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        """Cleanup."""
        del self.A, self.B, self.bias, self.residual, self.C
        self.extension = None
        self._verify_output_buffer = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=180,
            nsys_timeout_seconds=1200,
            nsys_preset_override="light",
        )
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=None,
            reduced_precision_time_ms=getattr(self, '_last_elapsed_ms', None),
            precision_type="fp16",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.bias is None or self.residual is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedMatmulPyTorchBenchmark()
