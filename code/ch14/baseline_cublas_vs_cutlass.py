"""baseline_cublas_vs_cutlass.py - Explicit cuBLAS baseline for the CUTLASS pair."""

from __future__ import annotations

import torch

from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


class BaselineCublasVsCutlassBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Supplementary comparison pair: explicit cuBLAS FP16 GEMM baseline."""

    story_metadata = {
        "pair_role": "comparison",
        "variant_role": "baseline",
        "chapter_alignment": "supplementary",
        "chapter_native_exemplar": False,
        "comparison_reason": (
            "Keeps the math fixed while comparing explicit vendor-library dispatch "
            "against explicit CUTLASS dispatch. This is not the chapter-native "
            "TorchInductor/Triton story."
        ),
        "comparison_axis": "explicit_cublas_vs_explicit_cutlass",
        "execution_pattern": "single_library_gemm",
        "optimization_mechanism": "explicit cuBLAS FP16 GEMM",
        "chapter_native_targets": ["model_compile_reduced_precision", "sliding_window", "triton_persistent"],
    }

    def __init__(self):
        super().__init__()
        self.A: torch.Tensor | None = None
        self.B: torch.Tensor | None = None
        self.C: torch.Tensor | None = None
        self.m = 4096
        self.n = 4096
        self.k = 4096
        self._cublas_gemm = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )
        self._verification_payload = None
        self._enable_nvtx = False
    
    def setup(self) -> None:
        """Setup: Initialize matrices and bind the explicit cuBLAS helper."""
        torch.manual_seed(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        self.C = None
        try:
            from core.benchmark.cutlass_binding import cublas_gemm_fp16
        except Exception as exc:
            raise RuntimeError(
                "SKIPPED: explicit cuBLAS/CUTLASS extension unavailable for baseline_cublas_vs_cutlass benchmark."
            ) from exc
        self._cublas_gemm = cublas_gemm_fp16

    def benchmark_fn(self) -> None:
        """Benchmark: explicit cuBLAS FP16 GEMM."""
        with nvtx_range("baseline_cublas_vs_cutlass", enable=self._enable_nvtx):
            if self.A is None or self.B is None or self._cublas_gemm is None:
                raise RuntimeError("Benchmark not initialized")
            self.C = self._cublas_gemm(self.A, self.B)
        if self.A is None or self.B is None or self.C is None:
            raise RuntimeError("Benchmark not initialized")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={
                "A": self.A.detach(),
                "B": self.B.detach(),
            },
            output=self.C.detach(),
            batch_size=1,
            parameter_count=0,
            output_tolerance=(0.1, 2.0),
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        self.C = None
        self._cublas_gemm = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
            backend_policy="fp32_strict",
        )

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_gemm_metrics
        metrics = compute_gemm_metrics(
            self.m, self.n, self.k,
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            precision="fp16",
            bytes_per_element=2,
        )
        metrics.update(
            {
                "story.comparison_pair": 1.0,
                "story.chapter_native_exemplar": 0.0,
                "gemm.explicit_cublas": 1.0,
                "gemm.explicit_cutlass": 0.0,
            }
        )
        return metrics

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.C is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineCublasVsCutlassBenchmark()
