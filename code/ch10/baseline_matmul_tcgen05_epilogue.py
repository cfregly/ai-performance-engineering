"""Baseline TMEM epilogue (separate matmul + bias + SiLU kernels)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from ch10.matmul_extension_tcgen05 import load_matmul_tcgen05_module
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.tcgen05_requirements import check_tcgen05_support


class BaselineMatmulTCGen05EpilogueBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: matmul followed by separate bias addition and SiLU activation.
    
    This is the naive approach that requires:
    1. Matmul kernel (write result to global memory)
    2. Bias addition kernel (read from global, write to global)
    3. SiLU activation kernel (read from global, write to global)
    
    The optimized version fuses these into a single kernel with TMEM-resident epilogue.
    """

    def __init__(self) -> None:
        super().__init__()
        available, reason = check_tcgen05_support(
            loader=None,
            module_name="ch10 matmul tcgen05 kernels",
        )
        self._tcgen05_available = available
        self._skip_reason = reason or "SKIPPED: tcgen05 matmul unavailable"
        self.module = None
        self.device = torch.device("cuda")
        # Choose dimensions where epilogue fusion is measurable:
        # keep K minimal (tcgen05 requires K%64==0) and use smaller M/N so
        # extra kernel launches and memory passes remain a visible fraction.
        self.M = 2048
        self.N = 2048
        self.K = 64
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._epilogue_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.register_workload_metadata(bytes_per_iteration=float((self.M * self.K + self.N * self.K) * 2))

    def setup(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        if self.module is None:
            self.module = load_matmul_tcgen05_module()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.A = torch.randn(self.M, self.K, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.N, self.K, device=self.device, dtype=torch.float16)
        # Match the tcgen05 fused epilogue: bias is promoted to FP32 before activation.
        self.bias = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.M, self.N, device=self.device, dtype=torch.float16)
        self._epilogue_buffer = torch.empty(self.M, self.N, device=self.device, dtype=torch.float32)
        self._verify_output_buffer = torch.empty(
            min(128, self.M),
            min(256, self.N),
            device=self.device,
            dtype=torch.float32,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        assert (
            self.A is not None
            and self.B is not None
            and self.bias is not None
            and self.output is not None
            and self._epilogue_buffer is not None
            and self.module is not None
        )
        with self._nvtx_range("baseline_matmul_tcgen05_bias_silu"):
            with torch.inference_mode():
                # Use the same tcgen05 GEMM kernel as optimized; keep bias+SiLU separate.
                C = self._epilogue_buffer
                C.copy_(self.module.matmul_tcgen05(self.A, self.B))
                # Step 2: Add bias (separate kernel launch)
                C.add_(self.bias)
                # Step 3: SiLU activation (separate kernel launch)
                F.silu(C, inplace=True)
                self.output.copy_(C)
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if (
            self.A is None
            or self.B is None
            or self.bias is None
            or self.output is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[
            : self._verify_output_buffer.shape[0],
            : self._verify_output_buffer.shape[1],
        ]
        self._verify_output_buffer.copy_(output_slice)
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B, "bias": self.bias},
            output=self._verify_output_buffer,
            batch_size=self.M,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(1e-2, 1e-2),
        )

    def teardown(self) -> None:
        self.A = None
        self.B = None
        self.bias = None
        self.output = None
        self._epilogue_buffer = None
        self._verify_output_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaselineMatmulTCGen05EpilogueBenchmark:
    return BaselineMatmulTCGen05EpilogueBenchmark()
