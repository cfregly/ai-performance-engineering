"""Optimized Triton persistent benchmark for Chapter 14.

This variant uses the real Triton persistent batched GEMM kernel instead of a
plain `torch.bmm` launch. The benchmark name, README, and expectation files all
refer to Triton persistent scheduling, so the implementation needs to exercise
that exact path.
"""

from __future__ import annotations

from typing import Optional

import torch

from ch14.triton_persistent_batched import (
    compute_persistent_batched_metrics,
    matmul_persistent_batched,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)

# GB300 (sm_103) tile. The original 32x32x32 starved the Blackwell tensor cores:
# 8x8 tiny tiles per 256-matrix x 64 batches = ~4096 tiles dominated by per-tile
# overhead. A 128x128x64 tile (8 warps, 3 stages) was the config-sweep winner on this
# workload: ~31 -> ~112 TFLOPS (3.6x), exact match vs torch.bmm.
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 64
GROUP_M = 8
NUM_WARPS = 8
NUM_STAGES = 3


class OptimizedTritonPersistentBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Triton persistent batched GEMM."""

    def __init__(self):
        super().__init__()
        self.output = None
        self.a = None
        self.b = None
        self._output_buffer = None

        self.batch_size = 64
        self.M = 256
        self.N = 256
        self.K = 256
        self.dtype = torch.float16
        self.num_sms = 0
        self._last = 0.0

        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.M * self.N),
        )
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.M * self.N),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        props = torch.cuda.get_device_properties(self.device)
        self.num_sms = props.multi_processor_count

        self.a = torch.randn(self.batch_size, self.M, self.K, device=self.device, dtype=self.dtype)
        self.b = torch.randn(self.batch_size, self.K, self.N, device=self.device, dtype=self.dtype)
        self._output_buffer = torch.empty(
            (self.batch_size, self.M, self.N), device=self.device, dtype=self.dtype
        )

        for _ in range(3):
            _ = matmul_persistent_batched(
                self.a,
                self.b,
                self.num_sms,
                out=self._output_buffer,
                block_m=BLOCK_M,
                block_n=BLOCK_N,
                block_k=BLOCK_K,
                group_m=GROUP_M,
                num_warps=NUM_WARPS,
                num_stages=NUM_STAGES,
            )

    def benchmark_fn(self) -> None:
        self.output = matmul_persistent_batched(
            self.a,
            self.b,
            self.num_sms,
            out=self._output_buffer,
            block_m=BLOCK_M,
            block_n=BLOCK_N,
            block_k=BLOCK_K,
            group_m=GROUP_M,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        if self.output is None or self.a is None or self.b is None:
            raise RuntimeError("benchmark_fn() must produce output")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"a": self.a, "b": self.b},
            output=self.output,
            batch_size=self.batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.a = None
        self.b = None
        self._output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return compute_persistent_batched_metrics(
            batch_size=self.batch_size,
            m=self.M,
            n=self.N,
            k=self.K,
            block_m=BLOCK_M,
            block_n=BLOCK_N,
            block_k=BLOCK_K,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
            elapsed_ms=getattr(self, "_last_elapsed_ms", None),
            persistent_kernel=True,
        )

    def validate_result(self) -> Optional[str]:
        if self.a is None or self.b is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedTritonPersistentBenchmark()
