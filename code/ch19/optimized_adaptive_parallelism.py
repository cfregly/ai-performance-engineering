"""Optimized adaptive-parallelism benchmark using vectorized routing rules."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch19.adaptive_parallelism_benchmark_common import (
    AdaptiveParallelismBenchmarkConfig,
    build_workload,
    classify_vectorized_out,
)


class OptimizedAdaptiveParallelismBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self, cfg: Optional[AdaptiveParallelismBenchmarkConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or AdaptiveParallelismBenchmarkConfig()
        self.workload: Optional[Dict[str, torch.Tensor]] = None
        self.output: Optional[torch.Tensor] = None
        self._result_buffer: Optional[torch.Tensor] = None
        self._steady_decode_mask: Optional[torch.Tensor] = None
        self._data_mask: Optional[torch.Tensor] = None
        self._long_prefill_mask: Optional[torch.Tensor] = None
        self._heavy_context_mask: Optional[torch.Tensor] = None
        self._pipeline_mask: Optional[torch.Tensor] = None
        self._hybrid_mask: Optional[torch.Tensor] = None
        self._doubled_decode_tokens: Optional[torch.Tensor] = None
        self._verify_input_buffers: Optional[Dict[str, torch.Tensor]] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.cfg.num_requests),
            tokens_per_iteration=float(self.cfg.num_requests),
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: adaptive_parallelism requires CUDA")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.workload = build_workload(self.cfg, self.device)
        self._result_buffer = torch.empty(
            self.cfg.num_requests,
            device=self.device,
            dtype=torch.int64,
        )
        self._steady_decode_mask = torch.empty(
            self.cfg.num_requests,
            device=self.device,
            dtype=torch.bool,
        )
        self._data_mask = torch.empty_like(self._steady_decode_mask)
        self._long_prefill_mask = torch.empty_like(self._steady_decode_mask)
        self._heavy_context_mask = torch.empty_like(self._steady_decode_mask)
        self._pipeline_mask = torch.empty_like(self._steady_decode_mask)
        self._hybrid_mask = torch.empty_like(self._steady_decode_mask)
        self._doubled_decode_tokens = torch.empty_like(self.workload["decode_tokens"])
        self._verify_input_buffers = {
            name: torch.empty(
                tensor.shape,
                device="cpu",
                dtype=tensor.dtype,
                pin_memory=True,
            )
            for name, tensor in self.workload.items()
        }
        self._verify_output_buffer = torch.empty(
            self.cfg.num_requests,
            device="cpu",
            dtype=torch.int64,
            pin_memory=True,
        )

    def benchmark_fn(self) -> None:
        if (
            self.workload is None
            or self._result_buffer is None
            or self._steady_decode_mask is None
            or self._data_mask is None
            or self._long_prefill_mask is None
            or self._heavy_context_mask is None
            or self._pipeline_mask is None
            or self._hybrid_mask is None
            or self._doubled_decode_tokens is None
        ):
            raise RuntimeError("adaptive_parallelism workload not initialized")
        with torch.inference_mode():
            self.output = classify_vectorized_out(
                self.workload,
                result=self._result_buffer,
                steady_decode=self._steady_decode_mask,
                data_mask=self._data_mask,
                long_prefill=self._long_prefill_mask,
                heavy_context=self._heavy_context_mask,
                pipeline_mask=self._pipeline_mask,
                hybrid_mask=self._hybrid_mask,
                doubled_decode_tokens=self._doubled_decode_tokens,
            )

    def capture_verification_payload(self) -> None:
        if self.workload is None or self.output is None or self._verify_input_buffers is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._verify_output_buffer is None:
            raise RuntimeError("benchmark verification buffers are not initialized")
        for name, tensor in self.workload.items():
            self._verify_input_buffers[name].copy_(tensor, non_blocking=False)
        self._verify_output_buffer.copy_(self.output, non_blocking=False)
        self._set_verification_payload(
            inputs=self._verify_input_buffers,
            output=self._verify_output_buffer,
            batch_size=self.cfg.num_requests,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.workload = None
        self.output = None
        self._result_buffer = None
        self._steady_decode_mask = None
        self._data_mask = None
        self._long_prefill_mask = None
        self._heavy_context_mask = None
        self._pipeline_mask = None
        self._hybrid_mask = None
        self._doubled_decode_tokens = None
        self._verify_input_buffers = None
        self._verify_output_buffer = None
        super().teardown()

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)


def get_benchmark() -> BaseBenchmark:
    return OptimizedAdaptiveParallelismBenchmark()
