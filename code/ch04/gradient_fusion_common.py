"""Shared single-GPU gradient fusion benchmark logic."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

FLOAT32_BYTES = torch.finfo(torch.float32).bits // 8
__all__ = ["GradientFusionBenchmark", "attach_benchmark_metadata"]


class GradientFusionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Single-GPU gradient fusion benchmark (fused vs unfused reductions)."""

    def __init__(
        self,
        *,
        fused: bool,
        num_tensors: int = 256,
        tensor_kb: int = 32,
        reduction_repeats: int = 16,
        equivalence_group: str = "ch04_gradient_fusion_single",
    ) -> None:
        super().__init__()
        self.fused = bool(fused)
        self.num_tensors = int(num_tensors)
        self.tensor_kb = int(tensor_kb)
        self.reduction_repeats = max(1, int(reduction_repeats))
        self._repeat_tail_range = range(1, self.reduction_repeats)
        self.signature_equivalence_group = equivalence_group
        self.signature_equivalence_ignore_fields = ("precision_flags",)
        numel = max(1, (self.tensor_kb * 1024) // FLOAT32_BYTES)
        total_bytes = self.num_tensors * numel * FLOAT32_BYTES
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.reduction_repeats),
            tokens_per_iteration=float(total_bytes * self.reduction_repeats),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.reduction_repeats),
            tokens_per_iteration=float(total_bytes * self.reduction_repeats),
        )
        self.tensors: list[torch.Tensor] = []
        self.fused_tensor: Optional[torch.Tensor] = None
        self._seed_tensor: Optional[torch.Tensor] = None
        self._tail_tensors: list[torch.Tensor] = []
        self.output: Optional[torch.Tensor] = None
        self._verify_input: Optional[torch.Tensor] = None
        self._accum_buffer: Optional[torch.Tensor] = None
        self._sum_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for gradient fusion benchmark")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        numel = max(1, (self.tensor_kb * 1024) // FLOAT32_BYTES)
        self.tensors = [
            torch.randn(numel, device=self.device, dtype=torch.float32)
            for _ in range(self.num_tensors)
        ]
        self.fused_tensor = torch.empty(
            self.num_tensors * numel,
            device=self.device,
            dtype=torch.float32,
        )
        offset = 0
        for tensor in self.tensors:
            next_offset = offset + numel
            self.fused_tensor[offset:next_offset].copy_(tensor.view(-1))
            offset = next_offset
        self._seed_tensor = self.tensors[0]
        self._tail_tensors = self.tensors[1:]
        self._verify_input = self.tensors[0]
        self._accum_buffer = torch.empty((), device=self.device, dtype=torch.float32)
        self._sum_buffer = torch.empty_like(self._accum_buffer)
        self._verify_output_buffer = torch.empty_like(self._accum_buffer)

    def benchmark_fn(self) -> None:
        if not self.tensors or self.fused_tensor is None or self._seed_tensor is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        accum = self._accum_buffer
        sum_buffer = self._sum_buffer
        if accum is None or sum_buffer is None:
            raise RuntimeError("setup() must initialize reduction buffers")
        if self.fused:
            torch.sum(self.fused_tensor, dim=None, out=accum)
            for _ in self._repeat_tail_range:
                torch.sum(self.fused_tensor, dim=None, out=sum_buffer)
                accum.add_(sum_buffer)
        else:
            torch.sum(self._seed_tensor, dim=None, out=accum)
            for tensor in self._tail_tensors:
                torch.sum(tensor, dim=None, out=sum_buffer)
                accum.add_(sum_buffer)
            for _ in self._repeat_tail_range:
                for tensor in self.tensors:
                    torch.sum(tensor, dim=None, out=sum_buffer)
                    accum.add_(sum_buffer)
        self.output = accum

    def capture_verification_payload(self) -> None:
        if self._verify_input is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"probe": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=int(self._verify_input.shape[0]),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-3, 1e-2),
            signature_overrides={
                "world_size": 1,
                "collective_type": "all_reduce",
            },
        )

    def teardown(self) -> None:
        self.tensors = []
        self.fused_tensor = None
        self._seed_tensor = None
        self._tail_tensors = []
        self.output = None
        self._verify_input = None
        self._accum_buffer = None
        self._sum_buffer = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
