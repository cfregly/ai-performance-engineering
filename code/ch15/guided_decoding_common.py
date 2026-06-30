"""Shared harness logic for Chapter 15 guided decoding benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.wrapper_utils import attach_benchmark_metadata  # noqa: F401
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


@dataclass(frozen=True)
class GuidedDecodingConfig:
    """Workload configuration for the guided decoding benchmark family."""

    batch_size: int = 32
    steps: int = 96
    vocab_size: int = 65536
    allowed_count: int = 8192
    output_slice: int = 256


class GuidedDecodingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Parameterized guided decoding benchmark with or without mask reuse."""

    def __init__(
        self,
        *,
        reuse_gpu_mask: bool,
        label: str,
        cfg: Optional[GuidedDecodingConfig] = None,
    ) -> None:
        super().__init__()
        self.reuse_gpu_mask = bool(reuse_gpu_mask)
        self.label = label
        self.cfg = cfg or GuidedDecodingConfig()
        self.batch_size = int(self.cfg.batch_size)
        self.steps = int(self.cfg.steps)
        self.vocab_size = int(self.cfg.vocab_size)
        self.allowed_count = int(self.cfg.allowed_count)
        self.output_slice = int(self.cfg.output_slice)
        self._step_range = range(self.steps)

        tokens = self.batch_size * self.steps
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

        self.logits: Optional[torch.Tensor] = None
        self.allowed_token_ids: Optional[torch.Tensor] = None
        self.allowed_slice_cpu: Optional[torch.Tensor] = None
        self.slice_ids: Optional[torch.Tensor] = None
        self.cpu_mask_buffer: Optional[torch.Tensor] = None
        self.gpu_mask_buffer: Optional[torch.Tensor] = None
        self.disallowed_mask_buffer: Optional[torch.Tensor] = None
        self.masked_logits_buffer: Optional[torch.Tensor] = None
        self.slice_ids_buffer: Optional[torch.Tensor] = None
        self.output_buffer: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.logits = torch.randn(
            self.batch_size,
            self.vocab_size,
            device=self.device,
            dtype=torch.float32,
        )
        self.allowed_token_ids = torch.randperm(
            self.vocab_size,
            device="cpu",
            dtype=torch.int64,
        )[: self.allowed_count]
        self.allowed_slice_cpu = self.allowed_token_ids[: self.output_slice]

        if self.reuse_gpu_mask:
            disallowed = torch.ones(self.vocab_size, dtype=torch.bool, device=self.device)
            disallowed[self.allowed_token_ids.to(self.device)] = False
            self.disallowed_mask_buffer = disallowed
            self.slice_ids = self.allowed_slice_cpu.to(self.device)
            self.cpu_mask_buffer = None
            self.gpu_mask_buffer = None
            self.slice_ids_buffer = None
        else:
            self.slice_ids = None
            self.cpu_mask_buffer = torch.empty(self.vocab_size, dtype=torch.bool, device="cpu")
            self.gpu_mask_buffer = torch.empty(self.vocab_size, dtype=torch.bool, device=self.device)
            self.disallowed_mask_buffer = torch.empty(self.vocab_size, dtype=torch.bool, device=self.device)
            self.slice_ids_buffer = torch.empty(self.output_slice, dtype=torch.int64, device=self.device)
        self.masked_logits_buffer = torch.empty_like(self.logits)
        self.output_buffer = torch.empty(
            self.batch_size,
            self.output_slice,
            dtype=self.logits.dtype,
            device=self.device,
        )
        self._step_range = range(self.steps)
        self.output = None

    def benchmark_fn(self) -> None:
        if self.logits is None or self.allowed_token_ids is None:
            raise RuntimeError("Benchmark not initialized")
        if self.masked_logits_buffer is None or self.output_buffer is None:
            raise RuntimeError("Output buffers not initialized")

        logits = self.logits
        allowed = self.allowed_token_ids
        allowed_slice = self.allowed_slice_cpu
        masked_logits = self.masked_logits_buffer
        output = self.output_buffer

        with torch.inference_mode(), self._nvtx_range(self.label):
            for _ in self._step_range:
                if self.reuse_gpu_mask:
                    if self.disallowed_mask_buffer is None or self.slice_ids is None:
                        raise RuntimeError("GPU mask state not initialized")
                    masked_logits.copy_(logits)
                    masked_logits.masked_fill_(self.disallowed_mask_buffer, float("-inf"))
                    torch.index_select(masked_logits, 1, self.slice_ids, out=output)
                    continue

                if (
                    self.cpu_mask_buffer is None
                    or self.gpu_mask_buffer is None
                    or self.disallowed_mask_buffer is None
                    or self.slice_ids_buffer is None
                    or allowed_slice is None
                ):
                    raise RuntimeError("CPU/GPU mask buffers not initialized")
                mask_cpu = self.cpu_mask_buffer
                mask_cpu.zero_()
                mask_cpu[allowed] = True
                self.gpu_mask_buffer.copy_(mask_cpu, non_blocking=False)
                torch.logical_not(self.gpu_mask_buffer, out=self.disallowed_mask_buffer)
                self.slice_ids_buffer.copy_(allowed_slice, non_blocking=False)
                masked_logits.copy_(logits)
                masked_logits.masked_fill_(self.disallowed_mask_buffer, float("-inf"))
                torch.index_select(masked_logits, 1, self.slice_ids_buffer, out=output)
        self.output = output

        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.logits is None or self.allowed_token_ids is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={
                "logits": self.logits,
                "allowed_token_ids": self.allowed_token_ids,
            },
            output=self.output,
            batch_size=int(self.batch_size),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.logits = None
        self.allowed_token_ids = None
        self.allowed_slice_cpu = None
        self.slice_ids = None
        self.cpu_mask_buffer = None
        self.gpu_mask_buffer = None
        self.disallowed_mask_buffer = None
        self.masked_logits_buffer = None
        self.slice_ids_buffer = None
        self.output_buffer = None
        self.output = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None
