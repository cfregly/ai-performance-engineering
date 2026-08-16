"""Optimized dynamic-precision decode loop for Chapter 19."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from ch19.dynamic_precision_benchmark_common import (
    DynamicPrecisionBenchmarkConfig,
    build_model,
    build_prompt,
    decode_dynamic_precision_preallocated as decode_dynamic_precision,
)
from ch19.dynamic_precision_switching import DynamicPrecisionWorkspace, PrecisionStats


class OptimizedDynamicPrecisionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self, cfg: Optional[DynamicPrecisionBenchmarkConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or DynamicPrecisionBenchmarkConfig()
        self.model = None
        self.prompt = None
        self.output: Optional[torch.Tensor] = None
        self.stats = None
        self._decode_workspace: Optional[DynamicPrecisionWorkspace] = None
        self._verify_prompt_buffer: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.cfg.batch_size),
            tokens_per_iteration=float(self.cfg.batch_size * self.cfg.max_steps),
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: dynamic_precision requires CUDA")
        self.prompt = build_prompt(self.cfg, self.device)
        self.model = build_model(self.cfg, self.device, dtype=torch.bfloat16)
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        output_shape = (self.cfg.batch_size, self.cfg.prompt_len + self.cfg.max_steps)
        top2_shape = (self.cfg.batch_size, 2)
        self._decode_workspace = DynamicPrecisionWorkspace(
            generated=torch.empty(output_shape, device=self.device, dtype=self.prompt.dtype),
            next_token=torch.empty((self.cfg.batch_size, 1), device=self.device, dtype=self.prompt.dtype),
            next_token_values=torch.empty((self.cfg.batch_size, 1), device=self.device, dtype=torch.float32),
            top2_values=torch.empty(top2_shape, device=self.device, dtype=torch.float32),
            top2_indices=torch.empty(top2_shape, device=self.device, dtype=torch.long),
            margin_values=torch.empty(self.cfg.batch_size, device=self.device, dtype=torch.float32),
            margin_mean=torch.empty((), device=self.device, dtype=torch.float32),
            ema_conf=torch.empty((), device=self.device, dtype=torch.float32),
            stats=PrecisionStats(),
        )
        self._decode_workspace.next_token_flat = self._decode_workspace.next_token.view(
            self.cfg.batch_size
        )
        self._decode_workspace.generated_token_views = self._decode_workspace.generated.unbind(
            dim=1
        )
        self._verify_prompt_buffer = torch.empty(
            self.prompt.shape,
            device="cpu",
            dtype=self.prompt.dtype,
            pin_memory=True,
        )
        self._verify_output_buffer = torch.empty(
            output_shape,
            device="cpu",
            dtype=self.prompt.dtype,
            pin_memory=True,
        )

    def benchmark_fn(self) -> None:
        if self.model is None or self.prompt is None or self._decode_workspace is None:
            raise RuntimeError("dynamic_precision workload not initialized")
        prior_invocations = self._decode_workspace.decode_invocations
        with torch.inference_mode():
            self.output, self.stats = decode_dynamic_precision(
                self.model,
                self.prompt,
                max_steps=self.cfg.max_steps,
                device=self.device,
                workspace=self._decode_workspace,
            )
        if self._decode_workspace.decode_invocations != prior_invocations + 1:
            raise RuntimeError("dynamic_precision decode invocation was not executed")
        if (
            self.stats.decode_steps != self.cfg.max_steps
            or self.stats.model_forwards != self.cfg.max_steps
            or self.stats.policy_evaluations != self.cfg.max_steps
        ):
            raise RuntimeError("dynamic_precision mechanism counters are incomplete")

    def capture_verification_payload(self) -> None:
        if (
            self.prompt is None
            or self.output is None
            or self.model is None
            or self._verify_prompt_buffer is None
            or self._verify_output_buffer is None
        ):
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_prompt_buffer.copy_(self.prompt, non_blocking=False)
        self._verify_output_buffer.copy_(self.output, non_blocking=False)
        self._set_verification_payload(
            inputs={"prompt": self._verify_prompt_buffer},
            output=self._verify_output_buffer,
            batch_size=self.cfg.batch_size,
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.prompt = None
        self.output = None
        self.stats = None
        self._decode_workspace = None
        self._verify_prompt_buffer = None
        self._verify_output_buffer = None
        super().teardown()

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        if self.stats is None:
            return None
        return {
            "dynamic_precision.total_tokens": float(self.stats.total_tokens),
            "dynamic_precision.fp4_tokens": float(self.stats.fp4_tokens),
            "dynamic_precision.fp8_tokens": float(self.stats.fp8_tokens),
            "dynamic_precision.fp16_tokens": float(self.stats.fp16_tokens),
            "dynamic_precision.precision_switches": float(self.stats.precision_switches),
            "dynamic_precision.avg_confidence": float(self.stats.avg_confidence),
            "dynamic_precision.decode_steps": float(self.stats.decode_steps),
            "dynamic_precision.model_forwards": float(self.stats.model_forwards),
            "dynamic_precision.policy_evaluations": float(self.stats.policy_evaluations),
            "dynamic_precision.decode_invocations": float(
                self._decode_workspace.decode_invocations
                if self._decode_workspace is not None
                else 0
            ),
        }

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)


def get_benchmark() -> BaseBenchmark:
    return OptimizedDynamicPrecisionBenchmark()
