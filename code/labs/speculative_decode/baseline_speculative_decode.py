"""Baseline speculative decoding: sequential greedy decode with the target model."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

from labs.speculative_decode.speculative_decode_common import TokenMLP, default_workload, scale_tail_dims_


class BaselineSpeculativeDecodeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline decode loop: one target-model forward pass per generated token."""

    def __init__(self) -> None:
        super().__init__()

        # Use FP32 for deterministic argmax stability across launch/shape variations.
        self.workload = default_workload(dtype=torch.float32)

        self.target_model: Optional[TokenMLP] = None
        self.input_ids: Optional[torch.Tensor] = None
        self._output_ids: Optional[torch.Tensor] = None
        self._output_step_views: list[torch.Tensor] = []
        self._output_token_views: list[torch.Tensor] = []
        self._next_token_values: Optional[torch.Tensor] = None
        self._next_token_ids: Optional[torch.Tensor] = None
        self._target_logits: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._payload_parameter_count = 0

        tokens = float(self.workload.total_tokens)
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens,
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)

        wl = self.workload
        self.target_model = TokenMLP(
            vocab_size=wl.vocab_size,
            hidden_size=wl.target_hidden,
            num_layers=wl.target_layers,
            device=self.device,
            dtype=wl.dtype,
        ).eval()
        scale_tail_dims_(self.target_model, wl.draft_hidden, wl.tail_scale)
        self._payload_parameter_count = sum(p.numel() for p in self.target_model.parameters())

        self.input_ids = torch.randint(0, wl.vocab_size, (1, 1), device=self.device, dtype=torch.int64)
        self._output_ids = torch.empty((1, wl.total_tokens + 1), device=self.device, dtype=torch.int64)
        self._output_step_views = [
            self._output_ids[:, token_idx : token_idx + 1] for token_idx in range(wl.total_tokens + 1)
        ]
        self._output_token_views = [
            self._output_ids[:, token_idx] for token_idx in range(wl.total_tokens + 1)
        ]
        self._next_token_values = torch.empty((1,), device=self.device, dtype=wl.dtype)
        self._next_token_ids = torch.empty((1,), device=self.device, dtype=torch.long)
        self._target_logits = torch.empty((1, 1, wl.vocab_size), device=self.device, dtype=wl.dtype)
        self.output = None
        self._synchronize()

    def benchmark_fn(self) -> None:
        if (
            self.target_model is None
            or self.input_ids is None
            or self._output_ids is None
            or self._next_token_values is None
            or self._next_token_ids is None
            or self._target_logits is None
            or len(self._output_step_views) != self.workload.total_tokens + 1
            or len(self._output_token_views) != self.workload.total_tokens + 1
        ):
            raise RuntimeError("Benchmark not initialized")

        wl = self.workload
        out = self._output_ids
        self._output_token_views[0].copy_(self.input_ids[:, 0])

        with torch.inference_mode():
            for t in range(wl.total_tokens):
                logits = self.target_model.forward_into(self._output_step_views[t], self._target_logits)
                torch.max(logits[:, 0, :], dim=-1, out=(self._next_token_values, self._next_token_ids))
                self._output_token_views[t + 1].copy_(self._next_token_ids)

        self.output = out

    def capture_verification_payload(self) -> None:
        if self.input_ids is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=self.output.float(),
            batch_size=1,
            parameter_count=self._payload_parameter_count,
            precision_flags={"bf16": False, "fp16": False, "fp8": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.target_model = None
        self.input_ids = None
        self._output_ids = None
        self._output_step_views = []
        self._output_token_views = []
        self._next_token_values = None
        self._next_token_ids = None
        self._target_logits = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if self.output.shape[-1] != self.workload.total_tokens + 1:
            return "Unexpected output shape"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineSpeculativeDecodeBenchmark()
