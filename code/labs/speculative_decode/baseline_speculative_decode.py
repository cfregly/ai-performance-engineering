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
        self._input_token_view: Optional[torch.Tensor] = None
        self._output_ids: Optional[torch.Tensor] = None
        self._output_step_views: list[torch.Tensor] = []
        self._output_token_views: list[torch.Tensor] = []
        self._view_counts: tuple[int, int] = (0, 0)
        self._expected_view_counts: tuple[int, int] = (0, 0)
        self._token_range = range(0)
        self._next_token_values: Optional[torch.Tensor] = None
        self._target_logits: Optional[torch.Tensor] = None
        self._target_logits_next: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
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
        self._input_token_view = self.input_ids[:, 0]
        self._output_ids = torch.empty((1, wl.total_tokens + 1), device=self.device, dtype=torch.int64)
        self._verify_output_buffer = torch.empty_like(self._output_ids, dtype=torch.float32)
        self._output_step_views = [
            self._output_ids[:, token_idx : token_idx + 1] for token_idx in range(wl.total_tokens + 1)
        ]
        self._output_token_views = [
            self._output_ids[:, token_idx] for token_idx in range(wl.total_tokens + 1)
        ]
        self._view_counts = (len(self._output_step_views), len(self._output_token_views))
        self._expected_view_counts = (wl.total_tokens + 1, wl.total_tokens + 1)
        self._token_range = range(wl.total_tokens)
        # torch.max requires value and index outputs with matching strides.
        self._next_token_values = torch.empty_strided(
            (1,),
            self._output_token_views[0].stride(),
            device=self.device,
            dtype=wl.dtype,
        )
        self._target_logits = torch.empty((1, 1, wl.vocab_size), device=self.device, dtype=wl.dtype)
        self._target_logits_next = self._target_logits[:, 0, :]
        self.output = None
        self._synchronize()

    def benchmark_fn(self) -> None:
        if (
            self.target_model is None
            or self.input_ids is None
            or self._input_token_view is None
            or self._output_ids is None
            or self._next_token_values is None
            or self._target_logits is None
            or self._target_logits_next is None
            or self._view_counts != self._expected_view_counts
        ):
            raise RuntimeError("Benchmark not initialized")

        out = self._output_ids
        target_forward_into = self.target_model.forward_into
        output_step_views = self._output_step_views
        output_token_views = self._output_token_views
        token_range = self._token_range
        target_logits = self._target_logits
        target_logits_next = self._target_logits_next
        next_token_values = self._next_token_values
        output_token_views[0].copy_(self._input_token_view)

        with torch.inference_mode():
            for t in token_range:
                target_forward_into(output_step_views[t], target_logits)
                output_token = output_token_views[t + 1]
                torch.max(target_logits_next, dim=-1, out=(next_token_values, output_token))

        self.output = out

    def capture_verification_payload(self) -> None:
        if self.input_ids is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output, non_blocking=False)
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=self._verify_output_buffer,
            batch_size=1,
            parameter_count=self._payload_parameter_count,
            precision_flags={"bf16": False, "fp16": False, "fp8": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.target_model = None
        self.input_ids = None
        self._input_token_view = None
        self._output_ids = None
        self._output_step_views = []
        self._output_token_views = []
        self._view_counts = (0, 0)
        self._expected_view_counts = (0, 0)
        self._token_range = range(0)
        self._next_token_values = None
        self._target_logits = None
        self._target_logits_next = None
        self.output = None
        self._verify_output_buffer = None
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
