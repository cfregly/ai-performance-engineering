"""Baseline per-token decode: one kernel launch per timestep."""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    build_decode_input_signature,
    get_decode_options,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


class BaselinePersistentDecodeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Naive decode: host loop over tokens, launch work per timestep."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.options = get_decode_options()
        self.inputs = None
        self.output: Optional[torch.Tensor] = None
        self._product_buffer: Optional[torch.Tensor] = None
        self._dot_buffer: Optional[torch.Tensor] = None
        self._decode_step_views: tuple[
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            ...,
        ] = ()
        self._output_view: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        batch, seq_len, head_dim = resolve_shapes()
        self.seq_len = seq_len
        self.batch = batch
        self.batch_size = batch
        self.head_dim = head_dim
        self.hidden_dim = head_dim
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.inputs = build_inputs(self.device)
        self.batch = int(self.inputs.q.shape[0])
        self.seq_len = int(self.inputs.q.shape[1])
        self.head_dim = self.inputs.q.shape[-1]
        self.hidden_dim = self.head_dim
        self._product_buffer = torch.empty(
            self.batch,
            self.head_dim,
            device=self.inputs.q.device,
            dtype=self.inputs.q.dtype,
        )
        self._dot_buffer = torch.empty(
            self.batch,
            1,
            device=self.inputs.q.device,
            dtype=self.inputs.q.dtype,
        )
        self._verify_output_buffer = torch.empty(
            1,
            min(8, self.seq_len),
            self.head_dim,
            device=self.inputs.out.device,
            dtype=torch.float32,
        )
        self._decode_step_views = tuple(
            zip(
                self.inputs.q.unbind(1),
                self.inputs.k.unbind(1),
                self.inputs.v.unbind(1),
                self.inputs.out.unbind(1),
                strict=True,
            )
        )
        self._output_view = self.inputs.out[:1, : min(8, self.inputs.out.shape[1])]
        self._synchronize()

    def _decode_step(
        self,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        out_t: torch.Tensor,
    ) -> None:
        assert self._product_buffer is not None
        assert self._dot_buffer is not None
        # Compute a simple dot per sequence for timestep t, then scale V.
        product = self._product_buffer
        dot = self._dot_buffer

        torch.mul(q_t, k_t, out=product)
        torch.sum(product, dim=-1, keepdim=True, out=dot)
        torch.mul(v_t, dot, out=out_t)

    def benchmark_fn(self) -> None:
        if self.inputs is None or self._output_view is None or not self._decode_step_views:
            raise RuntimeError("Inputs not initialized")

        with torch.inference_mode(), self._nvtx_range("baseline_per_token"):
            for q_t, k_t, v_t, out_t in self._decode_step_views:
                self._decode_step(q_t, k_t, v_t, out_t)
            self.output = self._output_view
        if self.inputs is None or self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={
                "q": self.inputs.q.detach(),
                "k": self.inputs.k.detach(),
                "v": self.inputs.v.detach(),
            },
            output=self._verify_output_buffer,
            batch_size=self.batch,
            parameter_count=0,
            precision_flags={
                "fp16": self.inputs.q.dtype == torch.float16,
                "bf16": self.inputs.q.dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None
        self.output = None
        self._product_buffer = None
        self._dot_buffer = None
        self._decode_step_views = ()
        self._output_view = None
        self._verify_output_buffer = None

    def get_config(self) -> BenchmarkConfig:
        # Keep iterations small; focus on relative speedups and profiling
        return BenchmarkConfig(
            iterations=12,
            warmup=5,
            measurement_timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "persistent_decode.batch_size": float(getattr(self, 'batch_size', 0)),
            "persistent_decode.seq_len": float(getattr(self, 'seq_len', 0)),
            "persistent_decode.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def get_input_signature(self) -> dict:
        return build_decode_input_signature(
            batch=self.batch,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            quantization=self.options.quantization,
        )

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None

def get_benchmark() -> BaseBenchmark:
    return BaselinePersistentDecodeBenchmark()
