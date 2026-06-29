"""Baseline prefill vs. decode microbench without TMA shaping.

Emits two phases for Nsight Systems:
- Prefill: sequential "TMA-like" bulk copies + compute on the default stream.
- Decode: per-token work (host-driven loop).
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.blackwell_requirements import ensure_blackwell_tma_supported
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


class BaselineTmaPrefillDecodeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Sequential copy/compute prefill followed by host-driven decode."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.batch_size = self.batch
        self.hidden_dim = self.head_dim
        self.prefill_chunks = 8
        self.prefill_chunk_elems = 128 * 128
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())

    def setup(self) -> None:
        ensure_blackwell_tma_supported("baseline_tma_prefill_decode")
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.inputs = build_inputs(self.device)
        self.prefill_src = torch.randn(
            self.prefill_chunks, self.prefill_chunk_elems, device=self.device
        )
        self.prefill_dst = torch.empty_like(self.prefill_src)
        self._verify_output_buffer = torch.empty(
            1,
            min(8, self.seq_len),
            self.head_dim,
            device=self.inputs.out.device,
            dtype=torch.float32,
        )
        self._synchronize()

    def _prefill_sequential(self) -> None:
        """Sequential copy + compute without any pipelining."""
        for idx in range(self.prefill_chunks):
            # Real copy operation - no artificial delays
            self.prefill_dst[idx].copy_(self.prefill_src[idx])
            # Add computation to simulate processing
            self.prefill_dst[idx].add_(self.prefill_src[idx])

    def _decode_host_loop(self) -> None:
        assert self.inputs is not None
        for t in range(self.seq_len):
            q_t = self.inputs.q[:, t, :]
            k_t = self.inputs.k[:, t, :]
            v_t = self.inputs.v[:, t, :]
            dot = (q_t * k_t).sum(dim=-1, keepdim=True)
            self.inputs.out[:, t, :] = v_t * dot

    def benchmark_fn(self) -> None:
        if self.inputs is None:
            raise RuntimeError("Inputs not initialized")

        with self._nvtx_range("prefill_baseline"):
            self._prefill_sequential()
        with self._nvtx_range("decode_baseline"):
            self._decode_host_loop()
        if self.inputs is not None:
            self.output = self.inputs.out[:1, : min(8, self.inputs.out.shape[1])].detach()
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
        self._verify_output_buffer = None

    def get_config(self) -> BenchmarkConfig:
        # Keep short; this is primarily for profiling with --profile / nsys
        return BenchmarkConfig(
            iterations=8,
            warmup=10,
            measurement_timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "tma_prefill_decode.batch_size": float(getattr(self, 'batch_size', 0)),
            "tma_prefill_decode.seq_len": float(getattr(self, 'seq_len', 0)),
            "tma_prefill_decode.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None

def get_benchmark() -> BaseBenchmark:
    return BaselineTmaPrefillDecodeBenchmark()
