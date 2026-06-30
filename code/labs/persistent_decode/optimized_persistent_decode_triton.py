"""Persistent decode in Triton with a device-side work queue."""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    build_decode_input_signature,
    get_decode_options,
    get_decode_profile,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


@triton.jit
def persistent_decode_kernel(
    Q_ptr,
    K_ptr,
    V_ptr,
    OUT_ptr,
    work_seq_ids_ptr,
    work_steps_ptr,
    num_items,
    head_dim: tl.constexpr,
    max_steps: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per sequence
    if pid >= num_items:
        return

    seq_id = tl.load(work_seq_ids_ptr + pid)
    num_step = tl.load(work_steps_ptr + pid)

    for t in range(0, max_steps):
        active = num_step > t
        base = (seq_id * max_steps + t) * head_dim
        offs = tl.arange(0, BLOCK_K)
        dot = tl.zeros((), dtype=tl.float32)

        for k0 in range(0, head_dim, BLOCK_K):
            k_idx = k0 + offs
            mask = (k_idx < head_dim) & active
            q = tl.load(Q_ptr + base + k_idx, mask=mask, other=0.0)
            k = tl.load(K_ptr + base + k_idx, mask=mask, other=0.0)
            dot += tl.sum(q * k, axis=0)

        # Write vector output = V * dot
        for k0 in range(0, head_dim, BLOCK_K):
            k_idx = k0 + offs
            mask = (k_idx < head_dim) & active
            v = tl.load(V_ptr + base + k_idx, mask=mask, other=0.0)
            tl.store(OUT_ptr + base + k_idx, v * dot, mask=mask)


class OptimizedPersistentDecodeTritonBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Persistent decode using Triton."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.options = get_decode_options()
        self.profile = get_decode_profile()
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.batch_size = self.batch
        self.hidden_dim = self.head_dim
        self.block_k = self.profile.block_k
        self.num_programs = self.profile.num_programs
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())
        self.output: Optional[torch.Tensor] = None
        self._output_view: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._num_items = min(self.batch, self.num_programs)
        self._launch_grid = (max(1, self._num_items),)
        self._block_k_const = self.block_k

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.inputs = build_inputs(self.device)
        self._output_view = self.inputs.out[:1, : min(8, self.inputs.out.shape[1])]
        self._verify_output_buffer = torch.empty_like(self._output_view, dtype=torch.float32)
        self._synchronize()

        # Precompile the Triton kernel to keep measurement under benchmark timeouts.
        persistent_decode_kernel[self._launch_grid](
            self.inputs.q,
            self.inputs.k,
            self.inputs.v,
            self.inputs.out,
            self.inputs.work_seq_ids,
            self.inputs.work_steps,
            self._num_items,
            head_dim=self.head_dim,
            max_steps=self.seq_len,
            BLOCK_K=self._block_k_const,
            num_warps=2,
            num_stages=1,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None or self._output_view is None:
            raise RuntimeError("Inputs not initialized")

        with torch.inference_mode(), self._nvtx_range("persistent_decode_triton"):
            persistent_decode_kernel[self._launch_grid](
                self.inputs.q,
                self.inputs.k,
                self.inputs.v,
                self.inputs.out,
                self.inputs.work_seq_ids,
                self.inputs.work_steps,
                self._num_items,
                head_dim=self.head_dim,
                max_steps=self.seq_len,
                BLOCK_K=self._block_k_const,
                num_warps=2,
                num_stages=1,
            )
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
        self._output_view = None
        self._verify_output_buffer = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=12,
            warmup=5,
            measurement_timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "persistent_decode_tr.batch_size": float(getattr(self, 'batch_size', 0)),
            "persistent_decode_tr.seq_len": float(getattr(self, 'seq_len', 0)),
            "persistent_decode_tr.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
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
    return OptimizedPersistentDecodeTritonBenchmark()
