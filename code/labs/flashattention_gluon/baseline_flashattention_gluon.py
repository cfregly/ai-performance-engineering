"""Baseline FlashAttention lab: unfused attention (explicit softmax path)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.utils.compile_utils import configure_tf32, restore_tf32
from labs.flashattention_gluon.flashattention_gluon_common import (
    FlashAttentionInputs,
    build_flashattention_inputs,
)


@contextmanager
def _strict_fp32_matmul():
    """Disable reduced-precision FP32 matmul, then restore the caller's policy."""

    previous_matmul_allow_tf32 = (
        bool(torch.backends.cuda.matmul.allow_tf32)
        if hasattr(torch.backends.cuda, "matmul")
        and hasattr(torch.backends.cuda.matmul, "allow_tf32")
        else None
    )
    previous_cudnn_allow_tf32 = (
        bool(torch.backends.cudnn.allow_tf32)
        if hasattr(torch.backends, "cudnn")
        and hasattr(torch.backends.cudnn, "allow_tf32")
        else None
    )
    previous_precision = (
        torch.get_float32_matmul_precision()
        if hasattr(torch, "get_float32_matmul_precision")
        else None
    )
    tf32_state = configure_tf32(matmul_precision="highest")
    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("highest")
        yield
    finally:
        restore_tf32(tf32_state)
        # configure_tf32() installs compatibility accessors whose shadow values
        # cannot always be reconstructed from the newer precision strings (for
        # example, the new API reports "tf32" while the shim tests for "high").
        # Restore the exact legacy booleans captured before that first patch.
        if previous_matmul_allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = previous_matmul_allow_tf32
        if previous_cudnn_allow_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = previous_cudnn_allow_tf32
        if previous_precision is not None and hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(previous_precision)


class BaselineFlashAttentionGluonBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Naive attention: QK^T -> softmax -> matmul V (no fusion, no warp specialization)."""

    def __init__(self) -> None:
        super().__init__()
        self.batch = 2
        self.batch_size = self.batch
        self.seq_len = 1024  # modest size to keep baseline cost reasonable
        self.heads = 8
        self.head_dim = 64
        self.hidden_dim = self.heads * self.head_dim
        self.dtype = torch.float16
        self.inputs: Optional[FlashAttentionInputs] = None
        self._k_t: Optional[torch.Tensor] = None
        self._scale = 0.0
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._payload_q: Optional[torch.Tensor] = None
        self._payload_k: Optional[torch.Tensor] = None
        self._payload_v: Optional[torch.Tensor] = None
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        self.inputs = build_flashattention_inputs(
            batch=self.batch,
            seq_len=self.seq_len,
            heads=self.heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self._k_t = self.inputs.k.transpose(-1, -2)
        self._scale = self.head_dim ** -0.5
        self._verify_output_buffer = torch.empty_like(self.inputs.q, dtype=torch.float32)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None:
            raise RuntimeError("FlashAttention inputs are not initialized")

        with torch.inference_mode():
            with self._nvtx_range("flashattention_baseline_unfused"):
                q = self.inputs.q
                k = self.inputs.k
                v = self.inputs.v
                if self._k_t is None:
                    raise RuntimeError("FlashAttention key transpose is not initialized")
                # Keep the FP16 input/output contract while avoiding a rounded
                # QK matrix, probability matrix, and PV accumulation. This is
                # the independent numerical reference for the tiled kernel.
                with _strict_fp32_matmul():
                    scores = torch.matmul(q.float(), self._k_t.float())
                    scores.mul_(self._scale)
                    probs = torch.softmax(scores, dim=-1)
                    result = torch.matmul(probs, v.float())
                self.output = result.to(dtype=q.dtype)
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")
        self._payload_k = k
        self._payload_q = q
        self._payload_v = v

    def capture_verification_payload(self) -> None:
        k = self._payload_k
        q = self._payload_q
        v = self._payload_v
        if q is None or k is None or v is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"q": q.detach(), "k": k.detach(), "v": v.detach()},
            output=self._verify_output_buffer,
            batch_size=self.batch,
            parameter_count=0,
            precision_flags={"fp16": True, "bf16": False, "tf32": torch.backends.cuda.matmul.allow_tf32},
            # The transport buffer is FP32, but every value was produced and
            # rounded as FP16. Use the canonical strict FP16 comparison bounds.
            output_tolerance=(1e-3, 1e-5),
        )

    def teardown(self) -> None:
        self.inputs = None
        self._k_t = None
        self.output = None
        self._verify_output_buffer = None
        self._payload_q = None
        self._payload_k = None
        self._payload_v = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics for performance analysis."""
        # Basic metrics - override in subclass for domain-specific values
        return {
            "flashattention_gluon.workload_size": float(getattr(self, 'batch_size', 0)),
        }

    def validate_result(self) -> Optional[str]:
        if self.inputs is None:
            return "FlashAttention inputs are not initialized"
        return None

def get_benchmark() -> BaseBenchmark:
    return BaselineFlashAttentionGluonBenchmark()
