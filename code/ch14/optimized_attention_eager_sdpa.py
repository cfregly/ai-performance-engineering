"""optimized_attention_eager_sdpa.py - fused SDPA attention benchmark."""

from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn.functional as F

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except ImportError:  # pragma: no cover - older PyTorch fallback
    SDPBackend = None  # type: ignore[assignment]
    sdpa_kernel = None  # type: ignore[assignment]

from typing import Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


def _flash_sdp_context():
    """Prefer the new sdpa_kernel API; fall back to no-op if unavailable."""
    if sdpa_kernel is None or SDPBackend is None or not hasattr(SDPBackend, "FLASH_ATTENTION"):
        return nullcontext()
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION])


class OptimizedAttentionEagerSDPABenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: uses fused scaled-dot-product attention."""
    
    def __init__(self):
        super().__init__()
        self.seq_len = 1024
        self.num_heads = 16
        self.head_dim = 64
        self.embed_dim = self.num_heads * self.head_dim  # 1024
        self.batch = 1
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.q = None
        self.k = None
        self.v = None
        self._q_bhsd: Optional[torch.Tensor] = None
        self._k_bhsd: Optional[torch.Tensor] = None
        self._v_bhsd: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._last = 0.0
        self.repeat_passes = 1
        tokens = self.seq_len * self.num_heads * self.repeat_passes
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.seq_len),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self._enable_nvtx = False
        self.register_workload_metadata(
            requests_per_iteration=float(self.seq_len),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: materialize query/key/value tensors (same workload as baseline)."""

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        shape = (self.seq_len, self.num_heads, self.head_dim)
        self.q = torch.randn(shape, device=self.device, dtype=self.dtype)
        self.k = torch.randn(shape, device=self.device, dtype=self.dtype)
        self.v = torch.randn(shape, device=self.device, dtype=self.dtype)
        self._q_bhsd = self.q.transpose(0, 1).unsqueeze(0)
        self._k_bhsd = self.k.transpose(0, 1).unsqueeze(0)
        self._v_bhsd = self.v.transpose(0, 1).unsqueeze(0)
        self._verify_output_buffer = torch.empty(
            (self.batch, self.seq_len, self.embed_dim * self.repeat_passes),
            device=self.device,
            dtype=self.dtype,
        )
        self._ensure_output_buffer(
            self.batch,
            self.seq_len,
            self.num_heads,
            self.head_dim,
            self._q_bhsd.device,
            self._q_bhsd.dtype,
        )
        for _ in range(3):
            with torch.inference_mode():
                _ = self._attention_bhsd(self._q_bhsd, self._k_bhsd, self._v_bhsd)
        torch.cuda.synchronize(self.device)

    def _ensure_output_buffer(
        self,
        batch_size: int,
        seq_len: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        output_shape = (batch_size, seq_len, num_heads * head_dim * self.repeat_passes)
        if (
            self._output_buffer is None
            or tuple(self._output_buffer.shape) != output_shape
            or self._output_buffer.device != device
            or self._output_buffer.dtype != dtype
        ):
            self._output_buffer = torch.empty(output_shape, device=device, dtype=dtype)
        return self._output_buffer

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_bhsd = q.transpose(0, 1).unsqueeze(0)
        k_bhsd = k.transpose(0, 1).unsqueeze(0)
        v_bhsd = v.transpose(0, 1).unsqueeze(0)
        return self._attention_bhsd(q_bhsd, k_bhsd, v_bhsd)

    def _attention_bhsd(
        self,
        q_bhsd: torch.Tensor,
        k_bhsd: torch.Tensor,
        v_bhsd: torch.Tensor,
    ) -> torch.Tensor:
        with _flash_sdp_context():
            out = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, dropout_p=0.0, is_causal=False)
        batch_size, num_heads, seq_len, head_dim = out.shape
        embed_dim = num_heads * head_dim
        output = self._ensure_output_buffer(
            batch_size,
            seq_len,
            num_heads,
            head_dim,
            out.device,
            out.dtype,
        )
        output_slice = output[:, :, :embed_dim].view(batch_size, seq_len, num_heads, head_dim)
        output_slice.copy_(out.transpose(1, 2))
        if self.repeat_passes > 1:
            for repeat_idx in range(1, self.repeat_passes):
                start = repeat_idx * embed_dim
                end = start + embed_dim
                output[:, :, start:end].copy_(output[:, :, :embed_dim])
        return output
    
    def benchmark_fn(self) -> None:
        """Benchmark: fused SDPA attention operations."""
        with (
            torch.inference_mode(),
            nvtx_range("optimized_attention_eager_sdpa", enable=self._enable_nvtx),
        ):
            if (
                self.q is None
                or self.k is None
                or self.v is None
                or self._q_bhsd is None
                or self._k_bhsd is None
                or self._v_bhsd is None
            ):
                raise RuntimeError("Tensors not initialized")
            out = self._attention_bhsd(self._q_bhsd, self._k_bhsd, self._v_bhsd)
            self.output = out
        if self.q is None or self.k is None or self.v is None or self.output is None:
            raise RuntimeError("Verification input/output not initialized")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={
                "q": self.q.detach(),
                "k": self.k.detach(),
                "v": self.v.detach(),
            },
            output=self._verify_output_buffer,
            batch_size=1,
            parameter_count=0,
            precision_flags={
                "fp16": self.dtype == torch.float16,
                "bf16": self.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.q = None
        self.k = None
        self.v = None
        self._q_bhsd = None
        self._k_bhsd = None
        self._v_bhsd = None
        self.output = None
        self._output_buffer = None
        self._verify_output_buffer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        total_flops = 2.0 * self.seq_len * self.seq_len * self.head_dim * self.num_heads
        total_bytes = float(self.seq_len * self.num_heads * self.head_dim * 3 * (2 if self.dtype != torch.float32 else 4))
        from core.benchmark.metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=total_flops,
            total_bytes=total_bytes,
            elapsed_ms=getattr(self, '_last_elapsed_ms', None),
            precision="bf16" if self.dtype == torch.bfloat16 else "fp16",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.q is None or self.k is None or self.v is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedAttentionEagerSDPABenchmark()
