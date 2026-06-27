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
        self._last = 0.0
        self.repeat_passes = 1
        tokens = self.seq_len * self.num_heads * self.repeat_passes
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.seq_len),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self.parameter_count: int = 0
        self.register_workload_metadata(
            requests_per_iteration=float(self.seq_len),
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: materialize query/key/value tensors (same workload as baseline)."""

        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        shape = (self.seq_len, self.num_heads, self.head_dim)
        self.q = torch.randn(shape, device=self.device, dtype=self.dtype)
        self.k = torch.randn(shape, device=self.device, dtype=self.dtype)
        self.v = torch.randn(shape, device=self.device, dtype=self.dtype)
        for _ in range(3):
            with torch.inference_mode():
                _ = self._attention(self.q, self.k, self.v)
        torch.cuda.synchronize(self.device)

    def _attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        q_bhsd = q.transpose(0, 1).unsqueeze(0)  # [1, H, S, D]
        k_bhsd = k.transpose(0, 1).unsqueeze(0)
        v_bhsd = v.transpose(0, 1).unsqueeze(0)
        with _flash_sdp_context():
            out = F.scaled_dot_product_attention(q_bhsd, k_bhsd, v_bhsd, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).contiguous().view(1, self.seq_len, self.embed_dim)
        if self.repeat_passes > 1:
            out = out.repeat(1, 1, self.repeat_passes)
        return out
    
    def benchmark_fn(self) -> None:
        """Benchmark: fused SDPA attention operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with (
            torch.inference_mode(),
            nvtx_range("optimized_attention_eager_sdpa", enable=enable_nvtx),
        ):
            if self.q is None or self.k is None or self.v is None:
                raise RuntimeError("Tensors not initialized")
            out = self._attention(self.q, self.k, self.v)
            self.output = out.detach()
        if self.q is None or self.k is None or self.v is None or self.output is None:
            raise RuntimeError("Verification input/output not initialized")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={
                "q": self.q.detach(),
                "k": self.k.detach(),
                "v": self.v.detach(),
            },
            output=self.output.detach().clone(),
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
