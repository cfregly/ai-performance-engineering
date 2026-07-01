"""Optimized Flash SDP attention benchmark with fail-fast backend checks."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range


def ensure_flash_sdp_available() -> None:
    """Fail fast by actually invoking the Flash kernel."""
    if not torch.cuda.is_available():
        raise RuntimeError("Flash SDP benchmark requires a CUDA device.")
    try:
        q = torch.randn(1, 1, 4, 64, device="cuda", dtype=torch.float16)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
            with torch.inference_mode():
                _ = F.scaled_dot_product_attention(q, q, q, is_causal=False)
        torch.cuda.synchronize()
    except Exception as exc:  # pragma: no cover - only hit on unsupported stacks
        raise RuntimeError(f"SKIPPED: Flash SDP kernel failed to run: {exc}") from exc


class FlashAttentionModule(nn.Module):
    """Attention block that forces Flash SDP backend."""

    def __init__(self, hidden_dim: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3, bias=False)
        self._flash_backends = [SDPBackend.FLASH_ATTENTION]
        self._qkv_buffer: Optional[torch.Tensor] = None
        self._qkv_weight_t: Optional[torch.Tensor] = None

    def cache_weight_views(self) -> None:
        self._qkv_weight_t = self.qkv.weight.t()

    def _ensure_qkv_buffer(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        rows = int(batch_size * seq_len)
        shape = (rows, self.hidden_dim * 3)
        view_shape = (batch_size, seq_len, self.hidden_dim * 3)
        if (
            self._qkv_buffer is None
            or self._qkv_buffer.size(0) < rows
            or self._qkv_buffer.device != x.device
            or self._qkv_buffer.dtype != x.dtype
        ):
            self._qkv_buffer = torch.empty(shape, device=x.device, dtype=x.dtype)
        return self._qkv_buffer[:rows].view(view_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        if torch.is_grad_enabled():
            qkv = self.qkv(x)
        else:
            if self._qkv_weight_t is None:
                self.cache_weight_views()
            qkv_buffer = self._ensure_qkv_buffer(x, B, T)
            qkv = torch.matmul(x, self._qkv_weight_t, out=qkv_buffer)
        qkv = qkv.view(B, T, 3, self.num_heads, self.hidden_dim // self.num_heads)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        with sdpa_kernel(self._flash_backends):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        out = out.transpose(1, 2).reshape(B, T, self.hidden_dim)
        return out


class OptimizedFlashSDPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Flash SDP attention (hardware-accelerated)."""

    def __init__(self):
        super().__init__()
        self.model: Optional[FlashAttentionModule] = None
        self.inputs: Optional[torch.Tensor] = None
        self.seq_len = 256
        self.batch = 8
        self.hidden = 512
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verify_input: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._enable_nvtx = False
        self._payload_parameter_count = 0
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        ensure_flash_sdp_available()
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        # Optimized: Flash SDP with fused kernel
        self.model = FlashAttentionModule(hidden_dim=self.hidden, num_heads=8).to(
            self.device, dtype=torch.float16
        )
        self.model.cache_weight_views()
        self.inputs = torch.randn(self.batch, self.seq_len, self.hidden, device=self.device, dtype=torch.float16)
        self._verify_input = self.inputs.detach().clone()
        self._verify_output_buffer = torch.empty(
            self.batch,
            self.seq_len,
            self.hidden,
            device=self.device,
            dtype=torch.float16,
        )
        self._payload_parameter_count = sum(p.numel() for p in self.model.parameters())
        # Warmup
        with torch.inference_mode():
            for _ in range(3):
                _ = self.model(self.inputs)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Model not initialized")
        with nvtx_range("flash_sdp_optimized", enable=self._enable_nvtx):
            with torch.inference_mode():
                self.output = self.model(self.inputs)
        if self._verify_input is None:
            raise RuntimeError("Verification input missing")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_input is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self._verify_input},
            output=self._verify_output_buffer,
            batch_size=self._verify_input.shape[0],
            parameter_count=self._payload_parameter_count,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.inputs is None:
            return "Model not initialized"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=10,
            warmup=5,
            measurement_timeout_seconds=90,
            setup_timeout_seconds=90,
            timing_method="wall_clock",
            full_device_sync=True,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=None,
            tpot_ms=None,
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

def get_benchmark() -> BaseBenchmark:
    return OptimizedFlashSDPBenchmark()
