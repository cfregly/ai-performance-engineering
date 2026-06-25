"""Baseline eager attention with per-head serial computation."""

from __future__ import annotations

import math
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)


class BaselineAttentionEagerSDPABenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: naive attention that iterates per head without fusion."""

    def __init__(self):
        super().__init__()
        self.q = None
        self.k = None
        self.v = None
        self.num_heads = 16
        self.head_dim = 64
        self.embed_dim = self.num_heads * self.head_dim  # 1024
        self.seq_len = 1024
        self._last = 0.0
        self.repeat_passes = 1
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        tokens = self.seq_len * self.num_heads * self.repeat_passes
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.seq_len),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._last_outputs: Optional[list[torch.Tensor]] = None
        self.parameter_count: int = 0
        self._verification_payload = None
        self.register_workload_metadata(
            requests_per_iteration=float(self.seq_len),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Setup: materialize query/key/value tensors."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        shape = (self.seq_len, self.num_heads, self.head_dim)
        self.q = torch.randn(shape, device=self.device, dtype=self.dtype)
        self.k = torch.randn(shape, device=self.device, dtype=self.dtype)
        self.v = torch.randn(shape, device=self.device, dtype=self.dtype)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: per-head attention computed serially."""
        # Use conditional NVTX ranges - only enabled when profiling

        from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_attention_eager_sdpa", enable=enable_nvtx):
            if self.q is None or self.k is None or self.v is None:
                raise RuntimeError("Tensors not initialized")
            scale = 1.0 / math.sqrt(self.head_dim)
            outputs = []
            for _ in range(self.repeat_passes):
                for head in range(self.num_heads):
                    qh = self.q[:, head, :]
                    kh = self.k[:, head, :]
                    vh = self.v[:, head, :]
                    scores = torch.matmul(qh, kh.transpose(0, 1)) * scale
                    attn = torch.softmax(scores, dim=-1)
                    outputs.append(torch.matmul(attn, vh))
            self._last_outputs = outputs
        if self._last_outputs is None or self.q is None or self.k is None or self.v is None:
            raise RuntimeError("Verification input/output not initialized")

    def capture_verification_payload(self) -> None:
        if self._last_outputs is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        stacked = torch.stack(self._last_outputs, dim=1)
        # Flatten heads into the embedding dimension to match optimized output
        self.output = stacked.permute(1, 0, 2).reshape(
            1, self.seq_len, self.embed_dim * self.repeat_passes
        ).contiguous()
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
        self.output = None
        self._last_outputs = None
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
    return BaselineAttentionEagerSDPABenchmark()
