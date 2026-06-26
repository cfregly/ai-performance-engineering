"""Optimized RoPE + Q projection + KV cache update (vectorized)."""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from ch18.rope_q_cache_common import RopeQCacheConfig, apply_rope_inplace, build_rope_tables


class OptimizedRopeQCacheBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: vectorized RoPE + single cache write per step."""

    def __init__(self, cfg: Optional[RopeQCacheConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or RopeQCacheConfig()
        self.inputs: Optional[torch.Tensor] = None
        self.q_weight: Optional[torch.Tensor] = None
        self.cos: Optional[torch.Tensor] = None
        self.sin: Optional[torch.Tensor] = None
        self.cache: Optional[torch.Tensor] = None
        self.rope_scratch: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.cfg.batch_size),
            tokens_per_iteration=float(self.cfg.tokens_per_iter),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.cfg.batch_size),
            tokens_per_iteration=float(self.cfg.tokens_per_iter),
        )

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for RoPE fusion benchmark")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.inputs = torch.randn(
            self.cfg.steps,
            self.cfg.batch_size,
            self.cfg.hidden_size,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self.q_weight = torch.randn(
            self.cfg.hidden_size,
            self.cfg.hidden_size,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self.cos, self.sin = build_rope_tables(
            self.cfg.max_seq_len,
            self.cfg.head_dim,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self.cache = torch.empty(
            self.cfg.batch_size,
            self.cfg.heads,
            self.cfg.max_seq_len,
            self.cfg.head_dim,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self.rope_scratch = torch.empty(
            self.cfg.batch_size,
            self.cfg.heads,
            self.cfg.head_dim // 2,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if (
            self.inputs is None
            or self.q_weight is None
            or self.cos is None
            or self.sin is None
            or self.cache is None
            or self.rope_scratch is None
        ):
            raise RuntimeError("Benchmark not initialized")
        with torch.inference_mode():
            for step in range(self.cfg.steps):
                x = self.inputs[step]
                q = x @ self.q_weight
                q = q.view(self.cfg.batch_size, self.cfg.heads, self.cfg.head_dim)
                cos_t = self.cos[step].view(1, 1, self.cfg.head_dim)
                sin_t = self.sin[step].view(1, 1, self.cfg.head_dim)
                q = apply_rope_inplace(q, cos_t, sin_t, self.rope_scratch)
                self.cache[:, :, step, :] = q
            self.output = self.cache[:, :, self.cfg.steps - 1, :]
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"inputs": self.inputs[:1].detach()},
            output=self.output.detach(),
            batch_size=self.cfg.batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": self.cfg.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(5e-2, 5e-1),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)


def get_benchmark() -> BaseBenchmark:
    return OptimizedRopeQCacheBenchmark()
