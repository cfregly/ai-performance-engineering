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
        self.q_buffer: Optional[torch.Tensor] = None
        self.q_heads: Optional[torch.Tensor] = None
        self._input_step_views: list[torch.Tensor] = []
        self._cache_step_views: list[torch.Tensor] = []
        self._cos_step_views: list[torch.Tensor] = []
        self._sin_step_views: list[torch.Tensor] = []
        self._step_groups: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self._input_step_count = 0
        self._cache_step_count = 0
        self._cos_step_count = 0
        self._sin_step_count = 0
        self._step_group_count = 0
        self._output_view: Optional[torch.Tensor] = None
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
        self.q_buffer = torch.empty(
            self.cfg.batch_size,
            self.cfg.hidden_size,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self.q_heads = self.q_buffer.view(self.cfg.batch_size, self.cfg.heads, self.cfg.head_dim)
        self._input_step_views = list(self.inputs.unbind(0))
        self._cache_step_views = [self.cache[:, :, step, :] for step in range(self.cfg.steps)]
        self._cos_step_views = [
            self.cos[step].view(1, 1, self.cfg.head_dim)
            for step in range(self.cfg.steps)
        ]
        self._sin_step_views = [
            self.sin[step].view(1, 1, self.cfg.head_dim)
            for step in range(self.cfg.steps)
        ]
        self._step_groups = list(
            zip(
                self._input_step_views,
                self._cache_step_views,
                self._cos_step_views,
                self._sin_step_views,
                strict=True,
            )
        )
        self._input_step_count = len(self._input_step_views)
        self._cache_step_count = len(self._cache_step_views)
        self._cos_step_count = len(self._cos_step_views)
        self._sin_step_count = len(self._sin_step_views)
        self._step_group_count = len(self._step_groups)
        self._output_view = self._cache_step_views[self.cfg.steps - 1]
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if (
            self.inputs is None
            or self.q_weight is None
            or self.cos is None
            or self.sin is None
            or self.cache is None
            or self.rope_scratch is None
            or self.q_buffer is None
            or self.q_heads is None
            or self._output_view is None
            or self._input_step_count != self.cfg.steps
            or self._cache_step_count != self.cfg.steps
            or self._cos_step_count != self.cfg.steps
            or self._sin_step_count != self.cfg.steps
            or self._step_group_count != self.cfg.steps
        ):
            raise RuntimeError("Benchmark not initialized")
        with torch.inference_mode():
            for x, cache_step, cos_t, sin_t in self._step_groups:
                torch.mm(x, self.q_weight, out=self.q_buffer)
                q = apply_rope_inplace(self.q_heads, cos_t, sin_t, self.rope_scratch)
                cache_step.copy_(q)
            self.output = self._output_view
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

    def teardown(self) -> None:
        self.inputs = None
        self.q_weight = None
        self.cos = None
        self.sin = None
        self.cache = None
        self.rope_scratch = None
        self.q_buffer = None
        self.q_heads = None
        self._input_step_views = []
        self._cache_step_views = []
        self._cos_step_views = []
        self._sin_step_views = []
        self._step_groups = []
        self._input_step_count = 0
        self._cache_step_count = 0
        self._cos_step_count = 0
        self._sin_step_count = 0
        self._step_group_count = 0
        self._output_view = None
        self.output = None
        torch.cuda.empty_cache()

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)


def get_benchmark() -> BaseBenchmark:
    return OptimizedRopeQCacheBenchmark()
