"""Shared helpers for sliding-window FlexAttention benchmarks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.utils.compile_utils import compile_callable

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    HAS_FLEX = True
except (ImportError, AttributeError):
    HAS_FLEX = False


@dataclass(frozen=True)
class SlidingWindowConfig:
    batch_size: int = 2
    num_heads: int = 16
    seq_len: int = 2048
    head_dim: int = 64
    window_size: int = 256
    dtype: torch.dtype = torch.bfloat16


def _sliding_window_causal_mask(window_size: int):
    def mask_fn(_b, _h, q_idx, kv_idx):
        return (q_idx >= kv_idx) & ((q_idx - kv_idx) <= window_size)

    return mask_fn


class SlidingWindowAttentionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    def __init__(self, *, use_flex: bool, cfg: Optional[SlidingWindowConfig] = None) -> None:
        super().__init__()
        self.use_flex = use_flex
        self.cfg = cfg or SlidingWindowConfig()
        self.q: Optional[torch.Tensor] = None
        self.k: Optional[torch.Tensor] = None
        self._k_t: Optional[torch.Tensor] = None
        self.v: Optional[torch.Tensor] = None
        self.mask: Optional[torch.Tensor] = None
        self.block_mask = None
        self.output: Optional[torch.Tensor] = None
        self.scale = 1.0 / math.sqrt(float(self.cfg.head_dim))
        self._mask_fill_value = 0.0
        self._compiled_flex = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Sliding-window attention requires CUDA")
        if self.use_flex and not HAS_FLEX:
            raise RuntimeError("SKIPPED: FlexAttention not available on this build.")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.q = torch.randn(
            self.cfg.batch_size,
            self.cfg.num_heads,
            self.cfg.seq_len,
            self.cfg.head_dim,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self.k = torch.randn_like(self.q)
        self.v = torch.randn_like(self.q)
        self._k_t = self.k.transpose(-2, -1)
        self.scale = 1.0 / math.sqrt(float(self.cfg.head_dim))
        self._mask_fill_value = float(torch.finfo(self.cfg.dtype).min)

        positions = torch.arange(self.cfg.seq_len, device=self.device)
        q_pos = positions[:, None]
        k_pos = positions[None, :]
        mask = (q_pos >= k_pos) & ((q_pos - k_pos) <= self.cfg.window_size)
        self.mask = mask.unsqueeze(0).unsqueeze(0)

        if self.use_flex:
            mask_fn = _sliding_window_causal_mask(self.cfg.window_size)
            self.block_mask = create_block_mask(
                mask_fn,
                B=self.cfg.batch_size,
                H=self.cfg.num_heads,
                Q_LEN=self.cfg.seq_len,
                KV_LEN=self.cfg.seq_len,
                device=self.device,
            )

            def flex_attn(q, k, v):
                return flex_attention(q, k, v, block_mask=self.block_mask)

            self._compiled_flex = compile_callable(flex_attn, mode="max-autotune")
            self._compiled_flex(self.q, self.k, self.v)

    def benchmark_fn(self) -> None:
        if self.q is None or self.k is None or self.v is None:
            raise RuntimeError("Benchmark tensors not initialized")
        if self.use_flex:
            if self._compiled_flex is None:
                raise RuntimeError("FlexAttention not compiled")
            with torch.inference_mode():
                self.output = self._compiled_flex(self.q, self.k, self.v)
            return

        if self.mask is None or self._k_t is None:
            raise RuntimeError("Mask not initialized")
        with torch.inference_mode():
            scores = torch.matmul(self.q, self._k_t)
            scores.mul_(self.scale)
            scores.masked_fill_(~self.mask, self._mask_fill_value)
            attn = torch.softmax(scores, dim=-1)
            self.output = torch.matmul(attn, self.v)

    def capture_verification_payload(self) -> None:
        if self.q is None or self.k is None or self.v is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        sparsity_ratio = float(self.cfg.window_size) / float(self.cfg.seq_len)
        self._set_verification_payload(
            inputs={"q": self.q, "k": self.k, "v": self.v},
            output=self.output,
            batch_size=self.cfg.batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": self.output.dtype == torch.float16,
                "bf16": self.output.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-2, 1e-2),
            signature_overrides={"sparsity_ratio": sparsity_ratio},
        )

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "benchmark_fn() did not produce output"
        return None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=5,
            warmup=5,
            timing_method="wall_clock",
        )
