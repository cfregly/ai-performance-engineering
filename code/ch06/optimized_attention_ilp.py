"""Optimized attention-score ILP benchmark with four independent chains."""

from __future__ import annotations

from typing import Optional

import torch

from ch06.cuda_extensions import load_ilp_extension
from ch06.workload_config import WORKLOAD
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class OptimizedAttentionILPBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized attention-score preprocessing with four-way ILP.

    The optimization is the instruction schedule, not a different attention
    algorithm. Each thread processes four independent score terms so the warp can
    issue more useful work while earlier operations are still in flight.
    """

    def __init__(self) -> None:
        super().__init__()
        self._extension = None
        self.attention_terms: Optional[torch.Tensor] = None
        self._buf0: Optional[torch.Tensor] = None
        self._buf1: Optional[torch.Tensor] = None
        self._output_view0: Optional[torch.Tensor] = None
        self._output_view1: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self.workload = WORKLOAD
        self.batch = self.workload.attention_batch
        self.embed_dim = self.workload.attention_embed_dim
        self.num_heads = self.workload.attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.tokens = self.workload.attention_tokens
        self.repeats = 8
        self._repeat_range = range(self.repeats)
        self.score_terms = self.batch * self.tokens * self.num_heads * self.head_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(self.batch * self.tokens),
        )

    def setup(self) -> None:
        """Build attention-shaped score terms and the CUDA ILP buffers."""
        self._extension = load_ilp_extension()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        query = torch.randn(
            self.batch,
            self.tokens,
            self.num_heads,
            self.head_dim,
            device=self.device,
            dtype=torch.float32,
        )
        key = torch.randn_like(query)
        # Keep magnitudes in the same stable range as the GEMM ILP pair so the
        # repeated square chain measures ILP rather than overflow behavior.
        self.attention_terms = (query * key * 0.1).contiguous().reshape(-1)
        self._buf0 = torch.empty_like(self.attention_terms)
        self._buf1 = torch.empty_like(self.attention_terms)
        self._output_view0 = self._buf0[:4096]
        self._output_view1 = self._buf1[:4096]
        self._verify_output_buffer = torch.empty_like(self._output_view0)
        self.output = None
        self._synchronize()

    def benchmark_fn(self) -> None:
        """Run the high-ILP transform over attention-score terms."""
        assert self._extension is not None
        assert self.attention_terms is not None
        assert self._buf0 is not None and self._buf1 is not None
        assert self._output_view0 is not None and self._output_view1 is not None
        with torch.inference_mode(), self._nvtx_range("optimized_attention_ilp"):
            src: torch.Tensor = self.attention_terms
            buf0: torch.Tensor = self._buf0
            buf1: torch.Tensor = self._buf1
            dst: torch.Tensor = buf0
            for _ in self._repeat_range:
                self._extension.unrolled_ilp(dst, src)
                src, dst = dst, (buf1 if dst is buf0 else buf0)
        self.output = self._output_view0 if src is buf0 else self._output_view1

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must produce output before verification")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"attention_terms": self.attention_terms},
            output=self._verify_output_buffer,
            batch_size=self.batch,
            parameter_count=0,
            output_tolerance=(1e-5, 1e-5),
            precision_flags={"fp16": False, "bf16": False, "fp8": False, "tf32": False},
        )

    def teardown(self) -> None:
        self._extension = None
        self.attention_terms = None
        self._buf0 = None
        self._buf1 = None
        self._output_view0 = None
        self._output_view1 = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=self.workload.ilp_iterations,
            warmup=self.workload.ilp_warmup,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=120,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        from core.benchmark.metrics import compute_kernel_fundamentals_metrics

        metrics = compute_kernel_fundamentals_metrics(
            num_elements=self.score_terms,
            num_iterations=self.repeats,
        )
        metrics.update(
            {
                "attention_ilp.batch": float(self.batch),
                "attention_ilp.tokens": float(self.tokens),
                "attention_ilp.heads": float(self.num_heads),
                "attention_ilp.head_dim": float(self.head_dim),
                "attention_ilp.independent_chains_per_thread": 4.0,
            }
        )
        return metrics

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output tensor not initialized"
        if self.output.shape[0] != 4096:
            return f"Output shape mismatch: expected 4096, got {self.output.shape[0]}"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedAttentionILPBenchmark()
