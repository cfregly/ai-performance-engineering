"""Optimized MoE backend selection (FlashInfer-style auto-tuning)."""

from __future__ import annotations

from typing import Optional, Callable, Dict

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from labs.moe_cuda.moe_backend_common import (
    MoEBackendConfig,
    MoEBackendWorkload,
    select_best_backend,
)


class OptimizedMoEBackendSelectionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: auto-select the fastest backend (vectorized vs trtllm-gen)."""

    def __init__(self, cfg: Optional[MoEBackendConfig] = None) -> None:
        super().__init__()
        self.cfg = cfg or MoEBackendConfig()
        self.workload: Optional[MoEBackendWorkload] = None
        self.output: Optional[torch.Tensor] = None
        self._backend_name: str = "vectorized"
        self._backend_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
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
            raise RuntimeError("SKIPPED: CUDA required for MoE backend selection")
        self.workload = MoEBackendWorkload(self.cfg, self.device)

        vectorized = self.workload.forward_vectorized
        try:
            compiled = torch.compile(vectorized, mode="max-autotune")
        except Exception as exc:
            raise RuntimeError("torch.compile failed for MoE backend selection benchmark.") from exc

        # Warmup to populate compile caches.
        with torch.inference_mode():
            _ = compiled(self.workload.x)
        torch.cuda.synchronize(self.device)

        with torch.inference_mode():
            reference = self.workload.forward_naive(self.workload.x).detach()
        torch.cuda.synchronize(self.device)

        candidates: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
            "vectorized": vectorized,
            "trtllm_gen": compiled,
        }

        # Only benchmark/select backends that preserve baseline math under benchmark tolerances.
        valid_candidates: Dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
        for name, fn in candidates.items():
            with torch.inference_mode():
                candidate_out = fn(self.workload.x).detach()
            if torch.allclose(candidate_out, reference, rtol=5e-2, atol=5e-1):
                valid_candidates[name] = fn

        if not valid_candidates:
            self._backend_name = "vectorized"
            self._backend_fn = vectorized
            return

        self._backend_name, self._backend_fn = select_best_backend(
            valid_candidates,
            self.workload.x,
        )

    def benchmark_fn(self) -> None:
        if self.workload is None or self._backend_fn is None:
            raise RuntimeError("Benchmark not initialized")
        with torch.inference_mode():
            self.output = self._backend_fn(self.workload.x)
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.workload is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={"x": self.workload.x.detach()},
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

    def get_custom_metrics(self) -> Optional[dict]:
        backend_flag = 1.0 if self._backend_name == "trtllm_gen" else 0.0
        return {
            "moe.backend.trtllm_gen": backend_flag,
        }

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=6, warmup=6)


def get_benchmark() -> BaseBenchmark:
    return OptimizedMoEBackendSelectionBenchmark()
