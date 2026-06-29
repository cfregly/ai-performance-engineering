"""Shared single-GPU disaggregated prefill/decode benchmark logic."""

from __future__ import annotations

from typing import List, Optional

import torch

from core.benchmark.verification import PrecisionFlags
from core.benchmark.wrapper_utils import attach_benchmark_metadata as attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.benchmark.verification_mixin import VerificationPayloadMixin
from ch17.prefill_decode_disagg_multigpu_common import PrefillDecodeConfig, TinyPrefillDecode


class _PrefillDecodeSingleGPUBase(VerificationPayloadMixin, BaseBenchmark):
    """Shared single-GPU disaggregated prefill/decode setup and verification logic."""

    def __init__(
        self,
        *,
        label: str,
        cfg: Optional[PrefillDecodeConfig] = None,
    ) -> None:
        super().__init__()
        self.label = label
        self.cfg = cfg or PrefillDecodeConfig()
        tokens = self.cfg.requests_per_rank * self.cfg.tokens_per_request
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.cfg.requests_per_rank),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.cfg.requests_per_rank),
            tokens_per_iteration=float(tokens),
        )
        self.prefill_model: Optional[TinyPrefillDecode] = None
        self.decode_model: Optional[TinyPrefillDecode] = None
        self.prompts: Optional[torch.Tensor] = None
        self.kv_caches: List[torch.Tensor] = []
        self._kv_host_staging: Optional[torch.Tensor] = None
        self._flat_prompts: Optional[torch.Tensor] = None
        self._output: Optional[torch.Tensor] = None
        self._pending_outputs: List[torch.Tensor] = []
        self._param_count = 0

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for prefill/decode disaggregation")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.prefill_model = TinyPrefillDecode(
            self.cfg.hidden_size, self.cfg.num_layers, self.device, self.cfg.dtype
        ).eval()
        self.decode_model = TinyPrefillDecode(
            self.cfg.hidden_size, self.cfg.num_layers, self.device, self.cfg.dtype
        ).eval()
        self.prompts = torch.randn(
            self.cfg.requests_per_rank,
            self.cfg.batch_size,
            self.cfg.context_window,
            self.cfg.hidden_size,
            device=self.device,
            dtype=self.cfg.dtype,
        )
        self._flat_prompts = self.prompts.view(
            self.cfg.requests_per_rank * self.cfg.batch_size,
            self.cfg.context_window,
            self.cfg.hidden_size,
        )
        self._param_count = sum(p.numel() for p in self.prefill_model.parameters()) + sum(
            p.numel() for p in self.decode_model.parameters()
        )
        self._pending_outputs = [torch.empty(0) for _ in range(self.cfg.requests_per_rank)]
        self._output = None
        torch.cuda.synchronize(self.device)

    def _allocate_host_staging(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        try:
            return torch.empty(shape, device="cpu", dtype=dtype, pin_memory=True)
        except RuntimeError:
            return torch.empty(shape, device="cpu", dtype=dtype)

    def _set_output(self, outputs: List[torch.Tensor]) -> None:
        self._pending_outputs = outputs
        self._output = None

    def capture_verification_payload(self) -> None:
        if self._output is None and self._pending_outputs:
            self._output = torch.stack(self._pending_outputs, dim=0)
        if self._output is None or self.prompts is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
        meta_dtype = torch.float32
        self._set_verification_payload(
            inputs={
                "prompt": self.prompts[0].detach().cpu(),
                "decode_tokens": torch.zeros((self.cfg.decode_tokens,), dtype=meta_dtype),
                "hidden_size": torch.zeros((self.cfg.hidden_size,), dtype=meta_dtype),
                "num_layers": torch.zeros((self.cfg.num_layers,), dtype=meta_dtype),
            },
            output=self._output,
            batch_size=int(self._output.shape[0]),
            parameter_count=int(self._param_count),
            precision_flags=PrecisionFlags(bf16=True, tf32=tf32_enabled),
            output_tolerance=(0.0, 0.0),
            signature_overrides={
                "world_size": 1,
                "pipeline_stages": 1,
                "pipeline_stage_boundaries": [(0, 0)],
                "per_rank_batch_size": self.cfg.requests_per_rank,
                "collective_type": "local_copy",
            },
        )

    def teardown(self) -> None:
        self.prefill_model = None
        self.decode_model = None
        self.prompts = None
        self.kv_caches = []
        self._kv_host_staging = None
        self._flat_prompts = None
        self._output = None
        self._pending_outputs = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=3, warmup=5, measurement_timeout_seconds=900)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


class BaselinePrefillDecodeSingleGPUBenchmark(_PrefillDecodeSingleGPUBase):
    """Single-GPU disaggregated prefill/decode baseline with host-staged KV handoff."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def setup(self) -> None:
        super().setup()
        self._kv_host_staging = self._allocate_host_staging(
            torch.Size(
                (
                    self.cfg.batch_size,
                    self.cfg.context_window,
                    self.cfg.hidden_size,
                )
            ),
            self.cfg.dtype,
        )

    def benchmark_fn(self) -> None:
        if self.prefill_model is None or self.decode_model is None or self.prompts is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        if self._kv_host_staging is None:
            raise RuntimeError("Baseline KV host staging buffer not initialized")

        outputs = self._pending_outputs
        if len(outputs) != self.cfg.requests_per_rank:
            raise RuntimeError("Decode output slots not initialized")
        output_idx = 0
        with torch.inference_mode():
            for idx in range(self.cfg.requests_per_rank):
                kv_cache, seed = self.prefill_model.prefill(self.prompts[idx])
                self._kv_host_staging.copy_(kv_cache, non_blocking=False)
                kv_cache.copy_(self._kv_host_staging, non_blocking=False)
                outputs[output_idx] = self.decode_model.decode(seed, kv_cache, self.cfg.decode_tokens)
                output_idx += 1

        self._set_output(outputs)


class OptimizedPrefillDecodeSingleGPUBenchmark(_PrefillDecodeSingleGPUBase):
    """Single-GPU disaggregated prefill/decode optimized with batched device-local decode."""

    def benchmark_fn(self) -> None:
        if self.prefill_model is None or self.decode_model is None or self._flat_prompts is None:
            raise RuntimeError("setup() must run before benchmark_fn()")

        with torch.inference_mode():
            kv_cache, seed = self.prefill_model.prefill(self._flat_prompts)
            decoded = self.decode_model.decode(seed, kv_cache, self.cfg.decode_tokens)

        self._output = decoded.view(
            self.cfg.requests_per_rank,
            self.cfg.batch_size,
            self.cfg.hidden_size,
        )
        self._pending_outputs.clear()
