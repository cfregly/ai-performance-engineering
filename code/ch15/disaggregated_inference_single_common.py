"""Shared single-GPU KV-handoff comparison benchmark logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from core.benchmark.verification import PrecisionFlags
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.benchmark.wrapper_utils import attach_benchmark_metadata
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.optimization.moe_inference import (
    MoeInferenceConfig,
    SimpleMoEGPT,
    allocate_kv_cache,
    env_override_int,
)


@dataclass(frozen=True)
class DisaggConfig:
    vocab_size: int = 16384
    hidden_size: int = 1024
    ffn_size: int = 1024
    num_layers: int = 1
    num_moe_layers: int = 1
    num_experts: int = 16
    top_k: int = 2
    batch_size: int = 1
    requests_per_rank: int = 128
    context_window: int = 4096
    decode_tokens: int = 8
    dtype: torch.dtype = torch.bfloat16

    @property
    def tokens_per_request(self) -> int:
        return self.context_window + self.decode_tokens


def _flatten_prompt_batches(prompts: torch.Tensor) -> torch.Tensor:
    """Collapse request and microbatch axes for batched prefill/decode."""
    requests_per_rank, batch_size, context_window = prompts.shape
    return prompts.reshape(requests_per_rank * batch_size, context_window)


def _format_batched_decode_output(
    final_tokens: torch.Tensor,
    *,
    requests_per_rank: int,
    batch_size: int,
) -> torch.Tensor:
    """Match the legacy per-request output layout used by verification."""
    flattened = final_tokens.reshape(requests_per_rank * batch_size, -1)
    if batch_size == 1:
        return flattened.squeeze(1)
    return flattened


def _build_moe_config(cfg: DisaggConfig) -> MoeInferenceConfig:
    return MoeInferenceConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        ffn_size=cfg.ffn_size,
        num_layers=cfg.num_layers,
        num_moe_layers=cfg.num_moe_layers,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        moe_layer_frequency=1,
        batch_size=cfg.batch_size,
        context_window=cfg.context_window,
        decode_tokens=cfg.decode_tokens,
        router_noise=0.0,
        dtype=cfg.dtype,
    )


def _apply_profile_overrides(cfg: DisaggConfig) -> DisaggConfig:
    return DisaggConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        ffn_size=cfg.ffn_size,
        num_layers=cfg.num_layers,
        num_moe_layers=cfg.num_moe_layers,
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        batch_size=env_override_int("AISP_NCU_PROFILE_BATCH", cfg.batch_size),
        requests_per_rank=env_override_int("AISP_NCU_PROFILE_REQUESTS", cfg.requests_per_rank),
        context_window=env_override_int("AISP_NCU_PROFILE_CONTEXT", cfg.context_window),
        decode_tokens=env_override_int("AISP_NCU_PROFILE_DECODE", cfg.decode_tokens),
        dtype=cfg.dtype,
    )


class _DisaggregatedInferenceSingleGPUBase(VerificationPayloadMixin, BaseBenchmark):
    """Shared single-GPU KV-handoff setup/verification logic."""

    ncu_env_overrides = {
        "AISP_NCU_PROFILE_REQUESTS": "4",
        "AISP_NCU_PROFILE_CONTEXT": "256",
        "AISP_NCU_PROFILE_DECODE": "8",
        "AISP_NCU_PROFILE_BATCH": "1",
    }

    def __init__(self, *, label: str, cfg: Optional[DisaggConfig] = None) -> None:
        super().__init__()
        self.label = label
        base_cfg = cfg or DisaggConfig()
        self.cfg = _apply_profile_overrides(base_cfg)
        tokens = self.cfg.requests_per_rank * self.cfg.tokens_per_request
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.cfg.requests_per_rank),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.cfg.requests_per_rank),
            tokens_per_iteration=float(tokens),
        )
        self.prefill_model: Optional[SimpleMoEGPT] = None
        self.decode_model: Optional[SimpleMoEGPT] = None
        self.prompts: Optional[torch.Tensor] = None
        self.kv_caches: List[torch.Tensor] = []
        self.batched_kv_cache: Optional[torch.Tensor] = None
        self._output: Optional[torch.Tensor] = None
        self._pending_outputs: List[torch.Tensor] = []
        self._param_count = 0

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for disaggregated inference")
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = _build_moe_config(self.cfg)
        self.prefill_model = SimpleMoEGPT(config, device=self.device).eval()
        self.decode_model = SimpleMoEGPT(config, device=self.device).eval()
        self.prompts = torch.randint(
            0,
            self.cfg.vocab_size,
            (self.cfg.requests_per_rank, self.cfg.batch_size, self.cfg.context_window),
            device=self.device,
            dtype=torch.long,
        )
        self._param_count = sum(p.numel() for p in self.prefill_model.parameters()) + sum(
            p.numel() for p in self.decode_model.parameters()
        )
        torch.cuda.synchronize(self.device)

    def _allocate_kv_cache(self) -> torch.Tensor:
        return allocate_kv_cache(
            self.cfg.batch_size,
            self.cfg.tokens_per_request,
            self.cfg.hidden_size,
            self.cfg.dtype,
            self.device,
        )

    def _run_decode_loop(self, kv_cache: torch.Tensor, seed_tokens: torch.Tensor) -> torch.Tensor:
        if self.decode_model is None:
            raise RuntimeError("Decode model not initialized")
        tokens = seed_tokens
        for step in range(self.cfg.decode_tokens):
            _, decode_logits = self.decode_model.decode(
                tokens,
                kv_cache=kv_cache,
                position=self.cfg.context_window + step,
            )
            tokens = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)
        return tokens.squeeze(0)

    def _set_output_from_tokens(self, outputs: List[torch.Tensor]) -> None:
        self._pending_outputs = outputs
        self._output = None

    def capture_verification_payload(self) -> None:
        if self.prompts is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        if self._output is None:
            if not self._pending_outputs:
                raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
            self._output = torch.cat(self._pending_outputs, dim=0)
        tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
        meta_dtype = torch.float32
        self._set_verification_payload(
            inputs={
                "prompt": self.prompts[0].detach().cpu(),
                "decode_tokens": torch.zeros((self.cfg.decode_tokens,), dtype=meta_dtype),
                "hidden_size": torch.zeros((self.cfg.hidden_size,), dtype=meta_dtype),
                "num_layers": torch.zeros((self.cfg.num_layers,), dtype=meta_dtype),
                "num_experts": torch.zeros((self.cfg.num_experts,), dtype=meta_dtype),
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
        self.batched_kv_cache = None
        self._output = None
        self._pending_outputs = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=3, warmup=5, measurement_timeout_seconds=900)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        metrics = {
            "story.comparison_pair": 1.0,
            "story.chapter_native_exemplar": 0.0,
            "single_gpu_kv_handoff.requests_per_rank": float(self.cfg.requests_per_rank),
            "single_gpu_kv_handoff.context_window": float(self.cfg.context_window),
            "single_gpu_kv_handoff.decode_tokens": float(self.cfg.decode_tokens),
            "single_gpu_kv_handoff.hidden_size": float(self.cfg.hidden_size),
            "single_gpu_kv_handoff.host_staged_kv": 0.0,
            "single_gpu_kv_handoff.device_resident_kv": 0.0,
        }
        metrics.update(self._variant_metrics())
        return metrics

    def _variant_metrics(self) -> dict[str, float]:
        raise NotImplementedError


class BaselineDisaggregatedInferenceSingleGPUBenchmark(_DisaggregatedInferenceSingleGPUBase):
    """Baseline single-GPU KV handoff with host-staged cache transfer."""

    allowed_benchmark_fn_antipatterns = ("host_transfer",)

    def benchmark_fn(self) -> None:
        if self.prefill_model is None or self.prompts is None:
            raise RuntimeError("setup() must run before benchmark_fn()")

        outputs: List[torch.Tensor] = []
        with torch.no_grad():
            for idx in range(self.cfg.requests_per_rank):
                prompt = self.prompts[idx]
                hidden, logits = self.prefill_model.prefill(prompt)
                seed_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                kv_cpu = hidden.cpu()
                kv_cache = self._allocate_kv_cache()
                kv_cache[:, : self.cfg.context_window] = kv_cpu.to(self.device)
                outputs.append(self._run_decode_loop(kv_cache, seed_tokens))

        self._set_output_from_tokens(outputs)

    def _variant_metrics(self) -> dict[str, float]:
        return {
            "single_gpu_kv_handoff.host_staged_kv": 1.0,
            "single_gpu_kv_handoff.device_resident_kv": 0.0,
        }


class OptimizedDisaggregatedInferenceSingleGPUBenchmark(_DisaggregatedInferenceSingleGPUBase):
    """Optimized single-GPU KV handoff with batched device-resident reuse."""

    def setup(self) -> None:
        super().setup()
        total_batch = self.cfg.requests_per_rank * self.cfg.batch_size
        self.batched_kv_cache = allocate_kv_cache(
            total_batch,
            self.cfg.tokens_per_request,
            self.cfg.hidden_size,
            self.cfg.dtype,
            self.device,
        )

    def benchmark_fn(self) -> None:
        if self.prefill_model is None or self.prompts is None:
            raise RuntimeError("setup() must run before benchmark_fn()")
        if self.batched_kv_cache is None:
            raise RuntimeError("Optimized KV cache not initialized")

        flat_prompts = _flatten_prompt_batches(self.prompts)
        with torch.no_grad():
            hidden, logits = self.prefill_model.prefill(flat_prompts)
            seed_tokens = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            self.batched_kv_cache[:, : self.cfg.context_window] = hidden
            tokens = seed_tokens
            for step in range(self.cfg.decode_tokens):
                _, decode_logits = self.decode_model.decode(
                    tokens,
                    kv_cache=self.batched_kv_cache,
                    position=self.cfg.context_window + step,
                )
                tokens = torch.argmax(decode_logits[:, -1, :], dim=-1, keepdim=True)

        self._output = _format_batched_decode_output(
            tokens,
            requests_per_rank=self.cfg.requests_per_rank,
            batch_size=self.cfg.batch_size,
        )

    def _variant_metrics(self) -> dict[str, float]:
        return {
            "single_gpu_kv_handoff.host_staged_kv": 0.0,
            "single_gpu_kv_handoff.device_resident_kv": 1.0,
        }
