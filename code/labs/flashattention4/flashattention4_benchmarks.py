"""Shared benchmark bases for the FlashAttention-4 lab family."""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.flashattention4.flashattention4_common import (
    BEST_AVAILABLE_PROVIDERS,
    FLASHATTENTION4_PROVIDER_ORDER,
    FlashAttention4Config,
    FlashAttention4Inputs,
    FlashAttention4Kernel,
    FlashAttention4Selection,
    build_reference_inputs,
    eager_flex_attention,
    emit_flashattention4_mode_table_artifacts,
    flashattention4_claim_type_id,
    flashattention4_provider_id,
    load_lab_config,
    resolve_best_available_attention_kernel,
    resolve_cuda_device,
    resolve_flashattention4_claim_type,
    resolve_flashattention4_kernel,
    resolve_flashattention4_mode_decision,
)
from labs.flashattention4.pipeline_model import PipelineStageProfile, estimate_pipeline


class FlashAttention4BenchmarkBase(VerificationPayloadMixin, BaseBenchmark):
    """Shared setup/verification lifecycle for FlashAttention-4 benchmarks."""

    default_claim_type = "educational"
    default_iterations = 8
    default_warmup = 5
    default_measurement_timeout_seconds = 90

    def __init__(self) -> None:
        super().__init__()
        self.config: FlashAttention4Config = self._build_lab_config()
        self.inputs: Optional[FlashAttention4Inputs] = None
        self.output: Optional[torch.Tensor] = None
        self._selected_provider: Optional[str] = None
        self._prev_matmul_allow_tf32: Optional[bool] = None
        self._prev_cudnn_allow_tf32: Optional[bool] = None
        self._sparsity_ratio = 1.0
        self._verify_output_buffer: Optional[torch.Tensor] = None
        tokens = self.config.batch * self.config.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.config.batch),
            tokens_per_iteration=float(tokens),
        )

    def _build_lab_config(self) -> FlashAttention4Config:
        return load_lab_config()

    def setup(self) -> None:
        resolve_cuda_device()
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self._prev_matmul_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
        self._prev_cudnn_allow_tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.inputs = build_reference_inputs(self.config, device=self.device, include_block_mask=True)
        self._verify_output_buffer = torch.empty(
            (1, 1, min(128, self.config.seq_len), min(16, self.config.head_dim)),
            device=self.device,
            dtype=torch.float32,
        )
        if self.inputs.dense_mask is not None:
            self._sparsity_ratio = float(self.inputs.dense_mask.float().mean())
        self._prepare_benchmark()
        runtime_config = self.get_config()
        claim_type = resolve_flashattention4_claim_type(
            getattr(runtime_config, "target_label", None),
            default=self.default_claim_type,
        )
        emit_flashattention4_mode_table_artifacts(
            runtime_config,
            current_mode=self.config.mode,
            run_claim_type=claim_type,
            selected_provider=self._selected_provider,
        )

    def _prepare_benchmark(self) -> None:
        raise NotImplementedError

    def _run_attention(self) -> torch.Tensor:
        raise NotImplementedError

    def _nvtx_label(self) -> str:
        raise NotImplementedError

    def _validate_state(self) -> Optional[str]:
        if self.inputs is None:
            return "FlashAttention-4 inputs are not initialized"
        return None

    def benchmark_fn(self) -> None:
        state_error = self._validate_state()
        if state_error is not None:
            raise RuntimeError(state_error)
        with torch.inference_mode():
            with self._nvtx_range(self._nvtx_label()):
                result = self._run_attention()
                self.output = result
        if self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("setup() and benchmark_fn() must run first")
        verify_output = self._verify_output_buffer
        output_slice = self.output[
            :1,
            :1,
            : min(self.output.shape[2], verify_output.shape[2]),
            : min(self.output.shape[3], verify_output.shape[3]),
        ].detach()
        verify_output = verify_output[:, :, : output_slice.shape[2], : output_slice.shape[3]]
        verify_output.copy_(output_slice, non_blocking=False)
        self._set_verification_payload(
            inputs={
                "q": self.inputs.q.detach(),
                "k": self.inputs.k.detach(),
                "v": self.inputs.v.detach(),
            },
            output=verify_output,
            batch_size=self.config.batch,
            parameter_count=0,
            precision_flags={
                "fp16": self.config.dtype == torch.float16,
                "bf16": self.config.dtype == torch.bfloat16,
                "tf32": False,
            },
            output_tolerance=(5e-2, 5e-1),
            signature_overrides={"sparsity_ratio": self._sparsity_ratio},
        )

    def teardown(self) -> None:
        self.inputs = None
        self.output = None
        self._verify_output_buffer = None
        self._selected_provider = None
        self._sparsity_ratio = 1.0
        if self._prev_matmul_allow_tf32 is not None:
            torch.backends.cuda.matmul.allow_tf32 = self._prev_matmul_allow_tf32
        if self._prev_cudnn_allow_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = self._prev_cudnn_allow_tf32
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        if self._config is None:
            self._config = BenchmarkConfig(
                iterations=self.default_iterations,
                warmup=self.default_warmup,
                measurement_timeout_seconds=self.default_measurement_timeout_seconds,
            )
        return self._config

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def _base_custom_metrics(self) -> dict[str, float]:
        runtime_config = self.get_config()
        claim_type = resolve_flashattention4_claim_type(
            getattr(runtime_config, "target_label", None),
            default=self.default_claim_type,
        )
        decision = resolve_flashattention4_mode_decision(self.config.mode)
        metrics = {
            "flashattention4.mode": self.config.mode,
            "flashattention4.block_size": float(self.config.block_size),
            "flashattention4.window_size": float(self.config.window_size),
            "flashattention4.allow_tf32": 0.0,
            "flashattention4.provider_id": flashattention4_provider_id(self._selected_provider or ""),
            "flashattention4.claim_type_id": flashattention4_claim_type_id(claim_type),
            "flashattention4.recommended_provider_id": flashattention4_provider_id(
                decision.recommended_backend
            ),
            "flashattention4.provider_matches_recommended": (
                1.0 if self._selected_provider == decision.recommended_backend else 0.0
            ),
        }
        for provider in FLASHATTENTION4_PROVIDER_ORDER:
            metrics[f"flashattention4.selected.{provider}"] = (
                1.0 if self._selected_provider == provider else 0.0
            )
        return metrics

    def validate_result(self) -> Optional[str]:
        state_error = self._validate_state()
        if state_error is not None:
            return state_error
        if self.output is None:
            return "benchmark_fn() did not produce output"
        return None


class BaselineFlashAttention4BenchmarkBase(FlashAttention4BenchmarkBase):
    """Uncompiled eager FlexAttention path."""

    def _prepare_benchmark(self) -> None:
        self._selected_provider = "eager_flex"

    def _run_attention(self) -> torch.Tensor:
        if self.inputs is None:
            raise RuntimeError("FlashAttention-4 inputs are not initialized")
        return eager_flex_attention(self.inputs)

    def _nvtx_label(self) -> str:
        return "flashattention4_baseline_eager"

    def get_custom_metrics(self) -> Optional[dict]:
        return self._base_custom_metrics()


class OptimizedFlashAttention4BenchmarkBase(FlashAttention4BenchmarkBase):
    """Compiled FlashAttention path with an explicit provider selection."""

    default_iterations = 10
    default_warmup = 6
    default_measurement_timeout_seconds = 120

    def __init__(self) -> None:
        super().__init__()
        self.kernel: Optional[FlashAttention4Kernel] = None

    def _prepare_benchmark(self) -> None:
        if self.inputs is None:
            raise RuntimeError("FlashAttention-4 inputs are not initialized")
        self.kernel = resolve_flashattention4_kernel(self.inputs, self.config)
        self._selected_provider = self.kernel.provider

    def _run_attention(self) -> torch.Tensor:
        if self.inputs is None or self.kernel is None:
            raise RuntimeError("FlashAttention-4 kernel is not initialized")
        return self.kernel.fn(self.inputs.q, self.inputs.k, self.inputs.v)

    def _nvtx_label(self) -> str:
        if self.kernel is None:
            raise RuntimeError("FlashAttention-4 kernel is not initialized")
        return f"flashattention4_optimized_{self.kernel.provider}"

    def _validate_state(self) -> Optional[str]:
        if self.inputs is None or self.kernel is None:
            return "FlashAttention-4 kernel is not initialized"
        return None

    def teardown(self) -> None:
        self.kernel = None
        super().teardown()

    def get_custom_metrics(self) -> Optional[dict]:
        tile_count = max(1, self.config.seq_len // max(1, self.config.block_size))
        pipeline = estimate_pipeline(tile_count, PipelineStageProfile())
        return {
            **self._base_custom_metrics(),
            "flashattention4.estimated_pipeline_speedup": pipeline.speedup,
            "flashattention4.pipeline_tiles": float(tile_count),
        }


class OptimizedBestAvailableAttentionBenchmarkBase(FlashAttention4BenchmarkBase):
    """Select the fastest correct backend on the current machine for the chosen mode."""

    default_claim_type = "absolute"
    default_iterations = 10
    default_warmup = 6
    default_measurement_timeout_seconds = 150

    def __init__(self) -> None:
        super().__init__()
        self.selection: Optional[FlashAttention4Selection] = None
        self.kernel: Optional[FlashAttention4Kernel] = None
        self._candidate_median_ms: dict[str, float] = {}
        self._candidate_errors: dict[str, str] = {}

    def _build_lab_config(self) -> FlashAttention4Config:
        return replace(load_lab_config(), backend="auto")

    def _prepare_benchmark(self) -> None:
        if self.inputs is None:
            raise RuntimeError("Best-available attention kernel is not initialized")
        self.selection = resolve_best_available_attention_kernel(self.inputs, self.config)
        self.kernel = self.selection.kernel
        self._selected_provider = self.selection.kernel.provider
        self._candidate_median_ms = dict(self.selection.candidate_median_ms)
        self._candidate_errors = dict(self.selection.candidate_errors)

    def _run_attention(self) -> torch.Tensor:
        if self.inputs is None or self.kernel is None:
            raise RuntimeError("Best-available attention kernel is not initialized")
        return self.kernel.fn(self.inputs.q, self.inputs.k, self.inputs.v)

    def _nvtx_label(self) -> str:
        if self.kernel is None:
            raise RuntimeError("Best-available attention kernel is not initialized")
        return f"flashattention4_best_available_{self.kernel.provider}"

    def _validate_state(self) -> Optional[str]:
        if self.inputs is None or self.kernel is None or self.selection is None:
            return "Best-available attention kernel is not initialized"
        return None

    def teardown(self) -> None:
        self.selection = None
        self.kernel = None
        self._candidate_median_ms = {}
        self._candidate_errors = {}
        super().teardown()

    def get_custom_metrics(self) -> Optional[dict]:
        metrics = {
            **self._base_custom_metrics(),
        }
        for provider in BEST_AVAILABLE_PROVIDERS:
            metrics[f"flashattention4.selected.{provider}"] = (
                1.0 if self._selected_provider == provider else 0.0
            )
        if not self._candidate_median_ms and not self._candidate_errors:
            return metrics

        metrics["flashattention4.selection_candidates"] = float(len(self._candidate_median_ms))
        metrics["flashattention4.selection_failures"] = float(len(self._candidate_errors))
        best_provider: Optional[str] = None
        second_provider: Optional[str] = None
        best_ms = float("inf")
        second_ms = float("inf")
        for provider, median_ms in self._candidate_median_ms.items():
            metrics[f"flashattention4.selection_ms.{provider}"] = median_ms
            candidate_key = (median_ms, provider)
            if best_provider is None or candidate_key < (best_ms, best_provider):
                second_provider, second_ms = best_provider, best_ms
                best_provider, best_ms = provider, median_ms
            elif second_provider is None or candidate_key < (second_ms, second_provider):
                second_provider, second_ms = provider, median_ms
        if second_provider is not None:
            metrics["flashattention4.best_margin_ms"] = second_ms - best_ms
        for provider in self._candidate_errors:
            metrics[f"flashattention4.failed.{provider}"] = 1.0
        return metrics
