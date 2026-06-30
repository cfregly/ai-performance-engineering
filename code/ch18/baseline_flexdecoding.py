"""Baseline harness for FlexDecoding with SDPA fallback."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from core.benchmark.cuda_event_timing import elapsed_ms
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.benchmark.verification_mixin import VerificationPayloadMixin
from ch18 import flexdecoding as flexdemo  # noqa: E402


class FlexDecodingHarness(VerificationPayloadMixin, BaseBenchmark):
    """Shared benchmarking harness for baseline/optimized FlexDecoding runs."""

    def __init__(
        self,
        *,
        use_flex_attention: bool,
        require_flex: bool,
        decode_tokens: int = 128,
        compile_enabled: bool = True,
    ):
        super().__init__()
        self.use_flex_attention = use_flex_attention
        self.require_flex = require_flex
        self.decode_tokens = decode_tokens
        self.compile_enabled = compile_enabled
        self.config = flexdemo.FlexDecodingConfig()
        self.model: Optional[flexdemo.FlexDecodingModule] = None
        self.prefill_tokens: Optional[torch.Tensor] = None
        self.decode_token: Optional[torch.Tensor] = None
        self._prefill_total_ms = 0.0
        self._prefill_count = 0
        self._decode_total_ms = 0.0
        self._decode_count = 0
        self._last_output: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self._verification_payload = None
        self._prefill_events: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._decode_events: Optional[List[tuple[torch.cuda.Event, torch.cuda.Event]]] = None
        self._decode_positions: List[int] = []
        self._decode_schedule: List[tuple[int, torch.cuda.Event, torch.cuda.Event]] = []
        self._decode_event_count = 0
        self._decode_position_count = 0
        self._decode_schedule_count = 0
        self._pending_iteration_metrics = False
        self._prefill_metric_values = [0.0]
        self._decode_metric_values = [0.0] * self.decode_tokens
        self._iteration_metric_payload: Dict[str, List[float]] = {
            "prefill_ms": self._prefill_metric_values,
            "decode_ms": self._decode_metric_values,
        }
        total_tokens = self.config.max_seq_len + self.decode_tokens
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(total_tokens),
        )
        # FlexDecoding benchmark: fixed dimensions

    # --------------------------------------------------------------------- setup
    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("FlexDecoding benchmarks require CUDA")
        if self.require_flex and not getattr(flexdemo, "HAS_FLEX", False):
            raise RuntimeError("FlexAttention is not available; optimized path cannot run.")

        self.model = flexdemo.FlexDecodingModule(
            self.config,
            use_flex_attention=self.use_flex_attention,
            compile_enabled=self.compile_enabled,
        ).to(
            self.device,
            dtype=self.config.dtype,
        ).eval()
        self.model.ensure_compiled()
        self.parameter_count = sum(p.numel() for p in self.model.parameters())

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        self.prefill_tokens = torch.randn(
            1,
            self.config.window * 2,
            self.config.dim,
            device=self.device,
            dtype=self.config.dtype,
        )
        self.decode_token = torch.randn(
            1,
            1,
            self.config.dim,
            device=self.device,
            dtype=self.config.dtype,
        )
        self._prefill_events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        self._decode_events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(self.decode_tokens)
        ]
        self._decode_event_count = len(self._decode_events)
        base_position = self.prefill_tokens.size(1)
        self._decode_positions = [base_position + pos for pos in range(self.decode_tokens)]
        self._decode_position_count = len(self._decode_positions)
        self._decode_schedule = [
            (position, start_evt, end_evt)
            for position, (start_evt, end_evt) in zip(
                self._decode_positions,
                self._decode_events,
                strict=True,
            )
        ]
        self._decode_schedule_count = len(self._decode_schedule)
        torch.cuda.synchronize(self.device)

    def _prefill_step(self) -> torch.Tensor:
        if self.model is None or self.prefill_tokens is None:
            raise RuntimeError("Model/tokens not initialized")
        return self.model.prefill(self.prefill_tokens)

    def _decode_step(self, token: torch.Tensor, position: int) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return self.model.decode(token, position)

    # --------------------------------------------------------------- benchmark_fn
    def benchmark_fn(self) -> Optional[Dict[str, List[float]]]:
        if self.model is None or self.prefill_tokens is None or self.decode_token is None:
            raise RuntimeError("Model/tokens not initialized")
        if self._prefill_events is None or self._decode_events is None:
            raise RuntimeError("Timing events not initialized")
        if self._decode_event_count != self.decode_tokens:
            raise RuntimeError("Timing event count mismatch")
        if self._decode_position_count != self.decode_tokens:
            raise RuntimeError("Decode positions not initialized")
        if self._decode_schedule_count != self.decode_tokens:
            raise RuntimeError("Decode schedule not initialized")

        current_stream = torch.cuda.current_stream(self.device)

        with torch.inference_mode():
            with self._nvtx_range("flex_prefill"):
                prefill_start, prefill_end = self._prefill_events
                prefill_start.record(current_stream)
                prefill_out = self._prefill_step()
                prefill_end.record(current_stream)

            with self._nvtx_range("flex_decode"):
                for position, start_evt, end_evt in self._decode_schedule:
                    start_evt.record(current_stream)
                    decode_out = self._decode_step(self.decode_token, position)
                    end_evt.record(current_stream)

        # Store last output for verification
        self._last_output = decode_out if "decode_out" in locals() else prefill_out
        self._pending_iteration_metrics = True
        if self._last_output is None or self.prefill_tokens is None or self.decode_token is None:
            raise RuntimeError("benchmark_fn() must produce output")
        return None

    def finalize_iteration_metrics(self) -> Optional[Dict[str, List[float]]]:
        if not self._pending_iteration_metrics or self._prefill_events is None or self._decode_events is None:
            return None
        prefill_time = elapsed_ms(self._prefill_events)
        decode_times = self._decode_metric_values
        if len(decode_times) != len(self._decode_events):
            decode_times = [0.0] * len(self._decode_events)
            self._decode_metric_values = decode_times
            self._iteration_metric_payload["decode_ms"] = decode_times
        decode_total_ms = 0.0
        for idx, event_pair in enumerate(self._decode_events):
            token_ms = elapsed_ms(event_pair)
            decode_times[idx] = token_ms
            decode_total_ms += token_ms
        self._pending_iteration_metrics = False
        self._prefill_total_ms += prefill_time
        self._prefill_count += 1
        self._decode_total_ms += decode_total_ms
        self._decode_count += len(self._decode_events)
        self._prefill_metric_values[0] = prefill_time
        return self._iteration_metric_payload

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        self._set_verification_payload(
            inputs={
                "prefill_tokens": self.prefill_tokens,
                "decode_token": self.decode_token,
            },
            output=self._last_output,
            batch_size=self.prefill_tokens.shape[0],
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": self._last_output.dtype == torch.float16,
                "bf16": self._last_output.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-1, 10.0),
        )

    # ---------------------------------------------------------------- lifecycle
    def teardown(self) -> None:
        self.model = None
        self.prefill_tokens = None
        self.decode_token = None
        self._decode_schedule = []
        self._decode_event_count = 0
        self._decode_position_count = 0
        self._decode_schedule_count = 0
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------- configs
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=4, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        self.finalize_iteration_metrics()
        if self._prefill_count == 0:
            return None
        return {
            "flex_prefill.mean_ms": float(self._prefill_total_ms / self._prefill_count),
            "flex_decode.mean_ms": float(
                self._decode_total_ms / self._decode_count if self._decode_count else 0.0
            ),
            "flex_decode.tokens_per_iter": float(self.decode_tokens),
            "flexdecode.uses_flex_attention": float(self.use_flex_attention),
        }

    def validate_result(self) -> Optional[str]:
        self.finalize_iteration_metrics()
        if self._prefill_count == 0:
            return "No prefill samples collected"
        if self._decode_count == 0:
            return "No decode samples collected"
        return None


class BaselineFlexDecodingBenchmark(FlexDecodingHarness):
    story_metadata = {
        "pair_role": "canonical",
        "variant_role": "baseline",
        "chapter_alignment": "native",
        "chapter_native_exemplar": True,
        "comparison_axis": "full_kv_mask_vs_windowed_kv_slice",
        "execution_pattern": "masked_full_cache_decode",
        "comparison_reason": (
            "This chapter-native FlexDecoding pair intentionally changes decode work: "
            "the baseline scores the full KV cache with a sliding-window mask, while "
            "the optimized path slices the cache to the active window before attention."
        ),
    }

    def __init__(self):
        super().__init__(use_flex_attention=False, require_flex=False, decode_tokens=512, compile_enabled=False)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        metrics = super().get_custom_metrics()
        if metrics is None:
            return None
        metrics.update(
            {
                "flexdecode.decode_kv_span_tokens": float(self.config.max_seq_len),
                "flexdecode.active_window_tokens": float(self.config.window + 1),
                "flexdecode.window_slice_decode": 0.0,
            }
        )
        return metrics


def get_benchmark() -> BaseBenchmark:
    return BaselineFlexDecodingBenchmark()
