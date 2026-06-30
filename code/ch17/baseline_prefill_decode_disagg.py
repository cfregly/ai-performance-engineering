"""Baseline monolithic prefill+decode benchmark (native to Chapter 17)."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ch17.prefill_decode_disagg_monolithic_common import SimpleLLM
from core.benchmark.cuda_event_timing import elapsed_ms
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (  # noqa: E402
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class BaselinePrefillDecodeMonolithicBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Monolithic prefill+decode baseline (no disaggregation)."""

    def __init__(self):
        super().__init__()
        self.model: Optional[SimpleLLM] = None
        self.prompt: Optional[torch.Tensor] = None
        self.kv_cache: Optional[torch.Tensor] = None
        self._empty_iteration_result: Dict[str, List[float]] = {}
        # Workload dimensions for signature matching
        self.batch_size = 1
        self.prefill_seq = 256
        self.decode_seq = 16
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=self.prefill_seq + self.decode_seq,
        )
        self.output: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self._verification_payload = None
        self._pending_ttft_pair: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._empty_tpot_pairs: List[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._pending_tpot_pairs: List[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._ttft_events: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._tpot_events: List[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._ttft_metric_values = [0.0]
        self._tpot_metric_values = [0.0] * self.decode_seq
        self._iteration_metric_payload: Dict[str, List[float]] = {
            "ttft_times_ms": self._ttft_metric_values,
            "tpot_times_ms": self._tpot_metric_values,
        }
        self._enable_nvtx = False
        self._ttft_total_ms = 0.0
        self._tpot_total_ms = 0.0
        self._ttft_count = 0
        self._tpot_count = 0

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        self.model = SimpleLLM(hidden_dim=1024, num_layers=12).to(self.device).to(torch.bfloat16).eval()
        self.prompt = torch.randint(0, 10000, (1, 256), device=self.device)
        with torch.inference_mode():
            self.kv_cache = self.model.prefill(self.prompt)
        torch.cuda.synchronize(self.device)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
        self._ttft_total_ms = 0.0
        self._tpot_total_ms = 0.0
        self._ttft_count = 0
        self._tpot_count = 0
        self._ttft_events = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        self._tpot_events = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(self.decode_seq)
        ]

    def _ensure_timing_payload(self, num_tokens: int) -> List[float]:
        if len(self._tpot_metric_values) != num_tokens:
            self._tpot_metric_values = [0.0] * num_tokens
            self._iteration_metric_payload["tpot_times_ms"] = self._tpot_metric_values
        return self._tpot_metric_values

    def _get_ttft_events(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        if self._ttft_events is None:
            self._ttft_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        return self._ttft_events

    def _get_tpot_events(self, num_tokens: int) -> List[tuple[torch.cuda.Event, torch.cuda.Event]]:
        if len(self._tpot_events) != num_tokens:
            self._tpot_events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(num_tokens)
            ]
        return self._tpot_events

    def benchmark_fn(self) -> Dict[str, List[float]]:
        if self.model is None or self.prompt is None:
            raise RuntimeError("Model or prompt not initialized")

        with nvtx_range("inference_monolithic", enable=self._enable_nvtx):
            with torch.inference_mode():
                ttft_events = self._get_ttft_events()
                request_start, prefill_end = ttft_events
                current_stream = torch.cuda.current_stream(device=self.device)
                request_start.record(current_stream)
                kv_cache = self.model.prefill(self.prompt)
                prefill_end.record(current_stream)

                num_tokens = self.decode_seq
                token_event_pairs = self._get_tpot_events(num_tokens)
                token_output = kv_cache
                for token_start, token_end in token_event_pairs:
                    token_start.record(current_stream)
                    token_output = self.model.decode_step(token_output)
                    token_end.record(current_stream)
                self.output = token_output

                self._pending_ttft_pair = ttft_events
                self._pending_tpot_pairs = token_event_pairs
                return self._empty_iteration_result

    def finalize_iteration_metrics(self) -> Optional[Dict[str, List[float]]]:
        if self._pending_ttft_pair is None:
            return None
        ttft_ms = elapsed_ms(self._pending_ttft_pair)
        pending_tpot_pairs = self._pending_tpot_pairs
        tpot_times_ms = self._ensure_timing_payload(len(pending_tpot_pairs))
        tpot_total_ms = 0.0
        for idx, event_pair in enumerate(pending_tpot_pairs):
            token_ms = elapsed_ms(event_pair)
            tpot_times_ms[idx] = token_ms
            tpot_total_ms += token_ms
        self._pending_ttft_pair = None
        self._pending_tpot_pairs = self._empty_tpot_pairs
        self._ttft_total_ms += ttft_ms
        self._tpot_total_ms += tpot_total_ms
        self._ttft_count += 1
        self._tpot_count += len(pending_tpot_pairs)
        self._ttft_metric_values[0] = ttft_ms
        return self._iteration_metric_payload

    def capture_verification_payload(self) -> None:
        self.finalize_iteration_metrics()
        if self.prompt is None or self.output is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        dtype = self.output.dtype
        self._set_verification_payload(
            inputs={"prompt": self.prompt},
            output=self.output,
            batch_size=int(self.batch_size),
            parameter_count=self.parameter_count,
            precision_flags={
                "fp16": dtype == torch.float16,
                "bf16": dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        self.model = None
        self.prompt = None
        self.kv_cache = None
        self.output = None
        self._ttft_events = None
        self._tpot_events = []
        self._pending_ttft_pair = None
        self._pending_tpot_pairs = self._empty_tpot_pairs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        self.finalize_iteration_metrics()
        if self._ttft_count <= 0:
            return None
        return {
            "monolithic.ttft_ms": float(self._ttft_total_ms / self._ttft_count),
            "monolithic.tpot_mean_ms": float(
                self._tpot_total_ms / self._tpot_count if self._tpot_count else 0.0
            ),
        }

    def validate_result(self) -> Optional[str]:
        self.finalize_iteration_metrics()
        if self._ttft_count <= 0:
            return "No TTFT samples recorded"
        if self._tpot_count <= 0:
            return "No TPOT samples recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery (Chapter 17 baseline)."""
    return BaselinePrefillDecodeMonolithicBenchmark()
