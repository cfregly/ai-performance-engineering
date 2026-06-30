"""Optimized disaggregated prefill/decode benchmark (Chapter 17).

Separates prefill (long context) and decode (short, latency-sensitive) phases onto
independent CUDA streams. Mirrors production scheduling that dedicates resources
for context building while keeping decode latency low.
"""

from __future__ import annotations

from typing import Dict, Optional

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


class OptimizedDisaggregatedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Prefill on a long context + decode on a short context using separate streams."""

    def __init__(self) -> None:
        super().__init__()
        # Match baseline dimensions for fair comparison.
        self.dtype = torch.bfloat16
        self.hidden = 1024
        self.prefill_seq = 256
        self.decode_seq = 16
        self.batch_size = 1

        self.model: Optional[SimpleLLM] = None
        self.prompt: Optional[torch.Tensor] = None
        self.prefill_stream: Optional[torch.cuda.Stream] = None
        self.decode_stream: Optional[torch.cuda.Stream] = None
        self._prefill_done: Optional[torch.cuda.Event] = None
        self._empty_iteration_result: Dict[str, list[float]] = {}
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * (self.prefill_seq + self.decode_seq)),
        )
        self.output: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self._verification_payload = None
        self._pending_ttft_pair: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._empty_tpot_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._pending_tpot_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._pending_tpot_count = 0
        self._ttft_events: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._tpot_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._tpot_event_count = 0
        self._ttft_metric_values = [0.0]
        self._tpot_metric_values = [0.0] * self.decode_seq
        self._tpot_metric_count = self.decode_seq
        self._iteration_metric_payload: Dict[str, list[float]] = {
            "ttft_times_ms": self._ttft_metric_values,
            "tpot_times_ms": self._tpot_metric_values,
        }
        self._enable_nvtx = False
        self._ttft_count = 0
        self._tpot_count = 0

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False

        self.model = SimpleLLM(hidden_dim=self.hidden, num_layers=12).to(self.device).to(self.dtype).eval()
        self.prompt = torch.randint(0, 10000, (self.batch_size, self.prefill_seq), device=self.device)

        self.prefill_stream = torch.cuda.Stream(device=self.device)
        self.decode_stream = torch.cuda.Stream(device=self.device)
        self._prefill_done = torch.cuda.Event()

        # Warm up to reduce first-iteration variance.
        with torch.inference_mode():
            kv_cache = self.model.prefill(self.prompt)
            _ = self.model.decode_step(kv_cache)
        torch.cuda.synchronize(self.device)
        self.parameter_count = sum(p.numel() for p in self.model.parameters())
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
        self._tpot_event_count = self.decode_seq
        self._pending_tpot_count = 0
        self._tpot_metric_count = self.decode_seq
        self._ttft_count = 0
        self._tpot_count = 0

    def _ensure_timing_payload(self, num_tokens: int) -> list[float]:
        if self._tpot_metric_count != num_tokens:
            self._tpot_metric_values = [0.0] * num_tokens
            self._tpot_metric_count = num_tokens
            self._iteration_metric_payload["tpot_times_ms"] = self._tpot_metric_values
        return self._tpot_metric_values

    def _get_ttft_events(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        if self._ttft_events is None:
            self._ttft_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        return self._ttft_events

    def _get_tpot_events(self, num_tokens: int) -> list[tuple[torch.cuda.Event, torch.cuda.Event]]:
        if self._tpot_event_count != num_tokens:
            self._tpot_events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(num_tokens)
            ]
            self._tpot_event_count = num_tokens
        return self._tpot_events

    def benchmark_fn(self) -> Dict[str, list[float]]:
        if (
            self.model is None
            or self.prompt is None
            or self.prefill_stream is None
            or self.decode_stream is None
            or self._prefill_done is None
            or self._ttft_events is None
            or self._tpot_event_count != self.decode_seq
        ):
            raise RuntimeError("Model/inputs/streams not initialized")

        with nvtx_range("optimized_disaggregated_multigpu.prefill_decode", enable=self._enable_nvtx):
            with torch.inference_mode():
                ttft_events = self._ttft_events
                request_start, prefill_end = ttft_events
                default_stream = torch.cuda.current_stream(device=self.device)
                request_start.record(default_stream)
                with torch.cuda.stream(self.prefill_stream):
                    self.prefill_stream.wait_stream(default_stream)
                    kv_cache = self.model.prefill(self.prompt)
                    self._prefill_done.record(self.prefill_stream)
                    prefill_end.record(self.prefill_stream)

                token_output = kv_cache
                token_event_pairs = self._tpot_events
                token_event_count = self._tpot_event_count
                with torch.cuda.stream(self.decode_stream):
                    self.decode_stream.wait_event(self._prefill_done)
                    for token_start, token_end in token_event_pairs:
                        token_start.record(self.decode_stream)
                        token_output = self.model.decode_step(token_output)
                        token_end.record(self.decode_stream)

                self.output = token_output
                self._pending_ttft_pair = ttft_events
                self._pending_tpot_pairs = token_event_pairs
                self._pending_tpot_count = token_event_count
                return self._empty_iteration_result

    def finalize_iteration_metrics(self) -> Optional[Dict[str, list[float]]]:
        if self._pending_ttft_pair is None:
            return None
        ttft_ms = elapsed_ms(self._pending_ttft_pair)
        pending_tpot_pairs = self._pending_tpot_pairs
        pending_tpot_count = self._pending_tpot_count
        tpot_times_ms = self._ensure_timing_payload(pending_tpot_count)
        tpot_total_ms = 0.0
        for idx, event_pair in enumerate(pending_tpot_pairs):
            token_ms = elapsed_ms(event_pair)
            tpot_times_ms[idx] = token_ms
            tpot_total_ms += token_ms
        self._pending_ttft_pair = None
        self._pending_tpot_pairs = self._empty_tpot_pairs
        self._pending_tpot_count = 0
        self._ttft_ms = ttft_ms
        self._tpot_ms = float(tpot_total_ms / pending_tpot_count) if pending_tpot_count else 0.0
        self.total_tokens = float(self.decode_seq)
        self.total_requests = float(self.batch_size)
        self.max_batch_size = float(self.batch_size)
        self._ttft_count += 1
        self._tpot_count += pending_tpot_count
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

    def get_custom_streams(self):
        if self.prefill_stream is None or self.decode_stream is None:
            return None
        return [self.prefill_stream, self.decode_stream]

    def teardown(self) -> None:
        self.model = None
        self.prompt = None
        self.prefill_stream = None
        self.decode_stream = None
        self._prefill_done = None
        self.output = None
        self._ttft_events = None
        self._tpot_events = []
        self._tpot_event_count = 0
        self._pending_ttft_pair = None
        self._pending_tpot_pairs = self._empty_tpot_pairs
        self._pending_tpot_count = 0
        self._tpot_metric_count = self.decode_seq
        self._ttft_count = 0
        self._tpot_count = 0
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        self.finalize_iteration_metrics()
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', None),
            tpot_ms=getattr(self, '_tpot_ms', None),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def validate_result(self) -> Optional[str]:
        self.finalize_iteration_metrics()
        if self.model is None or self.prompt is None:
            return "Model/inputs not initialized"
        if self._ttft_count <= 0:
            return "No TTFT samples recorded"
        if self._tpot_count <= 0:
            return "No TPOT samples recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedDisaggregatedBenchmark()
