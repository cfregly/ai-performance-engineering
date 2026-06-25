"""Optimized disaggregated prefill/decode benchmark (Chapter 17).

Separates prefill (long context) and decode (short, latency-sensitive) phases onto
independent CUDA streams. Mirrors production scheduling that dedicates resources
for context building while keeping decode latency low.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from ch17.prefill_decode_disagg_monolithic_common import SimpleLLM
from core.benchmark.cuda_event_timing import elapsed_ms, elapsed_ms_list
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
        self._history: Dict[str, list[float]] = {"ttft": [], "tpot": []}
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * (self.prefill_seq + self.decode_seq)),
        )
        self.output: Optional[torch.Tensor] = None
        self.parameter_count: int = 0
        self._verification_payload = None
        self._pending_ttft_pair: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._pending_tpot_pairs: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self._ttft_events: Optional[tuple[torch.cuda.Event, torch.cuda.Event]] = None
        self._tpot_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.model = SimpleLLM(hidden_dim=self.hidden, num_layers=12).to(self.device).to(self.dtype).eval()
        self.prompt = torch.randint(0, 10000, (self.batch_size, self.prefill_seq), device=self.device)

        self.prefill_stream = torch.cuda.Stream(device=self.device)
        self.decode_stream = torch.cuda.Stream(device=self.device)
        self._prefill_done = torch.cuda.Event()

        # Warm up to reduce first-iteration variance.
        with torch.no_grad():
            kv_cache = self.model.prefill(self.prompt)
            _ = self.model.decode(kv_cache, num_tokens=1)
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

    def _get_ttft_events(self) -> tuple[torch.cuda.Event, torch.cuda.Event]:
        if self._ttft_events is None:
            self._ttft_events = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
        return self._ttft_events

    def _get_tpot_events(self, num_tokens: int) -> list[tuple[torch.cuda.Event, torch.cuda.Event]]:
        if len(self._tpot_events) != num_tokens:
            self._tpot_events = [
                (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                for _ in range(num_tokens)
            ]
        return self._tpot_events

    def benchmark_fn(self) -> Dict[str, list[float]]:
        if self.model is None or self.prompt is None or self.prefill_stream is None or self.decode_stream is None or self._prefill_done is None:
            raise RuntimeError("Model/inputs/streams not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("optimized_disaggregated_multigpu.prefill_decode", enable=enable_nvtx):
            with torch.no_grad():
                ttft_events = self._get_ttft_events()
                request_start, prefill_end = ttft_events
                request_start.record()
                default_stream = torch.cuda.current_stream(device=self.device)
                with torch.cuda.stream(self.prefill_stream):
                    self.prefill_stream.wait_stream(default_stream)
                    kv_cache = self.model.prefill(self.prompt)
                    self._prefill_done.record(self.prefill_stream)
                    prefill_end.record()

                token_output = kv_cache
                token_event_pairs = self._get_tpot_events(self.decode_seq)
                for token_start, token_end in token_event_pairs:
                    with torch.cuda.stream(self.decode_stream):
                        token_start.record(self.decode_stream)
                        self.decode_stream.wait_event(self._prefill_done)
                        token_output = self.model.decode(token_output[:, -1:, :], num_tokens=1)
                        token_end.record()

                self.output = token_output
                self._pending_ttft_pair = ttft_events
                self._pending_tpot_pairs = token_event_pairs
                return {}

    def finalize_iteration_metrics(self) -> Optional[Dict[str, list[float]]]:
        if self._pending_ttft_pair is None:
            return None
        ttft_ms = elapsed_ms(self._pending_ttft_pair)
        tpot_times_ms = elapsed_ms_list(self._pending_tpot_pairs)
        self._pending_ttft_pair = None
        self._pending_tpot_pairs = []
        self._history["ttft"].append(ttft_ms)
        self._history["tpot"].extend(tpot_times_ms)
        self._ttft_ms = ttft_ms
        self._tpot_ms = float(sum(tpot_times_ms) / len(tpot_times_ms)) if tpot_times_ms else 0.0
        self.total_tokens = float(self.decode_seq)
        self.total_requests = float(self.batch_size)
        self.max_batch_size = float(self.batch_size)
        return {
            "ttft_times_ms": [ttft_ms],
            "tpot_times_ms": tpot_times_ms,
        }

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
        self._pending_ttft_pair = None
        self._pending_tpot_pairs = []
        self._history = {"ttft": [], "tpot": []}
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
        if not self._history["ttft"]:
            return "No TTFT samples recorded"
        if not self._history["tpot"]:
            return "No TPOT samples recorded"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedDisaggregatedBenchmark()

