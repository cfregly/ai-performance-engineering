"""Optimized native-TMA prefill vs. decode microbench with burst shaping (no fallbacks)."""

from __future__ import annotations

from collections import deque
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.persistent_decode_common import (
    build_inputs,
    get_stream_priorities,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)
from labs.persistent_decode.tma_extension import load_native_tma


class NativeTmaBurstConfig:
    """Burst shaping knobs for native TMA path."""

    def __init__(self, max_in_flight: int = 2) -> None:
        self.max_in_flight = max_in_flight


class OptimizedNativeTmaPrefillDecodeBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Prefill with native TMA bursts + graph-captured decode."""

    def __init__(self) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.batch_size = self.batch
        self.hidden_dim = self.head_dim
        self.prefill_chunks = 8
        self.prefill_chunk_elems = 128 * 128
        self.cfg = NativeTmaBurstConfig()
        self._prio_low, self._prio_high = get_stream_priorities()
        # Keep custom stream count <= 2 to satisfy the harness StreamAuditor.
        self.prefill_streams = [torch.cuda.Stream(priority=self._prio_low)]
        self.decode_stream = torch.cuda.Stream(priority=self._prio_high)
        self.register_workload_metadata(tokens_per_iteration=float(self.batch * self.seq_len))
        self.decode_graph = torch.cuda.CUDAGraph()
        self.graph_q = None
        self.graph_k = None
        self.graph_v = None
        self.graph_out = None
        self._tma_ext = None
        self._prefill_events: list[torch.cuda.Event] = []
        self._prefill_work: list[
            tuple[torch.cuda.Stream, torch.Tensor, torch.Tensor, torch.cuda.Event]
        ] = []
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())
        self.output: Optional[torch.Tensor] = None
        self._output_view: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.inputs = build_inputs(self.device)
        self._output_view = self.inputs.out[:1, : min(8, self.inputs.out.shape[1])]
        self._verify_output_buffer = torch.empty_like(self._output_view, dtype=torch.float32)
        self.prefill_src = torch.randn(
            self.prefill_chunks, self.prefill_chunk_elems, device=self.device
        )
        self.prefill_dst = torch.empty_like(self.prefill_src)
        self._tma_ext = load_native_tma()  # raises if unsupported
        self._prefill_events = [
            torch.cuda.Event(enable_timing=False, blocking=False)
            for _ in range(self.prefill_chunks)
        ]
        self._prefill_work = [
            (
                self.prefill_streams[idx % len(self.prefill_streams)],
                src,
                dst,
                event,
            )
            for idx, (src, dst, event) in enumerate(
                zip(
                    self.prefill_src.unbind(0),
                    self.prefill_dst.unbind(0),
                    self._prefill_events,
                    strict=True,
                )
            )
        ]

        self.graph_q = self.inputs.q.clone()
        self.graph_k = self.inputs.k.clone()
        self.graph_v = self.inputs.v.clone()
        self.graph_out = torch.empty_like(self.inputs.out)

        torch.cuda.synchronize()
        with torch.cuda.graph(self.decode_graph, stream=self.decode_stream):
            self._decode_body(self.graph_q, self.graph_k, self.graph_v, self.graph_out)
        torch.cuda.synchronize()

    def _decode_body(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, out: torch.Tensor
    ) -> None:
        for t in range(self.seq_len):
            q_t = q[:, t, :]
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            dot = (q_t * k_t).sum(dim=-1, keepdim=True)
            out[:, t, :] = v_t * dot

    def _prefill_shaped_native(self, *, async_only: bool = False) -> deque[torch.cuda.Event] | None:
        """Launch native TMA copies on multiple streams with an in-flight cap."""
        if self._tma_ext is None:
            raise RuntimeError("Native TMA extension not initialized")
        if len(self._prefill_work) != self.prefill_chunks:
            raise RuntimeError("Prefill work not initialized")
        events: deque[torch.cuda.Event] = deque()
        for stream, src, dst, evt in self._prefill_work:
            with torch.cuda.stream(stream):
                self._tma_ext.tma_copy(src, dst)
            evt.record(stream)
            events.append(evt)
            if len(events) > self.cfg.max_in_flight:
                stream.wait_event(events.popleft())
        if async_only:
            return events
        current_stream = torch.cuda.current_stream()
        for evt in events:
            current_stream.wait_event(evt)
        return None

    def _decode_graph(self) -> None:
        assert self.inputs is not None
        with torch.cuda.stream(self.decode_stream):
            self.graph_q.copy_(self.inputs.q)
            self.graph_k.copy_(self.inputs.k)
            self.graph_v.copy_(self.inputs.v)
            self.decode_graph.replay()
            self.inputs.out.copy_(self.graph_out)

    def benchmark_fn(self) -> None:
        if self.inputs is None or self._output_view is None:
            raise RuntimeError("Inputs not initialized")

        current_stream = torch.cuda.current_stream()
        with torch.inference_mode(), self._nvtx_range("prefill_native_shaped_low_pri"):
            pref_events = self._prefill_shaped_native(async_only=True)
        with torch.inference_mode(), self._nvtx_range("decode_graph_high_pri"):
            self._decode_graph()
        if pref_events:
            for evt in pref_events:
                current_stream.wait_event(evt)
        current_stream.wait_stream(self.decode_stream)
        if self.inputs is not None:
            self.output = self._output_view
        if self.inputs is None or self.output is None:
            raise RuntimeError("benchmark_fn() did not produce output")

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={
                "q": self.inputs.q.detach(),
                "k": self.inputs.k.detach(),
                "v": self.inputs.v.detach(),
            },
            output=self._verify_output_buffer,
            batch_size=self.batch,
            parameter_count=0,
            precision_flags={
                "fp16": self.inputs.q.dtype == torch.float16,
                "bf16": self.inputs.q.dtype == torch.bfloat16,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None
        self.output = None
        self._output_view = None
        self._verify_output_buffer = None
        self._prefill_events = []
        self._prefill_work = []

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=8,
            warmup=5,
            measurement_timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "native_tma_prefill_d.batch_size": float(getattr(self, 'batch_size', 0)),
            "native_tma_prefill_d.seq_len": float(getattr(self, 'seq_len', 0)),
            "native_tma_prefill_d.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None

def get_benchmark() -> BaseBenchmark:
    return OptimizedNativeTmaPrefillDecodeBenchmark()
