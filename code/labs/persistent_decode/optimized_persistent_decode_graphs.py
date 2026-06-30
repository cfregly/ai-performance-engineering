"""Persistent decode with CUDA Graphs: prefill + decode captured separately."""

from __future__ import annotations

from enum import Enum
from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.persistent_decode.optimized_persistent_decode_triton import persistent_decode_kernel
from labs.persistent_decode.persistent_decode_common import (
    build_decode_input_signature,
    build_inputs,
    get_decode_options,
    get_decode_profile,
    resolve_device,
    resolve_shapes,
    tokens_per_iteration,
)


class GraphMode(Enum):
    FULL = "full"
    PIECEWISE = "piecewise"
    FULL_AND_PIECEWISE = "full_and_piecewise"

    @classmethod
    def from_str(cls, raw: str | None) -> "GraphMode":
        normalized = (raw or cls.FULL_AND_PIECEWISE.value).strip().lower().replace("-", "_")
        for mode in cls:
            if normalized == mode.value:
                return mode
        return cls.FULL_AND_PIECEWISE


class OptimizedPersistentDecodeGraphsBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Capture prefill and persistent decode in CUDA Graphs."""

    def __init__(self, *, graph_mode: GraphMode | None = None, max_capture_seq: int | None = None) -> None:
        super().__init__()
        self.device = resolve_device()
        self.inputs = None
        self.options = get_decode_options()
        self.profile = get_decode_profile()
        self.batch, self.seq_len, self.head_dim = resolve_shapes()
        self.batch_size = self.batch
        self.hidden_dim = self.head_dim
        self.block_k = self.profile.block_k
        self.num_programs = self.profile.num_programs
        self.prefill_graph: torch.cuda.CUDAGraph | None = None
        self.decode_graph: torch.cuda.CUDAGraph | None = None
        self.full_graph: torch.cuda.CUDAGraph | None = None
        self.prefill_out: torch.Tensor | None = None
        self.graph_mode = graph_mode or GraphMode.FULL_AND_PIECEWISE
        self.max_capture_seq = max_capture_seq or self.seq_len
        self._ttft_metric_values: list[float] = [0.0]
        self._tpot_metric_values: list[float] = [0.0] * self.seq_len
        self._iteration_metric_payload: dict[str, list[float]] = {
            "ttft_times_ms": self._ttft_metric_values,
            "tpot_times_ms": self._tpot_metric_values,
        }
        self._pending_graph_path: str | None = None
        self._pending_start: torch.cuda.Event | None = None
        self._pending_prefill_end: torch.cuda.Event | None = None
        self._pending_decode_start: torch.cuda.Event | None = None
        self._pending_decode_end: torch.cuda.Event | None = None
        self._full_events: dict[str, torch.cuda.Event] = {}
        self._piecewise_events: dict[str, torch.cuda.Event] = {}
        self.register_workload_metadata(tokens_per_iteration=tokens_per_iteration())
        self.output: torch.Tensor | None = None
        self._output_view: torch.Tensor | None = None
        self._verify_output_buffer: torch.Tensor | None = None

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.inputs = build_inputs(self.device)
        self._output_view = self.inputs.out[:1, : min(8, self.inputs.out.shape[1])]
        self._verify_output_buffer = torch.empty_like(self._output_view, dtype=torch.float32)
        self.prefill_out = torch.empty((self.batch, self.seq_len), device=self.device, dtype=torch.float32)
        self._full_events = {
            "start": torch.cuda.Event(enable_timing=True),
            "end": torch.cuda.Event(enable_timing=True),
        }
        self._piecewise_events = {
            "start_prefill": torch.cuda.Event(enable_timing=True),
            "end_prefill": torch.cuda.Event(enable_timing=True),
            "start_decode": torch.cuda.Event(enable_timing=True),
            "end_decode": torch.cuda.Event(enable_timing=True),
        }
        torch.cuda.synchronize()
        self._capture_graphs()

    def _capture_graphs(self) -> None:
        self._capture_piecewise_graphs()
        self._capture_full_graph()

    def _capture_piecewise_graphs(self) -> None:
        # Capture prefill (toy: dot across head_dim per token)
        self.prefill_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.prefill_graph):
            qk = (self.inputs.q * self.inputs.k).sum(dim=-1)
            self.prefill_out.copy_(qk)

        # Capture persistent decode kernel
        self.decode_graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize()
        num_items = min(self.batch, self.num_programs)
        grid = (max(1, num_items),)
        BLOCK_K = self.block_k
        with torch.cuda.graph(self.decode_graph):
            persistent_decode_kernel[grid](
                self.inputs.q,
                self.inputs.k,
                self.inputs.v,
                self.inputs.out,
                self.inputs.work_seq_ids,
                self.inputs.work_steps,
                num_items,
                head_dim=self.head_dim,
                max_steps=self.seq_len,
                BLOCK_K=BLOCK_K,
                num_warps=2,
                num_stages=1,
            )
        torch.cuda.synchronize()

    def _capture_full_graph(self) -> None:
        if self.graph_mode == GraphMode.PIECEWISE:
            return
        self.full_graph = torch.cuda.CUDAGraph()
        num_items = min(self.batch, self.num_programs)
        grid = (max(1, num_items),)
        BLOCK_K = self.block_k
        with torch.cuda.graph(self.full_graph):
            qk = (self.inputs.q * self.inputs.k).sum(dim=-1)
            self.prefill_out.copy_(qk)
            persistent_decode_kernel[grid](
                self.inputs.q,
                self.inputs.k,
                self.inputs.v,
                self.inputs.out,
                self.inputs.work_seq_ids,
                self.inputs.work_steps,
                num_items,
                head_dim=self.head_dim,
                max_steps=self.seq_len,
                BLOCK_K=BLOCK_K,
                num_warps=2,
                num_stages=1,
            )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None or self.prefill_out is None or self._output_view is None:
            raise RuntimeError("Inputs not initialized")

        use_full = (
            self.graph_mode == GraphMode.FULL
            or (self.graph_mode == GraphMode.FULL_AND_PIECEWISE and self.seq_len <= self.max_capture_seq)
        )
        current_stream = torch.cuda.current_stream(self.device)
        if use_full and self.full_graph is not None:
            start = self._full_events["start"]
            end = self._full_events["end"]
            with torch.inference_mode(), self._nvtx_range("full_graph"):
                start.record(current_stream)
                self.full_graph.replay()
                end.record(current_stream)
            self._pending_graph_path = "full_graph"
            self._pending_start = start
            self._pending_prefill_end = end
            self._pending_decode_start = start
            self._pending_decode_end = end
            self.output = self._output_view
            return

        if self.prefill_graph is None or self.decode_graph is None:
            raise RuntimeError("Piecewise graphs not initialized")

        with torch.inference_mode(), self._nvtx_range(
            "piecewise_graph" if self.graph_mode != GraphMode.FULL_AND_PIECEWISE else "graph_fallback_piecewise"
        ):
            start_prefill = self._piecewise_events["start_prefill"]
            end_prefill = self._piecewise_events["end_prefill"]
            start_decode = self._piecewise_events["start_decode"]
            end_decode = self._piecewise_events["end_decode"]
            start_prefill.record(current_stream)
            self.prefill_graph.replay()
            end_prefill.record(current_stream)
            start_decode.record(current_stream)
            self.decode_graph.replay()
            end_decode.record(current_stream)
            self._pending_graph_path = "piecewise_graph"
            self._pending_start = start_prefill
            self._pending_prefill_end = end_prefill
            self._pending_decode_start = start_decode
            self._pending_decode_end = end_decode
        self.output = self._output_view

    def _ensure_tpot_metric_values(self) -> list[float]:
        if len(self._tpot_metric_values) != self.seq_len:
            self._tpot_metric_values = [0.0] * self.seq_len
            self._iteration_metric_payload["tpot_times_ms"] = self._tpot_metric_values
        return self._tpot_metric_values

    def finalize_iteration_metrics(self) -> dict[str, list[float]] | None:
        start = self._pending_start
        prefill_end = self._pending_prefill_end
        decode_start = self._pending_decode_start
        decode_end = self._pending_decode_end
        graph_path = self._pending_graph_path
        if start is None or prefill_end is None or decode_start is None or decode_end is None or graph_path is None:
            return None

        self._pending_graph_path = None
        self._pending_start = None
        self._pending_prefill_end = None
        self._pending_decode_start = None
        self._pending_decode_end = None

        ttft_ms = float(start.elapsed_time(prefill_end))
        decode_ms = float(decode_start.elapsed_time(decode_end))
        per_token_ms = decode_ms / max(1, self.seq_len)
        self._ttft_metric_values[0] = ttft_ms
        tpot_times_ms = self._ensure_tpot_metric_values()
        for idx in range(len(tpot_times_ms)):
            tpot_times_ms[idx] = per_token_ms
        return self._iteration_metric_payload

    def capture_verification_payload(self) -> None:
        if self.inputs is None or self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("setup() and benchmark_fn() must be called before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={
                "q": self.inputs.q,
                "k": self.inputs.k,
                "v": self.inputs.v,
            },
            output=self._verify_output_buffer,
            batch_size=self.batch,
            parameter_count=0,
            precision_flags={
                "fp16": self.inputs.q.dtype == torch.float16,
                "bf16": self.inputs.q.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def teardown(self) -> None:
        torch.cuda.empty_cache()
        self.inputs = None
        self.prefill_graph = None
        self.decode_graph = None
        self.full_graph = None
        self.prefill_out = None
        self._full_events = {}
        self._piecewise_events = {}
        self.output = None
        self._output_view = None
        self._verify_output_buffer = None
        self._pending_graph_path = None
        self._pending_start = None
        self._pending_prefill_end = None
        self._pending_decode_start = None
        self._pending_decode_end = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=12,
            warmup=5,
            enable_profiling=False,
            enable_ncu=False,
            enable_nsys=False,
            measurement_timeout_seconds=90,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return inference metrics."""
        return {
            "persistent_decode_gr.batch_size": float(getattr(self, 'batch_size', 0)),
            "persistent_decode_gr.seq_len": float(getattr(self, 'seq_len', 0)),
            "persistent_decode_gr.hidden_dim": float(getattr(self, 'hidden_dim', 0)),
        }

    def get_input_signature(self) -> dict:
        return build_decode_input_signature(
            batch=self.batch,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            quantization=self.options.quantization,
        )

    def validate_result(self) -> str | None:
        if self.inputs is None:
            return "Inputs not initialized"
        if not torch.isfinite(self.inputs.out).all():
            return "Non-finite output detected"
        return None

def get_benchmark() -> BaseBenchmark:
    return OptimizedPersistentDecodeGraphsBenchmark()
