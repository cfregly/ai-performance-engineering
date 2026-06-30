"""FlexDecoding decode path wrapped in CUDA Graphs for lower launch overhead."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from ch18.baseline_flexdecoding import FlexDecodingHarness  # noqa: E402


class OptimizedFlexDecodingGraphsBenchmark(FlexDecodingHarness):
    """Capture a single-token decode in a CUDA Graph and replay per token."""

    def __init__(self) -> None:
        super().__init__(
            use_flex_attention=False,
            require_flex=False,
            decode_tokens=512,
            compile_enabled=False,
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.capture_stream: torch.cuda.Stream | None = None
        self.static_decode_q: torch.Tensor | None = None
        self.static_decode_k: torch.Tensor | None = None
        self.static_decode_v: torch.Tensor | None = None
        self.static_decode_out: torch.Tensor | None = None
        self.base_position: int = 0

    def _run_warmup(self) -> None:
        """Compile and warm kernels before capture."""
        if self.model is None or self.prefill_tokens is None or self.decode_token is None:
            raise RuntimeError("Model/tokens not initialized")
        with torch.inference_mode():
            self.model.prefill(self.prefill_tokens)
            _ = self.model.decode(self.decode_token, self.base_position)
        torch.cuda.synchronize(self.device)

    def setup(self) -> None:
        self._initialize_and_capture()

    def _initialize_and_capture(self) -> None:
        super().setup()
        if self.model is None or self.prefill_tokens is None or self.decode_token is None:
            raise RuntimeError("Model/tokens not initialized")

        self.base_position = self.prefill_tokens.size(1)
        batch = self.decode_token.size(0)
        heads = self.model.cfg.heads
        head_dim = self.model.head_dim
        with torch.inference_mode():
            self.static_decode_q = self.model.q_proj(self.decode_token).view(batch, 1, heads, head_dim)
            self.static_decode_k = self.model.k_proj(self.decode_token).view(batch, 1, heads, head_dim)
            self.static_decode_v = self.model.v_proj(self.decode_token).view(batch, 1, heads, head_dim)
        self.static_decode_out = torch.empty_like(self.decode_token)
        self.capture_stream = torch.cuda.Stream(device=self.device)

        # Compile/warm outside capture to avoid lazy compile during graph capture.
        self._run_warmup()

        self.graph = torch.cuda.CUDAGraph()
        assert self.capture_stream is not None
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            if self.model is None:
                raise RuntimeError("Model not initialized for capture")
            if self.static_decode_q is None:
                raise RuntimeError("Static decode Q not initialized for capture")
            out = self.model.decode_attention(self.static_decode_q)
            self.static_decode_out.copy_(out)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[Dict[str, List[float]]]:
        if (
            self.model is None
            or self.prefill_tokens is None
            or self.decode_token is None
            or self.graph is None
            or self.capture_stream is None
            or self.static_decode_k is None
            or self.static_decode_v is None
            or self.static_decode_out is None
        ):
            raise RuntimeError("Graph path not initialized")

        self.model.clear_cache(batch=self.prefill_tokens.size(0))
        if self._prefill_events is None or self._decode_events is None:
            raise RuntimeError("Timing events not initialized")
        if self._decode_event_count != self.decode_tokens:
            raise RuntimeError("Timing event count mismatch")
        if self._decode_position_count != self.decode_tokens:
            raise RuntimeError("Decode positions not initialized")
        if self._decode_schedule_count != self.decode_tokens:
            raise RuntimeError("Decode schedule not initialized")

        default_stream = torch.cuda.current_stream(device=self.device)
        with torch.inference_mode():
            with self._nvtx_range("flex_prefill"):
                prefill_start, prefill_end = self._prefill_events
                prefill_start.record(default_stream)
                _ = self.model.prefill(self.prefill_tokens)
                prefill_end.record(default_stream)

            with self._nvtx_range("flex_decode_graph"):
                for position, start_evt, end_evt in self._decode_schedule:
                    start_evt.record(default_stream)
                    self.model._update_cache(self.static_decode_k, self.static_decode_v, position)
                    self.model._set_offset(position)
                    with torch.cuda.stream(self.capture_stream):
                        self.capture_stream.wait_stream(default_stream)
                        self.graph.replay()
                        end_evt.record(self.capture_stream)
                    default_stream.wait_stream(self.capture_stream)

        # Store last output for verification (graph replay writes into static_decode_out)
        self._last_output = self.static_decode_out
        self._pending_iteration_metrics = True
        return None

    def teardown(self) -> None:
        if self.model is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        super().teardown()
        self.graph = None
        self.capture_stream = None
        self.static_decode_q = None
        self.static_decode_k = None
        self.static_decode_v = None
        self.static_decode_out = None
        self.base_position = 0

def get_benchmark():
    return OptimizedFlexDecodingGraphsBenchmark()
