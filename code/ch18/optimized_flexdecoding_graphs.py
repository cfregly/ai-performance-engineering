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
        self.static_decode_in: torch.Tensor | None = None
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
        self.static_decode_in = torch.empty_like(self.decode_token)
        self.static_decode_out = torch.empty_like(self.decode_token)
        self.capture_stream = torch.cuda.Stream(device=self.device)

        # Compile/warm outside capture to avoid lazy compile during graph capture.
        self._run_warmup()

        self.graph = torch.cuda.CUDAGraph()
        assert self.capture_stream is not None
        with torch.cuda.graph(self.graph, stream=self.capture_stream):
            if self.model is None:
                raise RuntimeError("Model not initialized for capture")
            q = self.model.q_proj(self.static_decode_in).view(
                self.static_decode_in.size(0),
                1,
                self.model.cfg.heads,
                self.model.head_dim,
            )
            out = self.model.decode_attention(q)
            self.static_decode_out.copy_(out)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[Dict[str, List[float]]]:
        if (
            self.model is None
            or self.prefill_tokens is None
            or self.decode_token is None
            or self.graph is None
            or self.capture_stream is None
            or self.static_decode_in is None
            or self.static_decode_out is None
        ):
            raise RuntimeError("Graph path not initialized")

        self.model.clear_cache(batch=self.prefill_tokens.size(0))
        if self._prefill_events is None or self._decode_events is None:
            raise RuntimeError("Timing events not initialized")
        if len(self._decode_events) != self.decode_tokens:
            raise RuntimeError("Timing event count mismatch")

        with torch.no_grad():
            with self._nvtx_range("flex_prefill"):
                prefill_start, prefill_end = self._prefill_events
                prefill_start.record()
                _ = self.model.prefill(self.prefill_tokens)
                prefill_end.record()

            with self._nvtx_range("flex_decode_graph"):
                self.static_decode_in.copy_(self.decode_token)
                heads = self.model.cfg.heads
                head_dim = self.model.head_dim
                default_stream = torch.cuda.current_stream(device=self.device)
                for pos in range(self.decode_tokens):
                    start_evt, end_evt = self._decode_events[pos]
                    start_evt.record(default_stream)
                    k = self.model.k_proj(self.decode_token).view(1, 1, heads, head_dim)
                    v = self.model.v_proj(self.decode_token).view(1, 1, heads, head_dim)
                    self.model._update_cache(k, v, self.base_position + pos)
                    self.model._set_offset(self.base_position + pos)
                    with torch.cuda.stream(self.capture_stream):
                        self.capture_stream.wait_stream(default_stream)
                        self.graph.replay()
                        end_evt.record()
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
        self.static_decode_in = None
        self.static_decode_out = None
        self.base_position = 0

def get_benchmark():
    return OptimizedFlexDecodingGraphsBenchmark()

