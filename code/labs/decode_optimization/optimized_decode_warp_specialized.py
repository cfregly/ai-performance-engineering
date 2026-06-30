"""Optimized: decode-only CUDA Graph replay with persistent prefill state.

This variant targets the decode phase's kernel-launch overhead by capturing the
decode loop into a CUDA Graph once, then replaying it for each benchmark
iteration. The math is identical to the baseline; the speedup comes from
reducing host-side launch overhead and the resulting GPU bubbles.
"""

from __future__ import annotations

import torch

from labs.decode_optimization.decode_common import (
    DecodeBenchmark,
    DecodeConfig,
    attach_benchmark_metadata,
)


class CUDAGraphPersistentDecodeBenchmark(DecodeBenchmark):
    """Persistent prefill + CUDA Graph replay for the decode loop."""

    def setup(self) -> None:
        super().setup()

        # Prefill once and stash persistent state to amortize setup in benchmark_fn.
        self._copy_prompts_to_device()
        with torch.inference_mode(), self.sdpa_ctx_factory():
            prefill_state = self.prefill_fn(self.gpu_prompt)
        self._prefilled_state = prefill_state.detach().clone()
        self._prefilled_tokens = self.gpu_prompt[:, -1].detach().clone()

        # Decode-only benchmark: exclude prompt prefill from per-iteration workload.
        self.register_workload_metadata(
            requests_per_iteration=float(self.cfg.batch_size),
            tokens_per_iteration=float(self.cfg.batch_size * self.cfg.decode_tokens),
        )

        self._capture_decode_graph()

    def _capture_decode_graph(self) -> None:
        self._graph_stream = torch.cuda.Stream()
        self._decode_graph = torch.cuda.CUDAGraph()

        # Warm up on the capture stream so kernels, heuristics, and workspaces are ready.
        with torch.inference_mode(), self.sdpa_ctx_factory():
            with torch.cuda.stream(self._graph_stream):
                self.state_buffer.copy_(self._prefilled_state)
                self.current_tokens.copy_(self._prefilled_tokens)
                for _ in range(2):
                    for _ in range(self.cfg.decode_tokens):
                        next_state, next_token = self.decode_fn(self.current_tokens, self.state_buffer)
                        self.state_buffer.copy_(next_state)
                        self.current_tokens.copy_(next_token)
            torch.cuda.synchronize()

            # Reset to the same starting state/tokens before capture.
            self.state_buffer.copy_(self._prefilled_state)
            self.current_tokens.copy_(self._prefilled_tokens)
            torch.cuda.synchronize()

            with torch.cuda.graph(self._decode_graph, stream=self._graph_stream):
                for _ in range(self.cfg.decode_tokens):
                    next_state, next_token = self.decode_fn(self.current_tokens, self.state_buffer)
                    self.state_buffer.copy_(next_state)
                    self.current_tokens.copy_(next_token)
            torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        current_stream = torch.cuda.current_stream()
        with torch.cuda.stream(self._graph_stream):
            # Reset on the graph stream so replay consumes the intended state.
            self.state_buffer.copy_(self._prefilled_state)
            self.current_tokens.copy_(self._prefilled_tokens)
            self._decode_graph.replay()
        current_stream.wait_stream(self._graph_stream)

    def teardown(self) -> None:
        for attr in ("_decode_graph", "_graph_stream", "_prefilled_state", "_prefilled_tokens"):
            if hasattr(self, attr):
                setattr(self, attr, None)
        super().teardown()


def get_benchmark() -> DecodeBenchmark:
    cfg = DecodeConfig(
        batch_size=8,
        prompt_tokens=1024,
        decode_tokens=256,
        hidden_size=2048,
        use_fp8=False,
        use_pinned_host=True,
        use_copy_stream=True,
        use_compute_stream=True,
        use_cuda_graphs=False,  # graph handled explicitly in this benchmark
        graph_full_iteration=False,
        use_torch_compile=False,
        label="optimized_decode_warp_specialized",
    )
    return attach_benchmark_metadata(CUDAGraphPersistentDecodeBenchmark(cfg), __file__)
