"""Baseline for warp-specialized Triton decode: eager PyTorch math.

This baseline matches `optimized_decode_warp_specialized.py` exactly (same
prompt/decode lengths, hidden size, and host/stream settings) but uses the
standard PyTorch decode path. This keeps the Triton warp-specialized comparison
workload-equivalent.
"""

from __future__ import annotations

import torch

from labs.decode_optimization.decode_common import DecodeBenchmark, DecodeConfig, attach_benchmark_metadata


class PersistentPrefillBaselineBenchmark(DecodeBenchmark):
    """Baseline that matches the optimized persistent-prefill execution model."""

    def __init__(self, cfg: DecodeConfig) -> None:
        super().__init__(cfg)
        self._prefilled_state: torch.Tensor | None = None
        self._prefilled_tokens: torch.Tensor | None = None

    def setup(self) -> None:
        super().setup()
        # Match optimized path: do prefill ONCE and benchmark decode-only iterations.
        self._copy_prompts_to_device()
        with torch.inference_mode(), self.sdpa_ctx_factory():
            prefill_state = self.prefill_fn(self.gpu_prompt)
        self._prefilled_state = prefill_state.detach().clone()
        self._prefilled_tokens = self.gpu_prompt[:, -1].detach().clone()
        self.register_workload_metadata(
            requests_per_iteration=float(self.cfg.batch_size),
            tokens_per_iteration=float(self.cfg.batch_size * self.cfg.decode_tokens),
        )

    def benchmark_fn(self) -> None:
        if self._prefilled_state is None or self._prefilled_tokens is None:
            raise RuntimeError("setup() must run before benchmark_fn()")

        current_stream = torch.cuda.current_stream()
        stream = self.compute_stream or current_stream
        with torch.cuda.stream(stream), torch.inference_mode(), self.sdpa_ctx_factory():
            # Reset on the same stream that consumes the state to keep ordering explicit.
            self.state_buffer.copy_(self._prefilled_state)
            self.current_tokens.copy_(self._prefilled_tokens)
            for _ in self._decode_step_range:
                next_state, next_token = self.decode_fn(self.current_tokens, self.state_buffer)
                self.state_buffer.copy_(next_state)
                self.current_tokens.copy_(next_token)
        if self.compute_stream is not None:
            current_stream.wait_stream(stream)


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
        use_cuda_graphs=False,
        graph_full_iteration=False,
        use_torch_compile=False,
        label="baseline_decode_warp_specialized",
    )
    return attach_benchmark_metadata(PersistentPrefillBaselineBenchmark(cfg), __file__)
