"""Optimized FlexDecoding benchmark using compiled FlexAttention on sliding-window slices."""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from ch18.baseline_flexdecoding import FlexDecodingHarness  # noqa: E402


class OptimizedFlexDecodingBenchmark(FlexDecodingHarness):
    """Optimized path: compiled FlexAttention with sliding-window cache slicing."""

    story_metadata = {
        "pair_role": "canonical",
        "variant_role": "optimized",
        "chapter_alignment": "native",
        "chapter_native_exemplar": True,
        "comparison_axis": "full_kv_mask_vs_windowed_kv_slice",
        "execution_pattern": "windowed_kv_slice_decode",
        "comparison_reason": (
            "This chapter-native FlexDecoding path reduces decode work by slicing the "
            "KV cache to the active sliding window before attention."
        ),
        "optimization_mechanism": (
            "slice the KV cache to the active decode window and run the decode step "
            "through the compiled FlashAttention-backed path"
        ),
    }

    def __init__(self) -> None:
        super().__init__(
            use_flex_attention=True,
            require_flex=True,
            decode_tokens=512,
            compile_enabled=True,
        )
        self._flash_attention_backends = [SDPBackend.FLASH_ATTENTION]

    def setup(self) -> None:
        super().setup()
        window = self.config.window
        if window <= 0:
            raise RuntimeError("Sliding-window size must be positive")

    def _decode_step(self, token: torch.Tensor, position: int) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Windowed decode not initialized")
        window = self.config.window
        start = position - window
        if start < 0:
            raise RuntimeError("Windowed decode expects position >= window size")
        end = position + 1
        q, k, v = self.model._project_token(token)
        self.model._update_cache(k, v, position)
        self.model._set_offset(position)
        k_slice = self.model.k_cache[:, start:end]
        v_slice = self.model.v_cache[:, start:end]
        out = F.scaled_dot_product_attention(
            q.transpose(1, 2),
            k_slice.transpose(1, 2),
            v_slice.transpose(1, 2),
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        return self.model.o_proj(out.transpose(1, 2).reshape(token.shape[0], 1, self.config.dim))

    def benchmark_fn(self) -> Optional[Dict[str, List[float]]]:
        if self.model is None or self.prefill_tokens is None or self.decode_token is None:
            raise RuntimeError("Model/tokens not initialized")
        if self._prefill_events is None or self._decode_events is None:
            raise RuntimeError("Timing events not initialized")
        if len(self._decode_events) != self.decode_tokens:
            raise RuntimeError("Timing event count mismatch")

        base_position = self.prefill_tokens.size(1)

        with torch.inference_mode():
            with self._nvtx_range("flex_prefill"):
                prefill_start, prefill_end = self._prefill_events
                prefill_start.record()
                prefill_out = self._prefill_step()
                prefill_end.record()

            with self._nvtx_range("flex_decode"):
                with sdpa_kernel(self._flash_attention_backends):
                    for pos in range(self.decode_tokens):
                        start_evt, end_evt = self._decode_events[pos]
                        start_evt.record()
                        decode_out = self._decode_step(self.decode_token, base_position + pos)
                        end_evt.record()

        self._last_output = decode_out if "decode_out" in locals() else prefill_out
        self._pending_iteration_metrics = True
        if self._last_output is None or self.prefill_tokens is None or self.decode_token is None:
            raise RuntimeError("benchmark_fn() must produce output")
        return None

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        metrics = super().get_custom_metrics()
        if metrics is None:
            return None
        metrics.update(
            {
                "flexdecode.decode_kv_span_tokens": float(self.config.window + 1),
                "flexdecode.active_window_tokens": float(self.config.window + 1),
                "flexdecode.window_slice_decode": 1.0,
            }
        )
        return metrics


def get_benchmark():
    return OptimizedFlexDecodingBenchmark()
