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
        self._decode_base_position = 0
        self._decode_k_window_views: List[torch.Tensor] = []
        self._decode_v_window_views: List[torch.Tensor] = []
        self._decode_k_window_sdp_views: List[torch.Tensor] = []
        self._decode_v_window_sdp_views: List[torch.Tensor] = []

    def setup(self) -> None:
        super().setup()
        window = self.config.window
        if window <= 0:
            raise RuntimeError("Sliding-window size must be positive")
        if self.model is None or self.prefill_tokens is None:
            raise RuntimeError("Windowed decode setup did not initialize model/tokens")
        self._decode_base_position = self.prefill_tokens.size(1)
        self._decode_k_window_views = []
        self._decode_v_window_views = []
        self._decode_k_window_sdp_views = []
        self._decode_v_window_sdp_views = []
        for position in self._decode_positions:
            start = position - window
            if start < 0:
                raise RuntimeError("Windowed decode expects position >= window size")
            end = position + 1
            k_window = self.model.k_cache[:, start:end]
            v_window = self.model.v_cache[:, start:end]
            self._decode_k_window_views.append(k_window)
            self._decode_v_window_views.append(v_window)
            self._decode_k_window_sdp_views.append(k_window.transpose(1, 2))
            self._decode_v_window_sdp_views.append(v_window.transpose(1, 2))

    def _cache_window_views_for_position(self, position: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model is None:
            raise RuntimeError("Windowed decode not initialized")
        view_idx = position - self._decode_base_position
        if 0 <= view_idx < len(self._decode_k_window_views):
            return self._decode_k_window_sdp_views[view_idx], self._decode_v_window_sdp_views[view_idx]
        window = self.config.window
        start = position - window
        if start < 0:
            raise RuntimeError("Windowed decode expects position >= window size")
        end = position + 1
        k_slice = self.model.k_cache[:, start:end]
        v_slice = self.model.v_cache[:, start:end]
        return k_slice.transpose(1, 2), v_slice.transpose(1, 2)

    def _decode_step(self, token: torch.Tensor, position: int) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Windowed decode not initialized")
        q, k, v = self.model._project_token(token)
        return self._decode_projected_step(q, q.transpose(1, 2), k, v, position)

    def _decode_projected_step(
        self,
        q: torch.Tensor,
        q_sdp: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        position: int,
    ) -> torch.Tensor:
        if self.model is None:
            raise RuntimeError("Windowed decode not initialized")
        self.model._update_cache(k, v, position)
        self.model._set_offset(position)
        k_sdp, v_sdp = self._cache_window_views_for_position(position)
        out = F.scaled_dot_product_attention(
            q_sdp,
            k_sdp,
            v_sdp,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
        )
        return self.model.o_proj(out.transpose(1, 2).reshape(q.shape[0], 1, self.config.dim))

    def teardown(self) -> None:
        self._decode_base_position = 0
        self._decode_k_window_views = []
        self._decode_v_window_views = []
        self._decode_k_window_sdp_views = []
        self._decode_v_window_sdp_views = []
        super().teardown()

    def benchmark_fn(self) -> Optional[Dict[str, List[float]]]:
        if self.model is None or self.prefill_tokens is None or self.decode_token is None:
            raise RuntimeError("Model/tokens not initialized")
        if self._prefill_events is None or self._decode_events is None:
            raise RuntimeError("Timing events not initialized")
        if self._decode_event_count != self.decode_tokens:
            raise RuntimeError("Timing event count mismatch")
        if self._decode_position_count != self.decode_tokens:
            raise RuntimeError("Decode positions not initialized")
        if self._decode_schedule_count != self.decode_tokens:
            raise RuntimeError("Decode schedule not initialized")

        current_stream = torch.cuda.current_stream(self.device)

        with torch.inference_mode():
            with self._nvtx_range("flex_prefill"):
                prefill_start, prefill_end = self._prefill_events
                prefill_start.record(current_stream)
                prefill_out = self._prefill_step()
                prefill_end.record(current_stream)

            decode_q, decode_k, decode_v = self.model._project_token(self.decode_token)
            decode_q_sdp = decode_q.transpose(1, 2)

            with self._nvtx_range("flex_decode"):
                with sdpa_kernel(self._flash_attention_backends):
                    for position, start_evt, end_evt in self._decode_schedule:
                        start_evt.record(current_stream)
                        decode_out = self._decode_projected_step(
                            decode_q,
                            decode_q_sdp,
                            decode_k,
                            decode_v,
                            position,
                        )
                        end_evt.record(current_stream)

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
