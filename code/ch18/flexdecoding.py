from __future__ import annotations

# Environment (Oct-2025): CUDA 13.x r580+, torch 2.10.0+cu130, optional TE 2.8+
"""FlexDecoding showcase aligned with PyTorch 2.10 (CUDA 13.0)."""

import os

try:
    pass
except Exception:
    class arch_config:  # type: ignore[override]
        @staticmethod
        def is_blackwell() -> bool:
            import torch
            if not torch.cuda.is_available():
                return False
            major, minor = torch.cuda.get_device_capability()
            # Blackwell is sm_100 (major=10), Grace-Blackwell is sm_12x
            return major >= 10


import math
import time
from dataclasses import dataclass
from typing import Iterable, List

import torch

from core.utils.compile_utils import compile_callable

try:
    from torch.nn.attention import flex_attention
    HAS_FLEX = True
except (ImportError, AttributeError):
    HAS_FLEX = False

QUICK_MODE = any(
    os.getenv(flag, "0") == "1"
    for flag in ("QUICK_PROFILE", "BENCHMARK_QUICK", "RUN_ALL_CHAPTERS")
)
DEFAULT_COMPILE_MODE = "reduce-overhead" if QUICK_MODE else "default"
COMPILE_MODE = os.getenv("TORCH_COMPILE_MODE", DEFAULT_COMPILE_MODE)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _benchmark(label: str, fn, iters: int) -> float:
    _sync()
    if QUICK_MODE:
        iters = min(iters, 5)
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream()
        start.record(current_stream)
        with torch.inference_mode():
            for _ in range(iters):
                fn()
        end.record(current_stream)
        end.synchronize()
        elapsed = start.elapsed_time(end) / iters
    else:
        start = time.perf_counter()
        with torch.inference_mode():
            for _ in range(iters):
                fn()
        elapsed = (time.perf_counter() - start) * 1_000 / iters
    print(f"{label:<28}: {elapsed:7.3f} ms")
    return elapsed


def _score_mod_causal(window_size: int):
    """Causal mask with a sliding window."""
    def score_mod(score, _b, _h, q_idx, kv_idx):
        delta = q_idx - kv_idx
        in_window = (delta >= 0) & (delta <= window_size)
        return torch.where(in_window, score, score - 1e9)

    return score_mod


def _score_mod_causal_with_offset(window_size: int, offset: torch.Tensor):
    """Causal mask with a sliding window and captured offset."""
    def score_mod(score, _b, _h, q_idx, kv_idx):
        delta = q_idx + offset - kv_idx
        in_window = (delta >= 0) & (delta <= window_size)
        return torch.where(in_window, score, score - 1e9)

    return score_mod


@dataclass
class FlexDecodingConfig:
    dim: int = 512
    heads: int = 8
    max_seq_len: int = 16384
    window: int = 512
    dtype: torch.dtype = torch.bfloat16


class FlexDecodingModule(torch.nn.Module):
    def __init__(
        self,
        cfg: FlexDecodingConfig,
        *,
        use_flex_attention: bool = True,
        compile_enabled: bool = True,
    ):
        super().__init__()
        assert cfg.dim % cfg.heads == 0
        self.cfg = cfg
        self.head_dim = cfg.dim // cfg.heads

        self.q_proj = torch.nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k_proj = torch.nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.v_proj = torch.nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.o_proj = torch.nn.Linear(cfg.dim, cfg.dim, bias=False)

        self.register_buffer(
            "k_cache",
            torch.zeros(1, cfg.max_seq_len, cfg.heads, self.head_dim, dtype=cfg.dtype),
        )
        self.register_buffer(
            "v_cache",
            torch.zeros(1, cfg.max_seq_len, cfg.heads, self.head_dim, dtype=cfg.dtype),
        )
        self.register_buffer("offset", torch.zeros((), dtype=torch.long))

        self.prefill_impl = None
        self.decode_impl = None
        self.use_flex_attention = bool(use_flex_attention) and HAS_FLEX
        self.flex_enabled = self.use_flex_attention
        self.compile_enabled = compile_enabled
        assert torch.cuda.is_available(), "CUDA required for FlexDecoding demo"
        major, minor = torch.cuda.get_device_capability()
        # Blackwell is sm_100 (major=10), not sm_120
        if major < 10:
            raise RuntimeError(f"SKIPPED: FlexDecoding targets Blackwell (sm_100+); found sm_{major}{minor}")
        self.compile_mode = COMPILE_MODE
        if self.use_flex_attention and not QUICK_MODE:
            self.compile_mode = "max-autotune"

    def _compile(self, pattern: str = "causal") -> None:
        device = self.k_cache.device
        dtype = self.k_cache.dtype
        head_dim = self.head_dim
        heads = self.cfg.heads

        window = self.cfg.window
        prefill_len = window * 2
        max_kv_len = self.cfg.max_seq_len
        q_prefill = torch.zeros(1, heads, prefill_len, head_dim, device=device, dtype=dtype)
        kv_prefill = torch.zeros_like(q_prefill)
        q_decode = torch.zeros(1, heads, 1, head_dim, device=device, dtype=dtype)
        kv_decode = torch.zeros(1, heads, max_kv_len, head_dim, device=device, dtype=dtype)
        compile_kwargs = {"mode": self.compile_mode, "dynamic": None}

        def configure_sdpa_backend() -> None:
            position_index = torch.arange(max_kv_len, device=device)
            q_position_column = position_index.view(max_kv_len, 1)
            k_position_row = position_index.view(1, max_kv_len)

            def sdpa(q, k, v, offset):
                # Inputs expected as [batch, seq, heads, head_dim]
                qh = q.transpose(1, 2)
                kh = k.transpose(1, 2)
                vh = v.transpose(1, 2)
                scale = 1.0 / math.sqrt(float(qh.size(-1)))
                attn = torch.matmul(qh, kh.transpose(-2, -1)) * scale
                seq_q = attn.size(-2)
                seq_k = attn.size(-1)
                q_positions = q_position_column[:seq_q] + offset
                k_positions = k_position_row[:, :seq_k]
                delta = q_positions - k_positions
                in_window = (delta >= 0) & (delta <= window)
                attn.masked_fill_(~in_window, -1e9)
                probs = torch.softmax(attn, dim=-1)
                out = torch.matmul(probs, vh)
                return out.transpose(1, 2)

            offset_zero = torch.zeros((), device=device, dtype=torch.long)

            def sdpa_prefill(q, k, v):
                return sdpa(q, k, v, offset_zero)

            def sdpa_decode(q, k, v, offset):
                return sdpa(q, k, v, offset)

            if self.compile_enabled:
                compiled_prefill = compile_callable(sdpa_prefill, **compile_kwargs)
                compiled_decode = compile_callable(sdpa_decode, **compile_kwargs)
            else:
                compiled_prefill = sdpa_prefill
                compiled_decode = sdpa_decode

            q_prefill_eager = q_prefill.transpose(1, 2)
            kv_prefill_eager = kv_prefill.transpose(1, 2)
            q_decode_eager = q_decode.transpose(1, 2)
            kv_decode_eager = kv_decode.transpose(1, 2)
            compiled_prefill(q_prefill_eager, kv_prefill_eager, kv_prefill_eager)
            compiled_decode(q_decode_eager, kv_decode_eager, kv_decode_eager, offset_zero)

            self.prefill_impl = compiled_prefill
            self.decode_impl = compiled_decode
            self.flex_enabled = False

        if self.use_flex_attention:
            score_mod_prefill = _score_mod_causal(window)
            score_mod_decode = _score_mod_causal_with_offset(window, self.offset)
            def prefill(q, k, v):
                return flex_attention.flex_attention(
                    q, k, v, score_mod=score_mod_prefill
                )

            def decode(q, k, v):
                return flex_attention.flex_attention(
                    q, k, v, score_mod=score_mod_decode
                )

            if self.compile_enabled:
                self.prefill_impl = compile_callable(prefill, **compile_kwargs)
                self.decode_impl = compile_callable(decode, **compile_kwargs)
            else:
                self.prefill_impl = prefill
                self.decode_impl = decode

            self.prefill_impl(q_prefill, kv_prefill, kv_prefill)
            self.decode_impl(q_decode, kv_decode, kv_decode)
            self.flex_enabled = True
        else:
            configure_sdpa_backend()

    def ensure_compiled(self) -> None:
        if self.prefill_impl is None or self.decode_impl is None:
            self._compile()

    def clear_cache(self, batch: int) -> None:
        if self.k_cache.shape[0] != batch:
            device = self.k_cache.device
            dtype = self.k_cache.dtype
            self.k_cache = torch.zeros(
                batch,
                self.cfg.max_seq_len,
                self.cfg.heads,
                self.head_dim,
                device=device,
                dtype=dtype,
            )
            self.v_cache = torch.zeros_like(self.k_cache)
        else:
            self.k_cache.zero_()
            self.v_cache.zero_()

    def prefill(self, tokens: torch.Tensor, past: int = 0) -> torch.Tensor:
        batch, seqlen, _ = tokens.shape
        self.ensure_compiled()
        if self.k_cache.shape[0] != batch:
            self.clear_cache(batch)
        q = self.q_proj(tokens).view(batch, seqlen, self.cfg.heads, self.head_dim)
        k = self.k_proj(tokens).view(batch, seqlen, self.cfg.heads, self.head_dim)
        v = self.v_proj(tokens).view(batch, seqlen, self.cfg.heads, self.head_dim)

        self.k_cache[:, past:past + seqlen] = k
        self.v_cache[:, past:past + seqlen] = v

        if self.flex_enabled:
            out = self.prefill_impl(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)).transpose(1, 2)
        else:
            out = self.prefill_impl(q, k, v)
        return self.o_proj(out.reshape(batch, seqlen, self.cfg.dim))

    def _set_offset(self, position: int) -> None:
        self.offset.fill_(int(position))

    def _project_token(self, token: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, _, _ = token.shape
        q = self.q_proj(token).view(batch, 1, self.cfg.heads, self.head_dim)
        k = self.k_proj(token).view(batch, 1, self.cfg.heads, self.head_dim)
        v = self.v_proj(token).view(batch, 1, self.cfg.heads, self.head_dim)
        return q, k, v

    def _update_cache(self, k: torch.Tensor, v: torch.Tensor, position: int) -> None:
        self.k_cache[:, position:position + 1].copy_(k)
        self.v_cache[:, position:position + 1].copy_(v)

    def decode_attention(self, q: torch.Tensor) -> torch.Tensor:
        batch = q.shape[0]
        if self.flex_enabled:
            out = self.decode_impl(
                q.transpose(1, 2),
                self.k_cache.transpose(1, 2),
                self.v_cache.transpose(1, 2),
            ).transpose(1, 2)
        else:
            out = self.decode_impl(q, self.k_cache, self.v_cache, self.offset)
        return self.o_proj(out.reshape(batch, 1, self.cfg.dim))

    def decode(self, token: torch.Tensor, position: int) -> torch.Tensor:
        self.ensure_compiled()
        q, k, v = self._project_token(token)
        self._update_cache(k, v, position)
        self._set_offset(position)
        return self.decode_attention(q)


def jagged_batch(model: FlexDecodingModule, sequences: Iterable[torch.Tensor]) -> List[torch.Tensor]:
    """Simple jagged-batching utility for variable-length sequences."""
    outputs: List[torch.Tensor] = []
    past = 0
    for seq in sequences:
        out = model.prefill(seq, past=past)
        outputs.append(out)
        past += seq.size(1)
    return outputs


def benchmark(model: FlexDecodingModule) -> None:
    cfg = model.cfg
    device = next(model.parameters()).device

    prompt = torch.randn(1, cfg.window * 2, cfg.dim, device=device, dtype=cfg.dtype)
    token = torch.randn(1, 1, cfg.dim, device=device, dtype=cfg.dtype)

    print("\\nBenchmarking flexdecode (prefill/decode)")
    _benchmark("prefill", lambda: model.prefill(prompt), iters=10)
    _benchmark("decode", lambda: model.decode(token, cfg.window * 2), iters=50)


def paged_attention_demo() -> None:
    print("\\nPagedAttention-style block mapping")
    batch, heads, head_dim = 2, 4, 32
    block, blocks = 8, 16
    logical = torch.randn(batch, blocks * block, heads, head_dim)
    physical = torch.zeros(blocks, block, heads, head_dim)
    table = torch.randint(0, blocks, (batch, blocks))
    for b in range(batch):
        for blk in range(blocks):
            src = logical[b, blk * block : (blk + 1) * block]
            dst = table[b, blk]
            physical[dst].copy_(src)
    print(f"Logical {logical.shape} -> Physical {physical.shape}")


def main() -> None:
    device = _device()
    print("FlexDecoding example (PyTorch 2.10 / CUDA 13.0)")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    cfg = FlexDecodingConfig()
    model = FlexDecodingModule(cfg).to(device, dtype=cfg.dtype)
    model.ensure_compiled()

    with torch.inference_mode():
        prompt = torch.randn(1, 32, cfg.dim, device=device, dtype=cfg.dtype)
        print("\\nPrefill output shape", model.prefill(prompt).shape)
        token = torch.randn(1, 1, cfg.dim, device=device, dtype=cfg.dtype)
        print("Decode output shape", model.decode(token, 32).shape)

        sequences = [
            torch.randn(1, 16, cfg.dim, device=device, dtype=cfg.dtype),
            torch.randn(1, 32, cfg.dim, device=device, dtype=cfg.dtype),
            torch.randn(1, 64, cfg.dim, device=device, dtype=cfg.dtype),
        ]
    outs = jagged_batch(model, sequences)
    for idx, out in enumerate(outs):
        print(f"Sequence {idx}: {out.shape[1]} tokens emitted")

    if torch.cuda.is_available():
        benchmark(model)

    paged_attention_demo()

    print("\\nKey takeaways:")
    print("- torch.compile can specialize prefill vs decode paths.")
    print("- FlexAttention (when available) supplies custom score mods.")
    print("- Jagged batches illustrate variable sequence handling.")
    print("- Block remapping mirrors PagedAttention KV packing.")


if __name__ == "__main__":
    main()
