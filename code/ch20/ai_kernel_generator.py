#!/usr/bin/env python3
"""Chapter 20 FlexAttention demo targeting NVIDIA Blackwell (SM10x)."""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
from core.benchmark.utils import scalar_tensor_to_float
from core.utils.compile_utils import enable_tf32
try:
    from torch._dynamo import config as dynamo_config
except ImportError:  # pragma: no cover - older torch versions
    dynamo_config = None
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)

_COMPILED_FLEX = None
_DEVICE: torch.device = torch.device("cuda")
_REFERENCE_POSITION_CACHE: dict[tuple[int, int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}
_REFERENCE_CAUSAL_MASK_CACHE: dict[tuple[int, int, torch.device], torch.Tensor] = {}


def _using_cuda() -> bool:
    return _DEVICE.type == "cuda"


def _ensure_environment(device_name: str) -> None:
    try:
        import triton  # noqa: F401  # pylint: disable=unused-import
    except Exception as exc:  # pragma: no cover - import guard
        raise RuntimeError("Triton >= 3.5 required for Blackwell kernels") from exc

    global _DEVICE  # pylint: disable=global-statement
    if device_name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA device not detected; cannot run with --device cuda")
        torch.cuda.set_device(0)
        _ = torch.empty(1, device="cuda")  # ensure context created
        major, minor = torch.cuda.get_device_capability()
        if major < 10:
            print(f"Warning: expected Blackwell SM10x/SM12x GPU, detected SM{major}{minor}.")
        enable_tf32()
        _DEVICE = torch.device("cuda")
    else:
        _DEVICE = torch.device("cpu")


def _relative_position_score(score, _batch, _head, q_idx, kv_idx):
    # Scale relative position by approximately 1 / ln(2)
    return score + (q_idx - kv_idx) * 1.44269504


def _causal_mask(_batch, _head, q_idx, kv_idx):
    return q_idx >= kv_idx


def _reference_position_views(
    q_len: int,
    kv_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(q_len), int(kv_len), device)
    cached = _REFERENCE_POSITION_CACHE.get(key)
    if cached is None:
        q_pos = torch.arange(q_len, device=device).view(1, 1, q_len, 1)
        kv_pos = torch.arange(kv_len, device=device).view(1, 1, 1, kv_len)
        cached = (q_pos, kv_pos)
        _REFERENCE_POSITION_CACHE[key] = cached
    return cached


def _reference_causal_mask(q_len: int, kv_len: int, device: torch.device) -> torch.Tensor:
    key = (int(q_len), int(kv_len), device)
    mask = _REFERENCE_CAUSAL_MASK_CACHE.get(key)
    if mask is None:
        q_pos, kv_pos = _reference_position_views(q_len, kv_len, device)
        mask = kv_pos > q_pos
        _REFERENCE_CAUSAL_MASK_CACHE[key] = mask
    return mask


def _reference_attention(q, k, v, scale, causal):
    q_len = q.shape[2]
    kv_len = k.shape[2]
    logits = torch.einsum("bhqd,bhkd->bhqk", q, k).float()
    logits.mul_(scale)
    q_pos, kv_pos = _reference_position_views(q_len, kv_len, q.device)
    logits.add_(q_pos, alpha=1.44269504)
    logits.add_(kv_pos, alpha=-1.44269504)
    if causal:
        mask = _reference_causal_mask(q_len, kv_len, q.device)
        logits.masked_fill_(mask, float("-inf"))
    probs = torch.softmax(logits, dim=-1).to(q.dtype)
    return torch.einsum("bhqk,bhkd->bhqd", probs, v)


def _compiled_flex_attention():
    global _COMPILED_FLEX  # pylint: disable=global-statement
    if _COMPILED_FLEX is None:
        if _using_cuda():
            if dynamo_config is not None and hasattr(dynamo_config, "error_on_graph_break"):
                dynamo_config.error_on_graph_break = True  # type: ignore[attr-defined]
            elif dynamo_config is not None and hasattr(dynamo_config, "raise_on_graph_break"):
                dynamo_config.raise_on_graph_break = True  # type: ignore[attr-defined]
            _COMPILED_FLEX = torch.compile(flex_attention, fullgraph=True, dynamic=True)
        else:
            _COMPILED_FLEX = flex_attention
    return _COMPILED_FLEX


def _attention_case(*, batch, heads, seqlen, head_dim, dtype, causal=True):
    q = torch.randn(batch, heads, seqlen, head_dim, device=_DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(head_dim)
    block_mask = None
    if causal:
        block_mask = create_block_mask(
            _causal_mask, batch, heads, seqlen, seqlen, device=_DEVICE  # type: ignore[arg-type]
        )
    return q, k, v, scale, block_mask


def _run_fused_attention(fused_fn, q, k, v, scale, block_mask):
    return fused_fn(
        q,
        k,
        v,
        score_mod=_relative_position_score,
        block_mask=block_mask,
        scale=scale,
    )


def run_once(*, batch, heads, seqlen, head_dim, dtype, causal=True):
    q, k, v, scale, block_mask = _attention_case(
        batch=batch,
        heads=heads,
        seqlen=seqlen,
        head_dim=head_dim,
        dtype=dtype,
        causal=causal,
    )
    fused_fn = _compiled_flex_attention()
    out_fused = _run_fused_attention(fused_fn, q, k, v, scale, block_mask)
    out_ref = _reference_attention(q, k, v, scale, causal)
    return scalar_tensor_to_float((out_fused - out_ref).abs().max())


def benchmark(*, batch, heads, seqlen, head_dim, dtype, repeat=10):
    if not _using_cuda():
        seqlen = min(seqlen, 2048)
        repeat = min(repeat, 3)
    q, k, v, scale, block_mask = _attention_case(
        batch=batch,
        heads=heads,
        seqlen=seqlen,
        head_dim=head_dim,
        dtype=dtype,
    )
    fused_fn = _compiled_flex_attention()
    for _ in range(3):
        _run_fused_attention(fused_fn, q, k, v, scale, block_mask)
    count = max(repeat, 1)
    if _using_cuda():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream(_DEVICE)
        start.record(current_stream)
        for _ in range(count):
            _run_fused_attention(fused_fn, q, k, v, scale, block_mask)
        end.record(current_stream)
        end.synchronize()
        avg_ms = start.elapsed_time(end) / count
    else:
        start_time = time.perf_counter()
        for _ in range(count):
            _run_fused_attention(fused_fn, q, k, v, scale, block_mask)
        avg_ms = (time.perf_counter() - start_time) * 1000.0 / count
    print(
        "FlexAttention fused kernel: "
        f"{avg_ms:.1f} ms (device={_DEVICE}, B={batch}, H={heads}, Q={seqlen}, "
        f"D={head_dim}, dtype={dtype})"
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="FlexAttention verifier/benchmark for Blackwell"
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=8192)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument(
        "--dtype",
        choices=["fp16", "bf16", "fp32"],
        default="bf16",
        help="Tensor dtype for the benchmark",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default=os.environ.get("AI_KERNEL_DEVICE", "cuda"),
        help="Execution device (default: cuda)",
    )
    return parser.parse_args()


def _resolve_dtype(name: str):
    return {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[name]


def main() -> None:
    args = _parse_args()
    _ensure_environment(args.device)
    dtype = _resolve_dtype(args.dtype)
    seqlen = args.seqlen
    if not _using_cuda():
        if dtype is torch.float16:
            print("FP16 not supported on CPU fallback; promoting to bfloat16.")
            dtype = torch.bfloat16
        if dtype is not torch.float32:
            print("CPU fallback uses float32 for accuracy.")
            dtype = torch.float32
        if seqlen > 2048:
            print(f"Reducing sequence length from {seqlen} to 2048 for CPU fallback.")
            seqlen = 2048
    max_err = run_once(
        batch=args.batch,
        heads=args.heads,
        seqlen=seqlen,
        head_dim=args.head_dim,
        dtype=dtype,
    )
    print(f"max |FlexAttention - reference| = {max_err:.3e}")
    benchmark(
        batch=args.batch,
        heads=args.heads,
        seqlen=seqlen,
        head_dim=args.head_dim,
        dtype=dtype,
    )


if __name__ == "__main__":
    main()
