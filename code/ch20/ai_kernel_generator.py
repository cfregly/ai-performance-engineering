#!/usr/bin/env python3
"""Chapter 20 FlexAttention demo targeting NVIDIA Blackwell (SM10x)."""
from __future__ import annotations

import argparse
import math
import os
import time

import torch
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


def _reference_attention(q, k, v, scale, causal):
    bsz, num_heads, q_len, head_dim = q.shape
    kv_len = k.shape[2]
    logits = torch.einsum("bhqd,bhkd->bhqk", q, k)
    logits = logits * scale
    q_pos = torch.arange(q_len, device=q.device).view(1, 1, q_len, 1)
    kv_pos = torch.arange(kv_len, device=q.device).view(1, 1, 1, kv_len)
    logits = logits + (q_pos - kv_pos) * 1.44269504
    if causal:
        mask = kv_pos > q_pos
        logits = logits.masked_fill(mask, float("-inf"))
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


def run_once(*, batch, heads, seqlen, head_dim, dtype, causal=True):
    q = torch.randn(batch, heads, seqlen, head_dim, device=_DEVICE, dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    scale = 1.0 / math.sqrt(head_dim)
    block_mask = None
    if causal:
        block_mask = create_block_mask(
            _causal_mask, batch, heads, seqlen, seqlen, device=_DEVICE  # type: ignore[arg-type]
        )
    fused_fn = _compiled_flex_attention()
    out_fused = fused_fn(
        q,
        k,
        v,
        score_mod=_relative_position_score,
        block_mask=block_mask,
        scale=scale,
    )
    out_ref = _reference_attention(q, k, v, scale, causal)
    if _using_cuda():
        torch.cuda.synchronize()
    return (out_fused - out_ref).abs().max().item()


def benchmark(*, batch, heads, seqlen, head_dim, dtype, repeat=10):
    if not _using_cuda():
        seqlen = min(seqlen, 2048)
        repeat = min(repeat, 3)
    for _ in range(3):
        run_once(
            batch=batch,
            heads=heads,
            seqlen=seqlen,
            head_dim=head_dim,
            dtype=dtype,
        )
    count = max(repeat, 1)
    if _using_cuda():
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(count):
            run_once(
                batch=batch,
                heads=heads,
                seqlen=seqlen,
                head_dim=head_dim,
                dtype=dtype,
            )
        end.record()
        torch.cuda.synchronize()
        avg_ms = start.elapsed_time(end) / count
    else:
        start_time = time.time()
        for _ in range(count):
            run_once(
                batch=batch,
                heads=heads,
                seqlen=seqlen,
                head_dim=head_dim,
                dtype=dtype,
            )
        avg_ms = (time.time() - start_time) * 1000.0 / count
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
