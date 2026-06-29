"""FlexAttention CuTe (FlashAttention) tool.

This is a utility script (not a baseline/optimized benchmark pair). It exists so
users can validate that the FlashAttention CuTe backend is installed and working
on systems where FlexAttention DSL bindings may be unavailable.

Run via:
  python -m cli.aisp tools flex-attention-cute -- [args...]
"""

from __future__ import annotations

import argparse

import torch

from labs.flexattention.flexattention_common import build_qkv_inputs, resolve_device

try:
    from flash_attn.cute.interface import _flash_attn_fwd
except Exception as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "SKIPPED: flash-attn with CuTe DSL support is required (pip install flash-attn)"
    ) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FlashAttention CuTe forward kernel.")
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul (mostly irrelevant for bf16/fp16).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    device = resolve_device()
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)

    q, k, v = build_qkv_inputs(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        head_dim=args.head_dim,
        dtype=dtype,
        device=device,
    )

    with torch.inference_mode():
        for _ in range(args.warmup):
            _flash_attn_fwd(q, k, v)
        torch.cuda.synchronize()

    count = max(int(args.iters), 1)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        current_stream = torch.cuda.current_stream(device)
        start.record(current_stream)
        for _ in range(count):
            out = _flash_attn_fwd(q, k, v)
        end.record(current_stream)
        end.synchronize()
    elapsed_s = start.elapsed_time(end) / 1000.0

    output_tensor = out[0] if isinstance(out, (tuple, list)) else out
    tokens = float(args.batch * args.seq_len)
    iters = float(count)
    ms_per_iter = (elapsed_s * 1e3) / max(iters, 1.0)
    tok_per_s = (tokens * iters) / max(elapsed_s, 1e-12)

    print(f"CuTe FlashAttention fwd: {ms_per_iter:.4f} ms/iter, {tok_per_s:,.0f} tokens/s")
    print(f"Output: shape={tuple(output_tensor.shape)}, dtype={output_tensor.dtype}, device={output_tensor.device}")


if __name__ == "__main__":
    main()
