"""stream_priority_demo.py - Chapter 11 CUDA stream priority demo (tool).

Demonstrates creating CUDA streams with different priorities and measuring how
quickly a small high-priority workload completes while the GPU is busy with a
low-priority workload.
"""

from __future__ import annotations

import argparse

import torch


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Chapter 11 stream priority demo.")
    return torch.device("cuda")


def _make_stream(priority: int) -> torch.cuda.Stream:
    try:
        return torch.cuda.Stream(priority=priority)
    except TypeError as exc:
        raise RuntimeError(
            "This PyTorch build does not support torch.cuda.Stream(priority=...). "
            "Upgrade PyTorch/CUDA to use the stream priority demo."
        ) from exc


def _run_once(
    *,
    matrix_size: int,
    low_iters: int,
    high_vector_elems: int,
    low_priority: int,
    high_priority: int,
    dtype: torch.dtype,
) -> float:
    device = torch.device("cuda")
    low_stream = _make_stream(low_priority)
    high_stream = _make_stream(high_priority)

    a = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
    b = torch.randn(matrix_size, matrix_size, device=device, dtype=dtype)
    vec = torch.randn(high_vector_elems, device=device, dtype=dtype)
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.inference_mode():
        with torch.cuda.stream(low_stream):
            for _ in range(low_iters):
                _ = a @ b

        with torch.cuda.stream(high_stream):
            start.record(high_stream)
            _ = vec.mul(1.01).add(0.1)
            end.record(high_stream)

    end.synchronize()
    return float(start.elapsed_time(end))


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 11 stream priority demo")
    parser.add_argument("--matrix-size", type=int, default=4096, help="Square matmul size for the low-priority load.")
    parser.add_argument("--low-iters", type=int, default=8, help="Number of low-priority GEMMs to enqueue.")
    parser.add_argument("--high-vector-elems", type=int, default=64_000_000, help="Vector size for high-priority op.")
    parser.add_argument("--low-priority", type=int, default=0, help="Priority for the background stream.")
    parser.add_argument("--high-priority", type=int, default=-1, help="Priority for the foreground stream.")
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16", help="Matmul dtype.")
    args = parser.parse_args()

    _require_cuda()

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16

    ms = _run_once(
        matrix_size=int(args.matrix_size),
        low_iters=int(args.low_iters),
        high_vector_elems=int(args.high_vector_elems),
        low_priority=int(args.low_priority),
        high_priority=int(args.high_priority),
        dtype=dtype,
    )
    print(
        "high_priority_op_latency_ms="
        f"{ms:.3f} (low_priority={args.low_priority}, high_priority={args.high_priority})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
