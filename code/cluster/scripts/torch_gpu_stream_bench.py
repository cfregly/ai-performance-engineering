#!/usr/bin/env python3
"""GPU STREAM-style memory throughput microbenchmark using PyTorch CUDA ops."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import torch


@dataclass(frozen=True)
class OpSpec:
    name: str
    bytes_per_element: int
    fn: Callable[[], None]


def _parse_dtype(dtype: str) -> torch.dtype:
    d = dtype.strip().lower()
    if d == "fp32":
        return torch.float32
    if d == "fp16":
        return torch.float16
    if d == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}. Choose from fp32, fp16, bf16.")


def _time_op(op: OpSpec, numel: int, element_size: int, warmup: int, iters: int) -> Dict[str, float]:
    for _ in range(warmup):
        op.fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        op.fn()
    end.record()
    torch.cuda.synchronize()

    elapsed_ms = float(start.elapsed_time(end))
    elapsed_s = elapsed_ms / 1000.0 if elapsed_ms > 0 else 0.0
    bytes_per_iter = float(op.bytes_per_element * numel * element_size)
    total_bytes = bytes_per_iter * float(iters)
    bandwidth_gbps = (total_bytes / elapsed_s / 1e9) if elapsed_s > 0 else 0.0
    return {
        "operation": op.name,
        "time_ms": elapsed_ms,
        "bytes_per_iter": bytes_per_iter,
        "total_bytes": total_bytes,
        "bandwidth_gbps": bandwidth_gbps,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="GPU STREAM-style memory bandwidth benchmark.")
    p.add_argument("--run-id", required=True, help="Run identifier")
    p.add_argument("--label", required=True, help="Node/host label")
    p.add_argument("--device", type=int, default=0, help="CUDA device index (default: 0)")
    p.add_argument("--size-mb", type=int, default=1024, help="Vector size in MB (default: 1024)")
    p.add_argument("--iters", type=int, default=40, help="Measured iterations per op (default: 40)")
    p.add_argument("--warmup", type=int, default=10, help="Warmup iterations per op (default: 10)")
    p.add_argument("--dtype", default="fp32", help="Data type: fp32|fp16|bf16 (default: fp32)")
    p.add_argument("--output-json", required=True, help="Output JSON path")
    p.add_argument("--output-csv", required=True, help="Output CSV path")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for torch_gpu_stream_bench.py")
    if args.size_mb <= 0:
        raise SystemExit("--size-mb must be > 0")
    if args.iters <= 0:
        raise SystemExit("--iters must be > 0")
    if args.warmup < 0:
        raise SystemExit("--warmup must be >= 0")

    dtype = _parse_dtype(args.dtype)
    torch.cuda.set_device(args.device)
    device = torch.device(f"cuda:{args.device}")
    element_size = torch.finfo(dtype).bits // 8
    numel = (args.size_mb * 1024 * 1024) // element_size
    if numel <= 0:
        raise SystemExit(f"size_mb={args.size_mb} is too small for dtype={args.dtype}")

    x = torch.randn(numel, device=device, dtype=dtype)
    y = torch.empty_like(x)
    z = torch.empty_like(x)
    alpha = 0.75

    # Pre-touch allocations so timed loops capture steady-state behavior.
    y.copy_(x)
    z.copy_(x)
    torch.cuda.synchronize()

    ops: List[OpSpec] = [
        OpSpec(name="copy", bytes_per_element=2, fn=lambda: y.copy_(x)),
        OpSpec(name="scale", bytes_per_element=2, fn=lambda: torch.mul(x, alpha, out=y)),
        OpSpec(name="add", bytes_per_element=3, fn=lambda: torch.add(x, y, out=z)),
        OpSpec(name="triad", bytes_per_element=3, fn=lambda: torch.add(x, y, alpha=alpha, out=z)),
    ]

    rows = [_time_op(op, numel=numel, element_size=element_size, warmup=args.warmup, iters=args.iters) for op in ops]
    peak_bw = max((r["bandwidth_gbps"] for r in rows), default=0.0)
    triad_row = next((r for r in rows if r["operation"] == "triad"), None)
    stream_like_bw = triad_row["bandwidth_gbps"] if triad_row else peak_bw

    gpu_name = torch.cuda.get_device_name(args.device)
    payload = {
        "run_id": args.run_id,
        "label": args.label,
        "status": "ok",
        "device": args.device,
        "gpu_name": gpu_name,
        "dtype": str(dtype).replace("torch.", ""),
        "size_mb": args.size_mb,
        "numel": numel,
        "element_size_bytes": element_size,
        "iters": args.iters,
        "warmup": args.warmup,
        "peak_bandwidth_gbps": peak_bw,
        "stream_like_bandwidth_gbps": stream_like_bw,
        "operations": rows,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
    }

    out_json = Path(args.output_json)
    out_csv = Path(args.output_csv)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["operation", "bandwidth_gbps", "time_ms", "bytes_per_iter", "total_bytes"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
