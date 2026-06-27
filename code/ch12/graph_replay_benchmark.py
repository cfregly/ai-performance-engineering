"""graph_replay_benchmark.py - Chapter 12 CUDA graph replay benchmark (tool).

Measures eager launch overhead vs CUDA graph replay for a fixed-shape workload.
This is a diagnostic tool (not a baseline_/optimized_ benchmark pair).
"""

from __future__ import annotations

import argparse

import torch


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Chapter 12 CUDA graph replay tool.")
    return torch.device("cuda")


def _time_eager(model: torch.nn.Module, x: torch.Tensor, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    with torch.inference_mode():
        start.record()
        for _ in range(iters):
            _ = model(x)
        end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / max(iters, 1)


def _time_replay(g: torch.cuda.CUDAGraph, iters: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        g.replay()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / max(iters, 1)


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 12 CUDA graph replay benchmark tool")
    parser.add_argument("--batch", type=int, default=256, help="Batch size.")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden dimension.")
    parser.add_argument("--layers", type=int, default=2, help="Number of Linear+GELU blocks.")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=200, help="Timed iterations for eager and replay.")
    args = parser.parse_args()

    device = _require_cuda()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    layers = []
    for _ in range(int(args.layers)):
        layers.append(torch.nn.Linear(int(args.hidden), int(args.hidden), bias=False, device=device, dtype=torch.float16))
        layers.append(torch.nn.GELU())
    model = torch.nn.Sequential(*layers).eval()

    x = torch.randn(int(args.batch), int(args.hidden), device=device, dtype=torch.float16)

    with torch.inference_mode():
        for _ in range(int(args.warmup)):
            _ = model(x)
        torch.cuda.synchronize(device)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            _ = model(x)
        torch.cuda.synchronize(device)

        eager_ms = _time_eager(model, x, int(args.iters))
        replay_ms = _time_replay(g, int(args.iters))

    speedup = (eager_ms / replay_ms) if replay_ms > 0 else float("inf")
    print(f"eager_ms_per_iter={eager_ms:.6f}")
    print(f"graph_replay_ms_per_iter={replay_ms:.6f}")
    print(f"speedup={speedup:.2f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
