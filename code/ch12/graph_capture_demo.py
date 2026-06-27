"""graph_capture_demo.py - Chapter 12 CUDA graph capture demo (tool).

Captures a steady-state PyTorch workload into a CUDA graph and replays it.
This is a standalone tool script (not a baseline_/optimized_ benchmark pair).
"""

from __future__ import annotations

import argparse

import torch


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Chapter 12 CUDA graph demo.")
    return torch.device("cuda")


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 12 CUDA graph capture demo")
    parser.add_argument("--batch", type=int, default=256, help="Batch size.")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden dimension.")
    parser.add_argument("--layers", type=int, default=2, help="Number of Linear+GELU blocks.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations before capture.")
    parser.add_argument("--replays", type=int, default=50, help="Number of graph replays.")
    args = parser.parse_args()

    device = _require_cuda()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    layers = []
    for _ in range(int(args.layers)):
        layers.append(torch.nn.Linear(int(args.hidden), int(args.hidden), bias=False, device=device, dtype=torch.float16))
        layers.append(torch.nn.GELU())
    model = torch.nn.Sequential(*layers).eval()

    static_inp = torch.randn(int(args.batch), int(args.hidden), device=device, dtype=torch.float16)

    with torch.inference_mode():
        eager_out = model(static_inp)
        for _ in range(int(args.warmup)):
            _ = model(static_inp)
        torch.cuda.synchronize(device)

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            static_out = model(static_inp)

        for _ in range(int(args.replays)):
            g.replay()
        torch.cuda.synchronize(device)

    max_diff = (static_out.float() - eager_out.float()).abs().max().item()
    print(f"max_abs_diff_vs_eager={max_diff:.6e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
