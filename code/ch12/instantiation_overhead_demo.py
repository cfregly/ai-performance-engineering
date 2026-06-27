"""instantiation_overhead_demo.py - Chapter 12 CUDA graph instantiation overhead demo (tool).

Measures:
  - graph capture/instantiation overhead (host-side + allocator work)
  - steady-state replay cost
and contrasts them with repeatedly rebuilding graphs.
"""

from __future__ import annotations

import argparse
import time

import torch


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Chapter 12 graph instantiation demo.")
    return torch.device("cuda")


def _build_model(device: torch.device, hidden: int, layers: int) -> torch.nn.Module:
    blocks = []
    for _ in range(layers):
        blocks.append(torch.nn.Linear(hidden, hidden, bias=False, device=device, dtype=torch.float16))
        blocks.append(torch.nn.GELU())
    return torch.nn.Sequential(*blocks).eval()


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 12 CUDA graph instantiation overhead demo")
    parser.add_argument("--batch", type=int, default=256, help="Batch size.")
    parser.add_argument("--hidden", type=int, default=4096, help="Hidden dimension.")
    parser.add_argument("--layers", type=int, default=2, help="Number of Linear+GELU blocks.")
    parser.add_argument("--warmup", type=int, default=25, help="Warmup iterations before capture.")
    parser.add_argument("--replay-iters", type=int, default=200, help="Replay iterations for steady-state timing.")
    parser.add_argument("--rebuild-iters", type=int, default=5, help="Number of rebuild (capture+1 replay) cycles.")
    args = parser.parse_args()

    device = _require_cuda()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = _build_model(device, int(args.hidden), int(args.layers))
    x = torch.randn(int(args.batch), int(args.hidden), device=device, dtype=torch.float16)

    with torch.inference_mode():
        for _ in range(int(args.warmup)):
            _ = model(x)
        torch.cuda.synchronize(device)

        g = torch.cuda.CUDAGraph()
        t0 = time.perf_counter()
        with torch.cuda.graph(g):
            _ = model(x)
        torch.cuda.synchronize(device)
        capture_ms = (time.perf_counter() - t0) * 1000.0

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(int(args.replay_iters)):
            g.replay()
        end.record()
        end.synchronize()
        replay_ms = float(start.elapsed_time(end)) / max(int(args.replay_iters), 1)

    # Rebuild loop (illustrates why you normally capture once).
    rebuild_total_ms = 0.0
    for _ in range(int(args.rebuild_iters)):
        g2 = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.cuda.graph(g2):
            _ = model(x)
        g2.replay()
        torch.cuda.synchronize(device)
        rebuild_total_ms += (time.perf_counter() - t0) * 1000.0
    rebuild_ms = rebuild_total_ms / max(int(args.rebuild_iters), 1)

    print(f"capture_ms={capture_ms:.3f}")
    print(f"replay_ms_per_iter={replay_ms:.6f}")
    print(f"rebuild_capture_plus_replay_ms_avg={rebuild_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
