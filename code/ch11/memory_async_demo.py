"""memory_async_demo.py - Chapter 11 async copy + compute overlap demo (tool).

Shows how pinned host memory enables non-blocking H2D/D2H transfers that can
overlap with compute when scheduled on independent CUDA streams.
"""

from __future__ import annotations

import argparse

import torch


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Chapter 11 async memory demo.")
    return torch.device("cuda")


def _run_once(*, n_elems: int, matmul_size: int, pinned: bool) -> float:
    device = torch.device("cuda")
    copy_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    host = torch.randn(n_elems, device="cpu", dtype=torch.float32, pin_memory=pinned)
    dev = torch.empty(n_elems, device=device, dtype=torch.float32)

    a = torch.randn(matmul_size, matmul_size, device=device, dtype=torch.float16)
    b = torch.randn(matmul_size, matmul_size, device=device, dtype=torch.float16)

    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current = torch.cuda.current_stream(device)

    with torch.inference_mode():
        start.record(current)
        with torch.cuda.stream(copy_stream):
            dev.copy_(host, non_blocking=True)
        with torch.cuda.stream(compute_stream):
            _ = a @ b

        # Join both streams back onto the current stream so event timing
        # accounts for all asynchronous work.
        current.wait_stream(copy_stream)
        current.wait_stream(compute_stream)
        end.record(current)

    end.synchronize()
    return float(start.elapsed_time(end))


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 11 async memory overlap demo")
    parser.add_argument("--n-elems", type=int, default=64_000_000, help="Number of float32 elements to transfer.")
    parser.add_argument("--matmul-size", type=int, default=4096, help="Square matmul size for compute overlap.")
    args = parser.parse_args()

    _require_cuda()

    pinned_ms = _run_once(n_elems=int(args.n_elems), matmul_size=int(args.matmul_size), pinned=True)
    pageable_ms = _run_once(n_elems=int(args.n_elems), matmul_size=int(args.matmul_size), pinned=False)

    print(f"pinned_host_overlap_ms={pinned_ms:.3f}")
    print(f"pageable_host_overlap_ms={pageable_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
