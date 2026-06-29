from __future__ import annotations

"""PyTorch memory access patterns benchmark."""

import torch

N = 1 << 20

def benchmark_copy(style: str) -> float:
    src = torch.arange(N, device="cuda", dtype=torch.float32)
    dst = torch.empty_like(src)

    # Use CUDA Events for accurate GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()

    start_event.record(current_stream)
    if style == "scalar":
        dst.copy_(src)
    elif style == "vectorized":
        dst.copy_(src)
    end_event.record(current_stream)
    end_event.synchronize()

    return float(start_event.elapsed_time(end_event))  # Already in ms


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ch07.memory_access_pytorch")
    torch.cuda.init()
    ms = benchmark_copy("scalar")
    print(f"scalar copy: {ms:.2f} ms")
    ms = benchmark_copy("vectorized")
    print(f"vectorized copy: {ms:.2f} ms")


if __name__ == "__main__":
    main()
