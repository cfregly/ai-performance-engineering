from __future__ import annotations

"""PyTorch naive vs vectorized matmul benchmark."""

import torch

M = 512
N = 512
K = 512


def benchmark(op) -> float:
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")

    # Use CUDA Events for accurate GPU timing
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()

    start_event.record(current_stream)
    op(a, b)
    end_event.record(current_stream)
    end_event.synchronize()

    return float(start_event.elapsed_time(end_event))  # Already in ms


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for ch07.matmul_pytorch")
    torch.cuda.init()
    naive_time = benchmark(lambda x, y: torch.einsum("ik,kj->ij", x, y))
    optimized_time = benchmark(lambda x, y: torch.matmul(x, y))
    print(f"naive einsum: {naive_time:.2f} ms")
    print(f"torch.matmul: {optimized_time:.2f} ms")


if __name__ == "__main__":
    main()
