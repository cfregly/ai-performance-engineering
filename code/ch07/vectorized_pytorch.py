from __future__ import annotations

"""PyTorch vectorized vs. naive additions benchmark."""

import torch

N = 1 << 20


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a = torch.arange(N, device=device, dtype=torch.float32)
    b = 2 * a
    c = torch.empty_like(a)

    if device.type == "cuda":
        # Use CUDA Events for accurate GPU timing
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream(device)

        start_event.record(current_stream)
        for i in range(N):
            c[i] = a[i] + b[i]
        end_event.record(current_stream)
        end_event.synchronize()
        sequential_ms = start_event.elapsed_time(end_event)

        start_event.record(current_stream)
        c = a + b
        end_event.record(current_stream)
        end_event.synchronize()
        vector_ms = start_event.elapsed_time(end_event)
    else:
        # CPU timing
        import time

        start = time.perf_counter()
        for i in range(N):
            c[i] = a[i] + b[i]
        sequential_ms = (time.perf_counter() - start) * 1_000

        start = time.perf_counter()
        c = a + b
        vector_ms = (time.perf_counter() - start) * 1_000

    print(f"naive loop: {sequential_ms:.2f} ms, vectorized: {vector_ms:.2f} ms")


if __name__ == "__main__":
    main()
