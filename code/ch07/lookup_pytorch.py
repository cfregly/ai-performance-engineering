"""GPU gather benchmark comparing naive vs. coalesced access.

This replicates the Chapter 7 guidance with simple timing.
"""

from __future__ import annotations

import torch

N = 1 << 20
_TABLE_CACHE: dict[tuple[str, int | None], torch.Tensor] = {}


def _table_for(device: torch.device) -> torch.Tensor:
    key = (device.type, device.index)
    table = _TABLE_CACHE.get(key)
    if table is None or table.device != device:
        table = torch.arange(N, device=device, dtype=torch.float32)
        _TABLE_CACHE[key] = table
    return table


def run(
    indices: torch.Tensor,
    *,
    table: torch.Tensor | None = None,
    events: tuple[torch.cuda.Event, torch.cuda.Event] | None = None,
) -> float:
    if table is None:
        table = _table_for(indices.device)

    if indices.device.type == "cuda":
        # Use CUDA Events for accurate GPU timing
        if events is None:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
        else:
            start_event, end_event = events

        current_stream = torch.cuda.current_stream(indices.device)
        start_event.record(current_stream)
        _ = table[indices]
        end_event.record(current_stream)
        end_event.synchronize()

        return float(start_event.elapsed_time(end_event))  # Already in ms

    # CPU timing
    import time

    start = time.perf_counter()
    _ = table[indices]
    return (time.perf_counter() - start) * 1_000


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    table = torch.arange(N, device=device, dtype=torch.float32)
    events = (
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        if device.type == "cuda"
        else None
    )

    random_indices = torch.randint(0, N, (N,), device=device)
    ms = run(random_indices, table=table, events=events)
    print(f"random gather: {ms:.2f} ms")
    coalesced_indices = torch.arange(N, device=device)
    ms = run(coalesced_indices, table=table, events=events)
    print(f"coalesced gather: {ms:.2f} ms")


if __name__ == "__main__":
    main()
