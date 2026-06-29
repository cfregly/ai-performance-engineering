"""event_timing_demo.py - Chapter 11 CUDA event timing demo (tool).

Demonstrates a common timing pitfall:
  - Recording CUDA events on the *wrong* stream can under-measure work that runs
    on a different stream.

This tool uses `torch.cuda._sleep()` to create work on a non-default stream
without introducing tensor dependencies that can force implicit stream syncs.
"""

from __future__ import annotations

import argparse

import torch


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Chapter 11 event timing demo.")
    return torch.device("cuda")


def main() -> int:
    parser = argparse.ArgumentParser(description="Chapter 11 CUDA event timing demo")
    parser.add_argument(
        "--sleep-cycles",
        type=int,
        default=200_000_000,
        help="Cycles passed to torch.cuda._sleep() (controls kernel duration).",
    )
    args = parser.parse_args()

    device = _require_cuda()
    torch.cuda.synchronize(device)

    stream = torch.cuda.Stream()
    default_stream = torch.cuda.current_stream(device)

    bad_start = torch.cuda.Event(enable_timing=True)
    bad_end = torch.cuda.Event(enable_timing=True)
    good_start = torch.cuda.Event(enable_timing=True)
    good_end = torch.cuda.Event(enable_timing=True)
    fixed_start = torch.cuda.Event(enable_timing=True)
    fixed_end = torch.cuda.Event(enable_timing=True)

    # WRONG: events recorded on default stream while work runs on a different stream.
    with torch.inference_mode():
        bad_start.record(default_stream)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(int(args.sleep_cycles))
        bad_end.record(default_stream)
    bad_end.synchronize()
    bad_ms = float(bad_start.elapsed_time(bad_end))
    torch.cuda.synchronize(device)

    # CORRECT: record events on the stream that actually runs the work.
    with torch.inference_mode(), torch.cuda.stream(stream):
        good_start.record(stream)
        torch.cuda._sleep(int(args.sleep_cycles))
        good_end.record(stream)
    good_end.synchronize()
    good_ms = float(good_start.elapsed_time(good_end))

    # ALSO CORRECT: join the worker stream back to default, then time on default.
    with torch.inference_mode():
        fixed_start.record(default_stream)
        with torch.cuda.stream(stream):
            torch.cuda._sleep(int(args.sleep_cycles))
        default_stream.wait_stream(stream)
        fixed_end.record(default_stream)
    fixed_end.synchronize()
    fixed_ms = float(fixed_start.elapsed_time(fixed_end))

    print(f"bad_event_timing_ms={bad_ms:.3f}")
    print(f"good_event_timing_ms={good_ms:.3f}")
    print(f"fixed_event_timing_ms={fixed_ms:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
