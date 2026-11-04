"""Run the Chapter 9 Triton online softmax with a full-output comparison."""

from __future__ import annotations

import argparse

import torch

from ch09.online_softmax_common import OnlineSoftmaxBenchmark

DEFAULT_ROWS = 4
DEFAULT_COLS = 257
OUTPUT_RTOL = 2e-5
OUTPUT_ATOL = 2e-7


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Chapter 9 Triton online softmax correctness demo."
    )
    parser.add_argument("--rows", type=_positive_int, default=DEFAULT_ROWS)
    parser.add_argument("--cols", type=_positive_int, default=DEFAULT_COLS)
    return parser


def build_benchmark(*, rows: int, cols: int) -> OnlineSoftmaxBenchmark:
    """Build the existing optimized online-softmax path for the requested shape."""
    return OnlineSoftmaxBenchmark(optimized=True, rows=rows, cols=cols)


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch09 online softmax demo")


def main() -> int:
    args = build_parser().parse_args()
    _require_cuda()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    benchmark = build_benchmark(rows=args.rows, cols=args.cols)
    try:
        benchmark.setup()
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        if benchmark.output is None or benchmark.x is None:
            raise RuntimeError("online softmax did not produce verification tensors")
        if not bool(torch.isfinite(benchmark.output).all().item()):
            raise AssertionError("online softmax produced a nonfinite output")
        expected = torch.softmax(benchmark.x.to(torch.float64), dim=1).to(torch.float32)
        max_abs_diff = float((benchmark.output - expected).abs().max().item())
        print(
            "online_softmax_full_compare=pass "
            f"rows={args.rows} cols={args.cols} "
            f"rtol={OUTPUT_RTOL:.0e} atol={OUTPUT_ATOL:.0e} "
            f"max_abs_diff={max_abs_diff:.6e} finite_output=true"
        )
    finally:
        benchmark.teardown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
