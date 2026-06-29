"""Direct Triton-vs-CUDA comparison runner for the grouped Top-K lab."""

from __future__ import annotations

import argparse

import torch

from labs.top_k_kernel.top_k_kernel_common import TopKKernelBenchmark


FORWARD_CASES = [
    {"seq_len": 32768, "heads": 8, "kv_heads": 1, "top_k": 16, "batch_size": 4},
    {"seq_len": 32768, "heads": 8, "kv_heads": 1, "top_k": 16, "batch_size": 8},
    {"seq_len": 32768, "heads": 8, "kv_heads": 1, "top_k": 16, "batch_size": 16},
    {"seq_len": 32768, "heads": 16, "kv_heads": 1, "top_k": 16, "batch_size": 4},
    {"seq_len": 32768, "heads": 16, "kv_heads": 1, "top_k": 16, "batch_size": 8},
    {"seq_len": 32768, "heads": 16, "kv_heads": 1, "top_k": 16, "batch_size": 16},
]

BACKWARD_CASES = [
    {"seq_len": 1024, "heads": 16, "kv_heads": 1, "top_k": 16, "batch_size": 16},
    {"seq_len": 1024, "heads": 32, "kv_heads": 2, "top_k": 16, "batch_size": 16},
    {"seq_len": 2048, "heads": 16, "kv_heads": 1, "top_k": 16, "batch_size": 16},
    {"seq_len": 2048, "heads": 32, "kv_heads": 2, "top_k": 16, "batch_size": 16},
    {"seq_len": 4096, "heads": 16, "kv_heads": 1, "top_k": 16, "batch_size": 16},
    {"seq_len": 4096, "heads": 32, "kv_heads": 2, "top_k": 16, "batch_size": 16},
    {"seq_len": 8192, "heads": 16, "kv_heads": 1, "top_k": 16, "batch_size": 16},
    {"seq_len": 8192, "heads": 32, "kv_heads": 2, "top_k": 16, "batch_size": 16},
    {"seq_len": 16384, "heads": 16, "kv_heads": 1, "top_k": 16, "batch_size": 16},
    {"seq_len": 16384, "heads": 32, "kv_heads": 2, "top_k": 16, "batch_size": 16},
    {"seq_len": 32768, "heads": 16, "kv_heads": 1, "top_k": 16, "batch_size": 16},
    {"seq_len": 32768, "heads": 32, "kv_heads": 2, "top_k": 16, "batch_size": 16},
]


def _measure_case(backend: str, mode: str, case: dict[str, int]) -> float:
    bench = TopKKernelBenchmark(backend=backend, label=f"{backend}_{mode}")
    argv = [
        "--batch-size",
        str(case["batch_size"]),
        "--heads",
        str(case["heads"]),
        "--kv-heads",
        str(case["kv_heads"]),
        "--q-len",
        str(case["seq_len"]),
        "--compressed-k-len",
        str(case["seq_len"]),
        "--head-dim",
        "128",
        "--top-k",
        str(case["top_k"]),
        "--selection-block-size",
        "64",
        "--compress-stride",
        "1",
        "--mode",
        mode,
        "--dtype",
        "fp16",
    ]
    bench.apply_target_overrides(argv)
    bench.setup()
    warmup = 1
    iters = 3 if mode == "forward" else 1
    try:
        for _ in range(warmup):
            bench.benchmark_fn()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream()
        start.record(current_stream)
        for _ in range(iters):
            bench.benchmark_fn()
        end.record(current_stream)
        end.synchronize()
        return start.elapsed_time(end) / iters
    finally:
        bench.teardown()


def _print_table(title: str, mode: str, cases: list[dict[str, int]]) -> None:
    print(f"\n{title}")
    print("| seq_len | gqa | kv_heads | batch | Triton (ms) | CUDA (ms) | CUDA speedup vs Triton |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for case in cases:
        triton_ms = _measure_case("triton", mode, case)
        cuda_ms = _measure_case("cuda", mode, case)
        gqa = case["heads"] // case["kv_heads"]
        print(
            "| "
            f"{case['seq_len']} | {gqa} | {case['kv_heads']} | {case['batch_size']} | "
            f"{triton_ms:.2f} | {cuda_ms:.2f} | {triton_ms / cuda_ms:.2f}x |"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--mode",
        choices=("all", "forward", "fwd_bwd"),
        default="all",
        help="Subset of the slide-style matrix to run.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on how many cases to run from each selected matrix.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.mode in {"all", "forward"}:
        _print_table(
            "Forward Matrix",
            "forward",
            FORWARD_CASES if args.limit is None else FORWARD_CASES[: args.limit],
        )
    if args.mode in {"all", "fwd_bwd"}:
        _print_table(
            "Backward Matrix",
            "fwd_bwd",
            BACKWARD_CASES if args.limit is None else BACKWARD_CASES[: args.limit],
        )


if __name__ == "__main__":
    main()
