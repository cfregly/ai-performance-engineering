"""Direct matrix runner for the Blackwell grouped-GEMM optimization lab."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import torch

from core.harness.benchmark_harness import lock_gpu_clocks
from labs.blackwell_gemm_optimizations.blackwell_grouped_gemm_common import (
    BlackwellGroupedGemmWorkload,
    build_state,
    require_blackwell_grouped_gemm_support,
    run_variant,
)
from labs.blackwell_gemm_optimizations.blackwell_grouped_gemm_autotune import (
    experimental_variant_names,
    public_variant_names,
)


def _measure_variant(
    workload: BlackwellGroupedGemmWorkload,
    variant: str,
    *,
    experimental: str | None,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    device = torch.device("cuda")
    state = build_state(workload, device)
    packed = torch.empty(
        workload.num_experts * state.max_count,
        workload.hidden_dim,
        device=device,
        dtype=workload.dtype,
    )
    out = torch.empty(
        workload.num_experts,
        state.max_count,
        workload.expert_ffn_dim,
        device=device,
        dtype=workload.dtype,
    )
    for _ in range(warmup):
        run_variant(
            state,
            variant=variant,
            experimental=experimental,
            packed_tokens_flat=packed,
            output_buffer=out,
        )
    torch.cuda.synchronize(device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream(device)
    start.record(current_stream)
    for _ in range(repeats):
        run_variant(
            state,
            variant=variant,
            experimental=experimental,
            packed_tokens_flat=packed,
            output_buffer=out,
        )
    end.record(current_stream)
    end.synchronize()
    mean_ms = float(start.elapsed_time(end) / repeats)
    flops = 2.0 * workload.num_tokens * workload.hidden_dim * workload.expert_ffn_dim
    tflops = flops / (mean_ms * 1e-3) / 1e12
    return mean_ms, tflops


def _iter_variant_rows(include_experimental: bool) -> Iterable[tuple[str, str | None]]:
    for variant in public_variant_names():
        yield variant, None
    if include_experimental:
        for experimental in experimental_variant_names():
            yield "full_stack", experimental


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--token-counts",
        nargs="+",
        type=int,
        default=[1024, 2048, 4096, 8192],
        help="Token-count matrix to benchmark.",
    )
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--expert-ffn-dim", type=int, default=3072)
    parser.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    parser.add_argument("--histogram", choices=("balanced", "skewed"), default="balanced")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument(
        "--include-experimental",
        action="store_true",
        help="Also benchmark the negative-control schedules from the slide deck.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Optional CSV output path.",
    )
    args = parser.parse_args(argv)

    device = torch.device("cuda")
    require_blackwell_grouped_gemm_support(device)

    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    rows: list[tuple[str, int, float, float]] = []
    context = lock_gpu_clocks(device=0) if torch.cuda.is_available() else nullcontext()
    with context:
        for num_tokens in args.token_counts:
            workload = BlackwellGroupedGemmWorkload(
                num_tokens=num_tokens,
                num_experts=args.num_experts,
                hidden_dim=args.hidden_dim,
                expert_ffn_dim=args.expert_ffn_dim,
                dtype=dtype,
                histogram=args.histogram,
            )
            for variant, experimental in _iter_variant_rows(args.include_experimental):
                name = experimental or variant
                mean_ms, tflops = _measure_variant(
                    workload,
                    variant,
                    experimental=experimental,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                rows.append((name, num_tokens, mean_ms, tflops))

    print("| Variant | Tokens | Mean (ms) | TFLOPs/s |")
    print("| --- | ---: | ---: | ---: |")
    for name, num_tokens, mean_ms, tflops in rows:
        print(f"| `{name}` | `{num_tokens}` | `{mean_ms:.3f}` | `{tflops:.2f}` |")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        args.csv.write_text(
            "variant,num_tokens,mean_ms,tflops\n"
            + "".join(
                f"{name},{num_tokens},{mean_ms:.6f},{tflops:.6f}\n"
                for name, num_tokens, mean_ms, tflops in rows
            ),
            encoding="utf-8",
        )
        print(f"\nWrote {args.csv}")
    return 0


if __name__ == "__main__":  # pragma: no cover - manual runner
    raise SystemExit(main())
