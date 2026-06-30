"""TFLOPs-oriented microbenchmark for apples-to-apples FlashAttention-4 comparisons."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from core.benchmark.utils import calculate_tflops, scalar_tensor_to_float
from core.harness.benchmark_harness import lock_gpu_clocks
from labs.flashattention4.flashattention4_common import (
    FlashAttention4Config,
    build_flashattention4_mode_table_payload,
    build_reference_inputs,
    compile_flashattention4_provider,
    eager_flex_attention,
    estimate_attention_forward_flops,
    resolve_cuda_device,
    resolve_flashattention4_mode_decision,
)

COLFAX_SOURCE_URL = "https://research.colfax-intl.com/flashattention-4-algorithm-and-kernel-pipelining-co-design-for-asymmetric-hardware-scaling/"
PYTORCH_SOURCE_URL = "https://pytorch.org/blog/flexattention-flashattention-4-fast-and-flexible/"
COLFAX_B200_BF16_ACHIEVED_TFLOPS = 1605.0
COLFAX_FORWARD_SPEEDUP_VS_CUDNN = (1.1, 1.3)
COLFAX_FORWARD_SPEEDUP_VS_TRITON = (2.1, 2.7)
PYTORCH_STANDARD_FORWARD_SPEEDUP_VS_TRITON = (1.6, 3.2)
PYTORCH_FLEX_FORWARD_SPEEDUP_VS_TRITON = {
    "alibi": (1.2, 2.1),
    "windowed": (1.4, 2.1),
    "alibi_windowed": (1.4, 2.1),
}

PRESETS = {
    "public_blog": {"batch": 2, "heads": 8, "seq_len": 2048, "head_dim": 128},
    "peak_probe": {"batch": 8, "heads": 16, "seq_len": 4096, "head_dim": 128},
}


@dataclass(frozen=True)
class BenchBackend:
    name: str
    display_name: str
    kind: str
    provider: Optional[str] = None
    sdp_backend: Optional[SDPBackend] = None


@dataclass(frozen=True)
class TimingStats:
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


@dataclass(frozen=True)
class MicrobenchResult:
    mode: str
    backend: str
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float
    tflops: float
    pct_of_colfax_1605: float
    speedup_vs_triton_flex: Optional[float]
    speedup_vs_cudnn_sdpa: Optional[float]
    verification_max_diff: Optional[float]


def _timing_stats_from_samples(times_ms: list[float]) -> TimingStats:
    if not times_ms:
        raise ValueError("times_ms must contain at least one sample")

    count = len(times_ms)
    total = 0.0
    total_sq = 0.0
    min_ms = float("inf")
    max_ms = float("-inf")
    for value in times_ms:
        total += value
        total_sq += value * value
        min_ms = min(min_ms, value)
        max_ms = max(max_ms, value)

    times_ms.sort()
    midpoint = count // 2
    if count % 2:
        median_ms = times_ms[midpoint]
    else:
        median_ms = (times_ms[midpoint - 1] + times_ms[midpoint]) / 2.0

    mean_ms = total / count
    if count > 1:
        variance = (total_sq - (total * total / count)) / (count - 1)
        std_ms = variance**0.5 if variance > 0.0 else 0.0
    else:
        std_ms = 0.0

    return TimingStats(
        mean_ms=mean_ms,
        median_ms=median_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        std_ms=std_ms,
    )


BACKENDS = {
    "flash_backend": BenchBackend("flash_backend", "flash_backend", "flex", provider="flash_backend"),
    "triton_flex": BenchBackend("triton_flex", "triton_flex", "flex", provider="flex_compiled"),
    "flex_tma": BenchBackend("flex_tma", "flex_tma", "flex", provider="flex_tma"),
    "cudnn_sdpa": BenchBackend("cudnn_sdpa", "cudnn_sdpa", "sdpa", sdp_backend=SDPBackend.CUDNN_ATTENTION),
}
BACKEND_NOTES = {
    "flash_backend": "PyTorch FlexAttention with kernel_options BACKEND=FLASH.",
    "triton_flex": "Compiled FlexAttention with USE_TMA=False; closest local proxy for the blog's Triton baseline.",
    "flex_tma": "Compiled FlexAttention with USE_TMA=True and FORCE_USE_FLEX_ATTENTION.",
    "cudnn_sdpa": "PyTorch SDPA forced to cuDNN attention backend.",
}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark FA4-style attention backends in TFLOPs/s.")
    parser.add_argument("--preset", choices=tuple(PRESETS) + ("custom",), default="public_blog")
    parser.add_argument("--mode", nargs="+", default=None, help="Modes to benchmark (default: dense causal).")
    parser.add_argument("--backends", nargs="+", default=["flash_backend", "triton_flex", "flex_tma", "cudnn_sdpa"])
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--window-size", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--compile-mode", default="max-autotune")
    parser.add_argument(
        "--no-lock-gpu-clocks",
        dest="lock_gpu_clocks",
        action="store_false",
        help="Skip harness clock locking for local experimentation.",
    )
    parser.add_argument("--sm-clock-mhz", type=int, default=None, help="Optional SM application clock.")
    parser.add_argument("--mem-clock-mhz", type=int, default=None, help="Optional memory application clock.")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--json-out", type=Path, default=None)
    parser.set_defaults(lock_gpu_clocks=True)
    return parser


def _apply_preset(args: argparse.Namespace) -> None:
    if args.preset == "custom":
        return
    for key, value in PRESETS[args.preset].items():
        setattr(args, key.replace("-", "_"), value)


def _build_flex_config(args: argparse.Namespace, mode: str, *, batch: int, heads: int, seq_len: int) -> FlashAttention4Config:
    return FlashAttention4Config(
        batch=batch,
        heads=heads,
        seq_len=seq_len,
        head_dim=args.head_dim,
        block_size=args.block_size,
        window_size=args.window_size,
        dtype=torch.bfloat16,
        mode=mode,
        backend="auto",
        compile_mode=args.compile_mode,
    )


def _build_sdpa_callable(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    is_causal: bool,
    backend: SDPBackend,
) -> Callable[[], torch.Tensor]:
    def run() -> torch.Tensor:
        with torch.inference_mode(), sdpa_kernel(backends=[backend]):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=is_causal)

    return run


def _build_flex_callable(
    args: argparse.Namespace,
    mode: str,
    backend: BenchBackend,
    *,
    batch: int,
    heads: int,
    seq_len: int,
) -> tuple[Callable[[], torch.Tensor], FlashAttention4Config]:
    cfg = _build_flex_config(args, mode, batch=batch, heads=heads, seq_len=seq_len)
    inputs = build_reference_inputs(cfg, device=resolve_cuda_device(), include_block_mask=True)
    kernel = compile_flashattention4_provider(inputs, cfg, provider=backend.provider or "flex_compiled")

    def run() -> torch.Tensor:
        with torch.inference_mode():
            return kernel.fn(inputs.q, inputs.k, inputs.v)

    return run, cfg


def _build_reference_callable(
    args: argparse.Namespace,
    mode: str,
    *,
    batch: int,
    heads: int,
    seq_len: int,
) -> Callable[[], torch.Tensor]:
    cfg = _build_flex_config(args, mode, batch=batch, heads=heads, seq_len=seq_len)
    inputs = build_reference_inputs(cfg, device=resolve_cuda_device(), include_block_mask=True)

    def run() -> torch.Tensor:
        with torch.inference_mode():
            return eager_flex_attention(inputs)

    return run


def _backend_supports_mode(backend: BenchBackend, mode: str) -> bool:
    if backend.kind == "flex":
        return True
    return mode in {"dense", "causal"}


def _shadow_verification_shape(batch: int, heads: int, seq_len: int) -> tuple[int, int, int]:
    return min(batch, 1), min(heads, 2), min(seq_len, 512)


def _verify_backend(args: argparse.Namespace, mode: str, backend: BenchBackend) -> Optional[float]:
    if not _backend_supports_mode(backend, mode):
        return None

    batch, heads, seq_len = _shadow_verification_shape(args.batch, args.heads, args.seq_len)
    reference = _build_reference_callable(args, mode, batch=batch, heads=heads, seq_len=seq_len)
    reference_out = reference().float()

    if backend.kind == "flex":
        candidate, _ = _build_flex_callable(args, mode, backend, batch=batch, heads=heads, seq_len=seq_len)
    else:
        cfg = _build_flex_config(args, mode, batch=batch, heads=heads, seq_len=seq_len)
        inputs = build_reference_inputs(cfg, device=resolve_cuda_device(), include_block_mask=True)
        candidate = _build_sdpa_callable(
            inputs.q,
            inputs.k,
            inputs.v,
            is_causal=(mode == "causal"),
            backend=backend.sdp_backend or SDPBackend.CUDNN_ATTENTION,
        )

    candidate_out = candidate().float()
    if not torch.isfinite(candidate_out).all():
        raise RuntimeError("non-finite output")
    max_diff = scalar_tensor_to_float((candidate_out - reference_out).abs().max())
    if not torch.allclose(candidate_out, reference_out, atol=0.5, rtol=0.05):
        raise RuntimeError(f"max_diff={max_diff:.6f}")
    return max_diff


def _benchmark_cuda_callable(fn: Callable[[], torch.Tensor], *, warmup: int, iterations: int) -> TimingStats:
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    for _ in range(iterations):
        start.record(current_stream)
        _ = fn()
        end.record(current_stream)
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

    return _timing_stats_from_samples(times_ms)


def _build_benchmark_callable(args: argparse.Namespace, mode: str, backend: BenchBackend) -> Callable[[], torch.Tensor]:
    if backend.kind == "flex":
        fn, _ = _build_flex_callable(
            args,
            mode,
            backend,
            batch=args.batch,
            heads=args.heads,
            seq_len=args.seq_len,
        )
        return fn

    cfg = _build_flex_config(args, mode, batch=args.batch, heads=args.heads, seq_len=args.seq_len)
    inputs = build_reference_inputs(cfg, device=resolve_cuda_device(), include_block_mask=True)
    return _build_sdpa_callable(
        inputs.q,
        inputs.k,
        inputs.v,
        is_causal=(mode == "causal"),
        backend=backend.sdp_backend or SDPBackend.CUDNN_ATTENTION,
    )


def _format_range(label: str, observed: Optional[float], target: tuple[float, float]) -> str:
    if observed is None:
        return f"{label}: n/a (reported {target[0]:.2f}x-{target[1]:.2f}x)"
    verdict = "inside" if target[0] <= observed <= target[1] else "outside"
    return f"{label}: {observed:.2f}x ({verdict} reported {target[0]:.2f}x-{target[1]:.2f}x)"


def _print_mode_report(mode: str, args: argparse.Namespace, results: list[MicrobenchResult]) -> None:
    decision = resolve_flashattention4_mode_decision(mode)
    print()
    print(f"Mode: {mode} | shape B={args.batch} H={args.heads} S={args.seq_len} D={args.head_dim}")
    print(
        f"Claim type: reproduction | Recommended local backend: {decision.recommended_backend} | "
        f"Recommended local target: {decision.recommended_target}"
    )
    print("| Backend | Median (ms) | TFLOPs/s | % of Colfax 1605 | Speedup vs Triton | Speedup vs cuDNN | Verify max diff |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        triton_speedup = f"{result.speedup_vs_triton_flex:.2f}x" if result.speedup_vs_triton_flex is not None else "n/a"
        cudnn_speedup = f"{result.speedup_vs_cudnn_sdpa:.2f}x" if result.speedup_vs_cudnn_sdpa is not None else "n/a"
        verify = f"{result.verification_max_diff:.6f}" if result.verification_max_diff is not None else "n/a"
        print(
            f"| {result.backend} | {result.median_ms:.3f} | {result.tflops:.1f} | "
            f"{result.pct_of_colfax_1605:.1f}% | {triton_speedup} | {cudnn_speedup} | {verify} |"
        )

    flash_result = next((item for item in results if item.backend == "flash_backend"), None)
    if flash_result is None:
        return

    print()
    print(
        _format_range(
            "Colfax vs cuDNN",
            flash_result.speedup_vs_cudnn_sdpa,
            COLFAX_FORWARD_SPEEDUP_VS_CUDNN,
        )
    )
    print(
        _format_range(
            "Colfax vs Triton",
            flash_result.speedup_vs_triton_flex,
            COLFAX_FORWARD_SPEEDUP_VS_TRITON,
        )
    )
    if mode in {"dense", "causal"}:
        print(
            _format_range(
                "PyTorch GB200 standard-attention vs Triton",
                flash_result.speedup_vs_triton_flex,
                PYTORCH_STANDARD_FORWARD_SPEEDUP_VS_TRITON,
            )
        )
    elif mode in PYTORCH_FLEX_FORWARD_SPEEDUP_VS_TRITON:
        print(
            _format_range(
                "PyTorch GB200 Flex-only vs Triton",
                flash_result.speedup_vs_triton_flex,
                PYTORCH_FLEX_FORWARD_SPEEDUP_VS_TRITON[mode],
            )
        )


def main() -> None:
    args = _build_arg_parser().parse_args()
    _apply_preset(args)
    resolve_cuda_device()

    modes = args.mode if args.mode else ["dense", "causal"]
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    prev_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    try:
        payload: dict[str, object] = {
            "preset": args.preset,
            "claim_type": "reproduction",
            "benchmark_settings": {
                "warmup": args.warmup,
                "iterations": args.iterations,
                "compile_mode": args.compile_mode,
                "lock_gpu_clocks": bool(args.lock_gpu_clocks),
                "sm_clock_mhz": args.sm_clock_mhz,
                "mem_clock_mhz": args.mem_clock_mhz,
            },
            "shape": {
                "batch": args.batch,
                "heads": args.heads,
                "seq_len": args.seq_len,
                "head_dim": args.head_dim,
                "block_size": args.block_size,
                "window_size": args.window_size,
            },
            "backend_notes": BACKEND_NOTES,
            "modes": {},
            "sources": {
                "colfax": COLFAX_SOURCE_URL,
                "pytorch": PYTORCH_SOURCE_URL,
            },
        }
        lock_ctx = (
            lock_gpu_clocks(device=0, sm_clock_mhz=args.sm_clock_mhz, mem_clock_mhz=args.mem_clock_mhz)
            if args.lock_gpu_clocks
            else nullcontext()
        )
        with lock_ctx:
            for mode in modes:
                flops = estimate_attention_forward_flops(
                    batch=args.batch,
                    heads=args.heads,
                    q_seq_len=args.seq_len,
                    kv_seq_len=args.seq_len,
                    head_dim=args.head_dim,
                    mode=mode,
                    window_size=args.window_size,
                )
                mode_results: list[MicrobenchResult] = []
                mode_errors: dict[str, str] = {}
                raw_results: dict[str, TimingStats] = {}
                verification_diffs: dict[str, Optional[float]] = {}

                for backend_name in args.backends:
                    if backend_name not in BACKENDS:
                        raise ValueError(f"Unknown backend {backend_name!r}; expected one of {tuple(BACKENDS)}")
                    backend = BACKENDS[backend_name]
                    if not _backend_supports_mode(backend, mode):
                        continue
                    try:
                        verification_diffs[backend_name] = _verify_backend(args, mode, backend)
                        fn = _build_benchmark_callable(args, mode, backend)
                        raw_results[backend_name] = _benchmark_cuda_callable(
                            fn,
                            warmup=args.warmup,
                            iterations=args.iterations,
                        )
                    except Exception as exc:
                        mode_errors[backend_name] = f"{exc.__class__.__name__}: {str(exc).splitlines()[0]}"

                triton_median = raw_results.get("triton_flex").median_ms if "triton_flex" in raw_results else None
                cudnn_median = raw_results.get("cudnn_sdpa").median_ms if "cudnn_sdpa" in raw_results else None

                for backend_name, timing in raw_results.items():
                    tflops = calculate_tflops(flops, timing.median_ms)
                    mode_results.append(
                        MicrobenchResult(
                            mode=mode,
                            backend=backend_name,
                            median_ms=timing.median_ms,
                            mean_ms=timing.mean_ms,
                            min_ms=timing.min_ms,
                            max_ms=timing.max_ms,
                            std_ms=timing.std_ms,
                            tflops=tflops,
                            pct_of_colfax_1605=(tflops / COLFAX_B200_BF16_ACHIEVED_TFLOPS) * 100.0,
                            speedup_vs_triton_flex=(triton_median / timing.median_ms) if triton_median else None,
                            speedup_vs_cudnn_sdpa=(cudnn_median / timing.median_ms) if cudnn_median else None,
                            verification_max_diff=verification_diffs.get(backend_name),
                        )
                    )

                mode_results.sort(key=lambda item: item.median_ms)
                payload["modes"][mode] = {
                    "mode_table": build_flashattention4_mode_table_payload(
                        current_mode=mode,
                        run_claim_type="reproduction",
                        target_label="labs/flashattention4/tflops_microbench.py",
                        selected_provider=None,
                    ),
                    "estimated_forward_flops": flops,
                    "reported_forward_speedup_ranges": {
                        "colfax_vs_cudnn": COLFAX_FORWARD_SPEEDUP_VS_CUDNN,
                        "colfax_vs_triton": COLFAX_FORWARD_SPEEDUP_VS_TRITON,
                        "pytorch_vs_triton": (
                            PYTORCH_STANDARD_FORWARD_SPEEDUP_VS_TRITON
                            if mode in {"dense", "causal"}
                            else PYTORCH_FLEX_FORWARD_SPEEDUP_VS_TRITON.get(mode)
                        ),
                    },
                    "results": [asdict(item) for item in mode_results],
                    "errors": mode_errors,
                }

                if not args.json:
                    _print_mode_report(mode, args, mode_results)
                    if mode_errors:
                        print("Errors:")
                        for backend_name, message in mode_errors.items():
                            print(f"- {backend_name}: {message}")

        rendered_payload = json.dumps(payload, indent=2, sort_keys=True)
        if args.json_out is not None:
            args.json_out.write_text(rendered_payload + "\n", encoding="ascii")
        if args.json:
            print(rendered_payload)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_matmul_tf32
        torch.backends.cudnn.allow_tf32 = prev_cudnn_tf32


if __name__ == "__main__":
    main()
