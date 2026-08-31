#!/usr/bin/env python3
"""
Automate cross-precision power efficiency sweeps for GPT workloads.

This helper runs the chapter 16 large-model benchmark at multiple precision
settings while sampling GPU power via NVML. The output summarizes throughput,
average power, tokens per joule, and estimated cost per million tokens.

Example:
    python -m scripts.precision_power_sweep --sequence-length 4096 \
        --modes fp16 bf16 fp8_weight fp8_te --output-json precision_sweep.json
"""

from __future__ import annotations

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch

from core.common.device_utils import resolve_requested_device
from ch16.gpt_large_benchmark import (
    GPTConfig,
    Workload,
    run_workload,
    transformer_engine_available,
    transformer_engine_warning,
)

from core.utils.power_sampling import PowerSampler


@dataclass
class PrecisionRunConfig:
    name: str
    dtype: str
    fp8_mode: str
    description: str


@dataclass
class PrecisionRunResult:
    mode: str
    dtype: str
    fp8_mode: str
    throughput_tokens_per_s: Optional[float]
    latency_ms: Optional[float]
    avg_power_watts: Optional[float]
    max_power_watts: Optional[float]
    duration_s: Optional[float]
    energy_joules: Optional[float]
    tokens_per_joule: Optional[float]
    cost_per_million_tokens: Optional[float]
    notes: List[str]
    status: str
    message: Optional[str] = None
    raw_benchmark: Dict[str, object] = field(default_factory=dict)
    raw_power: Optional[Dict[str, object]] = None


AVAILABLE_MODES: Dict[str, PrecisionRunConfig] = {
    "fp16": PrecisionRunConfig(
        name="fp16",
        dtype="float16",
        fp8_mode="none",
        description="Standard FP16 autocast",
    ),
    "bf16": PrecisionRunConfig(
        name="bf16",
        dtype="bfloat16",
        fp8_mode="none",
        description="BF16 autocast",
    ),
    "fp8_weight": PrecisionRunConfig(
        name="fp8_weight",
        dtype="float16",
        fp8_mode="weight-only",
        description="Weight-only FP8 quantization",
    ),
    "fp8_te": PrecisionRunConfig(
        name="fp8_te",
        dtype="bfloat16",
        fp8_mode="transformer-engine",
        description="Transformer Engine FP8 autocast",
    ),
}


def parse_gpu_list(arg: Optional[str]) -> List[int]:
    if not arg:
        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
        return []
    values = []
    for item in arg.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(int(item))
    return values


def build_precision_list(modes: Iterable[str]) -> List[PrecisionRunConfig]:
    if not modes:
        return [
            AVAILABLE_MODES["fp16"],
            AVAILABLE_MODES["bf16"],
            AVAILABLE_MODES["fp8_weight"],
            AVAILABLE_MODES["fp8_te"],
        ]
    configs = []
    for mode in modes:
        key = mode.strip().lower()
        if key not in AVAILABLE_MODES:
            raise ValueError(
                f"Unknown precision mode '{mode}'. "
                f"Available: {', '.join(sorted(AVAILABLE_MODES))}"
            )
        configs.append(AVAILABLE_MODES[key])
    return configs


def calculate_cost_per_million_tokens(
    *,
    avg_power_watts: float,
    tokens_per_sec: float,
    electricity_cost: float,
    pue: float,
) -> float:
    if avg_power_watts <= 0 or tokens_per_sec <= 0:
        return 0.0
    joules_per_token = avg_power_watts / tokens_per_sec
    kwh_per_token = joules_per_token / 3_600_000.0
    return kwh_per_token * 1_000_000 * electricity_cost * pue


def run_precision_mode(
    *,
    mode: PrecisionRunConfig,
    device: torch.device,
    sequence_length: int,
    batch_size: int,
    warmup: int,
    iters: int,
    compile_mode: str,
    skip_compile: bool,
    sampler: PowerSampler,
    electricity_cost: float,
    pue: float,
    attention_backend: str,
    attention_window: Optional[int],
    max_seq_len: Optional[int],
    model_layers: int,
    model_d_model: int,
    model_heads: int,
    model_d_ff: int,
    model_vocab_size: int,
) -> PrecisionRunResult:
    dtype = torch.float16 if mode.dtype == "float16" else torch.bfloat16
    workload = Workload(
        batch=batch_size,
        seq_len=sequence_length,
        description=f"Batch={batch_size}, Seq={sequence_length}",
    )

    config = GPTConfig(
        vocab_size=model_vocab_size,
        n_layers=model_layers,
        n_heads=model_heads,
        d_model=model_d_model,
        d_ff=model_d_ff,
    )
    config.max_seq_len = max(
        config.max_seq_len,
        sequence_length,
        max_seq_len or sequence_length,
    )
    config.attention_backend = attention_backend
    config.attention_window = attention_window

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)

    sampler.start()
    benchmark_result = None
    error_message: Optional[str] = None
    try:
        benchmark_result = run_workload(
            config,
            workload,
            devices=[device],
            dtype=dtype,
            warmup=warmup,
            iters=iters,
            compile_mode=compile_mode,
            skip_compile=skip_compile,
            fp8_mode=mode.fp8_mode,
        )
    except RuntimeError as exc:
        error_message = str(exc)
    finally:
        power_metrics = sampler.stop()

    if benchmark_result is None:
        return PrecisionRunResult(
            mode=mode.name,
            dtype=mode.dtype,
            fp8_mode=mode.fp8_mode,
            throughput_tokens_per_s=None,
            latency_ms=None,
            avg_power_watts=power_metrics.get("avg_watts") if power_metrics else None,
            max_power_watts=power_metrics.get("max_watts") if power_metrics else None,
            duration_s=power_metrics.get("duration_s") if power_metrics else None,
            energy_joules=power_metrics.get("energy_joules") if power_metrics else None,
            tokens_per_joule=None,
            cost_per_million_tokens=None,
            notes=[],
            status="error",
            message=error_message,
            raw_power=power_metrics,
        )

    benchmark_data = asdict(benchmark_result)
    throughput = benchmark_result.compiled_tps or benchmark_result.eager_tps
    latency = benchmark_result.compiled_ms or benchmark_result.eager_ms
    notes = list(benchmark_result.notes)

    avg_power = power_metrics.get("avg_watts") if power_metrics else None
    duration_s = power_metrics.get("duration_s") if power_metrics else None
    energy_j = power_metrics.get("energy_joules") if power_metrics else None

    tokens_per_joule = (
        throughput / avg_power if throughput and avg_power else None
    )

    cost_per_million = (
        calculate_cost_per_million_tokens(
            avg_power_watts=avg_power,
            tokens_per_sec=throughput,
            electricity_cost=electricity_cost,
            pue=pue,
        )
        if throughput and avg_power
        else None
    )

    return PrecisionRunResult(
        mode=mode.name,
        dtype=mode.dtype,
        fp8_mode=mode.fp8_mode,
        throughput_tokens_per_s=throughput,
        latency_ms=latency,
        avg_power_watts=avg_power,
        max_power_watts=power_metrics.get("max_watts") if power_metrics else None,
        duration_s=duration_s,
        energy_joules=energy_j,
        tokens_per_joule=tokens_per_joule,
        cost_per_million_tokens=cost_per_million,
        notes=notes,
        status="success",
        raw_benchmark=benchmark_data,
        raw_power=power_metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-precision power efficiency sweeps.",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=4096,
        help="Sequence length to benchmark (default: 4096).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for the workload (default: 2).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Warmup iterations per measurement.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=4,
        help="Timed iterations per measurement.",
    )
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        help="torch.compile mode (default: reduce-overhead).",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Skip torch.compile and run eager path only.",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        help="Precision modes to run (choices: fp16, bf16, fp8_weight, fp8_te).",
    )
    parser.add_argument(
        "--device",
        help="Device to execute the benchmark (default: cuda if available).",
    )
    parser.add_argument(
        "--gpus",
        help="Comma-separated GPU indices for power sampling (default: all).",
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.2,
        help="NVML sampling interval in seconds (default: 0.2).",
    )
    parser.add_argument(
        "--electricity-cost",
        type=float,
        default=0.16,
        help="Electricity cost in USD per kWh (default: 0.16).",
    )
    parser.add_argument(
        "--pue",
        type=float,
        default=1.5,
        help="Power Usage Effectiveness factor (default: 1.5).",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["sdpa", "flex", "auto"],
        default="auto",
        help="Attention backend override (default: auto).",
    )
    parser.add_argument(
        "--attention-window",
        type=int,
        default=None,
        help="Optional attention window when using Flex attention.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        help="Override GPTConfig.max_seq_len (default: matches sequence length).",
    )
    parser.add_argument(
        "--model-layers",
        type=int,
        default=48,
        help="Number of transformer layers in GPTConfig (default: 48).",
    )
    parser.add_argument(
        "--model-d-model",
        type=int,
        default=8192,
        help="Model hidden size d_model (default: 8192).",
    )
    parser.add_argument(
        "--model-heads",
        type=int,
        default=64,
        help="Number of attention heads (default: 64).",
    )
    parser.add_argument(
        "--model-d-ff",
        type=int,
        default=32768,
        help="Feed-forward hidden size (default: 32768).",
    )
    parser.add_argument(
        "--model-vocab-size",
        type=int,
        default=50304,
        help="Vocabulary size for GPTConfig (default: 50304).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional JSON path for structured results.",
    )
    parser.add_argument(
        "--output-markdown",
        type=Path,
        help="Optional Markdown summary file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_requested_device(args.device)
    gpu_indices = parse_gpu_list(args.gpus)
    precision_modes = build_precision_list(args.modes or [])

    if device.type != "cuda" and gpu_indices:
        print("⚠ Power sampling requested but CUDA device not in use; ignoring GPU list.")
        gpu_indices = []

    sampler = PowerSampler(gpu_indices, args.sample_interval)
    results: List[PrecisionRunResult] = []

    try:
        for mode in precision_modes:
            print(f"\n== Running precision sweep: {mode.name} ({mode.description}) ==")
            if mode.fp8_mode == "transformer-engine" and not transformer_engine_available():
                print(f"  Transformer Engine unavailable ({transformer_engine_warning()}); skipping.")
                results.append(
                    PrecisionRunResult(
                        mode=mode.name,
                        dtype=mode.dtype,
                        fp8_mode=mode.fp8_mode,
                        throughput_tokens_per_s=None,
                        latency_ms=None,
                        avg_power_watts=None,
                        max_power_watts=None,
                        duration_s=None,
                        energy_joules=None,
                        tokens_per_joule=None,
                        cost_per_million_tokens=None,
                        notes=[],
                        status="error",
                        message=transformer_engine_warning(),
                    )
                )
                continue

            result = run_precision_mode(
                mode=mode,
                device=device,
                sequence_length=args.sequence_length,
                batch_size=args.batch_size,
                warmup=args.warmup,
                iters=args.iters,
                compile_mode=args.compile_mode,
                skip_compile=args.skip_compile,
                sampler=sampler,
                electricity_cost=args.electricity_cost,
                pue=args.pue,
                attention_backend=args.attention_backend,
                attention_window=args.attention_window,
                max_seq_len=args.max_seq_len,
                model_layers=args.model_layers,
                model_d_model=args.model_d_model,
                model_heads=args.model_heads,
                model_d_ff=args.model_d_ff,
                model_vocab_size=args.model_vocab_size,
            )
            results.append(result)
            if result.status == "success":
                print(
                    f"  Throughput: {result.throughput_tokens_per_s:,.0f} tokens/s "
                    f"({result.latency_ms or 0:.2f} ms)"
                )
                if result.avg_power_watts is not None:
                    print(
                        f"  Power: avg {result.avg_power_watts:.1f} W, "
                        f"max {result.max_power_watts or 0:.1f} W"
                    )
                if result.tokens_per_joule is not None:
                    print(
                        f"  Efficiency: {result.tokens_per_joule:.3f} tokens/joule "
                        f"(cost ${result.cost_per_million_tokens or 0:.4f} / M tokens)"
                    )
            else:
                print(f"  Run failed: {result.message}")
    finally:
        sampler.close()

    success_count = sum(1 for item in results if item.status == "success")
    print(f"\nCompleted {success_count}/{len(results)} precision modes.")

    if args.output_json:
        payload = {
            "sequence_length": args.sequence_length,
            "batch_size": args.batch_size,
            "warmup": args.warmup,
            "iters": args.iters,
            "compile_mode": args.compile_mode,
            "skip_compile": args.skip_compile,
            "device": str(device),
            "gpu_indices": gpu_indices,
            "modes": [mode.name for mode in precision_modes],
            "power_sampling_interval": args.sample_interval,
            "electricity_cost": args.electricity_cost,
            "pue": args.pue,
            "model_layers": args.model_layers,
            "model_d_model": args.model_d_model,
            "model_heads": args.model_heads,
            "model_d_ff": args.model_d_ff,
            "model_vocab_size": args.model_vocab_size,
            "results": [asdict(res) for res in results],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"JSON summary written to {args.output_json}")

    if args.output_markdown:
        lines = [
            "# Precision Power Sweep",
            "",
            "| Mode | Throughput (tok/s) | Avg Power (W) | Tokens/J | Cost per 1M tokens | Notes |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in results:
            if item.status != "success":
                lines.append(
                    f"| {item.mode} | failure | - | - | - | {item.message or ''} |"
                )
                continue
            throughput = (
                f"{item.throughput_tokens_per_s:,.0f}"
                if item.throughput_tokens_per_s
                else "-"
            )
            avg_power = (
                f"{item.avg_power_watts:.1f}" if item.avg_power_watts else "-"
            )
            tpj = (
                f"{item.tokens_per_joule:.4f}"
                if item.tokens_per_joule
                else "-"
            )
            cost = (
                f"${item.cost_per_million_tokens:.4f}"
                if item.cost_per_million_tokens
                else "-"
            )
            note_text = "; ".join(item.notes)
            lines.append(
                f"| {item.mode} | {throughput} | {avg_power} | {tpj} | {cost} | {note_text} |"
            )
        args.output_markdown.parent.mkdir(parents=True, exist_ok=True)
        args.output_markdown.write_text("\n".join(lines) + "\n")
        print(f"Markdown summary written to {args.output_markdown}")


if __name__ == "__main__":
    main()
