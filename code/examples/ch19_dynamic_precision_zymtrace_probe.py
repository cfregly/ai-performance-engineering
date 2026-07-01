#!/usr/bin/env python3
"""Profile the Chapter 19 dynamic-precision decode path under Zymtrace."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch


CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from ch19.dynamic_precision_benchmark_common import (  # noqa: E402
    DynamicPrecisionBenchmarkConfig,
    FixedDecodeWorkspace,
    build_model,
    build_prompt,
    decode_fixed_precision,
    decode_host_policy_baseline,
)
from ch19.dynamic_precision_switching import (  # noqa: E402
    DynamicPrecisionWorkspace,
    decode_with_dynamic_precision,
)


DTYPES = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a timed ch19 decode workload for Zymtrace capture.")
    parser.add_argument("--seconds", type=float, default=20.0, help="Minimum measured runtime.")
    parser.add_argument("--mode", choices=("dynamic", "fixed", "host"), default="dynamic")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--prompt-len", type=int, default=192)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--vocab-size", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def fixed_workspace(
    cfg: DynamicPrecisionBenchmarkConfig,
    prompt: torch.Tensor,
    device: torch.device,
    *,
    host_policy: bool,
) -> FixedDecodeWorkspace:
    output_shape = (cfg.batch_size, cfg.prompt_len + cfg.max_steps)
    token_shape = (cfg.batch_size, 1)
    kwargs = {}
    if host_policy:
        kwargs = {
            "host_logits_buffer": torch.empty(
                (cfg.batch_size, cfg.vocab_size),
                device="cpu",
                dtype=torch.float32,
                pin_memory=device.type == "cuda",
            ),
            "policy_metrics_buffer": torch.empty(4, device="cpu", dtype=torch.float32),
            "policy_metric_values": [0.0] * 4,
            "policy_top2_values": torch.empty((cfg.batch_size, 2), device="cpu", dtype=torch.float32),
            "policy_top2_indices": torch.empty((cfg.batch_size, 2), device="cpu", dtype=torch.long),
        }
    return FixedDecodeWorkspace(
        generated=torch.empty(output_shape, device=device, dtype=prompt.dtype),
        next_token=torch.empty(token_shape, device=device, dtype=prompt.dtype),
        next_token_values=torch.empty(token_shape, device=device, dtype=torch.float32),
        **kwargs,
    )


def dynamic_workspace(
    cfg: DynamicPrecisionBenchmarkConfig,
    prompt: torch.Tensor,
    device: torch.device,
) -> DynamicPrecisionWorkspace:
    output_shape = (cfg.batch_size, cfg.prompt_len + cfg.max_steps)
    token_shape = (cfg.batch_size, 1)
    top2_shape = (cfg.batch_size, 2)
    return DynamicPrecisionWorkspace(
        generated=torch.empty(output_shape, device=device, dtype=prompt.dtype),
        next_token=torch.empty(token_shape, device=device, dtype=prompt.dtype),
        next_token_values=torch.empty(token_shape, device=device, dtype=torch.float32),
        top2_values=torch.empty(top2_shape, device=device, dtype=torch.float32),
        top2_indices=torch.empty(top2_shape, device=device, dtype=torch.long),
        margin_values=torch.empty(cfg.batch_size, device=device, dtype=torch.float32),
        margin_mean=torch.empty((), device=device, dtype=torch.float32),
        ema_conf=torch.empty((), device=device, dtype=torch.float32),
    )


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiling probe")
    if args.seconds <= 0:
        raise ValueError("--seconds must be positive")

    device = torch.device(args.device)
    cfg = DynamicPrecisionBenchmarkConfig(
        batch_size=args.batch_size,
        prompt_len=args.prompt_len,
        max_steps=args.max_steps,
        vocab_size=args.vocab_size,
        hidden_dim=args.hidden_dim,
    )
    dtype = DTYPES[args.dtype]
    torch.manual_seed(1234)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(1234)

    prompt = build_prompt(cfg, device)
    model = build_model(cfg, device, dtype=dtype).eval()

    if args.mode == "dynamic":
        workspace = dynamic_workspace(cfg, prompt, device)

        def run_once() -> torch.Tensor:
            output, _ = decode_with_dynamic_precision(
                model,
                prompt,
                cfg.max_steps,
                device=device,
                enable_fp8=False,
                enable_fp4=False,
                reeval_interval=8,
                workspace=workspace,
            )
            return output

    elif args.mode == "host":
        workspace = fixed_workspace(cfg, prompt, device, host_policy=True)

        def run_once() -> torch.Tensor:
            return decode_host_policy_baseline(
                model,
                prompt,
                max_steps=cfg.max_steps,
                device=device,
                workspace=workspace,
            )

    else:
        workspace = fixed_workspace(cfg, prompt, device, host_policy=False)

        def run_once() -> torch.Tensor:
            return decode_fixed_precision(
                model,
                prompt,
                max_steps=cfg.max_steps,
                device=device,
                workspace=workspace,
            )

    with torch.inference_mode():
        for _ in range(args.warmup):
            output = run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
        deadline = time.perf_counter() + args.seconds
        iterations = 0
        if start_event is not None:
            start_event.record()
        wall_start = time.perf_counter()
        while time.perf_counter() < deadline:
            output = run_once()
            iterations += 1
        if end_event is not None:
            end_event.record()
            torch.cuda.synchronize(device)
            elapsed_ms = start_event.elapsed_time(end_event)
        else:
            elapsed_ms = (time.perf_counter() - wall_start) * 1000.0

    tokens = cfg.batch_size * cfg.max_steps * max(iterations, 1)
    checksum = int(output[0, -1].item())
    print(
        "ch19_dynamic_precision_zymtrace_probe "
        f"device={torch.cuda.get_device_name(device) if device.type == 'cuda' else device} "
        f"mode={args.mode} dtype={args.dtype} iterations={iterations} "
        f"avg_ms={elapsed_ms / max(iterations, 1):.3f} "
        f"tokens_per_s={tokens / max(elapsed_ms / 1000.0, 1e-9):.1f} "
        f"checksum={checksum} "
        f"cuda_injection={'set' if os.environ.get('CUDA_INJECTION64_PATH') else 'unset'}"
    )


if __name__ == "__main__":
    main()
