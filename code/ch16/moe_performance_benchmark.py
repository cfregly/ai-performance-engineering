"""
Mixture-of-Experts (MoE) performance benchmark for Blackwell GPUs.

The benchmark instantiates a transformer-style stack with a configurable
number of MoE layers and measures forward-pass throughput for synthetic
prompts. It supports both eager mode and `torch.compile`, reporting
tokens/second as well as per-iteration latency.
"""


from __future__ import annotations

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn

from core.optimization.moe_inference import MoEFeedForwardSortedDispatch


@dataclass
class MoEConfig:
    vocab_size: int = 50257
    d_model: int = 4096
    d_ff: int = 11008
    num_layers: int = 16
    num_moe_layers: int = 4
    num_experts: int = 8
    top_k: int = 2
    d_head: int = 64
    seq_len: int = 2048
    batch_size: int = 8


class ExpertMLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEFeedForward(MoEFeedForwardSortedDispatch):
    def __init__(self, config: MoEConfig) -> None:
        super().__init__(
            hidden=config.d_model,
            ffn=config.d_ff,
            num_experts=config.num_experts,
            top_k=config.top_k,
        )


class DenseFeedForward(nn.Module):
    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.net = ExpertMLP(config.d_model, config.d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionBlock(nn.Module):
    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=config.d_model,
            num_heads=config.d_model // config.d_head,
            dropout=0.0,
            batch_first=True,
        )
        self.ln = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.ln(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        return residual + attn_out


class MoELayer(nn.Module):
    def __init__(self, config: MoEConfig, moe: bool) -> None:
        super().__init__()
        self.attn = AttentionBlock(config)
        self.ff = MoEFeedForward(config) if moe else DenseFeedForward(config)
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn(x)
        residual = x
        x = self.norm(x)
        x = self.ff(x)
        return residual + x


class MoEModel(nn.Module):
    def __init__(self, config: MoEConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList(
            [
                MoELayer(config, moe=(idx < config.num_moe_layers))
                for idx in range(config.num_layers)
            ]
        )
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(input_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.head(hidden)


def benchmark_model(
    model: nn.Module,
    input_ids: torch.Tensor,
    *,
    warmup: int,
    iters: int,
) -> Dict[str, float]:
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize(input_ids.device)

    for _ in range(warmup):
        _ = model(input_ids)
    if input_ids.device.type == "cuda":
        torch.cuda.synchronize(input_ids.device)

    if input_ids.device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        current_stream = torch.cuda.current_stream(input_ids.device)
        start_event.record(current_stream)
        for _ in range(iters):
            _ = model(input_ids)
        end_event.record(current_stream)
        end_event.synchronize()
        elapsed = start_event.elapsed_time(end_event) / 1000.0
    else:
        start = time.perf_counter()
        for _ in range(iters):
            _ = model(input_ids)
        elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iters) * 1000.0
    tokens = input_ids.numel()
    throughput = tokens / (elapsed / iters)
    return {"latency_ms": avg_ms, "throughput_tok_s": throughput}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("MoE performance benchmark")
    parser.add_argument("--device", default="cuda", help="Execution device (cuda or cpu)")
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"], help="Compute dtype")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations")
    parser.add_argument("--iters", type=int, default=20, help="Benchmark iterations")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for the model")
    parser.add_argument("--output-json", type=Path, help="Optional results file")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=16)
    parser.add_argument("--num-moe-layers", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    config = MoEConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        num_experts=args.num_experts,
        top_k=max(1, args.top_k),
        num_layers=args.num_layers,
        num_moe_layers=min(args.num_moe_layers, args.num_layers),
    )

    model = MoEModel(config).to(device=device, dtype=dtype)
    model.eval()

    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")

    input_ids = torch.randint(
        0,
        config.vocab_size,
        (config.batch_size, config.seq_len),
        device=device,
        dtype=torch.int32,
    )

    metrics = benchmark_model(model, input_ids, warmup=args.warmup, iters=args.iters)
    info = {
        "config": asdict(config),
        "dtype": args.dtype,
        "device": str(device),
        "compile": args.compile,
        **metrics,
    }

    print("MoE Benchmark Results")
    print(f"  Device: {info['device']}")
    print(f"  Dtype: {info['dtype']}")
    print(f"  Compile: {info['compile']}")
    print(f"  Batch x Seq: {config.batch_size} x {config.seq_len}")
    print(f"  Experts: {config.num_experts}  Top-k: {config.top_k}")
    print(f"  Layers: {config.num_layers} (MoE {config.num_moe_layers})")
    print(f"  Latency: {metrics['latency_ms']:.2f} ms/iter")
    print(f"  Throughput: {metrics['throughput_tok_s']:.0f} tok/s")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
