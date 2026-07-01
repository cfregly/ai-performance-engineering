"""
GPT Large Model Benchmark for Blackwell GPUs
============================================

This script provides a runnable large-model benchmark sized for a single
NVIDIA B200 (SM 10.0) GPU. It measures forward-pass latency and throughput
for a GPT-style transformer under different sequence lengths and compares
PyTorch eager execution with torch.compile.

Highlights
----------
- Parameter count ~39.6B (48 layers, d_model=8192, n_heads=64, d_ff=32768)
- Default workloads sized for long-context inference (2048 & 4096 tokens)
- Configurable warmup/iteration counts to balance accuracy vs. runtime
- Optional JSON logging for reproducible performance reports
- Graceful handling of OOM conditions (recorded instead of crashing)
- Optional tensor-parallel partitioning across multiple GPUs to reduce per-device
  memory pressure

Example:
    python ch16/gpt_large_benchmark.py --iters 5 --output-json results.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional

from core.benchmark.utils import scalar_tensor_to_float
from core.harness.arch_config import prefer_flash_sdpa

import torch
import torch.nn as nn

from core.utils.compile_utils import enable_tf32, compile_model

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
except ImportError:  # pragma: no cover - flex attention may be unavailable
    flex_attention = None
    create_block_mask = None

try:
    from ch16.fp8_transformer_engine import (
        TransformerEngineUnavailable,
        convert_linear_layers as convert_linear_layers_to_te,
        fp8_autocast as te_fp8_autocast,
        make_delayed_scaling_recipe,
        transformer_engine_available,
        transformer_engine_warning,
    )
except Exception:  # pragma: no cover - TE helpers unavailable
    TransformerEngineUnavailable = RuntimeError  # type: ignore

    def transformer_engine_available() -> bool:
        return False

    def transformer_engine_warning() -> str:
        return "Transformer Engine not available"

    def convert_linear_layers_to_te(*_, **__):
        raise TransformerEngineUnavailable("Transformer Engine helpers missing")

    @contextlib.contextmanager
    def te_fp8_autocast(*_, **__):
        yield

    def make_delayed_scaling_recipe(**_):
        raise TransformerEngineUnavailable("Transformer Engine helpers missing")


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


@dataclass
class GPTConfig:
    vocab_size: int = 50304
    n_layers: int = 48
    n_heads: int = 64
    d_model: int = 8192
    d_ff: int = 32768
    max_seq_len: int = 8192
    attention_backend: str = "sdpa"
    attention_window: Optional[int] = None


class FP8Linear(nn.Module):
    """Weight-only FP8 linear layer with per-output scaling."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer(
            "weight_fp8",
            torch.zeros(out_features, in_features, dtype=torch.float8_e4m3fn, device=device),
        )
        self.register_buffer(
            "weight_scale",
            torch.zeros(out_features, 1, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "weight_scale_t",
            torch.zeros(1, out_features, dtype=torch.float32, device=device),
        )
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device), requires_grad=False)
            self.register_buffer(
                "bias_bf16",
                torch.zeros(out_features, dtype=torch.bfloat16, device=device),
            )
        else:
            self.register_parameter("bias", None)
            self.register_buffer("bias_bf16", None)

    @staticmethod
    def _quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = weight.to(torch.float32)
        max_abs = weight.abs().amax(dim=1, keepdim=True)
        scale = (max_abs / 448.0).clamp(min=1e-6)
        scaled = (weight / scale).clamp(-448, 448)
        return scaled.to(torch.float8_e4m3fn), scale.to(torch.float16)

    @classmethod
    def from_linear(cls, linear: nn.Linear, dtype: torch.dtype) -> "FP8Linear":
        device = linear.weight.device
        module = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            dtype=dtype,
            device=device,
        )
        fp8_weight, scale = cls._quantize_weight(linear.weight.detach().to(dtype))
        module.weight_fp8.copy_(fp8_weight)
        weight_scale = scale.to(torch.float32)
        module.weight_scale.copy_(weight_scale)
        module.weight_scale_t.copy_(weight_scale.transpose(0, 1).contiguous())
        if linear.bias is not None and module.bias is not None:
            module.bias.data.copy_(linear.bias.detach().to(dtype))
            module.bias_bf16.copy_(module.bias.detach().to(torch.bfloat16))
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x2d = x.reshape(-1, self.in_features)

        # Row-wise activation scaling
        act_abs = x2d.abs().amax(dim=1, keepdim=True)
        act_scale = (act_abs / 448.0).clamp_(min=1e-6).to(torch.float32)
        x_fp8 = (x2d / act_scale.to(x.dtype)).clamp_(-448, 448).to(torch.float8_e4m3fn)

        # Column-wise weight scaling (already stored)
        weight_scale = self.weight_scale_t  # shape (1, out_features)
        mat2 = self.weight_fp8.transpose(0, 1)  # shape (in_features, out_features)

        bias = self.bias_bf16 if self.bias_bf16 is not None else None

        out = torch._scaled_mm(
            x_fp8,
            mat2,
            act_scale,
            weight_scale,
            bias=bias,
            out_dtype=torch.bfloat16,
            use_fast_accum=False,
        )
        if out.dtype != x.dtype:
            out = out.to(x.dtype)
        return out.reshape(*original_shape[:-1], self.out_features)


def convert_linear_layers_to_fp8(module: nn.Module, *, dtype: torch.dtype) -> int:
    converted = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.MultiheadAttention):
            continue
        if isinstance(child, nn.Linear):
            fp8_linear = FP8Linear.from_linear(child, dtype=dtype)
            setattr(module, name, fp8_linear)
            converted += 1
        elif isinstance(child, FP8Linear):
            continue
        else:
            converted += convert_linear_layers_to_fp8(child, dtype=dtype)
    return converted


class MultiheadAttentionBackend(nn.Module):
    """Self-attention with optional FlexAttention execution."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.backend = config.attention_backend.lower()
        self.attention_window = config.attention_window
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out_proj = nn.Linear(config.d_model, config.d_model)
        self._block_mask_cache = {}

    def _effective_backend(self, x: torch.Tensor) -> str:
        backend = self.backend
        if backend == "auto":
            if (
                flex_attention is not None
                and x.device.type == "cuda"
                and self.head_dim % 16 == 0
            ):
                backend = "flex"
            else:
                backend = "sdpa"
        return backend

    def _window_mask_fn(self, _b, _h, q_idx, kv_idx):
        window = self.attention_window
        causal = kv_idx <= q_idx
        if window is not None:
            in_window = (q_idx - kv_idx) < window
            return causal & in_window
        return causal

    def _window_mask(self, batch: int, q_len: int, kv_len: int, device: torch.device):
        if create_block_mask is None:
            return None

        cache_key = (int(batch), int(self.n_heads), int(q_len), int(kv_len), device, self.attention_window)
        block_mask = self._block_mask_cache.get(cache_key)
        if block_mask is None:
            block_mask = create_block_mask(
                self._window_mask_fn,
                B=batch,
                H=self.n_heads,
                Q_LEN=q_len,
                KV_LEN=kv_len,
                device=device,
            )
            self._block_mask_cache[cache_key] = block_mask
        return block_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        backend = self._effective_backend(x)

        if backend == "flex":
            if flex_attention is None:
                raise RuntimeError("FlexAttention requested but not available")
            block_mask = self._window_mask(batch, seq_len, key.shape[2], device=key.device)
            attn = flex_attention(
                query,
                key,
                value,
                block_mask=block_mask,
                scale=self.head_dim ** -0.5,
            )
        else:
            with prefer_flash_sdpa():
                attn = torch.nn.functional.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    dropout_p=0.0,
                    is_causal=True,
                )

        attn = attn.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        return self.out_proj(attn)


class GPTBlock(nn.Module):
    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)
        self.attn = MultiheadAttentionBackend(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Linear(config.d_ff, config.d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.ln1(x)
        attn_out = self.attn(x)
        x = residual + attn_out
        residual = x
        x = self.ln2(x)
        x = residual + self.mlp(x)
        return x


class GPTModel(nn.Module):
    def __init__(
        self,
        config: GPTConfig,
        *,
        devices: List[torch.device],
        dtype: torch.dtype,
        fp8_mode: str = "none",
    ) -> None:
        super().__init__()
        self.config = config
        self.devices = devices
        self.fp8_mode = fp8_mode
        self.embed = nn.Embedding(config.vocab_size, config.d_model).to(devices[0], dtype=dtype)

        layers_per_partition = (config.n_layers + len(devices) - 1) // len(devices)
        self.blocks = nn.ModuleList()
        self.block_devices: List[torch.device] = []
        for idx in range(config.n_layers):
            device = devices[min(idx // layers_per_partition, len(devices) - 1)]
            block = GPTBlock(config).to(device, dtype=dtype)
            self.blocks.append(block)
            self.block_devices.append(device)
        self.block_device_groups = self._build_block_device_groups()

        final_device = self.block_devices[-1] if self.block_devices else devices[0]
        self.final_device = final_device
        self._return_output_to_first_device = final_device != devices[0]
        self.ln_f = nn.LayerNorm(config.d_model).to(final_device, dtype=dtype)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False).to(final_device, dtype=dtype)

        if fp8_mode == "weight-only":
            converted = convert_linear_layers_to_fp8(self, dtype=dtype)
            self.register_buffer(
                "_fp8_converted_linear_layers",
                torch.tensor(converted, device=devices[0]),
            )
        elif fp8_mode == "transformer-engine":
            try:
                converted = convert_linear_layers_to_te(self, params_dtype=dtype)
            except TransformerEngineUnavailable as exc:
                raise RuntimeError(transformer_engine_warning()) from exc
            self.register_buffer(
                "_te_converted_linear_layers",
                torch.tensor(converted, device=devices[0]),
            )

    def _build_block_device_groups(self) -> List[tuple[torch.device, int, int]]:
        if not self.block_devices:
            return []
        groups: List[tuple[torch.device, int, int]] = []
        start = 0
        current_device = self.block_devices[0]
        for idx, device in enumerate(self.block_devices[1:], start=1):
            if device == current_device:
                continue
            groups.append((current_device, start, idx))
            current_device = device
            start = idx
        groups.append((current_device, start, len(self.block_devices)))
        return groups

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device != self.devices[0]:
            x = x.to(self.devices[0], non_blocking=True)
        x = self.embed(x)
        for device, start, end in self.block_device_groups:
            if x.device != device:
                x = x.to(device, non_blocking=True)
            for block in self.blocks[start:end]:
                x = block(x)
        x = self.ln_f(x)
        x = self.lm_head(x)
        if self._return_output_to_first_device:
            x = x.to(self.devices[0], non_blocking=True)
        return x


# ---------------------------------------------------------------------------
# Benchmark utilities
# ---------------------------------------------------------------------------


@dataclass
class Workload:
    batch: int
    seq_len: int
    description: str


@dataclass
class BenchmarkResult:
    config: str
    batch: int
    seq_len: int
    attention_backend: str
    precision_mode: str
    eager_ms: Optional[float]
    eager_tps: Optional[float]
    compiled_ms: Optional[float]
    compiled_tps: Optional[float]
    speedup: Optional[float]
    eager_peak_mem_gb: Optional[float]
    compiled_peak_mem_gb: Optional[float]
    notes: List[str]


def count_parameters(config: GPTConfig) -> float:
    embed = config.vocab_size * config.d_model
    ln = 2 * config.d_model * config.n_layers + config.d_model
    attn = config.n_layers * (
        3 * config.d_model * config.d_model  # qkv
        + config.d_model * config.d_model  # out proj
    )
    mlp = config.n_layers * (
        config.d_model * config.d_ff + config.d_ff * config.d_model
    )
    head = config.vocab_size * config.d_model
    total = embed + ln + attn + mlp + head
    return total


@torch.inference_mode()
def benchmark_model(
    model: nn.Module,
    inputs: torch.Tensor,
    warmup: int,
    iters: int,
    devices: List[torch.device],
    precision_ctx_factory: Optional[Callable[[], contextlib.AbstractContextManager]] = None,
) -> Dict[str, float]:
    """Measure average latency/throughput and peak memory."""
    for dev in devices:
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
    ctx_factory = precision_ctx_factory or contextlib.nullcontext
    with ctx_factory():
        for _ in range(warmup):
            model(inputs)
    for dev in devices:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

    start = time.perf_counter()
    with ctx_factory():
        for _ in range(iters):
            model(inputs)
    for dev in devices:
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
    elapsed = time.perf_counter() - start

    avg_ms = (elapsed / iters) * 1e3
    tokens_per_iter = inputs.shape[0] * inputs.shape[1]
    throughput = tokens_per_iter / (elapsed / iters)
    peak_mem_gb = 0.0
    for dev in devices:
        if dev.type == "cuda":
            peak_mem_gb = max(peak_mem_gb, torch.cuda.max_memory_allocated(dev) / 1e9)
    return {
        "avg_ms": avg_ms,
        "throughput": throughput,
        "peak_mem_gb": peak_mem_gb,
    }


def run_workload(
    config: GPTConfig,
    workload: Workload,
    *,
    devices: List[torch.device],
    dtype: torch.dtype,
    warmup: int,
    iters: int,
    compile_mode: str,
    skip_compile: bool,
    fp8_mode: str,
) -> BenchmarkResult:
    notes: List[str] = []
    x = torch.randint(
        0,
        config.vocab_size,
        (workload.batch, workload.seq_len),
        device=devices[0],
        dtype=torch.int32,
    )
    activation_mem_gb = (
        workload.batch * workload.seq_len * config.d_model * torch.finfo(dtype).bits / 8
    ) / 1e9
    notes.append(f"Activation footprint ~ {activation_mem_gb:.2f} GB")

    notes.append(f"Tensor-parallel GPUs: {len(devices)}")

    resolved_fp8_mode = fp8_mode
    if fp8_mode == "auto":
        if transformer_engine_available():
            resolved_fp8_mode = "transformer-engine"
        else:
            resolved_fp8_mode = "weight-only"
            notes.append(transformer_engine_warning())

    precision_ctx_factory: Optional[Callable[[], contextlib.AbstractContextManager]] = None
    model: Optional[GPTModel] = None
    while True:
        try:
            model = GPTModel(
                config,
                devices=devices,
                dtype=dtype,
                fp8_mode=resolved_fp8_mode,
            )
            break
        except RuntimeError as exc:
            if resolved_fp8_mode == "transformer-engine":
                notes.append(f"Transformer Engine FP8 unavailable ({exc})")
                resolved_fp8_mode = "none"
                continue
            raise

    if resolved_fp8_mode == "transformer-engine":
        try:
            recipe = make_delayed_scaling_recipe()
            precision_ctx_factory = lambda: te_fp8_autocast(enabled=True, recipe=recipe)
        except TransformerEngineUnavailable as exc:  # pragma: no cover - config dependent
            notes.append(f"TE autocast fallback ({exc})")
            resolved_fp8_mode = "none"
            model = GPTModel(
                config,
                devices=devices,
                dtype=dtype,
                fp8_mode=resolved_fp8_mode,
            )

    if precision_ctx_factory is None:
        precision_ctx_factory = contextlib.nullcontext

    notes.append(f"Precision mode: {resolved_fp8_mode}")
    notes.append(f"Attention backend: {config.attention_backend}")

    model.eval()
    for dev in devices:
        if dev.type == "cuda":
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()

    eager_result: Optional[Dict[str, float]] = None
    compiled_result: Optional[Dict[str, float]] = None

    try:
        eager_result = benchmark_model(model, x, warmup, iters, devices, precision_ctx_factory)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            notes.append("Eager: OOM")
        else:
            raise

    if not skip_compile and eager_result is not None:
        compiled_model = compile_model(
            model,
            mode=compile_mode,
        )
        try:
            compiled_result = benchmark_model(
                compiled_model,
                x,
                warmup,
                iters,
                devices,
                precision_ctx_factory,
            )
        except RuntimeError as exc:
            message = str(exc).lower()
            if "out of memory" in message:
                raise RuntimeError("SKIPPED: compiled benchmark OOM") from exc
            raise
        finally:
            del compiled_model

    if model is not None:
        del model
    for dev in devices:
        if dev.type == "cuda":
            with torch.cuda.device(dev):
                torch.cuda.empty_cache()

    eager_ms = eager_result["avg_ms"] if eager_result else None
    eager_tps = eager_result["throughput"] if eager_result else None
    eager_peak_mem = eager_result["peak_mem_gb"] if eager_result else None
    compiled_ms = compiled_result["avg_ms"] if compiled_result else None
    compiled_tps = compiled_result["throughput"] if compiled_result else None
    compiled_peak_mem = (
        compiled_result["peak_mem_gb"] if compiled_result else None
    )
    speedup = (
        eager_ms / compiled_ms
        if eager_ms is not None and compiled_ms is not None and compiled_ms > 0
        else None
    )

    return BenchmarkResult(
        config=workload.description,
        batch=workload.batch,
        seq_len=workload.seq_len,
        attention_backend=config.attention_backend,
        precision_mode=resolved_fp8_mode,
        eager_ms=eager_ms,
        eager_tps=eager_tps,
        compiled_ms=compiled_ms,
        compiled_tps=compiled_tps,
        speedup=speedup,
        eager_peak_mem_gb=eager_peak_mem,
        compiled_peak_mem_gb=compiled_peak_mem,
        notes=notes,
    )


def validate_multi_gpu_equivalence(
    config: GPTConfig,
    devices: List[torch.device],
    dtype: torch.dtype,
) -> Optional[float]:
    """Sanity-check outputs between 1 GPU and tensor-parallel configuration."""
    if len(devices) < 2:
        return None

    sample_config = GPTConfig(
        vocab_size=min(config.vocab_size, 32000),
        n_layers=min(config.n_layers, 4),
        n_heads=config.n_heads,
        d_model=config.d_model,
        d_ff=config.d_ff,
        max_seq_len=min(config.max_seq_len, 1024),
        attention_backend=config.attention_backend,
        attention_window=config.attention_window,
    )
    seq_len = min(sample_config.max_seq_len, 256)
    batch = 2

    torch.manual_seed(0)
    if devices[0].type == "cuda":
        torch.cuda.manual_seed_all(0)
    single_device = [devices[0]]
    base_model = GPTModel(
        sample_config,
        devices=single_device,
        dtype=dtype,
        fp8_mode="none",
    )

    torch.manual_seed(0)
    if devices[0].type == "cuda":
        torch.cuda.manual_seed_all(0)
    multi_model = GPTModel(
        sample_config,
        devices=devices,
        dtype=dtype,
        fp8_mode="none",
    )
    multi_model.load_state_dict(base_model.state_dict())

    inputs = torch.randint(
        0,
        sample_config.vocab_size,
        (batch, seq_len),
        device=devices[0],
        dtype=torch.int32,
    )

    base_model.eval()
    multi_model.eval()
    with torch.inference_mode():
        ref = base_model(inputs)
        out = multi_model(inputs)
    diff = scalar_tensor_to_float(torch.max(torch.abs(ref.to(devices[0]) - out.to(devices[0]))))

    del base_model
    del multi_model
    if devices[0].type == "cuda":
        torch.cuda.empty_cache()
    return diff


def format_result(result: BenchmarkResult) -> str:
    eager_ms = f"{result.eager_ms:.2f} ms" if result.eager_ms else "n/a"
    eager_tps = f"{result.eager_tps:,.0f}" if result.eager_tps else "n/a"
    compiled_ms = (
        f"{result.compiled_ms:.2f} ms" if result.compiled_ms else "n/a"
    )
    compiled_tps = (
        f"{result.compiled_tps:,.0f}" if result.compiled_tps else "n/a"
    )
    speedup = f"{result.speedup:.2f}x" if result.speedup else "n/a"
    eager_mem = (
        f"{result.eager_peak_mem_gb:.2f} GB"
        if result.eager_peak_mem_gb
        else "n/a"
    )
    compiled_mem = (
        f"{result.compiled_peak_mem_gb:.2f} GB"
        if result.compiled_peak_mem_gb
        else "n/a"
    )
    notes = "; ".join(result.notes)
    return (
        f"{result.config:<28}"
        f"[attn={result.attention_backend:<9}] "
        f"[precision={result.precision_mode:<18}] "
        f"Eager: {eager_ms:<12} | {eager_tps:>10} tok/s | {eager_mem:<10}"
        f"  Compiled: {compiled_ms:<12} | {compiled_tps:>10} tok/s | {compiled_mem:<10}"
        f"  Speedup: {speedup:<6}  Notes: {notes}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark a large GPT-style model on NVIDIA B200 GPUs."
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device to run on (default: cuda).",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16"],
        help="Computation dtype.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=2,
        help="Number of warmup iterations per measurement.",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=4,
        help="Number of timed iterations per measurement.",
    )
    parser.add_argument(
        "--compile-mode",
        default="reduce-overhead",
        help="torch.compile mode to use (default: reduce-overhead).",
    )
    parser.add_argument(
        "--skip-compile",
        action="store_true",
        help="Run eager mode only.",
    )
    parser.add_argument(
        "--tensor-parallel-gpus",
        type=int,
        default=1,
        help="Number of GPUs to partition transformer layers across.",
    )
    parser.add_argument(
        "--fp8-weights",
        action="store_true",
        help="Quantize linear layers to FP8 with per-channel scaling (weight-only).",
    )
    parser.add_argument(
        "--fp8-mode",
        choices=["none", "weight-only", "transformer-engine", "auto"],
        default="auto",
        help="Select the FP8 execution mode.",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["sdpa", "flex", "auto"],
        default="auto",
        help="Attention implementation to use (Flex requires PyTorch 2.1+).",
    )
    parser.add_argument(
        "--attention-window",
        type=int,
        default=None,
        help="Optional sliding window (tokens) for attention when using Flex.",
    )
    parser.add_argument(
        "--validate-multi-gpu",
        action="store_true",
        help="Run a quick numerical equivalence check across tensor-parallel GPUs.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional path to write JSON results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but not available.")

    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    fp8_mode = args.fp8_mode
    if args.fp8_weights:
        if args.fp8_mode != "weight-only":
            print("[info] --fp8-weights flag detected; forcing fp8-mode=weight-only for compatibility.")
        fp8_mode = "weight-only"
    if args.tensor_parallel_gpus > 1 and args.device != "cuda":
        raise ValueError("tensor-parallel-gpus > 1 is only supported with CUDA devices.")
    if args.device == "cuda":
        available = torch.cuda.device_count()
        if available < args.tensor_parallel_gpus:
            raise ValueError(
                f"Requested {args.tensor_parallel_gpus} GPUs but only {available} available."
            )
        devices = [torch.device(f"cuda:{idx}") for idx in range(args.tensor_parallel_gpus)]
        torch.cuda.set_device(devices[0])
    else:
        devices = [torch.device(args.device)]

    if devices[0].type == "cuda":
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cudnn.benchmark = True
        enable_tf32()

    config = GPTConfig()
    config.attention_backend = args.attention_backend
    config.attention_window = args.attention_window
    config.max_seq_len = max(config.max_seq_len, 16384)
    total_params = count_parameters(config)
    param_gb = total_params * torch.finfo(dtype).bits / 8 / 1e9

    workloads = [
        Workload(8, 1024, "Batch=8, Seq=1024"),
        Workload(4, 2048, "Batch=4, Seq=2048"),
        Workload(2, 4096, "Batch=2, Seq=4096"),
        Workload(2, 8192, "Batch=2, Seq=8192"),
        Workload(1, 12288, "Batch=1, Seq=12K"),
        Workload(1, 16384, "Batch=1, Seq=16K"),
    ]

    print("=" * 96)
    print("GPT LARGE MODEL BENCHMARK")
    print("=" * 96)
    print(f"Parameters: {total_params / 1e9:.2f}B (~{param_gb:.2f} GB in {args.dtype})")
    print(f"Warmup: {args.warmup}  Iters: {args.iters}  Compile mode: {args.compile_mode}")
    device_summary = ", ".join(str(dev) for dev in devices)
    print(f"Devices: {device_summary}  Tensor-parallel GPUs: {len(devices)}")
    print(f"Dtype: {args.dtype}")
    window_msg = f"{config.attention_window} tokens" if config.attention_window else "full context"
    print(f"Attention backend: {config.attention_backend} (window={window_msg})")
    print(f"Requested FP8 mode: {fp8_mode}")
    if fp8_mode in {"transformer-engine", "auto"}:
        if transformer_engine_available():
            print("Transformer Engine: available")
        else:
            print(f"Transformer Engine: unavailable ({transformer_engine_warning()})")
    if args.skip_compile:
        print("torch.compile: disabled")
    print("=" * 96)

    if args.validate_multi_gpu and len(devices) > 1:
        try:
            diff = validate_multi_gpu_equivalence(config, devices, dtype)
            if diff is not None:
                print(f"Multi-GPU validation (max |Δ|): {diff:.3e}")
        except RuntimeError as exc:
            print(f"Multi-GPU validation failed: {exc}")

    results: List[BenchmarkResult] = []
    for workload in workloads:
        print(f"\n--- {workload.description} ---")
        try:
            result = run_workload(
                config,
                workload,
                devices=devices,
                dtype=dtype,
                warmup=args.warmup,
                iters=args.iters,
                compile_mode=args.compile_mode,
                skip_compile=args.skip_compile,
                fp8_mode=fp8_mode,
            )
        except RuntimeError as exc:
            print(f"Failed: {exc}")
            continue

        results.append(result)
        print(format_result(result))

    if args.output_json and results:

        payload = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "device": str(devices[0]),
            "devices": device_summary,
            "dtype": args.dtype,
            "warmup": args.warmup,
            "iters": args.iters,
            "compile_mode": None if args.skip_compile else args.compile_mode,
            "fp8_mode": fp8_mode,
            "attention_backend": config.attention_backend,
            "attention_window": config.attention_window,
            "tensor_parallel_gpus": len(devices),
            "parameters_billion": total_params / 1e9,
            "param_memory_gb": param_gb,
            "results": [asdict(result) for result in results],
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote results to {args.output_json}")


if __name__ == "__main__":
    main()
