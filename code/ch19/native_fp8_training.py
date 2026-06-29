#!/usr/bin/env python3
"""Transformer Engine FP8 training demo for Chapter 19.

Highlights current (Oct 2025) best practices on NVIDIA Blackwell with
CUDA 13 + PyTorch 2.10:
  * Prefer BF16 when using AMP; rely on Transformer Engine (TE) for MXFP8/NVFP4 paths.
  * Switch to cudaMallocAsync allocator without respawning when possible.
  * Pair FP8 paths with torch.compile(mode="reduce-overhead") for steady decode shapes.
"""

from __future__ import annotations

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
from core.utils.compile_utils import compile_model
import contextlib
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ch19.arch_config import TORCH_COMPILE_KW



try:
    from transformer_engine.pytorch import Linear as TELinear
    from transformer_engine.pytorch import LayerNorm as TELayerNorm
    from transformer_engine.pytorch import fp8_autocast

    _TE_AVAILABLE = True
except Exception:  # pragma: no cover - allow CPU/dev boxes without TE
    _TE_AVAILABLE = False

    class _NullCtx(contextlib.ContextDecorator):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fp8_autocast(**_: object) -> contextlib.AbstractContextManager[None]:
        return _NullCtx()

    class TELinear(nn.Linear):  # type: ignore[misc]
        """Fallback shim so code paths remain import-safe."""

    class TELayerNorm(nn.LayerNorm):  # type: ignore[misc]
        """Fallback shim matching TE interface."""


def _amp_dtype(prefer_bfloat16: bool) -> torch.dtype:
    return torch.bfloat16 if prefer_bfloat16 else torch.float16


class FP8MLP(nn.Module):
    """Stacked MLP with optional FP8 (Transformer Engine) linear layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 4,
        *,
        use_fp8: bool = True,
    ) -> None:
        super().__init__()
        self.use_fp8 = bool(use_fp8 and _TE_AVAILABLE)

        def _linear(in_features: int, out_features: int) -> nn.Module:
            if self.use_fp8:
                return TELinear(in_features, out_features, bias=True)
            return nn.Linear(in_features, out_features, bias=True)

        layers: Iterable[nn.Module] = []
        layers = [
            _linear(input_dim, hidden_dim),
            TELayerNorm(hidden_dim) if self.use_fp8 else nn.LayerNorm(hidden_dim),
            nn.GELU(),
        ]
        for _ in range(num_layers - 2):
            layers.extend(
                [
                    _linear(hidden_dim, hidden_dim),
                    TELayerNorm(hidden_dim) if self.use_fp8 else nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                ]
            )
        layers.append(_linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fp8:
            # Transformer Engine exposes MXFP8 kernels when fp8_autocast is enabled.
            with fp8_autocast(enabled=True):
                return self.net(x)
        return self.net(x)


@dataclass
class BenchmarkResult:
    time_ms: float
    memory_mb: float
    loss: float


def _training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    target: torch.Tensor,
    *,
    amp_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else contextlib.nullcontext()
    )
    with autocast_ctx:
        output = model(inputs)
        loss = F.mse_loss(output, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach()


def benchmark_fp8_training() -> Dict[str, BenchmarkResult]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device required for FP8 benchmark.")

    device = torch.device("cuda")
    batch_size = 32
    seq_len = 512
    input_dim = 2048
    hidden_dim = 8192

    configs: Tuple[Tuple[str, bool, Optional[torch.dtype]], ...] = (
        ("BF16", False, torch.bfloat16),
        ("FP16", False, torch.float16),
        ("FP8", True, None),
    )

    results: Dict[str, BenchmarkResult] = {}

    for name, use_fp8, amp_dtype in configs:
        if use_fp8 and not _TE_AVAILABLE:
            print(f"[skip] Transformer Engine not available; skipping {name} benchmark.")
            continue

        model = FP8MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=input_dim,
            num_layers=4,
            use_fp8=use_fp8,
        ).to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

        inputs = torch.randn(batch_size, seq_len, input_dim, device=device)
        targets = torch.randn(batch_size, seq_len, input_dim, device=device)
        if amp_dtype is not None:
            inputs = inputs.to(amp_dtype)
            targets = targets.to(amp_dtype)
        loss_value_buffer = torch.empty(1, dtype=torch.float64, device=device)
        loss_tensor: torch.Tensor | None = None

        # Warmup
        for _ in range(5):
            loss_tensor = _training_step(
                model,
                optimizer,
                inputs,
                targets,
                amp_dtype=amp_dtype if not use_fp8 else None,
            )

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        current_stream = torch.cuda.current_stream(device)
        start.record(current_stream)
        for _ in range(20):
            loss_tensor = _training_step(
                model,
                optimizer,
                inputs,
                targets,
                amp_dtype=amp_dtype if not use_fp8 else None,
            )
        end.record(current_stream)
        end.synchronize()

        time_ms = start.elapsed_time(end) / 20.0
        memory_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
        if loss_tensor is None:
            raise RuntimeError("No FP8 training iterations ran")
        loss_value_buffer[0].copy_(loss_tensor)
        loss_val = float(loss_value_buffer.detach().cpu()[0])

        results[name] = BenchmarkResult(time_ms=time_ms, memory_mb=memory_mb, loss=loss_val)
        print(f"{name:<4}  time={time_ms:6.2f} ms | memory={memory_mb:7.1f} MB | loss={loss_val:.5f}")

        del model, optimizer, inputs, targets
        torch.cuda.empty_cache()

    return results


def demonstrate_fp8_compile() -> None:
    if not torch.cuda.is_available() or not _TE_AVAILABLE:
        print("[skip] FP8 compile demo requires CUDA device and Transformer Engine.")
        return

    device = torch.device("cuda")
    model = FP8MLP(2048, 8192, 2048, num_layers=4, use_fp8=True).to(device)
    compiled = compile_model(model, **TORCH_COMPILE_KW)

    sample = torch.randn(32, 512, 2048, device=device)
    for _ in range(4):
        compiled(sample)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream(device)
    start.record(current_stream)
    for _ in range(20):
        compiled(sample)
    end.record(current_stream)
    end.synchronize()
    print(f"FP8 + torch.compile throughput: {start.elapsed_time(end)/20.0:6.2f} ms/iter")


def main() -> None:
    print("=" * 80)
    print("Chapter 19 — Transformer Engine FP8 Benchmark")
    print("=" * 80)
    try:
        results = benchmark_fp8_training()
    except RuntimeError as exc:
        print(f"[error] {exc}")
        return

    if "FP8" in results and "BF16" in results:
        bf16 = results["BF16"]
        fp8 = results["FP8"]
        print("\nComparisons vs BF16:")
        print(f"  Throughput speedup: {bf16.time_ms / fp8.time_ms:5.2f}×")
        print(f"  Memory savings:     {(bf16.memory_mb - fp8.memory_mb) / bf16.memory_mb * 100.0:5.1f}%")

    demonstrate_fp8_compile()


if __name__ == "__main__":
    main()
