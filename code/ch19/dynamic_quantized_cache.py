#!/usr/bin/env python3
"""Dynamic KV cache quantization helpers for Chapter 19."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from core.benchmark.utils import scalar_tensor_to_float

if torch.cuda.is_available():
    try:
        _CC_MAJOR, _CC_MINOR = torch.cuda.get_device_capability()
    except Exception:  # pragma: no cover - defensive
        _CC_MAJOR = 0
    _ARCH_PREFERS_BF16 = _CC_MAJOR >= 9
else:  # CPU-only fallback
    _ARCH_PREFERS_BF16 = False

try:
    from transformers.cache_utils import QuantizedCacheConfig
    _TRANSFORMERS_HAS_QUANTIZED = True
except Exception:  # pragma: no cover - transformers not installed
    QuantizedCacheConfig = None  # type: ignore[assignment]
    _TRANSFORMERS_HAS_QUANTIZED = False



SUPPORTED_BACKENDS = {"hqq", "quanto"}


@dataclass
class QuantizedCachePolicy:
    backend: str = "hqq"
    start_bits: int = 8
    fallback_bits: int = 4
    q_group_size: int = 64
    axis_key: int = 1
    axis_value: int = 1
    compute_dtype: torch.dtype = torch.bfloat16

    def __post_init__(self) -> None:
        backend_lc = self.backend.lower()
        if backend_lc not in SUPPORTED_BACKENDS:
            raise ValueError(f"backend must be one of {SUPPORTED_BACKENDS}, got {self.backend!r}")
        self.backend = backend_lc
        if self.compute_dtype not in {torch.bfloat16, torch.float16, torch.float32}:
            raise ValueError("compute_dtype must be bf16, fp16, or fp32")

    def to_config(self):
        payload: Dict[str, object] = {
            "backend": self.backend,
            "nbits": self.start_bits,
            "fallback_bits": self.fallback_bits,
            "q_group_size": self.q_group_size,
            "axis_key": self.axis_key,
            "axis_value": self.axis_value,
            "compute_dtype": self.compute_dtype,
        }
        if _TRANSFORMERS_HAS_QUANTIZED and QuantizedCacheConfig is not None:
            return QuantizedCacheConfig(**payload)  # type: ignore[call-arg]
        return payload


def make_quantized_cache_config(
    backend: str = "hqq",
    *,
    start_bits: int = 8,
    fallback_bits: int = 4,
    compute_dtype: Optional[torch.dtype] = None,
    q_group_size: int = 64,
) -> QuantizedCachePolicy:
    dtype = compute_dtype if compute_dtype is not None else (torch.bfloat16 if _ARCH_PREFERS_BF16 else torch.float16)
    return QuantizedCachePolicy(
        backend=backend,
        start_bits=start_bits,
        fallback_bits=fallback_bits,
        q_group_size=q_group_size,
        compute_dtype=dtype,
    )


def build_generation_kwargs(
    use_quantized_cache: bool,
    *,
    backend: str = "hqq",
    offload: bool = False,
) -> Dict[str, object]:
    if not use_quantized_cache:
        impl = "auto"
        cache_cfg = None
    else:
        impl = "quantized"
        cache_cfg = make_quantized_cache_config(backend).to_config()

    if offload:
        impl = "offloaded"

    kwargs: Dict[str, object] = {"cache_implementation": impl}
    if cache_cfg is not None:
        kwargs["cache_config"] = cache_cfg
    return kwargs


def demo_quantized_cache(device: torch.device, bits: int = 8) -> None:
    """
    Emulate QuantizedCache behaviour locally without Transformers.
    Uses signed INT8/INT4 quantisation depending on the supplied bits.
    """
    torch.manual_seed(0)
    vocab_proj = torch.randn(8, 16, 128, device=device, dtype=torch.float32)

    qmax = 127 if bits == 8 else 7
    scale = vocab_proj.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6) / qmax
    quant = torch.clamp((vocab_proj / scale).round(), -qmax, qmax)
    dequant = quant * scale
    max_error = scalar_tensor_to_float((vocab_proj - dequant).abs().max())
    print(f"[demo] {bits}-bit cache emulation max error: {max_error:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamic quantized KV cache demo.")
    parser.add_argument("--bits", type=int, default=8, choices=(8, 4))
    parser.add_argument("--offload", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = make_quantized_cache_config(start_bits=args.bits, fallback_bits=4)
    gen_kwargs = build_generation_kwargs(use_quantized_cache=True, backend=policy.backend, offload=args.offload)
    print("Generation kwargs:", gen_kwargs)

    demo_quantized_cache(device, bits=args.bits)


if __name__ == "__main__":
    main()
