"""Common helpers for persistent decode lab."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
from typing import Optional, Tuple

import torch
from core.benchmark.verification import InputSignature, PrecisionFlags
from core.common.device_utils import require_cuda_device

resolve_device = partial(require_cuda_device, "CUDA device required for persistent decode lab")


def get_stream_priorities() -> Tuple[int, int]:
    """Return (low_priority, high_priority) for the current CUDA device."""
    low, high = torch.cuda.Stream.priority_range()
    return low, high


@dataclass
class DecodeProfile:
    tier: str
    block_k: int
    num_programs: int


@dataclass
class DecodeOptions:
    tier: str = "medium"  # small | medium | large
    quantization: str = "fp32"  # fp32 | fp16 | int4
    block_k: Optional[int] = None
    num_programs: Optional[int] = None


_PROFILE_BY_TIER = {
    "small": DecodeProfile(tier="small", block_k=32, num_programs=4),
    "medium": DecodeProfile(tier="medium", block_k=64, num_programs=8),
    "large": DecodeProfile(tier="large", block_k=64, num_programs=12),
}

_OPTIONS = DecodeOptions()


def set_decode_options(opts: DecodeOptions) -> None:
    """Update global decode options (set via CLI flags in right-sized benchmarks)."""
    global _OPTIONS
    _OPTIONS = opts


def get_decode_options() -> DecodeOptions:
    return _OPTIONS


def get_decode_profile() -> DecodeProfile:
    base = _PROFILE_BY_TIER.get(_OPTIONS.tier, _PROFILE_BY_TIER["medium"])
    prof = replace(base)
    if _OPTIONS.block_k is not None:
        prof.block_k = _OPTIONS.block_k
    if _OPTIONS.num_programs is not None:
        prof.num_programs = _OPTIONS.num_programs
    if prof.num_programs <= 0 or prof.block_k <= 0 or prof.block_k & (prof.block_k - 1):
        raise ValueError("num_programs must be positive and block_k a positive power of two")
    return prof


def resolve_shapes() -> Tuple[int, int, int]:
    """
    Resolve (batch, seq_len, head_dim) with tier-aware defaults.

    - small: lighter shapes to mimic a smaller decode slice
    - medium: default shapes (8 x 32) for continuity with earlier labs
    - large: heavier batch to mimic a bigger slice
    """
    if _OPTIONS.tier == "small":
        return 4, 32, 64
    if _OPTIONS.tier == "large":
        return 12, 64, 64
    return 8, 32, 64


@dataclass
class DecodeInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    work_seq_ids: torch.Tensor
    work_steps: torch.Tensor
    work_counter: torch.Tensor


def validate_decode_output(inputs: DecodeInputs) -> None:
    """Check every sequence/token against independent vectorized dot-times-V math."""
    actual = inputs.out
    if actual.shape != inputs.q.shape or not torch.isfinite(actual).all():
        raise AssertionError("Decode output has wrong shape or non-finite values")
    if any(actual.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
           for tensor in (inputs.q, inputs.k, inputs.v)):
        raise AssertionError("Decode output aliases an input")
    reference = ((inputs.q.float() * inputs.k.float()).sum(-1, keepdim=True) * inputs.v.float()).to(actual.dtype)
    # Retain the existing pair's numerical budget; the complete-output and
    # dropped-sequence checks must pass before its harness payload is accepted.
    for index in range(actual.shape[0]):
        if reference[index].count_nonzero() and not actual[index].count_nonzero():
            raise AssertionError(f"Decode sequence {index} was not computed")
    torch.testing.assert_close(actual, reference, rtol=0.1, atol=1.0)


def build_inputs(
    batch: int | torch.device | None = None,
    seq_len: int | None = None,
    head_dim: int | None = None,
    device: torch.device | None = None,
) -> DecodeInputs:
    """
    Construct input tensors for the persistent decode extension.

    Backwards compatible with previous signature:
    - build_inputs(device)
    - build_inputs(batch, seq_len, head_dim, device)
    """
    # Signature shim: if the first arg is a device, treat it as `device` only.
    if isinstance(batch, torch.device):
        device = batch
        batch = None
    if device is None:
        device = resolve_device()

    # Default shapes unless explicitly overridden
    if batch is None or seq_len is None or head_dim is None:
        batch, seq_len, head_dim = resolve_shapes()

    generator = torch.Generator(device=device).manual_seed(torch.initial_seed())

    quant = _OPTIONS.quantization.lower()
    dtype = torch.float32 if quant == "fp32" else torch.float16

    q = torch.randn(batch, seq_len, head_dim, device=device, dtype=dtype, generator=generator)
    k = torch.randn(batch, seq_len, head_dim, device=device, dtype=dtype, generator=generator)
    v = torch.randn(batch, seq_len, head_dim, device=device, dtype=dtype, generator=generator)

    if quant == "int4":
        q = _fake_int4(q)
        k = _fake_int4(k)
        v = _fake_int4(v)

    # Output buffer matches q/k/v shape for per-token decode writes.
    out = torch.zeros(batch, seq_len, head_dim, device=device, dtype=dtype)

    work_seq_ids = torch.arange(batch, device=device, dtype=torch.int32)
    work_steps = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
    work_counter = torch.zeros(1, device=device, dtype=torch.int32)

    return DecodeInputs(
        q=q,
        k=k,
        v=v,
        out=out,
        work_seq_ids=work_seq_ids,
        work_steps=work_steps,
        work_counter=work_counter,
    )


def tokens_per_iteration() -> float:
    batch, seq_len, _ = resolve_shapes()
    return float(batch * seq_len)


def build_decode_input_signature(
    *,
    batch: int,
    seq_len: int,
    head_dim: int,
    quantization: str,
) -> dict:
    """Publish a static workload signature without running the decode kernels."""
    quant = str(quantization).lower()
    use_fp16 = quant in {"fp16", "int4"}
    input_dtype = "float16" if use_fp16 else "float32"
    tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
    signature = InputSignature(
        batch_size=batch,
        shapes={
            "q": (batch, seq_len, head_dim),
            "k": (batch, seq_len, head_dim),
            "v": (batch, seq_len, head_dim),
            "output": (1, min(8, seq_len), head_dim),
        },
        dtypes={
            "q": input_dtype,
            "k": input_dtype,
            "v": input_dtype,
            "output": "float32",
        },
        parameter_count=0,
        precision_flags=PrecisionFlags(fp16=use_fp16, tf32=tf32_enabled),
    )
    signature.quantization_mode = quant
    return signature.to_dict()


def _fake_int4(t: torch.Tensor) -> torch.Tensor:
    """Toy fake-quantization to simulate INT4 decode pools."""
    max_abs = t.abs().amax()
    if max_abs == 0:
        return t.half()
    scale = max_abs / 7.0
    q = torch.round(t / scale).clamp(-8, 7)
    return (q * scale).half()
