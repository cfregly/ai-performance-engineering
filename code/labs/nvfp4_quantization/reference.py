"""Executable NVFP4 format reference, with real packed values and block scales.

E2M1 uses round-to-nearest/even and two nibbles per byte (even column low).
Each group of 16 values has an E4M3FN scale. The tensor multiplier g gives
dequantized values = E2M1 * E4M3 / g. Scale storage can be linear or 128x4 tiled.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as functional

from core.utils.nvfp4_layout import to_blocked


@dataclass
class Quantized:
    packed: torch.Tensor
    scales: torch.Tensor
    shape: tuple[int, int, int]
    swizzled: bool
    residual: torch.Tensor | None = None


def encode_e2m1(values):
    magnitude = values.abs()
    codes = (
        (magnitude > 0.25).to(torch.uint8)
        + (magnitude >= 0.75).to(torch.uint8)
        + (magnitude > 1.25).to(torch.uint8)
        + (magnitude >= 1.75).to(torch.uint8)
        + (magnitude > 2.5).to(torch.uint8)
        + (magnitude >= 3.5).to(torch.uint8)
        + (magnitude > 5).to(torch.uint8)
    )
    return codes | (torch.signbit(values).to(torch.uint8) << 3)


def pack_e2m1(values):
    if values.shape[-1] % 2:
        raise ValueError("E2M1 packing requires an even final dimension")
    codes = encode_e2m1(values)
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_e2m1(packed):
    if packed.dtype != torch.uint8:
        raise ValueError("packed E2M1 storage must be uint8")
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2)
    levels = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=packed.device)
    return levels[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)


def swizzle_scales(scales):
    if scales.ndim != 3 or scales.dtype != torch.uint8:
        raise ValueError("scale storage must be [batch, row, group] uint8")
    batch, rows, groups = scales.shape
    padded_rows, padded_groups = (rows + 127) // 128 * 128, (groups + 3) // 4 * 4
    padded = functional.pad(scales, (0, padded_groups - groups, 0, padded_rows - rows))
    return to_blocked(padded).reshape(batch, padded_rows, padded_groups)


def unswizzle_scales(scales, rows, groups):
    batch, padded_rows, padded_groups = scales.shape
    if (
        scales.dtype != torch.uint8
        or padded_rows % 128
        or padded_groups % 4
        or not (0 < rows <= padded_rows and 0 < groups <= padded_groups)
    ):
        raise ValueError("invalid 128x4 scale layout or logical shape")
    tiles = scales.reshape(batch, padded_rows // 128, padded_groups // 4, 32, 4, 4)
    return (
        tiles.permute(0, 1, 4, 3, 2, 5)
        .reshape(batch, padded_rows, padded_groups)[:, :rows, :groups]
        .contiguous()
    )


def quantize_blocks(values, global_scale, *, swizzled=False, valid_rows=None):
    if (
        values.ndim != 3
        or values.shape[-1] % 16
        or min(values.shape) <= 0
        or not values.is_floating_point()
    ):
        raise ValueError(
            "NVFP4 input must be a nonempty floating [batch, row, K] tensor with K divisible by 16"
        )
    batch, rows, cols = values.shape
    if (
        global_scale.shape != (batch,)
        or global_scale.dtype != torch.float32
        or global_scale.device != values.device
    ):
        raise ValueError(
            "global_scale must be a same-device float32 vector, one multiplier per batch"
        )
    if valid_rows is not None:
        if (
            valid_rows.shape != (batch,)
            or valid_rows.dtype != torch.int32
            or valid_rows.device != values.device
        ):
            raise ValueError("valid_rows must be a same-device int32 vector, one count per batch")
        values = torch.where(
            torch.arange(rows, device=values.device)[None, :, None] < valid_rows[:, None, None],
            values,
            0.0,
        )
    blocks = values.float().reshape(batch, rows, cols // 16, 16)
    multiplier = global_scale[:, None, None]
    scales = (blocks.abs().amax(-1) * (1.0 / 6.0) * multiplier).to(torch.float8_e4m3fn)
    scale_values = scales.float()
    inverse = torch.where(scale_values == 0, 0.0, multiplier / scale_values)
    normalized = blocks * inverse[..., None]
    packed = pack_e2m1(normalized.reshape(batch, rows, cols))
    scale_bytes = scales.view(torch.uint8)
    return Quantized(
        packed,
        swizzle_scales(scale_bytes) if swizzled else scale_bytes,
        (batch, rows, cols),
        swizzled,
    )


def dequantize(result: Quantized, global_scale):
    batch, rows, cols = result.shape
    scales = unswizzle_scales(result.scales, rows, cols // 16) if result.swizzled else result.scales
    factors = scales.view(torch.float8_e4m3fn).float() / global_scale[:, None, None]
    return (
        unpack_e2m1(result.packed)
        .reshape(batch, rows, cols // 16, 16)
        .mul(factors[..., None])
        .reshape(result.shape)
    )


def evaluate_reference(
    kind, x, global_scale, *, residual=None, weight=None, valid_rows=None, eps=1e-6
):
    if x.dtype != torch.bfloat16:
        raise ValueError("task inputs use BF16")
    residual_output = None
    if kind == "add_rmsnorm":
        if (
            residual is None
            or residual.shape != x.shape
            or residual.dtype != x.dtype
            or weight is None
            or weight.shape != (x.shape[-1],)
            or weight.dtype != x.dtype
        ):
            raise ValueError("add_rmsnorm requires matching BF16 residual and row weight")
        residual_output = x + residual
        h = residual_output.float()
        normalized = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps) * weight.float()
        values = normalized.to(x.dtype)
    elif kind == "silu_mul":
        if x.shape[-1] % 32:
            raise ValueError("SiLU/multiply input width must be divisible by 32")
        gate, up = x.float().chunk(2, dim=-1)
        values = (functional.silu(gate) * up).to(x.dtype)
    elif kind == "quantize":
        values = x
    else:
        raise ValueError(f"unknown NVFP4 operation: {kind}")
    result = quantize_blocks(
        values, global_scale, swizzled=kind == "silu_mul", valid_rows=valid_rows
    )
    result.residual = residual_output
    return result
