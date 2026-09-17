"""Fuse BF16 transform, E4M3 scale construction, E2M1 packing and scale stores."""

import triton
import triton.language as tl


@triton.jit
def fused_quantize(
    input_ptr,
    residual_ptr,
    weight_ptr,
    global_ptr,
    valid_rows_ptr,
    packed_ptr,
    scales_ptr,
    residual_out_ptr,
    num_rows: tl.constexpr,
    num_cols: tl.constexpr,
    kind: tl.constexpr,
    padded_rows: tl.constexpr,
    padded_groups: tl.constexpr,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    index = tl.program_id(0)
    batch = index // num_rows
    row = index % num_rows
    offsets = tl.arange(0, block_size)
    active = row < tl.load(valid_rows_ptr + batch)
    input_width = num_cols * 2 if kind == "silu_mul" else num_cols
    x = tl.load(
        input_ptr + index * input_width + offsets, (offsets < num_cols) & active, other=0
    ).to(tl.float32)
    if kind == "add_rmsnorm":
        r = tl.load(
            residual_ptr + index * num_cols + offsets, (offsets < num_cols) & active, other=0
        ).to(tl.float32)
        h = (x + r).to(tl.bfloat16).to(tl.float32)
        tl.store(residual_out_ptr + index * num_cols + offsets, h, offsets < num_cols)
        variance = tl.sum(h * h, axis=0) / num_cols
        weight = tl.load(weight_ptr + offsets, offsets < num_cols, other=0).to(tl.float32)
        values = (h * tl.rsqrt(variance + eps) * weight).to(tl.bfloat16).to(tl.float32)
    elif kind == "silu_mul":
        up = tl.load(
            input_ptr + index * input_width + num_cols + offsets,
            (offsets < num_cols) & active,
            other=0,
        ).to(tl.float32)
        values = (x / (1.0 + tl.exp(-x)) * up).to(tl.bfloat16).to(tl.float32)
    else:
        values = x
    blocks = tl.reshape(values, (block_size // 16, 16))
    multiplier = tl.load(global_ptr + batch)
    scale = (
        (tl.max(tl.abs(blocks), axis=1) * (1.0 / 6.0) * multiplier).to(tl.float8e4nv).to(tl.float32)
    )
    inverse = tl.where(scale == 0, 0.0, multiplier / scale)
    normalized = tl.reshape(blocks * inverse[:, None], (block_size,))
    a = tl.abs(normalized)
    code = (
        (a > 0.25).to(tl.int32)
        + (a >= 0.75).to(tl.int32)
        + (a > 1.25).to(tl.int32)
        + (a >= 1.75).to(tl.int32)
        + (a > 2.5).to(tl.int32)
        + (a >= 3.5).to(tl.int32)
        + (a > 5.0).to(tl.int32)
    )
    sign = (normalized.to(tl.int32, bitcast=True) >> 31) & 1
    code = code | (sign << 3)
    pairs = tl.reshape(code, (block_size // 2, 2))
    packed = tl.sum(pairs << (tl.arange(0, 2)[None, :] * 4), axis=1).to(tl.uint8)
    byte_offsets = tl.arange(0, block_size // 2)
    tl.store(
        packed_ptr + index * (num_cols // 2) + byte_offsets, packed, byte_offsets < num_cols // 2
    )
    groups = tl.arange(0, block_size // 16)
    if kind == "silu_mul":
        scale_offset = (
            batch * padded_rows * padded_groups
            + (row // 128) * (padded_groups // 4) * 512
            + (groups // 4) * 512
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + groups % 4
        )
    else:
        scale_offset = index * (num_cols // 16) + groups
    scale_bytes = scale.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(scales_ptr + scale_offset, scale_bytes, groups < num_cols // 16)
