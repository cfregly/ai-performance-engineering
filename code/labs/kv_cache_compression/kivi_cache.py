"""KIVI-style asymmetric 2-bit KV storage with a BF16/FP16 residual tail.

Keys are grouped along tokens per channel. Values are grouped along channels
per token. The recent-token tail stays at input precision. Packed and unpacked
variants use identical quantization; only code storage differs between them.
"""

from dataclasses import dataclass

import torch


@dataclass
class AffineGroups:
    codes: torch.Tensor
    minimum: torch.Tensor
    scale: torch.Tensor
    packed: bool

    def decode(self):
        if self.packed:
            codes = torch.stack(
                tuple((self.codes >> shift) & 3 for shift in (0, 2, 4, 6)), dim=-1
            ).flatten(-2)
        else:
            codes = self.codes
        return codes.float() * self.scale + self.minimum

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.codes, self.minimum, self.scale))


def quantize_groups(values, *, packed):
    if values.shape[-1] % 4 or not values.is_floating_point():
        raise ValueError("quantization groups must be floating and divisible by four")
    x = values.float()
    minimum, maximum = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    scale = (maximum - minimum) / 3
    # Constant groups use code zero and preserve their value in the minimum.
    inverse = torch.where(scale > 0, scale.reciprocal(), 0.0)
    codes = ((x - minimum) * inverse).round().clamp(0, 3).to(torch.uint8)
    if packed:
        codes = (
            codes[..., 0::4]
            | (codes[..., 1::4] << 2)
            | (codes[..., 2::4] << 4)
            | (codes[..., 3::4] << 6)
        )
    return AffineGroups(codes, minimum, scale, packed)


@dataclass
class KiviCache:
    keys: AffineGroups | None
    values: AffineGroups | None
    key_tail: torch.Tensor
    value_tail: torch.Tensor
    shape: tuple[int, int, int, int]
    prefix_tokens: int
    group_size: int

    def decode(self):
        batch, heads, _, dim = self.shape
        if self.prefix_tokens == 0:
            return self.key_tail.clone(), self.value_tail.clone()
        key_prefix = (
            self.keys.decode().transpose(-1, -2).reshape(batch, heads, self.prefix_tokens, dim)
        )
        value_prefix = self.values.decode().reshape(batch, heads, self.prefix_tokens, dim)
        return (
            torch.cat((key_prefix.to(self.key_tail.dtype), self.key_tail), dim=2),
            torch.cat((value_prefix.to(self.value_tail.dtype), self.value_tail), dim=2),
        )

    @property
    def nbytes(self):
        tails = (
            self.key_tail.numel() * self.key_tail.element_size()
            + self.value_tail.numel() * self.value_tail.element_size()
        )
        return tails + (self.keys.nbytes + self.values.nbytes if self.keys is not None else 0)


@torch.no_grad()
def encode_cache(keys, values, *, group_size=32, residual_tokens=32, packed=True):
    if (
        keys.ndim != 4
        or keys.shape != values.shape
        or keys.dtype != values.dtype
        or keys.device != values.device
    ):
        raise ValueError(
            "keys and values must have matching [batch, heads, tokens, dim] shapes, dtype and device"
        )
    if keys.dtype not in (torch.bfloat16, torch.float16) or min(keys.shape) <= 0:
        raise ValueError("nonempty BF16 or FP16 KV tensors are required")
    if (
        type(group_size) is not int
        or group_size <= 0
        or group_size % 4
        or keys.shape[-1] % group_size
    ):
        raise ValueError("group_size must be positive, divisible by four, and divide head_dim")
    if type(residual_tokens) is not int or residual_tokens < 0:
        raise ValueError("residual_tokens must be nonnegative")
    batch, heads, tokens, dim = keys.shape
    prefix = max(0, (tokens - residual_tokens) // group_size * group_size)
    key_groups = value_groups = None
    if prefix:
        key_groups = quantize_groups(
            keys[:, :, :prefix, :]
            .reshape(batch, heads, prefix // group_size, group_size, dim)
            .transpose(-1, -2),
            packed=packed,
        )
        value_groups = quantize_groups(
            values[:, :, :prefix, :].reshape(batch, heads, prefix, dim // group_size, group_size),
            packed=packed,
        )
    # Views would retain the entire original storage and defeat compression.
    return KiviCache(
        key_groups,
        value_groups,
        keys[:, :, prefix:, :].clone(),
        values[:, :, prefix:, :].clone(),
        tuple(keys.shape),
        prefix,
        group_size,
    )
