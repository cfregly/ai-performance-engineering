"""Independent grouped GEMM oracle: decode E2M1 bytes, apply original scales, FP64 GEMM.

This deliberately does not import the custom kernel or its reordered-scale helpers.
The baseline includes dequantization and FP64 multiplication in its measured path;
it is an accuracy/performance reference, not the former conservative tcgen05 route.
"""

from __future__ import annotations

import torch


def dequantize_fp4(packed: torch.Tensor, scales: torch.Tensor, k: int) -> torch.Tensor:
    """Decode [rows, K/2, batch] storage with one E4M3 scale per 16 values."""
    if packed.ndim != 3 or k <= 0 or k % 2 or packed.shape[1] != k // 2:
        raise ValueError("Expected packed [rows, K/2, batch] with positive even K")
    if tuple(scales.shape) != (packed.shape[0], (k + 15) // 16, packed.shape[2]):
        raise ValueError("Original (unreordered) scale shape does not match FP4 input")
    raw = packed.view(torch.uint8)
    low, high = raw & 15, raw >> 4
    codes = torch.stack((low, high), dim=2).reshape(packed.shape[0], k, packed.shape[2])
    # E2M1 magnitudes are exactly representable in binary; sign is nibble bit 3.
    magnitudes = torch.tensor((0, .5, 1, 1.5, 2, 3, 4, 6), dtype=torch.float64, device=raw.device)
    values = magnitudes[(codes & 7).long()] * torch.where((codes & 8) != 0, -1.0, 1.0)
    scale_values = scales.to(device=raw.device, dtype=torch.float64).repeat_interleave(16, dim=1)[:, :k]
    return values * scale_values


def reference_group_gemm(data, *, write_output: bool = True) -> list[torch.Tensor]:
    abc, original_scales, _reordered, sizes = data[:4]
    if not (len(abc) == len(original_scales) == len(sizes)):
        raise ValueError("Grouped input lengths differ")
    outputs = []
    for (a, b, c), (sfa, sfb), (m, n, k, batches) in zip(abc, original_scales, sizes):
        if tuple(c.shape) != (m, n, batches):
            raise ValueError("C shape does not match problem size")
        a64 = dequantize_fp4(a, sfa, k).permute(2, 0, 1)
        b64 = dequantize_fp4(b, sfb, k).permute(2, 0, 1)
        expected = torch.bmm(a64, b64.transpose(1, 2)).permute(1, 2, 0).to(c.dtype)
        if write_output:
            c.copy_(expected)
            outputs.append(c)
        else:
            outputs.append(expected)
    return outputs


def prepare_reference(data_list):
    """Move original scales once; no kernel metadata or custom packing is shared."""
    return [(abc, [(sa.to(a.device), sb.to(b.device))
                   for (a, b, _), (sa, sb) in zip(abc, scales)], reordered, sizes)
            for abc, scales, reordered, sizes in data_list]


def assert_group_outputs(actual, expected) -> None:
    """Check every element against independent storage, including finite values."""
    if len(actual) != len(expected) or not actual:
        raise AssertionError("Grouped output count differs from independent reference")
    for index, (got, ref) in enumerate(zip(actual, expected)):
        if got.untyped_storage().data_ptr() == ref.untyped_storage().data_ptr():
            raise AssertionError(f"Group {index}: reference aliases candidate storage")
        if not torch.isfinite(got).all() or not torch.isfinite(ref).all():
            raise AssertionError(f"Group {index}: non-finite output/reference")
        # Existing FP16 output contract; GPU qualification of each route remains required.
        torch.testing.assert_close(got, ref, rtol=1e-3, atol=1e-3, check_dtype=True)
