"""Hierarchical Triton scan and carry propagation."""

import triton
import triton.language as tl


@triton.jit
def scan_tiles(input_ptr, output_ptr, totals_ptr, numel: tl.constexpr, block_size: tl.constexpr):
    tile = tl.program_id(0)
    offsets = tile * block_size + tl.arange(0, block_size)
    values = tl.load(input_ptr + offsets, offsets < numel, other=0)
    prefixes = tl.cumsum(values, axis=0)
    tl.store(output_ptr + offsets, prefixes, offsets < numel)
    tl.store(totals_ptr + tile, tl.sum(values, axis=0))


@triton.jit
def add_tile_carries(output_ptr, scanned_totals_ptr, numel: tl.constexpr, block_size: tl.constexpr):
    tile = tl.program_id(0)
    offsets = tile * block_size + tl.arange(0, block_size)
    carry = tl.load(scanned_totals_ptr + tile - 1, tile > 0, other=0)
    values = tl.load(output_ptr + offsets, offsets < numel, other=0)
    tl.store(output_ptr + offsets, values + carry, offsets < numel)
