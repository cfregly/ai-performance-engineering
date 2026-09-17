"""Online row normalization without materialized intermediates."""

import triton
import triton.language as tl


@triton.jit
def online_softmax_rows(input_ptr, output_ptr, num_cols: tl.constexpr, block_size: tl.constexpr):
    row = tl.program_id(0)
    lane = tl.arange(0, block_size)
    maximum = tl.full((), float("-inf"), tl.float32)
    mass = tl.full((), 0.0, tl.float32)
    for start in range(tl.cdiv(num_cols, block_size)):
        offsets = start * block_size + lane
        values = tl.load(
            input_ptr + row * num_cols + offsets, offsets < num_cols, other=float("-inf")
        )
        new_maximum = tl.maximum(maximum, tl.max(values, axis=0))
        old_weight = tl.where(maximum == float("-inf"), 0.0, tl.exp(maximum - new_maximum))
        block_mass = tl.where(
            new_maximum == float("-inf"),
            0.0,
            tl.sum(tl.exp(values - new_maximum), axis=0),
        )
        mass = mass * old_weight + block_mass
        maximum = new_maximum
    for start in range(tl.cdiv(num_cols, block_size)):
        offsets = start * block_size + lane
        values = tl.load(
            input_ptr + row * num_cols + offsets, offsets < num_cols, other=float("-inf")
        )
        tl.store(
            output_ptr + row * num_cols + offsets,
            tl.exp(values - maximum) / mass,
            offsets < num_cols,
        )
