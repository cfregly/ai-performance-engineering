"""Independent FP32 decode acceptance, including formerly tolerated corruption."""
from __future__ import annotations

import pytest
import torch

from labs.persistent_decode.persistent_decode_common import (
    DecodeInputs,
    validate_decode_output,
)


def inputs(q, k, v):
    output = ((q * k).sum(-1, keepdim=True) * v).clone()
    return DecodeInputs(q, k, v, output, torch.arange(q.shape[0]),
                        torch.full((q.shape[0],), q.shape[1]), torch.zeros(1))


@pytest.mark.parametrize("shape", [(4, 32, 64), (8, 32, 64), (12, 64, 64)])
def test_fp32_roundoff_budget_accepts_all_public_tiers(shape):
    generator = torch.Generator().manual_seed(7123)
    data = inputs(*(torch.randn(shape, generator=generator) for _ in range(3)))
    validate_decode_output(data)


def test_fp32_guard_rejects_small_tail_corruption_accepted_by_old_tolerance():
    q = torch.ones(2, 9, 64)
    data = inputs(q, q.clone(), torch.full_like(q, 0.125))
    reference = data.out.clone()
    data.out[-1, -1, -1] += 0.25
    torch.testing.assert_close(data.out, reference, rtol=0.1, atol=1.0)
    with pytest.raises(AssertionError, match="independent FP32 roundoff budget"):
        validate_decode_output(data)


def test_fp32_cancellation_uses_sum_of_absolute_products():
    q = torch.tensor([1e8, 1.0, -1e8, 1.0] * 16).reshape(1, 1, 64)
    data = inputs(q, torch.ones_like(q), torch.ones_like(q))
    # A legal FP32 serial reduction loses the unit after +1e8 on every group.
    total = torch.zeros(())
    for value in q.reshape(-1):
        total += value
    data.out.fill_(total.item())
    validate_decode_output(data)


@pytest.mark.parametrize("field", ["q", "k", "v"])
def test_fp32_guard_rejects_nonfinite_input_even_with_finite_output(field):
    q = torch.ones(1, 2, 64)
    data = inputs(q, q.clone(), q.clone())
    getattr(data, field)[0, 0, -1] = float("inf")
    with pytest.raises(AssertionError, match="Decode input has non-finite"):
        validate_decode_output(data)
