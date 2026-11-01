"""Hand-built bit fixtures, independent of the classifier's implementation. No CUDA work."""
from fractions import Fraction

import pytest

torch = pytest.importorskip("torch")
from divide_not_reciprocal import constants as C
from divide_not_reciprocal.ulp import Tally, ordered, patterns, tally


def floats(bits):
    signed = [b if b < (1 << 31) else b - (1 << 32) for b in bits]
    return torch.tensor(signed, dtype=torch.int32).view(torch.float32)


@pytest.mark.parametrize("p,q,kind", [
    (0x80000001, 0x80000000, "away1"),  # -minsub versus -zero: the reported defect
    (0x80000000, 0x80000001, "toward1"),
    (0x00000001, 0x00000000, "away1"),
    (0x00000000, 0x00000001, "toward1"),
    (0x00000000, 0x80000000, "same"),
    (0x80000000, 0x00000000, "same"),
    (0x3F800001, 0x3F800000, "away1"),
    (0xBF800001, 0xBF800000, "away1"),
    (0x3F800000, 0x3F800001, "toward1"),
    (0xBF800000, 0xBF800001, "toward1"),
    (0x00000001, 0x80000001, "far"),
    (0x7F800000, 0x7F7FFFFF, "away1"),
    (0xFF800000, 0xFF7FFFFF, "away1"),
    (0x7FC00001, 0xFFC00002, "nan_both"),
    (0x7FC00001, 0x3F800000, "nan_one"),
])
def test_manual_classes(p, q, kind):
    t = tally(floats([p]), floats([q]), torch.ones(1, dtype=torch.float32))
    classes = ("same", "away1", "toward1", "far", "nan_both", "nan_one")
    assert {k: getattr(t, k) for k in classes} == {k: int(k == kind) for k in classes}
    assert sum(getattr(t, k) for k in classes) == t.inputs == 1


def test_bit_pattern_wrap_and_signed_zero_order():
    x = patterns((1 << 31) - 2, (1 << 31) + 3, "cpu")
    assert x.view(torch.int32).tolist() == [2147483646, 2147483647, -2147483648, -2147483647, -2147483646]
    assert ordered(floats([0x80000001, 0x80000000, 0, 1])).tolist() == [-1, 0, 0, 1]
    assert patterns(1 << 32, 1 << 32, "cpu").numel() == 0


@pytest.mark.parametrize("lo,hi", [(-1, 1), (0, (1 << 32) + 1), (2, 1), (0.0, 1), (False, 1)])
def test_pattern_bounds_reject_wrapping(lo, hi):
    with pytest.raises(ValueError):
        patterns(lo, hi, "cpu")


def test_shape_dtype_and_mask_contracts():
    x = torch.ones(2, dtype=torch.float32)
    for p, q, mask in [(x[:1], x, None), (x.double(), x, None), (x, x, torch.ones(2)),
                       (x, x, torch.ones(1, dtype=torch.bool))]:
        with pytest.raises(ValueError):
            tally(p, q, x, mask)
    with pytest.raises(ValueError, match="share a device"):
        tally(x, torch.empty(2, dtype=torch.float32, device="meta"), x)
    assert tally(x, x, x, torch.zeros(2, dtype=torch.bool)).inputs == 0


def test_undefined_gap_is_explicit_and_survives_aggregation():
    bad = tally(floats([0x7FC00001]), floats([0x3F800000]), torch.ones(1))
    assert bad.max_gap_tol is None and bad.isclose_fail == 1 and bad.nan_one == 1
    assert Tally(max_gap_tol=10).add(bad).max_gap_tol is None
    assert Tally(max_gap_tol=None).add(Tally(max_gap_tol=10)).max_gap_tol is None


@pytest.mark.parametrize("value", [Fraction(0), Fraction(-1), Fraction(2) ** -127, Fraction(2) ** 128])
def test_fraction_rounder_rejects_outside_documented_normal_domain(value):
    with pytest.raises(ValueError):
        C.round_f32(value)


def test_known_five_divided_by_six_pair():
    q = C.round_f32(Fraction(5, 6))
    p = C.round_f32(5 * C.FL_SIXTH)
    assert C.f32_bits(float(q)) == 0x3F555555
    assert C.f32_bits(float(p)) == 0x3F555556
