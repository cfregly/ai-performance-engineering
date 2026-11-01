"""The post's arithmetic, re-derived in pure Python (no GPU, no torch): what the GPU count is compared against."""

from fractions import Fraction

import pytest

from divide_not_reciprocal import constants as C
from divide_not_reciprocal.ulp import ordered_py


def test_fl_sixth_is_one_sixth_times_one_plus_two_to_minus_25():
    assert C.f32_bits(1.0 / 6.0) == 0x3E2AAAAB
    assert C.FL_SIXTH_REL_ERROR == Fraction(1, 2 ** 25)          # claims-allow: fl(1/6) = 1/6 x (1 + 2^-25)
    assert C.FL_SIXTH == C.round_f32(Fraction(1, 6))


def test_predicted_count_per_binade_is_the_residue_count():
    # 2^23 is not a multiple of 3, so "one third" is 2,796,202 or 2,796,203 per binade; the residue count says which.
    assert C.PREDICTED_FLIPS_PER_BINADE == 2_796_202
    assert C.PREDICTED_FLIPS_CENSUS == 2 * 251 * 2_796_202 == 1_403_693_404
    assert abs(C.PREDICTED_FLIPS_PER_BINADE - (1 << 23) / 3) < 1


@pytest.mark.parametrize("m", list(range(1 << 23, (1 << 23) + 3000, 7)) + list(range((1 << 24) - 3000, 1 << 24, 11)) + list(range(12_582_912 - 1500, 12_582_912 + 1500, 13)))
def test_landing_rule_matches_exact_rounding(m):
    """For an input M x 2^0 the rounded reciprocal multiply differs from the rounded quotient exactly when the
    derivation says it does: the quotient lands at 1/3 of a last-place unit."""
    q = C.round_f32(Fraction(m, 6))
    p = C.round_f32(Fraction(m) * C.FL_SIXTH)
    assert (p != q) == C.flips_predicted(m)
    if p != q:
        ulp = Fraction(1, 1 << 23) * Fraction(2) ** (q.numerator.bit_length() - q.denominator.bit_length())
        assert p > q                                             # away from zero
        assert p - q == ulp                                      # exactly the next positive representable value


def test_shift_is_a_quarter_to_a_half_ulp():
    # the exact product x * fl(1/6) sits above x/6 by (x/6) * 2^-25: significand/4 of the quotient's ulp, so [1/4, 1/2)
    for sig in (Fraction(1), Fraction(3, 2), Fraction(2) - Fraction(1, 1 << 23)):
        shift_ulps = sig * Fraction(1, 2 ** 25) / Fraction(1, 2 ** 23)
        assert Fraction(1, 4) <= shift_ulps < Fraction(1, 2)


def test_tolerance_over_ulp_is_84():
    assert round(C.TOLERANCE_OVER_ULP) == 84
    assert C.NORMAL_INPUTS == 4_261_412_864
    assert C.NORMAL_INPUTS_BELOW_CENSUS == 50_331_648
    assert C.NORMAL_INPUTS_WITH_SUBNORMAL_QUOTIENT == 41_943_040


def test_ordered_ints_make_neighbours_differ_by_one():
    one = C.f32_bits(1.0)
    assert ordered_py(one + 1) - ordered_py(one) == 1
    neg = C.f32_bits(-1.0)
    assert ordered_py(neg + 1) - ordered_py(neg) == -1             # the next pattern of a negative is farther from zero
    assert ordered_py(C.f32_bits(0.0)) == ordered_py(C.f32_bits(-0.0)) == 0
