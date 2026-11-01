"""The post's inputs and predictions, each with the source or the short derivation it comes from.

Pure Python, so the derivation tests run anywhere. Nothing here is measured: the GPU modules measure and the
report puts each measurement next to the prediction stated here.
"""

from __future__ import annotations

import struct
from fractions import Fraction

DIVISOR = 6                                   # the post's case: hardsigmoid's x / 6 (torch-hardsig-docs)
RTOL = 1e-05                                  # torch.allclose default rtol (torch-allclose)
ATOL = 1e-08                                  # torch.allclose default atol (torch-allclose)
DIV_FULL_MAX_ULP = 2                          # PTX ISA: div.full.f32 maximum ulp error (ptx-div-full)
DIV_APPROX_MAX_ULP = 2                        # PTX bound when 2^-126 <= |divisor| <= 2^126. Divisor6 meets it (ptx-div-approx).
STORED_BITS = 23                              # float32: one implicit plus 23 stored significand bits (nv-fp-ieee)
EXP_BITS = 8
BIAS = 127
PATTERNS = 1 << 32                            # every float32 bit pattern (2^32)
NORMAL_BINADES_PER_SIGN = (1 << EXP_BITS) - 2  # exponents 1..254
NORMAL_INPUTS = 2 * NORMAL_BINADES_PER_SIGN * (1 << STORED_BITS)   # 4,261,412,864
# Census: finite |x| >= 2^-123, exponent fields 4..254 (251 whole binades per sign).
# Field3 significands >=1.5 also have normal quotients, outside this whole-binade census.
CENSUS_EXP_MIN = 4
CENSUS_BINADES_PER_SIGN = (1 << EXP_BITS) - 1 - CENSUS_EXP_MIN   # 251
NORMAL_INPUTS_BELOW_CENSUS = 2 * (CENSUS_EXP_MIN - 1) * (1 << STORED_BITS)  # 50,331,648
NORMAL_INPUTS_WITH_SUBNORMAL_QUOTIENT = 5 * (1 << STORED_BITS)  # 41,943,040, both signs


def f32_bits(x: float) -> int:
    return struct.unpack("<I", struct.pack("<f", x))[0]


def f32_from_bits(b: int) -> float:
    return struct.unpack("<f", struct.pack("<I", b & 0xFFFFFFFF))[0]


def round_f32(v: Fraction) -> Fraction:
    """A positive rational rounded to the nearest float32 (ties to even), as an exact rational. Normal range only."""
    if not Fraction(2) ** -126 <= v <= Fraction((1 << 24) - 1) * Fraction(2) ** 104:
        raise ValueError("round_f32 requires a positive value in the finite normal float32 range")
    e = 0
    while v >= 2:
        v /= 2
        e += 1
    while v < 1:
        v *= 2
        e -= 1
    scaled = v * (1 << STORED_BITS)              # significand in units of the last place
    q, r = divmod(scaled.numerator, scaled.denominator)
    half = Fraction(r, scaled.denominator)
    if half > Fraction(1, 2) or (half == Fraction(1, 2) and q % 2 == 1):
        q += 1
    return Fraction(q, 1 << STORED_BITS) * Fraction(2) ** e


FL_SIXTH = Fraction(f32_from_bits(f32_bits(1.0 / 6.0)))         # fl(1/6), the stored reciprocal
FL_SIXTH_REL_ERROR = FL_SIXTH / Fraction(1, DIVISOR) - 1          # 2^-25, upward


def landing(m: int) -> Fraction:
    """Where M/6 sits between two floats, in units of the quotient's last place, for a 24-bit integer M.

    One in three: x = M * 2^e, and x/6 in last-place units is 4M/3 (significand under 1.5, so the
    quotient is in the lower of its two possible binades) or 2M/3. The fractional part is 0, 1/3, or 2/3.
    """
    assert (1 << STORED_BITS) <= m < (1 << (STORED_BITS + 1))
    units = Fraction(4 * m, 3) if m < 3 * (1 << (STORED_BITS - 1)) else Fraction(2 * m, 3)
    return units - (units.numerator // units.denominator)


def flips_predicted(m: int) -> bool:
    """The derivation's rule: only the 1/3 landing spot is carried past the midpoint by a shift of 1/4 to 1/2."""
    return landing(m) == Fraction(1, 3)


def predicted_flips_per_binade() -> int:
    """How many of the 2^23 significands in one binade flip, by counting residue classes rather than inputs."""
    lo, mid, hi = 1 << STORED_BITS, 3 * (1 << (STORED_BITS - 1)), 1 << (STORED_BITS + 1)
    # 4M/3 has fractional part 1/3 when M = 1 (mod 3); 2M/3 has it when M = 2 (mod 3).
    def count(a: int, b: int, residue: int) -> int:
        first = a + (residue - a) % 3
        return 0 if first >= b else (b - 1 - first) // 3 + 1
    return count(lo, mid, 1) + count(mid, hi, 2)


PREDICTED_FLIPS_PER_BINADE = predicted_flips_per_binade()
PREDICTED_FLIPS_CENSUS = 2 * CENSUS_BINADES_PER_SIGN * PREDICTED_FLIPS_PER_BINADE
ULP_REL_MAX = 2.0 ** -STORED_BITS             # one ulp relative to the leading bit: 2^-23
TOLERANCE_OVER_ULP = RTOL / ULP_REL_MAX        # 83.9: rtol is 84 times the largest 1-ulp gap (84)
