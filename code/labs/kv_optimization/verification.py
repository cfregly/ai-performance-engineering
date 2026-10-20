"""Accuracy bounds shared by the standard KV-cache benchmark pair."""

# E4M3 has three explicit mantissa bits, so normal values have at most 1/16
# relative rounding error. The small absolute term covers values that scale into
# the E4M3 subnormal range, where the spacing is a fixed 2**-9 in scaled units
# instead of a fraction of the value. It still rejects order-one errors in
# N(0, 1) cache data.
FP8_KV_OUTPUT_TOLERANCE = (1.0 / 16.0, 2.0e-3)


__all__ = ["FP8_KV_OUTPUT_TOLERANCE"]
