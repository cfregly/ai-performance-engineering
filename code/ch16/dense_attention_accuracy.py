"""Shared numerical policy for the Chapter 16 dense-attention family."""


# The eager FP16 and fused SDPA paths execute the same projections and causal
# attention math. Early rows average over few keys and peak near 1.26. Rows past
# 2048 average over thousands of keys and stay below 0.04. The whole output has a
# standard deviation near 0.017. At the harness seed and the full shape on the CPU,
# the two paths differ by at most 4.9e-4, which uses about 16% of this bound. The
# bound rejects an all-zero output, one scaled by 0.9, and one whose last 64 rows
# are zero. The earlier (5e-2, 5e-2) accepted an output whose second half was zero.
DENSE_ATTENTION_OUTPUT_TOLERANCE = (2e-2, 2e-3)


__all__ = ["DENSE_ATTENTION_OUTPUT_TOLERANCE"]
