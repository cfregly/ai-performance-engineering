"""Shared numerical policy for the Chapter 16 dense-attention family."""


# The eager FP16 and fused SDPA paths execute the same projections and attention
# math. This matches the reviewed Chapter 10 FlashAttention bound and rejects an
# all-zero output for the workload's observed output scale.
DENSE_ATTENTION_OUTPUT_TOLERANCE = (5e-2, 5e-2)


__all__ = ["DENSE_ATTENTION_OUTPUT_TOLERANCE"]
