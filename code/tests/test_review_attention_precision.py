"""Real CUDA coverage for attention probability accumulation precision."""

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_attention_preserves_small_probabilities(dtype, causal, seed):
    from labs.flashattention_gluon.flashattention_gluon_common import gluon_flash_attention

    generator = torch.Generator(device="cuda").manual_seed(seed)
    shape = (2, 3, 65, 40)
    q = torch.full(shape, 0.25, device="cuda", dtype=dtype)
    k = -torch.rand(shape, device="cuda", dtype=dtype, generator=generator) - 0.5
    v = torch.randn(shape, device="cuda", dtype=dtype, generator=generator)
    # Also exercise the documented strided output contract.
    out = torch.empty((2, 3, 65, 80), device="cuda", dtype=dtype)[..., ::2]
    with torch.nn.attention.sdpa_kernel([torch.nn.attention.SDPBackend.MATH]):
        # Harness imports can enable TF32 globally. FP64 provides a stable
        # reference even when the surrounding suite changes matmul precision.
        expected = F.scaled_dot_product_attention(
            q.double(), k.double(), v.double(), is_causal=causal
        ).to(dtype)
    actual = gluon_flash_attention(q, k, v, causal=causal, out=out)
    assert actual is out
    torch.testing.assert_close(actual, expected)
