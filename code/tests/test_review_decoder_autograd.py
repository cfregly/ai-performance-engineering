"""Exercise the real decoder projection with trainable weights and plain inputs."""

import pytest
import torch

from ch16.inference_optimizations_blackwell import OptimizedDecoderLayer
from core.utils.compile_utils import configure_tf32, restore_tf32


@pytest.fixture(autouse=True)
def preserve_ieee_reference_policy():
    precision = torch.get_float32_matmul_precision()
    previous = configure_tf32(enable_matmul=False, enable_cudnn=False)
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        restore_tf32(previous)
        torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_decoder_parameter_gradients_without_input_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires actual CUDA")
    layer = OptimizedDecoderLayer(32, 4, device=device, use_flex_attention=False)
    inputs = torch.randn(2, 5, 32, device=device)
    output = layer(inputs)
    assert output.requires_grad
    output.square().mean().backward()
    for projection in (layer.q_proj, layer.k_proj, layer.v_proj, layer.o_proj):
        assert projection.weight.grad is not None
        assert torch.isfinite(projection.weight.grad).all()
        assert torch.count_nonzero(projection.weight.grad) > 0
    with torch.no_grad():
        inference_output = layer(inputs)
    torch.testing.assert_close(inference_output, output.detach())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires actual CUDA compilation")
def test_compiled_decoder_dynamic_shapes_match_eager():
    layer = OptimizedDecoderLayer(64, 4, device="cuda", use_flex_attention=False)
    compiled = torch.compile(layer, dynamic=True, fullgraph=False)
    with torch.no_grad():
        for batch, length in ((2, 5), (3, 9), (2, 5)):
            inputs = torch.randn(batch, length, 64, device="cuda")
            expected = layer(inputs).clone()
            actual = compiled(inputs)
            torch.testing.assert_close(actual, expected)
