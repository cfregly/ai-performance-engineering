"""Retirement contracts and real references; CUDA/Triton execution stays gated."""
from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from labs.moe_optimization_journey.level4_triton import GroupedMoEExperts
from labs.moe_optimization_journey import triton_fused_moe as retired
from labs.moe_optimization_journey import triton_kernels


LAB = Path(__file__).resolve().parents[1] / "labs" / "moe_optimization_journey"


def test_retired_cli_reports_nonacceptance_without_a_benchmark_result():
    result = subprocess.run([sys.executable, str(LAB / "triton_fused_moe.py")],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 3
    assert result.stdout == ""
    assert "Retired incomplete Triton MoE" in result.stderr
    assert "not a fused Triton FFN" in result.stderr
    assert "TFLOPS" not in result.stderr and "ms =" not in result.stderr


def test_removed_moe_kernel_has_no_launchable_compatibility_alias():
    assert not hasattr(retired, "fused_moe_expert_kernel")
    with pytest.raises(retired.RetiredMoEKernelError, match="No kernel or benchmark result"):
        retired.triton_fused_moe(None, None, None, None, None, None, 8, 4096, 11008)


def test_unused_raw_experiments_are_absent_from_the_active_module_source():
    # Source/API inventory only. No Triton module or device is simulated here.
    module = ast.parse((LAB / "triton_kernels.py").read_text())
    definitions = {node.name for node in ast.walk(module) if isinstance(node, ast.FunctionDef)}
    assert "fused_expert_ffn_kernel" not in definitions
    assert "grouped_gemm_kernel" not in definitions
    assert {"fused_silu_mul_kernel", "fused_silu_mul", "fused_gate_up_proj"} <= definitions
    assert not hasattr(triton_kernels, "fused_expert_ffn_kernel")
    assert not hasattr(triton_kernels, "grouped_gemm_kernel")


@pytest.mark.parametrize("gate_grad,up_grad", [(True, False), (False, True), (True, True)])
def test_cpu_activation_preserves_requested_gradients_against_analytic_derivative(gate_grad, up_grad):
    gate = torch.linspace(-3, 3, 15, dtype=torch.float64).reshape(3, 5).t().detach().requires_grad_(gate_grad)
    up = torch.linspace(-1, 2, 15, dtype=torch.float64).reshape(3, 5).t().detach().requires_grad_(up_grad)
    coefficient = torch.linspace(-0.5, 1.5, 15, dtype=torch.float64).reshape(5, 3)
    assert not gate.is_contiguous() and not up.is_contiguous()
    assert triton_kernels.fused_silu_mul_backend(gate, up) == "pytorch_autograd"
    output = triton_kernels.fused_silu_mul(gate, up)
    sigmoid = gate.detach().sigmoid()
    torch.testing.assert_close(output, gate.detach() * sigmoid * up.detach(), rtol=1e-12, atol=1e-12)
    (output * coefficient).sum().backward()
    if gate_grad:
        expected = coefficient * up.detach() * sigmoid * (1 + gate.detach() * (1 - sigmoid))
        torch.testing.assert_close(gate.grad, expected, rtol=1e-12, atol=1e-12)
    else:
        assert gate.grad is None
    if up_grad:
        torch.testing.assert_close(up.grad, coefficient * gate.detach() * sigmoid, rtol=1e-12, atol=1e-12)
    else:
        assert up.grad is None


def test_cpu_activation_aliases_accumulate_gradients_without_mutating_storage():
    original = torch.linspace(-2, 2, 20, dtype=torch.float64).reshape(4, 5)
    base = original.clone().requires_grad_()
    reference = original.clone().requires_grad_()
    output = triton_kernels.fused_silu_mul(base[:, :-1], base[:, 1:])
    expected = F.silu(reference[:, :-1]) * reference[:, 1:]
    output.square().sum().backward()
    expected.square().sum().backward()
    torch.testing.assert_close(base.grad, reference.grad, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(base, original, rtol=0, atol=0)


@pytest.mark.parametrize("shape", [(0,), (3, 0), (2, 5)])
def test_cpu_activation_keeps_explicit_fallback_and_empty_shapes(shape):
    from labs.moe_optimization_journey import moe_model

    gate, up = torch.ones(shape, dtype=torch.float64), torch.full(shape, 2.0, dtype=torch.float64)
    assert moe_model.TRITON_AVAILABLE is triton_kernels.TRITON_AVAILABLE
    assert triton_kernels.fused_silu_mul_backend(gate, up) == "pytorch_cpu"
    output = triton_kernels.fused_silu_mul(gate, up)
    torch.testing.assert_close(output, F.silu(gate) * up)
    assert output.shape == gate.shape and output.dtype == gate.dtype


@pytest.mark.parametrize("kind,error,message", [
    ("shape", ValueError, "matching shapes"),
    ("dtype", TypeError, "matching dtypes"),
    ("device", ValueError, "same device"),
    ("integer", TypeError, "floating point"),
    ("sparse", ValueError, "strided tensor layouts"),
])
def test_activation_rejects_invalid_pairs_before_backend_execution(kind, error, message):
    gate = torch.ones(2, 3)
    up = gate.clone()
    if kind == "shape":
        up = torch.ones(2, 4)
    elif kind == "dtype":
        up = up.double()
    elif kind == "device":
        up = torch.empty(2, 3, device="meta")  # Real metadata tensor, not fake CUDA.
    elif kind == "integer":
        gate, up = gate.long(), up.long()
    elif kind == "sparse":
        gate, up = gate.to_sparse(), up.to_sparse()
    with pytest.raises(error, match=message):
        triton_kernels.fused_silu_mul(gate, up)


def test_cpu_full_fused_model_path_retains_projection_and_input_gradients():
    from labs.moe_optimization_journey.moe_model import MoEExperts, MoEOptimizations

    torch.manual_seed(991)
    model = MoEExperts(3, 5, 9, MoEOptimizations(use_fused=True)).double()
    x = torch.randn(4, 5, dtype=torch.float64, requires_grad=True)
    indices = torch.tensor([[2, 0], [0, 2], [2, 0], [0, 2]])
    weights = torch.tensor([[0.25, 0.75]] * 4, dtype=torch.float64, requires_grad=True)
    original_x = x.detach().clone()
    ref_x, ref_weights = x.detach().clone().requires_grad_(), weights.detach().clone().requires_grad_()
    ref_w1 = model.w1_stacked.detach().clone().requires_grad_()
    ref_w2 = model.w2_stacked.detach().clone().requires_grad_()
    ref_w3 = model.w3_stacked.detach().clone().requires_grad_()
    rows = []
    for token in range(len(x)):
        contributions = []
        for route in range(2):
            expert = int(indices[token, route])
            hidden = F.silu(ref_x[token] @ ref_w1[expert]) * (ref_x[token] @ ref_w3[expert])
            contributions.append((hidden @ ref_w2[expert]) * ref_weights[token, route])
        rows.append(sum(contributions))
    expected = torch.stack(rows)
    output = model(x, indices, weights, num_experts_per_tok=2)
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    output.square().sum().backward()
    expected.square().sum().backward()
    for actual, reference in zip((x, weights, model.w1_stacked, model.w2_stacked, model.w3_stacked),
                                 (ref_x, ref_weights, ref_w1, ref_w2, ref_w3), strict=True):
        torch.testing.assert_close(actual.grad, reference.grad, rtol=1e-12, atol=1e-12)
    assert x.grad is not None and torch.count_nonzero(x.grad) > 0
    assert weights.grad is not None and torch.count_nonzero(weights.grad) > 0
    for parameter in (model.w1_stacked, model.w2_stacked, model.w3_stacked):
        assert parameter.grad is not None
        assert torch.count_nonzero(parameter.grad[[0, 2]]) > 0
        assert torch.count_nonzero(parameter.grad[1]) == 0
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)


def _tail_case(hidden, intermediate, *, device="cpu", dtype=torch.float64):
    """Only the final hidden/intermediate columns contribute; expert 1 is empty."""
    layer = GroupedMoEExperts(3, hidden, intermediate).to(device=device, dtype=dtype).eval()
    with torch.no_grad():
        layer.w1.zero_()
        layer.w2.zero_()
        layer.w3.zero_()
        for expert in range(3):
            layer.w1[expert, -1, -1] = expert + 1
            layer.w3[expert, -1, -1] = expert + 2
            layer.w2[expert, -1, 0] = expert + 3
            layer.w2[expert, -1, -1] = expert + 4
    x = torch.zeros(5, hidden, device=device, dtype=dtype)
    x[:, -1] = torch.arange(1, 6, device=device, dtype=dtype) / 4
    indices = torch.tensor([[2, 0], [0, 2], [2, 0], [2, 0], [0, 2]], device=device)
    weights = torch.tensor([[0.25, 0.75], [0.5, 0.5], [0.75, 0.25], [0.25, 0.75], [0.5, 0.5]],
                           device=device, dtype=dtype)
    # Independent scalar oracle for this sparse fixture, evaluated on CPU FP64.
    expected = torch.zeros(5, hidden, dtype=torch.float64)
    for token in range(5):
        value = torch.tensor((token + 1) / 4, dtype=torch.float64)
        for route in range(2):
            expert = int(indices[token, route].cpu())
            activation = F.silu(value * (expert + 1)) * value * (expert + 2)
            route_weight = float(weights[token, route].cpu())
            expected[token, 0] += activation * (expert + 3) * route_weight
            expected[token, -1] += activation * (expert + 4) * route_weight
    return layer, x, indices, weights, expected


@pytest.mark.parametrize("hidden,intermediate", [(65, 129), (129, 193)])
def test_explicit_pytorch_reference_covers_tail_columns_and_combines_routes(hidden, intermediate):
    layer, x, indices, weights, expected = _tail_case(hidden, intermediate)
    with torch.inference_mode():
        output = layer(x, indices, weights)
    torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
    assert torch.count_nonzero(output[:, -1]) == len(x)
    assert torch.count_nonzero(output[:, 1:-1]) == 0
    # The retired first-64-intermediate-column computation sees only zeros in
    # this fixture. These are real reference results, not retired-kernel runs.
    assert torch.count_nonzero(layer.w1[:, :, :64]) == 0
    assert torch.count_nonzero(layer.w3[:, :, :64]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA required; no CPU result qualifies a GPU kernel")
def test_actual_cuda_sorted_pytorch_reference_covers_full_output():
    from core.utils.compile_utils import tf32_override

    layer, x, indices, weights, expected = _tail_case(129, 193, device="cuda", dtype=torch.float32)
    with torch.inference_mode(), tf32_override(enable_matmul=False):
        output = layer(x, indices, weights)
    torch.cuda.synchronize()
    torch.testing.assert_close(output.cpu().double(), expected, rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA and Triton execution required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(3, 129), (5, 193), (2, 1025)])
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "sliced"])
def test_active_triton_activation_matches_full_reference_and_updates_tail(dtype, shape, layout):
    pytest.importorskip("triton", reason="Actual installed Triton required")
    from labs.moe_optimization_journey import triton_kernels

    assert not hasattr(triton_kernels, "fused_expert_ffn_kernel")
    assert not hasattr(triton_kernels, "grouped_gemm_kernel")
    gate = torch.linspace(-3, 3, shape[0] * shape[1], device="cuda", dtype=torch.float32).reshape(shape).to(dtype)
    up = torch.cos(gate.float()).to(dtype)
    if layout == "transposed":
        gate, up = gate.t(), up.t()
    elif layout == "sliced":
        gate = torch.stack((gate, gate), dim=-1)[..., 0]
        up = torch.stack((up, up), dim=-1)[..., 1]
    tolerance = {torch.float32: (1e-5, 1e-6), torch.float16: (2e-3, 2e-3), torch.bfloat16: (2e-2, 2e-2)}[dtype]
    with torch.inference_mode():
        assert triton_kernels.fused_silu_mul_backend(gate, up) == "triton_inference"
        first = triton_kernels.fused_silu_mul(gate, up)
        expected = F.silu(gate) * up
        torch.cuda.synchronize()
        torch.testing.assert_close(first, expected, rtol=tolerance[0], atol=tolerance[1])
        first_tail = first[-1, -1].clone()
        gate[-1, -1] = -2
        up[-1, -1] = 2
        second = triton_kernels.fused_silu_mul(gate, up)
        torch.cuda.synchronize()
        torch.testing.assert_close(second, F.silu(gate) * up, rtol=tolerance[0], atol=tolerance[1])
        assert not torch.equal(second[-1, -1], first_tail)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA needed for differentiable CUDA route")
def test_cuda_trainable_activation_uses_pytorch_and_preserves_backward():
    gate = torch.randn(3, 129, device="cuda", requires_grad=True)
    up = torch.randn(3, 129, device="cuda", requires_grad=True)
    ref_gate, ref_up = gate.detach().clone().requires_grad_(), up.detach().clone().requires_grad_()
    assert triton_kernels.fused_silu_mul_backend(gate, up) == "pytorch_autograd"
    result = triton_kernels.fused_silu_mul(gate, up)
    expected = F.silu(ref_gate) * ref_up
    result.square().sum().backward()
    expected.square().sum().backward()
    torch.cuda.synchronize()
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(gate.grad, ref_gate.grad)
    torch.testing.assert_close(up.grad, ref_up.grad)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.device_count() < 2,
                    reason="Actual two CUDA devices required for noncurrent-device inference")
def test_triton_inference_honors_noncurrent_device_and_restores_caller():
    pytest.importorskip("triton", reason="Actual installed Triton required")
    with torch.cuda.device(0), torch.inference_mode():
        gate = torch.linspace(-2, 2, 259, device="cuda:1").reshape(7, 37).t()
        up = torch.full_like(gate, 0.75)
        output = triton_kernels.fused_silu_mul(gate, up)
        assert torch.cuda.current_device() == 0
        assert output.device == gate.device
        torch.cuda.synchronize(1)
        torch.testing.assert_close(output, F.silu(gate) * up, rtol=1e-5, atol=1e-6)
