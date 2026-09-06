from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

from ch04.reduction_common import ReusableReductionMlp


def _reference(model: ReusableReductionMlp) -> nn.Sequential:
    return nn.Sequential(
        copy.deepcopy(model.fc1),
        nn.ReLU(inplace=True),
        copy.deepcopy(model.fc2),
    ).eval()


@pytest.mark.parametrize("shape", [(7, 4), (2, 3, 4)])
@pytest.mark.parametrize("prepared", [False, True])
def test_reduction_mlp_no_grad_matches_same_weight_sequential(
    shape: tuple[int, ...],
    prepared: bool,
) -> None:
    torch.manual_seed(7)
    model = ReusableReductionMlp(hidden_dim=4, inner_dim=9).eval()
    reference = _reference(model)
    x = torch.randn(shape)

    with torch.inference_mode():
        expected = reference(x)
        if prepared:
            model.prepare_forward_buffers(x)
            actual = model.forward_prepared(x)
        else:
            actual = model(x)

    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    ("fc1_bias", "fc2_bias"),
    [(False, False), (False, True), (True, False)],
)
def test_reduction_mlp_bias_free_branches_preserve_math(
    fc1_bias: bool,
    fc2_bias: bool,
) -> None:
    torch.manual_seed(11)
    model = ReusableReductionMlp(hidden_dim=4, inner_dim=7).eval()
    if not fc1_bias:
        model.fc1.bias = None
    if not fc2_bias:
        model.fc2.bias = None
    reference = _reference(model)
    x = torch.randn(2, 5, 4)

    with torch.inference_mode():
        expected = reference(x)
        actual = model(x)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_reduction_mlp_bias_path_uses_flattened_addmm_outputs(monkeypatch) -> None:
    torch.manual_seed(19)
    model = ReusableReductionMlp(hidden_dim=4, inner_dim=6).eval()
    x = torch.randn(2, 3, 4)
    expected = _reference(model)(x)
    real_addmm = torch.addmm
    calls: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []

    def record_addmm(
        input_tensor: torch.Tensor,
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        *,
        out: torch.Tensor,
    ) -> torch.Tensor:
        calls.append((tuple(mat1.shape), tuple(out.shape), out.data_ptr()))
        return real_addmm(input_tensor, mat1, mat2, out=out)

    monkeypatch.setattr(torch, "addmm", record_addmm)
    with torch.inference_mode():
        actual = model(x)

    assert calls == [
        ((6, 4), (6, 6), model._fc1_buffer.data_ptr()),
        ((6, 6), (6, 4), model._fc2_buffer.data_ptr()),
    ]
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_reduction_mlp_gradient_path_matches_same_weight_sequential() -> None:
    torch.manual_seed(23)
    model = ReusableReductionMlp(hidden_dim=4, inner_dim=8).eval()
    reference = _reference(model)
    actual_input = torch.randn(2, 3, 4, requires_grad=True)
    reference_input = actual_input.detach().clone().requires_grad_(True)

    actual = model(actual_input)
    expected = reference(reference_input)
    actual.square().sum().backward()
    expected.square().sum().backward()

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_input.grad, reference_input.grad, rtol=1e-6, atol=1e-6)
    for parameter, reference_parameter in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, reference_parameter.grad, rtol=1e-6, atol=1e-6)
    assert model._fc1_buffer is None
    assert model._fc2_buffer is None
