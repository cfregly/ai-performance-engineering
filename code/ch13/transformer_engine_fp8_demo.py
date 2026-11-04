"""Run one Transformer Engine FP8 forward/backward step against FP16."""

from __future__ import annotations

import argparse
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional

from ch13.optimized_precisionfp8_te import (
    TEFP8MLP,
    _create_delayed_scaling_recipe,
    _load_transformer_engine,
)
from ch13.te_runtime_common import get_te_precision_output_tolerances

DEFAULT_BATCH = 16
DEFAULT_HIDDEN = 256


@dataclass(frozen=True)
class TrainingStepResult:
    output: torch.Tensor
    loss: torch.Tensor
    input_gradient: torch.Tensor
    parameter_gradients: dict[str, torch.Tensor]


def _positive_multiple_of_16(raw: str) -> int:
    value = int(raw)
    if value <= 0 or value % 16:
        raise argparse.ArgumentTypeError("value must be a positive multiple of 16")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a same-weight Transformer Engine FP8 and FP16 training-step comparison."
    )
    parser.add_argument("--batch", type=_positive_multiple_of_16, default=DEFAULT_BATCH)
    parser.add_argument("--hidden", type=_positive_multiple_of_16, default=DEFAULT_HIDDEN)
    return parser


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13 Transformer Engine FP8 demo")
    return torch.device("cuda")


def _run_training_step(
    model: torch.nn.Module,
    source: torch.Tensor,
    target: torch.Tensor,
    precision_context: AbstractContextManager[Any],
) -> TrainingStepResult:
    model.zero_grad(set_to_none=True)
    step_input = source.detach().clone().requires_grad_(True)
    with precision_context:
        output = model(step_input)
        loss = functional.mse_loss(output, target)
    loss.backward()

    if step_input.grad is None:
        raise AssertionError("training step did not produce an input gradient")
    parameter_gradients: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            raise AssertionError(f"training step did not produce a gradient for {name}")
        parameter_gradients[name] = parameter.grad.detach().clone()
    return TrainingStepResult(
        output=output.detach().clone(),
        loss=loss.detach().clone(),
        input_gradient=step_input.grad.detach().clone(),
        parameter_gradients=parameter_gradients,
    )


def assert_finite_training_step(label: str, result: TrainingStepResult) -> None:
    tensors = {
        "output": result.output,
        "loss": result.loss,
        "input_gradient": result.input_gradient,
        **{
            f"parameter_gradient.{name}": gradient
            for name, gradient in result.parameter_gradients.items()
        },
    }
    for name, tensor in tensors.items():
        if not bool(torch.isfinite(tensor).all().item()):
            raise AssertionError(f"{label} {name} contains nonfinite values")


def main() -> int:
    args = build_parser().parse_args()
    device = _require_cuda()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    _, fp8_autocast, te_recipe = _load_transformer_engine()
    recipe = _create_delayed_scaling_recipe(te_recipe)
    model = TEFP8MLP(hidden_dim=args.hidden).to(device=device, dtype=torch.float16).train()
    source = torch.randn(args.batch, args.hidden, device=device, dtype=torch.float16)
    target = torch.randn_like(source)
    initial_parameters = {
        name: parameter.detach().clone() for name, parameter in model.named_parameters()
    }

    fp16_result = _run_training_step(model, source, target, nullcontext())
    fp8_result = _run_training_step(
        model,
        source,
        target,
        fp8_autocast(enabled=True, fp8_recipe=recipe),
    )
    assert_finite_training_step("FP16", fp16_result)
    assert_finite_training_step("FP8", fp8_result)

    for name, parameter in model.named_parameters():
        torch.testing.assert_close(parameter, initial_parameters[name], rtol=0.0, atol=0.0)
    rtol, atol = get_te_precision_output_tolerances()["prediction"]
    torch.testing.assert_close(fp8_result.output, fp16_result.output, rtol=rtol, atol=atol)
    max_abs_diff = float((fp8_result.output - fp16_result.output).abs().max().item())
    torch.cuda.synchronize(device)

    print(
        "transformer_engine_fp8_forward_backward=pass "
        f"batch={args.batch} hidden={args.hidden} finite_output_loss_grads=true"
    )
    print(
        "same_weight_fp16_compare=pass "
        f"rtol={rtol:.1f} atol={atol:.1f} max_abs_diff={max_abs_diff:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
