from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from labs.train_distributed.pipeline import (
    PipelineConfig,
    PipelineExperiment,
    _build_toy_model,
)


def _cpu_experiment(
    model: nn.Sequential, *, schedule: str = "gpipe"
) -> PipelineExperiment:
    experiment = PipelineExperiment.__new__(PipelineExperiment)
    experiment.config = PipelineConfig(
        schedule=schedule,
        n_stages=2,
        batch_size=4,
        micro_batch_size=1,
        input_dim=3,
        hidden_dim=5,
        depth=6,
        learning_rate=1e-3,
        non_blocking=False,
        dtype=torch.float64,
        dual_window=2,
    )
    experiment.devices = ["cpu", "cpu"]
    experiment.model = model
    experiment.stages = experiment._split_into_stages()
    experiment.criterion = nn.MSELoss()
    return experiment


def _staged_forward_backward(
    stages: list[nn.Sequential], inputs: torch.Tensor, targets: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    stage_inputs = [inputs]
    stage_outputs = []
    value = inputs
    for stage_id, stage in enumerate(stages):
        if stage_id:
            value = value.detach().requires_grad_(True)
            stage_inputs.append(value)
        value = stage(value)
        stage_outputs.append(value)

    loss = nn.functional.mse_loss(value, targets)
    loss.backward()
    for stage_id in reversed(range(1, len(stages))):
        boundary_grad = stage_inputs[stage_id].grad
        assert boundary_grad is not None
        stage_outputs[stage_id - 1].backward(boundary_grad)
    return value, loss


def test_default_partition_reproduces_inplace_leaf_failure_without_fix() -> None:
    torch.manual_seed(7)
    model = _build_toy_model(input_dim=3, hidden_dim=5, depth=6).to(torch.float64)
    per_stage = (len(model) + 1) // 2
    first_stage = nn.Sequential(*model[:per_stage])
    unsafe_second_stage = nn.Sequential(*model[per_stage:])
    assert isinstance(unsafe_second_stage[0], nn.ReLU)
    assert unsafe_second_stage[0].inplace is True

    transported = first_stage(
        torch.randn(2, 3, dtype=torch.float64, requires_grad=True)
    ).detach().requires_grad_(True)
    with pytest.raises(
        RuntimeError,
        match="leaf Variable that requires grad is being used in an in-place operation",
    ):
        unsafe_second_stage(transported)


def test_stage_boundary_matches_monolithic_outputs_loss_and_full_gradients() -> None:
    torch.manual_seed(11)
    reference = _build_toy_model(input_dim=3, hidden_dim=5, depth=6).to(torch.float64)
    staged_model = deepcopy(reference)
    experiment = _cpu_experiment(staged_model)

    boundary_relu = experiment.stages[1][0]
    assert isinstance(boundary_relu, nn.ReLU)
    assert boundary_relu.inplace is False

    values = torch.randn(4, 3, dtype=torch.float64)
    targets = torch.randn(4, 3, dtype=torch.float64)
    reference_input = values.detach().clone().requires_grad_(True)
    staged_input = values.detach().clone().requires_grad_(True)

    reference_output = reference(reference_input)
    reference_loss = nn.functional.mse_loss(reference_output, targets)
    reference_loss.backward()
    staged_output, staged_loss = _staged_forward_backward(
        experiment.stages, staged_input, targets
    )

    torch.testing.assert_close(staged_output, reference_output, rtol=0, atol=0)
    torch.testing.assert_close(staged_loss, reference_loss, rtol=0, atol=0)
    torch.testing.assert_close(staged_input.grad, reference_input.grad, rtol=1e-12, atol=1e-12)
    for staged_parameter, reference_parameter in zip(
        staged_model.parameters(), reference.parameters(), strict=True
    ):
        assert staged_parameter.grad is not None
        assert reference_parameter.grad is not None
        torch.testing.assert_close(
            staged_parameter.grad,
            reference_parameter.grad,
            rtol=1e-12,
            atol=1e-12,
        )


@pytest.mark.parametrize("schedule", ["gpipe", "1f1b", "dualpipe", "dualpipev"])
def test_every_schedule_trains_through_a_relu_stage_boundary(schedule: str) -> None:
    torch.manual_seed(17)
    experiment = _cpu_experiment(
        _build_toy_model(input_dim=3, hidden_dim=5, depth=6), schedule=schedule
    )
    experiment.optimizers = [
        torch.optim.Adam(stage.parameters(), lr=experiment.config.learning_rate)
        for stage in experiment.stages
    ]
    before = [parameter.detach().clone() for parameter in experiment.model.parameters()]
    loss, telemetry = experiment.run_batch(
        torch.randn(4, 3, dtype=torch.float64),
        torch.randn(4, 3, dtype=torch.float64),
    )

    assert torch.isfinite(torch.tensor(loss))
    assert all(stage.forward_ops == experiment.config.n_micro for stage in telemetry.stage_stats)
    assert all(stage.backward_ops == experiment.config.n_micro for stage in telemetry.stage_stats)
    assert any(
        not torch.equal(previous, current)
        for previous, current in zip(before, experiment.model.parameters(), strict=True)
    )
