from __future__ import annotations

import importlib

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

ZERO3_MODULES = (
    "labs.train_distributed.baseline_zero3",
    "labs.train_distributed.baseline_zero3_multigpu",
)


def test_inplace_relu_reproduces_full_backward_hook_view_failure() -> None:
    module = importlib.import_module(ZERO3_MODULES[0])
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(inplace=True))
    module.attach_zero3_hooks(model, {})

    with pytest.raises(RuntimeError, match="view.*modified inplace"):
        model(torch.randn(2, 4)).sum().backward()


@pytest.mark.parametrize("module_name", ZERO3_MODULES)
def test_zero3_model_backward_hooks_preserve_full_gradients(
    module_name: str,
    tmp_path,
) -> None:
    module = importlib.import_module(module_name)
    rendezvous = tmp_path / "gloo-rendezvous"
    dist.init_process_group(
        "gloo",
        init_method=rendezvous.as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        torch.manual_seed(11)
        reference = module._build_model(hidden_size=4, device="cpu")
        hooked = module._build_model(hidden_size=4, device="cpu")
        hooked.load_state_dict(reference.state_dict())

        shard_map = {param: module.ParamShard(param) for param in hooked.parameters()}
        module.attach_zero3_hooks(hooked, shard_map)

        inputs = torch.randn(3, 4)
        reference_inputs = inputs.detach().clone().requires_grad_()
        hooked_inputs = inputs.detach().clone().requires_grad_()
        targets = torch.randn(3, 4)
        reference_output = reference(reference_inputs)
        hooked_output = hooked(hooked_inputs)
        nn.functional.mse_loss(reference_output, targets).backward()
        nn.functional.mse_loss(hooked_output, targets).backward()

        torch.testing.assert_close(hooked_output, reference_output)
        for hooked_param, reference_param in zip(
            hooked.parameters(), reference.parameters(), strict=True
        ):
            assert hooked_param.grad is not None
            assert reference_param.grad is not None
            torch.testing.assert_close(hooked_param.grad, reference_param.grad)
        torch.testing.assert_close(hooked_inputs.grad, reference_inputs.grad)
    finally:
        dist.destroy_process_group()
