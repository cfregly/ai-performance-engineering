"""Real CPU controls for the optimized DDP accumulation schedule."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
from torch.nn.parallel import DistributedDataParallel

from labs.train_distributed.training_utils.gradient_accumulation import (
    build_gradient_accumulation_plan,
    ddp_static_graph_enabled,
    gradient_sync_context,
)


def _model() -> nn.Linear:
    model = nn.Linear(2, 1, bias=True, dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[0.25, -0.5]], dtype=torch.float64))
        model.bias.copy_(torch.tensor([0.125], dtype=torch.float64))
    return model


def _sample(rank: int, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    inputs = torch.tensor([[float(step + 1), float(rank + 1) / 2.0]], dtype=torch.float64)
    target = torch.tensor([[float(step - rank) / 4.0]], dtype=torch.float64)
    return inputs, target


def _reference_parameters(
    world_size: int,
    microbatches: int,
    grad_accum: int,
) -> torch.Tensor:
    replicas = [_model() for _ in range(world_size)]
    optimizers = [torch.optim.SGD(model.parameters(), lr=1.0 / 64.0) for model in replicas]
    plan = build_gradient_accumulation_plan(microbatches, grad_accum)
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)

    for step, accumulation in enumerate(plan):
        for rank, model in enumerate(replicas):
            inputs, target = _sample(rank, step)
            loss = nn.functional.mse_loss(model(inputs), target) / accumulation.group_size
            loss.backward()
        if accumulation.should_step:
            parameter_groups = zip(*(tuple(model.parameters()) for model in replicas), strict=True)
            for parameters in parameter_groups:
                mean_gradient = torch.stack([parameter.grad for parameter in parameters]).mean(
                    dim=0
                )
                for parameter in parameters:
                    parameter.grad.copy_(mean_gradient)
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

    return torch.cat([parameter.detach().reshape(-1) for parameter in replicas[0].parameters()])


@dataclass
class _CommState:
    process_group: dist.ProcessGroup
    calls: int = 0


def _counted_allreduce(state, bucket):
    state.calls += 1
    return default_hooks.allreduce_hook(state.process_group, bucket)


def _run_case(
    rank: int,
    world_size: int,
    microbatches: int,
    grad_accum: int,
) -> dict[str, object]:
    static_graph = ddp_static_graph_enabled(grad_accum)
    model = DistributedDataParallel(
        _model(),
        static_graph=static_graph,
        bucket_cap_mb=50,
        gradient_as_bucket_view=True,
    )
    comm_state = _CommState(dist.group.WORLD)
    model.register_comm_hook(comm_state, _counted_allreduce)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0 / 64.0)
    optimizer.zero_grad(set_to_none=True)
    plan = build_gradient_accumulation_plan(microbatches, grad_accum)
    updates = 0

    for step, accumulation in enumerate(plan):
        inputs, target = _sample(rank, step)
        with gradient_sync_context(model, accumulation, distributed=True):
            loss = nn.functional.mse_loss(model(inputs), target) / accumulation.group_size
            loss.backward()
        if accumulation.should_step:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1

    parameters = torch.cat(
        [parameter.detach().reshape(-1) for parameter in model.module.parameters()]
    )
    gathered = [torch.empty_like(parameters) for _ in range(world_size)]
    dist.all_gather(gathered, parameters)
    for peer_parameters in gathered:
        torch.testing.assert_close(parameters, peer_parameters, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        parameters,
        _reference_parameters(world_size, microbatches, grad_accum),
        rtol=0.0,
        atol=1e-12,
    )
    expected_updates = math.ceil(microbatches / grad_accum)
    assert updates == expected_updates
    assert comm_state.calls == expected_updates
    return {
        "microbatches": microbatches,
        "grad_accum": grad_accum,
        "optimizer_updates": updates,
        "allreduce_calls": comm_state.calls,
        "static_graph": static_graph,
        "parameters": parameters.tolist(),
    }


def _gloo_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    output_dir: str,
) -> None:
    torch.set_num_threads(1)
    report: dict[str, object] = {"rank": rank, "world_size": world_size}
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=20),
    )
    try:
        report["cases"] = [
            _run_case(rank, world_size, microbatches=5, grad_accum=3),
            _run_case(rank, world_size, microbatches=2, grad_accum=3),
            _run_case(rank, world_size, microbatches=5, grad_accum=1),
        ]
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        Path(output_dir, f"rank-{rank}.json").write_text(
            json.dumps(report, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        dist.destroy_process_group()


def _join_processes(context: mp.ProcessContext, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    try:
        while not context.join(timeout=1, grace_period=1):
            if time.monotonic() >= deadline:
                pytest.fail("Actual CPU DDP accumulation control exceeded 30 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=2)
            if process.is_alive():
                process.kill()
                process.join(timeout=2)


def test_partial_accumulation_plan_uses_actual_group_size() -> None:
    plan = build_gradient_accumulation_plan(total_microbatches=5, grad_accum=3)
    assert [(step.group_size, step.should_step) for step in plan] == [
        (3, False),
        (3, False),
        (3, True),
        (2, False),
        (2, True),
    ]
    short_plan = build_gradient_accumulation_plan(total_microbatches=2, grad_accum=3)
    assert [(step.group_size, step.should_step) for step in short_plan] == [
        (2, False),
        (2, True),
    ]
    default_plan = build_gradient_accumulation_plan(total_microbatches=3, grad_accum=1)
    assert [(step.group_size, step.should_step) for step in default_plan] == [
        (1, True),
        (1, True),
        (1, True),
    ]
    assert ddp_static_graph_enabled(1) is True
    assert ddp_static_graph_enabled(3) is False


@pytest.mark.parametrize(
    ("microbatches", "grad_accum"),
    [(0, 1), (1, 0), (-1, 1), (1, -1)],
)
def test_accumulation_plan_rejects_nonpositive_values(
    microbatches: int,
    grad_accum: int,
) -> None:
    with pytest.raises(ValueError, match="must be positive"):
        build_gradient_accumulation_plan(microbatches, grad_accum)


@pytest.mark.parametrize("world_size", [1, 2])
@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Actual CPU DDP control requires Gloo",
)
def test_real_cpu_ddp_partial_accumulation_matches_grouped_reference(
    tmp_path: Path,
    world_size: int,
) -> None:
    output_dir = tmp_path / f"world-{world_size}"
    output_dir.mkdir()
    context = mp.spawn(
        _gloo_worker,
        args=(
            world_size,
            (tmp_path / f"gloo-{world_size}").as_uri(),
            str(output_dir),
        ),
        nprocs=world_size,
        join=False,
    )
    _join_processes(context)

    reports = [
        json.loads((output_dir / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    assert all(report["status"] == "PASS" for report in reports)
    for report in reports:
        assert [case["optimizer_updates"] for case in report["cases"]] == [2, 1, 5]
        assert [case["allreduce_calls"] for case in report["cases"]] == [2, 1, 5]
        assert [case["static_graph"] for case in report["cases"]] == [False, False, True]
