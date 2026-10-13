"""W2-048 regressions for the multi-GPU 1F1B pipeline benchmark.

The benchmark's own timed iteration runs here on CPU in real Gloo processes.
Hooks only record when each stage's forward and backward layers start. The
buffers, schedule and point-to-point transfers are the benchmark's code.
"""

from __future__ import annotations

import ast
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import ch04.optimized_pipeline_parallel_multigpu_1f1b as pipeline_1f1b

_SEQ = 2
_HIDDEN = 16
_MICRO_BATCH = 2
_LAYERS_PER_STAGE = 2
_ITERATIONS = 2
_CPU = torch.device("cpu")
_SOURCE = Path(pipeline_1f1b.__file__)


def _stage_layers(stage: int) -> tuple[nn.ModuleList, nn.ModuleList]:
    fwd_layers, bwd_layers = pipeline_1f1b._build_stage_layers(_HIDDEN, _LAYERS_PER_STAGE, _CPU)
    generator = torch.Generator().manual_seed(1000 + stage)
    with torch.no_grad():
        for layer in (*fwd_layers, *bwd_layers):
            # Nonnegative weights keep every ReLU active, so each output
            # depends on every value of the microbatch that produced it.
            layer.weight.copy_(torch.rand(_HIDDEN, _HIDDEN, generator=generator) * (2.0 / _HIDDEN))
    return fwd_layers, bwd_layers


def _rank0_inputs(num_micro_batches: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(7)
    offsets = torch.arange(num_micro_batches, dtype=torch.float32).repeat_interleave(_MICRO_BATCH)
    values = torch.rand(num_micro_batches * _MICRO_BATCH, _SEQ, _HIDDEN, generator=generator)
    return (values + offsets[:, None, None]).to(torch.bfloat16)


def _reference_outputs(world_size: int, num_micro_batches: int) -> torch.Tensor:
    stages = [_stage_layers(stage) for stage in range(world_size)]
    inputs = _rank0_inputs(num_micro_batches)
    outputs = []
    with torch.inference_mode():
        for micro_idx in range(num_micro_batches):
            x = inputs.narrow(0, micro_idx * _MICRO_BATCH, _MICRO_BATCH)
            for fwd_layers, _ in stages:
                for layer in fwd_layers:
                    x = torch.relu_(layer(x))
            for _, bwd_layers in reversed(stages):
                for layer in bwd_layers:
                    x = torch.relu_(layer(x))
            outputs.append(x)
    return torch.stack(outputs)


def _gloo_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    micro_batch_counts: list[int],
    result_dir: str,
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        fwd_layers, bwd_layers = _stage_layers(rank)
        events: list[str] = []
        final_outputs: list[torch.Tensor] = []
        fwd_layers[0].register_forward_pre_hook(lambda _module, _args: events.append("F"))
        bwd_layers[0].register_forward_pre_hook(lambda _module, _args: events.append("B"))
        if rank == 0:
            bwd_layers[-1].register_forward_hook(
                lambda _module, _args, output: final_outputs.append(output)
            )

        report: dict[str, list[list[str]]] = {}
        for num_micro_batches in micro_batch_counts:
            run_iteration = pipeline_1f1b._build_1f1b_iteration(
                rank=rank,
                world_size=world_size,
                num_micro_batches=num_micro_batches,
                micro_batch_size=_MICRO_BATCH,
                seq_length=_SEQ,
                hidden=_HIDDEN,
                device=_CPU,
                inputs=_rank0_inputs(num_micro_batches) if rank == 0 else None,
                fwd_layers=fwd_layers,
                bwd_layers=bwd_layers,
            )
            orders: list[list[str]] = []
            with torch.inference_mode():
                for iteration in range(_ITERATIONS):
                    events.clear()
                    final_outputs.clear()
                    run_iteration()
                    orders.append(list(events))
                    if rank == 0:
                        torch.save(
                            torch.stack(final_outputs).clone(),
                            Path(result_dir) / f"outputs-m{num_micro_batches}-i{iteration}.pt",
                        )
            report[str(num_micro_batches)] = orders
        (Path(result_dir) / f"rank-{rank}.json").write_text(json.dumps(report), encoding="utf-8")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _live_activation_peak(order: list[str]) -> int:
    live = 0
    peak = 0
    for event in order:
        live += 1 if event == "F" else -1
        peak = max(peak, live)
    return peak


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
@pytest.mark.parametrize("world_size", [2, 3, 4])
def test_real_gloo_stages_run_staggered_1f1b_and_produce_reference_outputs(
    tmp_path: Path,
    world_size: int,
) -> None:
    # The harness default is one microbatch per stage. The larger count gives
    # every stage several steady-state steps.
    micro_batch_counts = [world_size, world_size + 2]
    mp.spawn(
        _gloo_worker,
        args=(
            world_size,
            (tmp_path / "gloo.rdzv").as_uri(),
            micro_batch_counts,
            str(tmp_path),
        ),
        nprocs=world_size,
        join=True,
    )

    reports = [
        json.loads((tmp_path / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(world_size)
    ]
    for num_micro_batches in micro_batch_counts:
        reference = _reference_outputs(world_size, num_micro_batches)
        for iteration in range(_ITERATIONS):
            actual = torch.load(
                tmp_path / f"outputs-m{num_micro_batches}-i{iteration}.pt",
                weights_only=True,
            )
            torch.testing.assert_close(actual, reference, rtol=0.0, atol=0.0)

            for rank, report in enumerate(reports):
                order = report[str(num_micro_batches)][iteration]
                assert order.count("F") == num_micro_batches
                assert order.count("B") == num_micro_batches
                if rank == 0:
                    # Stage 0 must forward one microbatch per stage before its
                    # first backward. Otherwise forward world_size - 1 waits for
                    # microbatch 0's gradient to cross the whole pipeline.
                    assert order.index("B") >= world_size
                    assert _live_activation_peak(order) <= world_size + 1
                    first_backward = order.index("B")
                    assert "FF" not in "".join(order[first_backward:])
                else:
                    warmup = world_size - rank - 1
                    assert order == (
                        ["F"] * warmup
                        + ["F", "B"] * (num_micro_batches - warmup)
                        + ["B"] * warmup
                    )
                    assert _live_activation_peak(order) == world_size - rank


def test_build_rejects_too_few_microbatches_before_any_transfer() -> None:
    fwd_layers, bwd_layers = _stage_layers(0)
    with pytest.raises(ValueError, match="at least one steady-state microbatch"):
        pipeline_1f1b._build_1f1b_iteration(
            rank=0,
            world_size=4,
            num_micro_batches=3,
            micro_batch_size=_MICRO_BATCH,
            seq_length=_SEQ,
            hidden=_HIDDEN,
            device=_CPU,
            inputs=_rank0_inputs(3),
            fwd_layers=fwd_layers,
            bwd_layers=bwd_layers,
        )


@pytest.mark.parametrize(("rank", "has_inputs"), [(0, False), (1, True)])
def test_build_rejects_input_ownership_on_the_wrong_stage(rank: int, has_inputs: bool) -> None:
    fwd_layers, bwd_layers = _stage_layers(rank)
    with pytest.raises(RuntimeError, match="Only rank zero owns the pipeline input batch"):
        pipeline_1f1b._build_1f1b_iteration(
            rank=rank,
            world_size=2,
            num_micro_batches=2,
            micro_batch_size=_MICRO_BATCH,
            seq_length=_SEQ,
            hidden=_HIDDEN,
            device=_CPU,
            inputs=_rank0_inputs(2) if has_inputs else None,
            fwd_layers=fwd_layers,
            bwd_layers=bwd_layers,
        )


def test_worker_times_the_tested_iteration_and_has_no_second_schedule() -> None:
    tree = ast.parse(_SOURCE.read_text(encoding="utf-8"))
    functions = {
        node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    worker_calls = {
        node.func.id
        for node in ast.walk(functions["_run_worker"])
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "_build_1f1b_iteration" in worker_calls
    assert "run_iteration" in worker_calls

    direct_p2p = [
        ast.unparse(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in {"send", "recv", "isend", "irecv", "batch_isend_irecv"}
    ]
    assert direct_p2p == []
