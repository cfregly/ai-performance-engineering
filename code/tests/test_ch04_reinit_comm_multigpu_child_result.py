from __future__ import annotations

import shutil
import time
from pathlib import Path

import pytest
import torch

from ch04 import reinit_comm_multigpu_worker as worker
from ch04.baseline_reinit_comm_multigpu import (
    get_benchmark as get_baseline_benchmark,
)
from ch04.optimized_reinit_comm_multigpu import (
    get_benchmark as get_optimized_benchmark,
)
from ch04.reinit_comm_multigpu_result import (
    LAUNCH_MONOTONIC_NS_ENV,
    LAUNCH_WALL_NS_ENV,
    RESULT_CALLBACK,
    write_reinit_comm_child_result,
)
from core.harness.benchmark_harness import LaunchVia


@pytest.mark.parametrize(
    ("factory", "variant"),
    (
        (get_baseline_benchmark, "baseline"),
        (get_optimized_benchmark, "optimized"),
    ),
)
def test_reinit_comm_pair_declares_real_two_rank_worker(factory, variant) -> None:
    benchmark = factory()
    config = benchmark.get_config()
    assert config.launch_via == LaunchVia.TORCHRUN
    assert config.nproc_per_node == 2
    assert config.iterations == 5
    assert config.warmup == 5
    assert benchmark._workload.bytes_per_iteration == 4.0
    workload = benchmark.get_workload_metadata()
    assert workload is not None
    assert workload.requests_per_iteration == 1.0
    assert workload.bytes_per_iteration == 4.0

    spec = benchmark.get_torchrun_spec(config)
    try:
        assert spec.script_path is not None
        assert spec.script_path.name == "reinit_comm_multigpu_worker.py"
        assert spec.script_args == ["--variant", variant]
        assert spec.config_arg_map == {
            "iterations": "--iterations",
            "warmup": "--warmup",
        }
        assert spec.result_callback == RESULT_CALLBACK
        assert spec.timing_source == "rank0_time_per_iter_ms"
        assert spec.timing_iterations_per_sample == 5
    finally:
        context = benchmark._reinit_comm_result_context
        assert context is not None
        shutil.rmtree(context["result_dir"])


@pytest.mark.parametrize(
    ("factory", "variant"),
    (
        (get_baseline_benchmark, "baseline"),
        (get_optimized_benchmark, "optimized"),
    ),
)
def test_child_callback_exposes_all_rank_inputs_and_outputs(
    monkeypatch: pytest.MonkeyPatch, factory, variant
) -> None:
    benchmark = factory()
    env = benchmark.prepare_reinit_comm_child_result(
        variant=variant,
        world_size=2,
        iterations=5,
        warmup=5,
    )
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    launch_wall_ns = time.time_ns() - 1_000_000
    launch_monotonic_ns = time.monotonic_ns() - 1_000_000
    monkeypatch.setenv(LAUNCH_WALL_NS_ENV, str(launch_wall_ns))
    monkeypatch.setenv(LAUNCH_MONOTONIC_NS_ENV, str(launch_monotonic_ns))

    inputs = (torch.tensor([[1.25]]), torch.tensor([[2.5]]))
    expected = torch.tensor([[3.75]])
    for rank, input_tensor in enumerate(inputs):
        assert write_reinit_comm_child_result(
            variant=variant,
            rank=rank,
            world_size=2,
            iterations=5,
            warmup=5,
            input_tensor=input_tensor,
            output=expected,
            time_per_iter_ms=1.25,
        )
    finish_wall_ns = time.time_ns() + 1_000_000
    finish_monotonic_ns = time.monotonic_ns() + 1_000_000
    benchmark.consume_reinit_comm_child_results(
        launch_wall_ns=launch_wall_ns,
        launch_monotonic_ns=launch_monotonic_ns,
        finish_wall_ns=finish_wall_ns,
        finish_monotonic_ns=finish_monotonic_ns,
        returncode=0,
    )

    verify_inputs = benchmark.get_verify_inputs()
    assert set(verify_inputs) == {"rank_0_input", "rank_1_input"}
    assert torch.equal(verify_inputs["rank_0_input"], inputs[0])
    assert torch.equal(verify_inputs["rank_1_input"], inputs[1])
    assert torch.equal(benchmark.get_verify_output(), expected.expand(2, 1, 1))
    signature = benchmark.get_input_signature()
    assert signature.world_size == 2
    assert signature.ranks == [0, 1]
    assert signature.collective_type == "all_reduce"
    assert signature.collective_algorithm == "nccl_sum"
    assert benchmark.validate_result() is None
    benchmark.capture_verification_payload()
    context = benchmark._reinit_comm_result_context
    assert context is not None
    assert context["retention"] == "cleaned-after-success"
    assert not Path(context["result_dir"]).exists()


def test_callback_rejects_stale_rank_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = get_baseline_benchmark()
    env = benchmark.prepare_reinit_comm_child_result(
        variant="baseline", world_size=2, iterations=5, warmup=5
    )
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    launch_wall_ns = time.time_ns() - 1_000_000
    launch_monotonic_ns = time.monotonic_ns() - 1_000_000
    monkeypatch.setenv(LAUNCH_WALL_NS_ENV, str(launch_wall_ns))
    monkeypatch.setenv(LAUNCH_MONOTONIC_NS_ENV, str(launch_monotonic_ns))
    value = torch.tensor([[2.0]])
    for rank in range(2):
        write_reinit_comm_child_result(
            variant="baseline",
            rank=rank,
            world_size=2,
            iterations=5,
            warmup=5,
            input_tensor=value,
            output=value * 2,
            time_per_iter_ms=2.5,
        )
    context = benchmark._reinit_comm_result_context
    assert context is not None
    first = Path(context["result_dir"]) / "rank-0.pt"
    payload = torch.load(first, map_location="cpu", weights_only=True)
    payload["created_wall_ns"] = 0
    torch.save(payload, first)
    try:
        with pytest.raises(RuntimeError, match="Stale wall clock"):
            benchmark.consume_reinit_comm_child_results(
                launch_wall_ns=launch_wall_ns,
                launch_monotonic_ns=launch_monotonic_ns,
                finish_wall_ns=time.time_ns() + 1_000_000,
                finish_monotonic_ns=time.monotonic_ns() + 1_000_000,
                returncode=0,
            )
        assert context["retention"] == "retained-stale-result"
    finally:
        shutil.rmtree(context["result_dir"])


def test_worker_iteration_reinitializes_only_the_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"initialized": True}
    events: list[str] = []

    monkeypatch.setattr(worker.dist, "is_initialized", lambda: state["initialized"])

    def destroy() -> None:
        events.append("destroy")
        state["initialized"] = False

    def initialize(local_rank: int) -> None:
        assert local_rank == 0
        events.append("init")
        state["initialized"] = True

    def all_reduce(output: torch.Tensor, *, op) -> None:
        assert op == worker.dist.ReduceOp.SUM
        events.append("all_reduce")
        output.mul_(2)

    monkeypatch.setattr(worker.dist, "destroy_process_group", destroy)
    monkeypatch.setattr(worker, "_initialize_process_group", initialize)
    monkeypatch.setattr(worker.dist, "all_reduce", all_reduce)
    monkeypatch.setattr(worker.torch.cuda, "set_device", lambda _: None)
    input_tensor = torch.tensor([[3.0]])
    output = torch.empty_like(input_tensor)

    worker._run_iteration("baseline", local_rank=0, input_tensor=input_tensor, output=output)
    assert events == ["destroy", "init", "all_reduce"]
    assert torch.equal(output, torch.tensor([[6.0]]))

    events.clear()
    worker._run_iteration("optimized", local_rank=0, input_tensor=input_tensor, output=output)
    assert events == ["all_reduce"]
    assert torch.equal(output, torch.tensor([[6.0]]))
