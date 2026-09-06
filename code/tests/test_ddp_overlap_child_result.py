from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ch04.ddp_no_overlap import (
    BaselineNoOverlapBenchmark,
    _run_no_overlap_step,
)
from ch04.ddp_no_overlap import MultiLayerNet as BaselineNetwork
from ch04.ddp_overlap import MultiLayerNet as OptimizedNetwork
from ch04.ddp_overlap import (
    OptimizedOverlapDdpBenchmark,
    _run_overlap_step,
)
from ch04.ddp_overlap_result import (
    RESULT_CALLBACK,
    DdpOverlapResultContract,
    DdpOverlapWorkloadResult,
    make_result_contract,
    reference_training_outputs,
    validate_workload_result,
    write_ddp_overlap_child_result,
)


def _tiny_contract(variant: str) -> DdpOverlapResultContract:
    return make_result_contract(
        variant=variant,
        world_size=2,
        batch_size=4,
        hidden_size=8,
        iterations=2,
        warmup=1,
        tf32=False,
    )


def _gloo_rank(
    rank: int,
    rendezvous: str,
    contract_payload: dict,
    result_env: dict[str, str],
    launch_wall_ns: int,
) -> None:
    os.environ.update(result_env)
    os.environ["AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS"] = str(launch_wall_ns)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        contract = DdpOverlapResultContract.from_dict(contract_payload)
        torch.manual_seed(contract.seed)
        network_type = (
            BaselineNetwork if contract.variant == "no-overlap" else OptimizedNetwork
        )
        base_model = network_type(contract.hidden_size)
        model: torch.nn.Module
        if contract.variant == "overlap":
            model = torch.nn.parallel.DistributedDataParallel(
                base_model,
                gradient_as_bucket_view=True,
                broadcast_buffers=False,
                static_graph=True,
            )
        else:
            model = base_model
        optimizer = torch.optim.SGD(model.parameters(), lr=contract.learning_rate)
        model_parameters = tuple(model.parameters())
        initial_state = {
            name: tensor.detach().clone()
            for name, tensor in base_model.state_dict().items()
        }
        data = torch.randn(contract.batch_size, contract.hidden_size)
        target = torch.randn(contract.batch_size, 1)

        def step() -> torch.Tensor:
            if contract.variant == "no-overlap":
                return _run_no_overlap_step(
                    model,
                    optimizer,
                    model_parameters,
                    data,
                    target,
                    world_size=contract.world_size,
                )
            return _run_overlap_step(model, optimizer, data, target)

        for _ in range(contract.warmup):
            step()
        dist.barrier()
        started = time.perf_counter()
        timed_output: torch.Tensor | None = None
        for _ in range(contract.iterations):
            timed_output = step()
        local_elapsed_ms = (time.perf_counter() - started) * 1000.0
        elapsed = torch.tensor([local_elapsed_ms], dtype=torch.float64)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        if timed_output is None:  # pragma: no cover - positive contract iterations
            raise RuntimeError("Gloo DDP control did not produce a timed output")
        with torch.no_grad():
            post_update_output = base_model(data).detach().clone()
        reference_timed, reference_post = reference_training_outputs(
            initial_state=initial_state,
            data=data,
            target=target,
            steps=contract.warmup + contract.iterations,
            learning_rate=contract.learning_rate,
        )
        result = DdpOverlapWorkloadResult(
            contract=contract,
            rank=rank,
            data=data,
            target=target,
            timed_output=timed_output.detach().clone(),
            post_update_output=post_update_output,
            reference_timed_output=reference_timed,
            reference_post_update_output=reference_post,
            time_per_iter_ms=float(elapsed.item()) / contract.iterations,
        )
        validate_workload_result(result)
        assert write_ddp_overlap_child_result(result) is True
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_gloo_result(
    benchmark: BaselineNoOverlapBenchmark | OptimizedOverlapDdpBenchmark,
    contract: DdpOverlapResultContract,
    tmp_path: Path,
) -> tuple[torch.Tensor, object]:
    result_env = benchmark.prepare_ddp_overlap_child_result(contract)
    result_dir = benchmark._ddp_overlap_result_context["result_dir"]
    launch_wall_ns = time.time_ns()
    rendezvous = tmp_path / f"gloo-{contract.variant}"
    try:
        mp.spawn(
            _gloo_rank,
            args=(str(rendezvous), contract.to_dict(), result_env, launch_wall_ns),
            nprocs=2,
            join=True,
        )
        benchmark.consume_ddp_overlap_child_results(
            launch_wall_ns=launch_wall_ns,
            finish_wall_ns=time.time_ns(),
            returncode=0,
        )
        benchmark._prepare_verification_payload()
        assert benchmark.validate_result() is None
        inputs = benchmark.get_verify_inputs()
        assert inputs["data"].shape == (
            contract.world_size,
            contract.batch_size,
            contract.hidden_size,
        )
        assert inputs["target"].shape == (
            contract.world_size,
            contract.batch_size,
            1,
        )
        return benchmark.get_verify_output(), benchmark.get_input_signature()
    finally:
        if result_dir.exists():
            shutil.rmtree(result_dir)


def _synthetic_result(
    contract: DdpOverlapResultContract,
    *,
    rank: int,
) -> DdpOverlapWorkloadResult:
    torch.manual_seed(contract.seed)
    model = BaselineNetwork(contract.hidden_size)
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    data = torch.randn(contract.batch_size, contract.hidden_size)
    target = torch.randn(contract.batch_size, 1)
    timed, post = reference_training_outputs(
        initial_state=state,
        data=data,
        target=target,
        steps=contract.warmup + contract.iterations,
        learning_rate=contract.learning_rate,
    )
    return DdpOverlapWorkloadResult(
        contract=contract,
        rank=rank,
        data=data,
        target=target,
        timed_output=timed,
        post_update_output=post,
        reference_timed_output=timed.clone(),
        reference_post_update_output=post.clone(),
        time_per_iter_ms=1.0,
    )


@pytest.mark.parametrize(
    ("benchmark_type", "variant"),
    [
        (BaselineNoOverlapBenchmark, "no-overlap"),
        (OptimizedOverlapDdpBenchmark, "overlap"),
    ],
)
def test_pair_declares_fresh_result_callback_and_child_cuda_timing(
    benchmark_type: type,
    variant: str,
) -> None:
    benchmark = benchmark_type()
    config = benchmark.get_config()
    spec = benchmark.get_torchrun_spec(config)
    result_dir = benchmark._ddp_overlap_result_context["result_dir"]
    try:
        assert config.nproc_per_node >= 2
        assert spec.script_path is not None and spec.script_path.name == "ddp_worker.py"
        assert spec.script_args == ["--variant", variant]
        assert spec.result_callback == RESULT_CALLBACK
        assert spec.timing_source == "rank0_time_per_iter_ms"
        assert spec.timing_iterations_per_sample == config.iterations
        assert spec.config_arg_map == {
            "iterations": "--iterations",
            "warmup": "--warmup",
        }
    finally:
        shutil.rmtree(result_dir)


@pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo is unavailable")
def test_real_two_rank_gloo_steps_match_functional_oracle_and_each_other(
    tmp_path: Path,
) -> None:
    baseline_output, baseline_signature = _run_gloo_result(
        BaselineNoOverlapBenchmark(),
        _tiny_contract("no-overlap"),
        tmp_path,
    )
    optimized_output, optimized_signature = _run_gloo_result(
        OptimizedOverlapDdpBenchmark(),
        _tiny_contract("overlap"),
        tmp_path,
    )

    assert baseline_output.shape == (2, 4, 1)
    torch.testing.assert_close(optimized_output, baseline_output, rtol=0, atol=0)
    assert optimized_signature == baseline_signature


def test_child_result_requires_every_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = BaselineNoOverlapBenchmark()
    contract = _tiny_contract("no-overlap")
    result_env = benchmark.prepare_ddp_overlap_child_result(contract)
    result_dir = benchmark._ddp_overlap_result_context["result_dir"]
    launch_wall_ns = time.time_ns()
    for name, value in {
        **result_env,
        "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS": str(launch_wall_ns),
    }.items():
        monkeypatch.setenv(name, value)
    write_ddp_overlap_child_result(_synthetic_result(contract, rank=0))
    try:
        with pytest.raises(RuntimeError, match="rank quorum is incomplete"):
            benchmark.consume_ddp_overlap_child_results(
                launch_wall_ns=launch_wall_ns,
                finish_wall_ns=time.time_ns(),
                returncode=0,
            )
    finally:
        shutil.rmtree(result_dir)


def test_child_result_rejects_corrupted_full_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = OptimizedOverlapDdpBenchmark()
    contract = _tiny_contract("overlap")
    result_env = benchmark.prepare_ddp_overlap_child_result(contract)
    result_dir = benchmark._ddp_overlap_result_context["result_dir"]
    launch_wall_ns = time.time_ns()
    for name, value in {
        **result_env,
        "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS": str(launch_wall_ns),
    }.items():
        monkeypatch.setenv(name, value)
    for rank in range(contract.world_size):
        write_ddp_overlap_child_result(_synthetic_result(contract, rank=rank))
    rank_one = result_dir / "rank-1.pt"
    payload = torch.load(rank_one, map_location="cpu", weights_only=True)
    payload["timed_output"] = payload["timed_output"].clone()
    payload["timed_output"][0, 0] += 10.0
    torch.save(payload, rank_one)
    try:
        with pytest.raises(RuntimeError, match="independent reference"):
            benchmark.consume_ddp_overlap_child_results(
                launch_wall_ns=launch_wall_ns,
                finish_wall_ns=time.time_ns(),
                returncode=0,
            )
    finally:
        shutil.rmtree(result_dir)
