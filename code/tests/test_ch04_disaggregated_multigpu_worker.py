from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ch04.baseline_disaggregated_multigpu import BaselineDisaggregatedBenchmark
from ch04.disaggregated_multigpu_result import (
    DisaggregatedResultContract,
    make_result_contract,
    write_disaggregated_child_result,
)
from ch04.disaggregated_multigpu_worker import (
    _make_inputs,
    _make_models,
    _reference_output,
    _run_iteration,
)
from ch04.optimized_disaggregated_multigpu import OptimizedDisaggregatedBenchmark
from core.harness.benchmark_harness import LaunchVia


def _tiny_contract(variant: str) -> DisaggregatedResultContract:
    return make_result_contract(
        variant=variant,
        world_size=2,
        batch_size=2,
        prefill_len=4,
        hidden_dim=8,
        iterations=1,
        warmup=0,
        seed=42,
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
        contract = DisaggregatedResultContract.from_dict(contract_payload)
        device = torch.device("cpu")
        prefill_model, decode_model = _make_models(
            contract.variant,
            hidden_dim=contract.hidden_dim,
            device=device,
            seed=contract.seed,
            wrap_ddp=False,
        )
        prefill_input, decode_input = _make_inputs(contract, device)
        reference = _reference_output(
            prefill_model,
            decode_model,
            prefill_input,
            decode_input,
        )
        timed = torch.cat(
            _run_iteration(
                contract.variant,
                prefill_model,
                decode_model,
                prefill_input,
                decode_input,
                world_size=contract.world_size,
            ),
            dim=1,
        )
        write_disaggregated_child_result(
            contract=contract,
            rank=rank,
            prefill_input=prefill_input,
            decode_input=decode_input,
            reference_output=reference,
            timed_output=timed,
        )
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _run_gloo_result(
    benchmark: BaselineDisaggregatedBenchmark | OptimizedDisaggregatedBenchmark,
    contract: DisaggregatedResultContract,
    tmp_path: Path,
) -> torch.Tensor:
    result_env = benchmark.prepare_disaggregated_child_result(contract)
    launch_wall_ns = time.time_ns()
    rendezvous = tmp_path / f"gloo-{contract.variant}"
    mp.spawn(
        _gloo_rank,
        args=(str(rendezvous), contract.to_dict(), result_env, launch_wall_ns),
        nprocs=2,
        join=True,
    )
    benchmark.consume_disaggregated_child_results(
        launch_wall_ns=launch_wall_ns,
        finish_wall_ns=time.time_ns(),
        returncode=0,
    )
    return benchmark.get_verify_output()


@pytest.mark.parametrize(
    ("benchmark_type", "variant"),
    [
        (BaselineDisaggregatedBenchmark, "baseline"),
        (OptimizedDisaggregatedBenchmark, "optimized"),
    ],
)
def test_factories_declare_explicit_two_rank_worker_and_timing(
    benchmark_type: type, variant: str
) -> None:
    benchmark = benchmark_type()
    config = benchmark.get_config()
    spec = benchmark.get_torchrun_spec(config)
    try:
        assert config.launch_via is LaunchVia.TORCHRUN
        assert config.nproc_per_node == 2
        assert config.multi_gpu_required is True
        assert config.iterations == 10
        assert config.warmup == 5
        assert spec.script_path is not None
        assert spec.script_path.name == "benchmark_worker.py"
        assert spec.script_args == [
            "--module",
            "ch04.disaggregated_multigpu_worker",
            "--callable",
            "main",
            "--",
            "--variant",
            variant,
        ]
        assert spec.result_callback == "consume_disaggregated_child_results"
        assert spec.timing_source == "rank0_time_per_iter_ms"
        assert spec.timing_iterations_per_sample == 10
        assert spec.config_arg_map == {
            "iterations": "--iterations",
            "warmup": "--warmup",
        }
    finally:
        shutil.rmtree(benchmark._disaggregated_result_context["result_dir"])


def test_real_two_rank_gloo_outputs_match_across_variants(tmp_path: Path) -> None:
    baseline = _run_gloo_result(
        BaselineDisaggregatedBenchmark(), _tiny_contract("baseline"), tmp_path
    )
    optimized = _run_gloo_result(
        OptimizedDisaggregatedBenchmark(), _tiny_contract("optimized"), tmp_path
    )

    assert baseline.shape == (2, 2, 5, 8)
    torch.testing.assert_close(optimized, baseline, rtol=1e-5, atol=1e-5)


def test_child_result_requires_every_rank(monkeypatch: pytest.MonkeyPatch) -> None:
    benchmark = BaselineDisaggregatedBenchmark()
    contract = _tiny_contract("baseline")
    result_env = benchmark.prepare_disaggregated_child_result(contract)
    for name, value in result_env.items():
        monkeypatch.setenv(name, value)
    launch_wall_ns = time.time_ns()
    monkeypatch.setenv("AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS", str(launch_wall_ns))
    model, _ = _make_models(
        "baseline",
        hidden_dim=contract.hidden_dim,
        device=torch.device("cpu"),
        seed=contract.seed,
        wrap_ddp=False,
    )
    prefill, decode = _make_inputs(contract, torch.device("cpu"))
    output = _reference_output(model, model, prefill, decode)
    write_disaggregated_child_result(
        contract=contract,
        rank=0,
        prefill_input=prefill,
        decode_input=decode,
        reference_output=output,
        timed_output=output,
    )
    try:
        with pytest.raises(RuntimeError, match="rank quorum is incomplete"):
            benchmark.consume_disaggregated_child_results(
                launch_wall_ns=launch_wall_ns,
                finish_wall_ns=time.time_ns(),
                returncode=0,
            )
    finally:
        shutil.rmtree(benchmark._disaggregated_result_context["result_dir"])


def test_child_result_rejects_missing_full_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = OptimizedDisaggregatedBenchmark()
    contract = _tiny_contract("optimized")
    result_env = benchmark.prepare_disaggregated_child_result(contract)
    launch_wall_ns = time.time_ns()
    for name, value in {
        **result_env,
        "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS": str(launch_wall_ns),
    }.items():
        monkeypatch.setenv(name, value)
    model, decode_model = _make_models(
        "optimized",
        hidden_dim=contract.hidden_dim,
        device=torch.device("cpu"),
        seed=contract.seed,
        wrap_ddp=False,
    )
    prefill, decode = _make_inputs(contract, torch.device("cpu"))
    output = _reference_output(model, decode_model, prefill, decode)
    for rank in range(2):
        write_disaggregated_child_result(
            contract=contract,
            rank=rank,
            prefill_input=prefill,
            decode_input=decode,
            reference_output=output,
            timed_output=output,
        )
    result_dir = benchmark._disaggregated_result_context["result_dir"]
    rank_one = result_dir / "rank-1.pt"
    payload = torch.load(rank_one, map_location="cpu", weights_only=True)
    del payload["timed_output"]
    torch.save(payload, rank_one)
    try:
        with pytest.raises(RuntimeError, match="timed_output shape/dtype/value mismatch"):
            benchmark.consume_disaggregated_child_results(
                launch_wall_ns=launch_wall_ns,
                finish_wall_ns=time.time_ns(),
                returncode=0,
            )
    finally:
        shutil.rmtree(result_dir)


def test_child_result_rejects_rank_runtime_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = BaselineDisaggregatedBenchmark()
    contract = _tiny_contract("baseline")
    result_env = benchmark.prepare_disaggregated_child_result(contract)
    launch_wall_ns = time.time_ns()
    for name, value in {
        **result_env,
        "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS": str(launch_wall_ns),
    }.items():
        monkeypatch.setenv(name, value)
    model, _ = _make_models(
        "baseline",
        hidden_dim=contract.hidden_dim,
        device=torch.device("cpu"),
        seed=contract.seed,
        wrap_ddp=False,
    )
    prefill, decode = _make_inputs(contract, torch.device("cpu"))
    output = _reference_output(model, model, prefill, decode)
    for rank in range(2):
        write_disaggregated_child_result(
            contract=contract,
            rank=rank,
            prefill_input=prefill,
            decode_input=decode,
            reference_output=output,
            timed_output=output,
        )
    result_dir = benchmark._disaggregated_result_context["result_dir"]
    rank_one = result_dir / "rank-1.pt"
    payload = torch.load(rank_one, map_location="cpu", weights_only=True)
    payload["prefill_input"] = payload["prefill_input"].clone()
    payload["prefill_input"][0, 0, 0] += 1.0
    torch.save(payload, rank_one)
    try:
        with pytest.raises(RuntimeError, match="seed-42 runtime parity failed"):
            benchmark.consume_disaggregated_child_results(
                launch_wall_ns=launch_wall_ns,
                finish_wall_ns=time.time_ns(),
                returncode=0,
            )
    finally:
        shutil.rmtree(result_dir)
