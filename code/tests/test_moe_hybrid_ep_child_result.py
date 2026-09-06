"""CPU controls for the hybrid-EP worker result and launch contract."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from core.harness.benchmark_harness import BenchmarkConfig, LaunchVia
from labs.fullstack_cluster import moe_hybrid_ep_common as common
from labs.fullstack_cluster.baseline_moe_hybrid_ep import get_benchmark
from labs.fullstack_cluster.moe_hybrid_ep_common import (
    BASELINE_MOE_HYBRID_EP_NVTX_RANGE,
    OPTIMIZED_MOE_HYBRID_EP_NVTX_RANGE,
)
from labs.fullstack_cluster.moe_hybrid_ep_result import (
    MOE_HYBRID_EP_LAUNCH_WALL_NS_ENV,
    MOE_HYBRID_EP_RESULT_CALLBACK,
    MOE_HYBRID_EP_RESULT_DIR_ENV,
    MoEHybridEPResultContract,
    _input_signature,
)
from labs.fullstack_cluster.optimized_moe_hybrid_ep import (
    get_benchmark as get_optimized_benchmark,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = REPO_ROOT / "code"


def _small_contract(*, label: str = "baseline_moe_hybrid_ep") -> MoEHybridEPResultContract:
    return MoEHybridEPResultContract(
        label=label,
        variant="optimized" if label.startswith("optimized") else "baseline",
        world_size=2,
        iterations=3,
        warmup_steps=2,
        tokens_per_rank=3,
        hidden_size=4,
        num_experts=4,
        local_experts=2,
        top_k=2,
        route_mode="uniform",
        dtype=str(torch.float32),
        learning_rate=2e-4,
        aux_loss_scale=1e-2,
        tf32=False,
        profile_range=BASELINE_MOE_HYBRID_EP_NVTX_RANGE,
    )


def test_single_rank_signature_declares_local_work_without_a_collective() -> None:
    contract = replace(_small_contract(), world_size=1, num_experts=2)
    contract.validate()
    signature = _input_signature(contract)
    assert signature.validate(strict=True) == []
    assert signature.world_size == 1
    assert signature.ranks == [0]
    assert signature.shapes["output"] == (contract.tokens_per_rank, contract.hidden_size)
    assert signature.collective_type is None
    assert signature.collective_algorithm is None
    distributed = _input_signature(_small_contract())
    assert distributed.validate(strict=True) == []
    assert distributed.collective_type == "all_to_all"
    assert distributed.collective_algorithm == "bidirectional_expert_route_exchange"


def _write_rank_in_real_subprocess(
    *,
    transport: dict[str, str],
    contract: MoEHybridEPResultContract,
    launch_wall_ns: int,
    rank: int,
) -> subprocess.CompletedProcess[str]:
    worker = r"""
import json
import os
import torch

from labs.fullstack_cluster.moe_hybrid_ep_result import (
    MoEHybridEPResultContract,
    write_moe_hybrid_ep_child_result,
)

contract = MoEHybridEPResultContract.from_dict(json.loads(os.environ["TEST_CONTRACT"]))
rank = int(os.environ["RANK"])
values = torch.arange(
    contract.tokens_per_rank * contract.hidden_size,
    dtype=torch.float32,
).reshape(contract.tokens_per_rank, contract.hidden_size)
inputs = values + rank
targets = values / 7 + rank
routes = torch.stack(
    (
        torch.arange(contract.tokens_per_rank) % contract.num_experts,
        (torch.arange(contract.tokens_per_rank) + 1) % contract.num_experts,
    ),
    dim=1,
).to(torch.int64)
output = inputs * 0.5 + targets * 0.25
hidden = contract.hidden_size
parameter_count = (
    (2 * hidden * hidden)
    + (hidden * contract.num_experts)
    + (contract.local_experts * 12 * hidden * hidden)
)
written = write_moe_hybrid_ep_child_result(
    contract=contract,
    rank=rank,
    parameter_count=parameter_count,
    inputs=inputs,
    targets=targets,
    route_assignments=routes,
    verify_output=output,
    reference_route_assignments=routes.clone(),
    reference_output=output.clone(),
    custom_metrics={"moe.step.total_ms": 1.25, "moe.step.loss": float(rank + 1)},
    time_per_iter_ms=1.25 + rank * 0.125,
)
assert written
"""
    env = os.environ.copy()
    env.update(transport)
    env.update(
        {
            "PYTHONPATH": str(CODE_ROOT),
            "TEST_CONTRACT": json.dumps(contract.to_dict()),
            MOE_HYBRID_EP_LAUNCH_WALL_NS_ENV: str(launch_wall_ns),
            "RANK": str(rank),
        }
    )
    return subprocess.run(
        [sys.executable, "-c", worker],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )


@pytest.mark.parametrize(
    ("factory", "expected_range"),
    [
        (get_benchmark, BASELINE_MOE_HYBRID_EP_NVTX_RANGE),
        (get_optimized_benchmark, OPTIMIZED_MOE_HYBRID_EP_NVTX_RANGE),
    ],
)
def test_torchrun_spec_declares_worker_timing_result_callback_and_app_range(
    factory,
    expected_range: str,
) -> None:
    benchmark = factory()
    config = BenchmarkConfig(
        launch_via=LaunchVia.TORCHRUN,
        nproc_per_node=2,
        nnodes="1",
        iterations=3,
        warmup=5,
    )

    spec = benchmark.get_torchrun_spec(config)

    assert spec.result_callback == MOE_HYBRID_EP_RESULT_CALLBACK
    assert spec.timing_source == "rank0_time_per_iter_ms"
    assert spec.timing_iterations_per_sample == 3
    assert spec.config_arg_map == {
        "iterations": "--iters",
        "warmup": "--warmup-steps",
    }
    assert MOE_HYBRID_EP_RESULT_DIR_ENV in spec.env
    assert benchmark.get_config().nsys_nvtx_include == [expected_range]
    with pytest.raises(RuntimeError, match="cannot be prepared before launch"):
        benchmark._prepare_verification_payload()


def test_pair_signature_uses_semantic_workload_and_ignores_variant_label() -> None:
    baseline = _small_contract()
    optimized = replace(
        baseline,
        label="optimized_moe_hybrid_ep",
        variant="optimized",
        profile_range=OPTIMIZED_MOE_HYBRID_EP_NVTX_RANGE,
    )

    assert _input_signature(baseline).to_dict() == _input_signature(optimized).to_dict()


def test_rank0_timing_is_computed_from_trainer_cuda_events() -> None:
    source = Path(common.__file__).read_text(encoding="utf-8")
    forward_section = source.split("def forward_loss", maxsplit=1)[1].split(
        "class HybridEPTrainer", maxsplit=1
    )[0]
    step_section = source.split("def run_step", maxsplit=1)[1].split(
        "def _reduce_metrics", maxsplit=1
    )[0]

    assert "rank_time_per_iter_ms" not in forward_section
    assert "rank_time_per_iter_ms = float(total_start.elapsed_time(total_end))" in step_section
    assert step_section.index("torch.cuda.synchronize()") < step_section.index(
        "rank_time_per_iter_ms ="
    )


def test_real_cpu_subprocess_rank_quorum_exposes_full_outputs_and_inputs() -> None:
    benchmark = get_benchmark()
    contract = _small_contract()
    transport = benchmark.prepare_moe_hybrid_ep_child_result(contract)
    result_dir = Path(transport[MOE_HYBRID_EP_RESULT_DIR_ENV])
    launch_wall_ns = time.time_ns()

    for rank in range(contract.world_size):
        result = _write_rank_in_real_subprocess(
            transport=transport,
            contract=contract,
            launch_wall_ns=launch_wall_ns,
            rank=rank,
        )
        assert result.returncode == 0, result.stderr

    finish_wall_ns = time.time_ns()
    benchmark.consume_moe_hybrid_ep_child_results(
        launch_wall_ns=launch_wall_ns,
        finish_wall_ns=finish_wall_ns,
        returncode=0,
        stdout="rank0 time_per_iter_ms: 1.250000000\n",
    )

    output = benchmark.get_verify_output()
    inputs = benchmark.get_verify_inputs()
    signature = benchmark.get_input_signature()
    assert output.shape == (6, 4)
    assert output.numel() > 1
    assert inputs["inputs"].shape == (6, 4)
    assert inputs["targets"].shape == (6, 4)
    assert inputs["route_assignments"].shape == (6, 2)
    torch.testing.assert_close(output, inputs["reference_output"], rtol=1e-5, atol=1e-8)
    assert signature.world_size == 2
    assert signature.ranks == [0, 1]
    assert signature.collective_type == "all_to_all"
    assert signature.collective_algorithm == "bidirectional_expert_route_exchange"
    assert benchmark.validate_result() is None
    assert benchmark.get_custom_metrics()["moe.step.total_ms"] == 1.25
    assert benchmark._moe_hybrid_ep_result_context["retention"] == "cleaned-after-success"
    assert not result_dir.exists()


def test_missing_rank_receipt_is_retained_and_cannot_verify() -> None:
    benchmark = get_benchmark()
    contract = _small_contract()
    transport = benchmark.prepare_moe_hybrid_ep_child_result(contract)
    result_dir = Path(transport[MOE_HYBRID_EP_RESULT_DIR_ENV])
    launch_wall_ns = time.time_ns()
    result = _write_rank_in_real_subprocess(
        transport=transport,
        contract=contract,
        launch_wall_ns=launch_wall_ns,
        rank=0,
    )
    assert result.returncode == 0, result.stderr

    with pytest.raises(RuntimeError, match="rank quorum is incomplete"):
        benchmark.consume_moe_hybrid_ep_child_results(
            launch_wall_ns=launch_wall_ns,
            finish_wall_ns=time.time_ns(),
            returncode=0,
            stdout="rank0 time_per_iter_ms: 1.250000000\n",
        )

    assert result_dir.exists()
    assert benchmark.validate_result() == "Fresh full-rank hybrid-EP worker output is missing"
    assert benchmark._moe_hybrid_ep_result_context["retention"] == (
        "retained-incomplete-rank-quorum"
    )

    second = _write_rank_in_real_subprocess(
        transport=transport,
        contract=contract,
        launch_wall_ns=launch_wall_ns,
        rank=1,
    )
    assert second.returncode == 0, second.stderr
    with pytest.raises(RuntimeError, match="launch identity mismatch"):
        benchmark.consume_moe_hybrid_ep_child_results(
            launch_wall_ns=launch_wall_ns - 1,
            finish_wall_ns=time.time_ns(),
            returncode=0,
            stdout="rank0 time_per_iter_ms: 1.250000000\n",
        )
    assert result_dir.exists()
    assert benchmark._moe_hybrid_ep_result_context["retention"] == (
        "retained-invalid-child-result"
    )


def test_worker_rejects_output_that_does_not_match_full_reference(monkeypatch, tmp_path) -> None:
    from labs.fullstack_cluster.moe_hybrid_ep_result import (
        MOE_HYBRID_EP_RESULT_LABEL_ENV,
        MOE_HYBRID_EP_RESULT_TOKEN_ENV,
        write_moe_hybrid_ep_child_result,
    )

    contract = _small_contract()
    monkeypatch.setenv(MOE_HYBRID_EP_RESULT_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(MOE_HYBRID_EP_RESULT_TOKEN_ENV, "fresh-token")
    monkeypatch.setenv(MOE_HYBRID_EP_RESULT_LABEL_ENV, contract.label)
    monkeypatch.setenv(MOE_HYBRID_EP_LAUNCH_WALL_NS_ENV, str(time.time_ns()))
    inputs = torch.ones(3, 4)
    routes = torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.int64)
    parameter_count = (2 * 4 * 4) + (4 * 4) + (2 * 12 * 4 * 4)

    with pytest.raises(RuntimeError, match="full timed output differs"):
        write_moe_hybrid_ep_child_result(
            contract=contract,
            rank=0,
            parameter_count=parameter_count,
            inputs=inputs,
            targets=inputs,
            route_assignments=routes,
            verify_output=torch.zeros(3, 4),
            reference_route_assignments=routes,
            reference_output=torch.ones(3, 4),
            custom_metrics={"moe.step.total_ms": 1.0},
            time_per_iter_ms=1.0,
        )
    assert not list(tmp_path.glob("rank-*.pt"))


def test_canonical_replay_loads_exact_state_and_data_outside_candidate(monkeypatch) -> None:
    instances = []

    class FakeTrainer:
        def __init__(self, args, topology, *, optimized):
            self.args = args
            self.topology = topology
            self.optimized = optimized
            self.model = torch.nn.Linear(2, 2, bias=False)
            self.inputs = torch.zeros(2, 2)
            self.targets = torch.zeros(2, 2)
            self.calls = 0
            instances.append(self)

        def run_step(self):
            self.calls += 1
            output = self.inputs + self.model.weight[0, 0]
            routes = torch.tensor([[0], [1]], dtype=torch.int64)
            return common.StepArtifacts(
                metrics={"moe.step.total_ms": 1.0},
                loss=0.0,
                output=output,
                route_assignments=routes,
            )

    monkeypatch.setattr(common, "HybridEPTrainer", FakeTrainer)
    args = SimpleNamespace(iters=2, overlap_mode="local_remote")
    initial_state = {"weight": torch.full((2, 2), 3.0)}
    inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    targets = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    output, routes = common._run_canonical_reference(
        args=args,
        topology=SimpleNamespace(),
        initial_state=initial_state,
        inputs=inputs,
        targets=targets,
        warmup_steps=2,
    )

    reference = instances[0]
    assert reference.optimized is False
    assert reference.args.overlap_mode == "disabled"
    assert reference.calls == 4
    torch.testing.assert_close(reference.inputs, inputs)
    torch.testing.assert_close(reference.targets, targets)
    torch.testing.assert_close(output, inputs + 3.0)
    torch.testing.assert_close(routes, torch.tensor([[0], [1]], dtype=torch.int64))
