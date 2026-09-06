from __future__ import annotations

import importlib
import shutil
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ch04.gradient_compression_common import GradientCompressionBenchmark
from ch04.gradient_compression_multigpu_result import (
    GradientCompressionResultContract,
    input_signature,
    make_result_contract,
)
from ch04.gradient_compression_multigpu_worker import (
    _make_buffers,
    _reduce_payload,
    execute_workload,
)
from core.benchmark.verification import (
    get_signature_equivalence_spec,
    signature_workload_dict,
)
from core.harness.benchmark_harness import LaunchVia

FACTORIES = (
    ("baseline", "fp16", False),
    ("optimized", "fp16", False),
    ("baseline", "int8", False),
    ("optimized", "int8", False),
    ("baseline", "fp16", True),
    ("optimized", "fp16", True),
    ("baseline", "int8", True),
    ("optimized", "int8", True),
)


def _contract(
    variant: str,
    compression: str,
    comm_only: bool,
    *,
    tensor_size_mb: int = 2,
) -> GradientCompressionResultContract:
    bucket_mb = 0
    if variant == "baseline" and not comm_only:
        bucket_mb = 1
    tolerance = (1e-3, 1e-2) if compression == "fp16" else (1e-1, 5e-1)
    return make_result_contract(
        variant=variant,
        pair_compression=compression,
        comm_only=comm_only,
        world_size=2,
        tensor_size_mb=tensor_size_mb,
        bucket_mb=bucket_mb,
        iterations=2,
        warmup=1,
        seed=42,
        output_tolerance=tolerance,
    )


def _gloo_worker(
    rank: int,
    rendezvous: str,
    result_contract: GradientCompressionResultContract,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=2,
    )
    try:
        bucketed = _contract("baseline", "fp16", False)
        bucketed_input = torch.ones(bucketed.numel, dtype=torch.float32)
        bucketed_buffers = _make_buffers(bucketed, bucketed_input)
        assert bucketed_buffers.compressed is not None
        bucketed_buffers.compressed.fill_(rank + 1)
        bucketed_storage = bucketed_buffers.compressed.data_ptr()
        bucketed_output = _reduce_payload(
            bucketed,
            bucketed_buffers.compressed,
            bucketed_buffers,
            group=None,
            recorder=None,
        )
        assert bucketed_output.data_ptr() == bucketed_storage
        assert torch.equal(bucketed_output, torch.full_like(bucketed_output, 3))

        comm_only = _contract("optimized", "fp16", True)
        comm_input = torch.ones(comm_only.numel, dtype=torch.float16).mul_(rank + 1)
        comm_before = comm_input.clone()
        comm_buffers = _make_buffers(
            comm_only, torch.empty(comm_only.numel, dtype=torch.float32)
        )
        comm_output = _reduce_payload(
            comm_only,
            comm_input,
            comm_buffers,
            group=None,
            recorder=None,
        )
        assert comm_output.data_ptr() != comm_input.data_ptr()
        assert torch.equal(comm_input, comm_before)
        assert torch.equal(comm_output, torch.full_like(comm_output, 3))

        for variant, compression, comm_only in FACTORIES:
            contract = _contract(variant, compression, comm_only)
            rng_before = torch.random.get_rng_state().clone()
            _, output, reference, receipt = execute_workload(
                contract,
                rank=rank,
                device=torch.device("cpu"),
                publish_result=contract == result_contract,
            )
            torch.testing.assert_close(
                output,
                reference,
                rtol=contract.output_rtol,
                atol=contract.output_atol,
            )
            assert len(receipt.collective_launch_ns) == contract.expected_collectives_per_rank
            assert len(receipt.collective_completion_ns) == contract.expected_collectives_per_rank
            assert torch.equal(torch.random.get_rng_state(), rng_before)
    finally:
        dist.destroy_process_group()


def _make_parent(contract: GradientCompressionResultContract) -> GradientCompressionBenchmark:
    return GradientCompressionBenchmark(
        compression=contract.effective_compression,
        equivalence_group="test_gradient_compression_multigpu",
        output_tolerance=(contract.output_rtol, contract.output_atol),
        tensor_size_mb=contract.tensor_size_mb,
        multi_gpu=True,
        comm_only=contract.comm_only,
        use_prealloc_buffers=contract.variant == "optimized",
        bucket_mb=contract.bucket_mb,
        variant=contract.variant,
        pair_compression=contract.pair_compression,
    )


def test_multigpu_factories_declare_real_two_rank_worker_and_one_gib_gradient() -> None:
    for variant, compression, comm_only in FACTORIES:
        suffix = "_comm_only" if comm_only else ""
        module_name = (
            f"ch04.{variant}_gradient_compression_{compression}{suffix}_multigpu"
        )
        benchmark = importlib.import_module(module_name).get_benchmark()
        config = benchmark.get_config()
        spec = benchmark.get_torchrun_spec(config)
        context = benchmark._gradient_compression_result_context
        try:
            assert benchmark.tensor_size_mb == 1024
            assert config.launch_via is LaunchVia.TORCHRUN
            assert config.nproc_per_node == config.required_world_size == 2
            assert config.adaptive_iterations is False
            assert spec.result_callback == "consume_gradient_compression_child_results"
            assert spec.timing_source == "rank0_time_per_iter_ms"
            assert spec.timing_iterations_per_sample == config.iterations == 10
            assert spec.config_arg_map == {
                "iterations": "--iterations",
                "warmup": "--warmup",
            }
            assert spec.script_path is not None
            assert "ch04.gradient_compression_multigpu_worker" in spec.script_args
            contract = context["contract"]
            assert contract.variant == variant
            assert contract.pair_compression == compression
            assert contract.comm_only is comm_only
            assert contract.full_gradient_bytes == 1024**3
            assert context["raw_result_tensor_bytes"] == 5 * 1024**3
        finally:
            shutil.rmtree(context["result_dir"])


def test_pair_signatures_preserve_same_workload_while_precision_is_explicit() -> None:
    for compression in ("fp16", "int8"):
        baseline = _make_parent(_contract("baseline", compression, True))
        optimized = _make_parent(_contract("optimized", compression, True))
        baseline_contract = _contract("baseline", compression, True)
        optimized_contract = _contract("optimized", compression, True)
        equivalence = get_signature_equivalence_spec(baseline)
        assert equivalence == get_signature_equivalence_spec(optimized)
        assert signature_workload_dict(
            input_signature(baseline_contract), equivalence=equivalence
        ) == signature_workload_dict(
            input_signature(optimized_contract), equivalence=equivalence
        )
        baseline_precision = input_signature(baseline_contract).precision_flags
        optimized_precision = input_signature(optimized_contract).precision_flags
        if compression == "fp16":
            assert baseline_precision != optimized_precision
        else:
            # PrecisionFlags has no INT8 member; both stored verification views
            # are truthfully FP32, while the result contract records INT8 wire work.
            assert baseline_precision == optimized_precision


def test_real_two_rank_gloo_work_and_fresh_full_output_quorum(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_contract = _contract("optimized", "int8", True)
    parent = _make_parent(result_contract)
    result_env = parent.prepare_gradient_compression_child_result(result_contract)
    for key, value in result_env.items():
        monkeypatch.setenv(key, value)
    launch_wall_ns = time.time_ns()
    monkeypatch.setenv(
        "AISP_TORCHRUN_RESULT_LAUNCH_WALL_NS", str(launch_wall_ns)
    )
    rendezvous = tmp_path / "gloo-rendezvous"
    mp.start_processes(
        _gloo_worker,
        args=(str(rendezvous), result_contract),
        nprocs=2,
        join=True,
        start_method="spawn",
    )
    parent.consume_gradient_compression_child_results(
        launch_wall_ns=launch_wall_ns,
        finish_wall_ns=time.time_ns(),
        returncode=0,
    )

    assert parent.validate_result() is None
    assert parent.get_verify_output().shape == (result_contract.numel,)
    assert set(parent.get_verify_inputs()) == {
        "rank_0_input",
        "rank_1_input",
        "reference",
    }
    bundle = parent._gradient_compression_result_bundle
    assert bundle["raw_result_tensor_bytes"] == result_contract.raw_result_tensor_bytes
    assert bundle["collective_algorithm_evidence"] == "declared_only"
    assert bundle["process_collective_mode"] == "functional_out_of_place_constant_payload"
    assert parent._gradient_compression_result_context["retention"] == "cleaned-after-success"
    assert not Path(parent._gradient_compression_result_context["result_dir"]).exists()


def test_missing_rank_receipt_fails_and_retains_result_directory() -> None:
    contract = _contract("optimized", "fp16", False, tensor_size_mb=1)
    parent = _make_parent(contract)
    parent.prepare_gradient_compression_child_result(contract)
    context = parent._gradient_compression_result_context
    with pytest.raises(RuntimeError, match="rank quorum is incomplete"):
        parent.consume_gradient_compression_child_results(
            launch_wall_ns=time.time_ns() - 1_000,
            finish_wall_ns=time.time_ns(),
            returncode=0,
        )
    try:
        assert context["retention"] == "retained-incomplete-rank-quorum"
        assert Path(context["result_dir"]).is_dir()
    finally:
        shutil.rmtree(context["result_dir"])


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"world_size": 1}, "exactly two ranks"),
        ({"bucket_mb": 0}, "requires its declared small bucket"),
        ({"pair_compression": "fp8"}, "Unsupported gradient-compression pair"),
        ({"iterations": 0}, "iterations must be a positive integer"),
    ],
)
def test_invalid_contracts_fail_closed(changes: dict[str, object], match: str) -> None:
    values = _contract("baseline", "fp16", False).to_dict()
    values.update(changes)
    with pytest.raises((TypeError, ValueError), match=match):
        GradientCompressionResultContract.from_dict(values)
