from __future__ import annotations

import json
import os
import shutil
import sys
import time
import uuid
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.distributed as dist

from ch04.pipeline_parallel_common import (
    CONTRACT_ENV,
    ITERATIONS_ENV,
    LAUNCH_MONOTONIC_NS_ENV,
    LAUNCH_WALL_NS_ENV,
    PIPELINE_RESULT_LAYERS_ENV,
    PIPELINE_RESULT_SCHEDULE_ENV,
    PIPELINE_RESULT_SHAPE_ENV,
    PIPELINE_RESULT_SOURCE_ENV,
    PIPELINE_RESULT_VARIANT_ENV,
    RESULT_DIR_ENV,
    RUN_ID_ENV,
    WORLD_SIZE_ENV,
    PipelineIterationCapture,
    make_pipeline_child_result_contract,
    run_1f1b_iteration,
    run_gpipe_iteration,
    validate_pipeline_child_result_bundle,
    verify_and_concatenate_pipeline_capture,
    write_pipeline_child_result,
)
from core.harness.benchmark_harness import BenchmarkConfig

_WORLD_SIZE = 2
_MICROBATCHES = 8
_SHAPE = (_MICROBATCHES, 1, 2)
_LAYERS = 2


def _run_environment(
    *,
    result_dir: Path,
    run_id: str,
    source: str,
    variant: str,
    schedule: str,
    launch_wall_ns: int,
    launch_monotonic_ns: int,
) -> dict[str, str]:
    contract = make_pipeline_child_result_contract(
        batch_size=_SHAPE[0],
        parameter_count=1,
    )
    return {
        RESULT_DIR_ENV: str(result_dir),
        RUN_ID_ENV: run_id,
        CONTRACT_ENV: json.dumps(contract.to_dict(), sort_keys=True, separators=(",", ":")),
        WORLD_SIZE_ENV: str(_WORLD_SIZE),
        ITERATIONS_ENV: "1",
        LAUNCH_WALL_NS_ENV: str(launch_wall_ns),
        LAUNCH_MONOTONIC_NS_ENV: str(launch_monotonic_ns),
        PIPELINE_RESULT_SOURCE_ENV: source,
        PIPELINE_RESULT_VARIANT_ENV: variant,
        PIPELINE_RESULT_SCHEDULE_ENV: schedule,
        PIPELINE_RESULT_SHAPE_ENV: json.dumps(_SHAPE, separators=(",", ":")),
        PIPELINE_RESULT_LAYERS_ENV: str(_LAYERS),
    }


def _stage(value: torch.Tensor, increment: float) -> torch.Tensor:
    return value + increment


def _gloo_worker(
    rank: int,
    rendezvous: str,
    run_environments: list[dict[str, str]],
) -> None:
    os.environ["GLOO_SOCKET_IFNAME"] = "lo0" if sys.platform == "darwin" else "lo"
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=_WORLD_SIZE,
    )
    try:
        rank0_input = torch.arange(
            torch.tensor(_SHAPE).prod().item(), dtype=torch.float32
        ).reshape(_SHAPE).to(torch.bfloat16)
        forward_increment = float(rank + 1)
        backward_increment = float((rank + 1) * 10)

        def forward_step(value: torch.Tensor) -> torch.Tensor:
            return _stage(value, forward_increment)

        def backward_step(_activation: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            return _stage(value, backward_increment)

        forward_stages: list[Callable[[torch.Tensor], torch.Tensor]] = [
            lambda value, increment=float(stage + 1): _stage(value, increment)
            for stage in range(_WORLD_SIZE)
        ]
        backward_stages: list[Callable[[torch.Tensor], torch.Tensor]] = [
            lambda value, increment=float((stage + 1) * 10): _stage(value, increment)
            for stage in range(_WORLD_SIZE)
        ]

        for run_index, environment in enumerate(run_environments):
            os.environ.update(environment)
            os.environ["RANK"] = str(rank)
            os.environ["WORLD_SIZE"] = str(_WORLD_SIZE)
            recv_forward = (
                [torch.empty((1, 1, 2), dtype=torch.bfloat16) for _ in range(_MICROBATCHES)]
                if rank > 0
                else []
            )
            recv_backward = (
                [torch.empty((1, 1, 2), dtype=torch.bfloat16) for _ in range(_MICROBATCHES)]
                if rank < _WORLD_SIZE - 1
                else []
            )
            capture = PipelineIterationCapture.create(_MICROBATCHES)
            started = time.perf_counter()
            if run_index == 0:
                run_gpipe_iteration(
                    rank=rank,
                    world_size=_WORLD_SIZE,
                    num_micro_batches=_MICROBATCHES,
                    get_rank0_microbatch=lambda index: rank0_input[index : index + 1],
                    recv_forward_buffers=recv_forward,
                    recv_backward_buffers=recv_backward,
                    forward_step=forward_step,
                    backward_step=backward_step,
                    capture=capture,
                )
            else:
                run_1f1b_iteration(
                    rank=rank,
                    world_size=_WORLD_SIZE,
                    num_micro_batches=_MICROBATCHES,
                    get_rank0_microbatch=lambda index: rank0_input[index : index + 1],
                    recv_forward_buffers=recv_forward,
                    recv_backward_buffers=recv_backward,
                    forward_step=forward_step,
                    backward_step=backward_step,
                    activation_slots=[None] * max(_WORLD_SIZE - rank - 1, 1),
                    capture=capture,
                )
            elapsed_ms = max((time.perf_counter() - started) * 1000.0, 1e-9)
            verify_input, verify_output = verify_and_concatenate_pipeline_capture(
                rank=rank,
                capture=capture,
                forward_stages=forward_stages,
                backward_stages=backward_stages,
                tolerance=(0.0, 0.0),
            )
            result_dir = Path(environment[RESULT_DIR_ENV])
            stdout_path = result_dir.with_name(f"{result_dir.name}.rank-{rank}.stdout")
            with (
                stdout_path.open("w", encoding="utf-8") as stdout_handle,
                redirect_stdout(stdout_handle),
            ):
                assert write_pipeline_child_result(
                    verify_input=verify_input,
                    verify_output=verify_output,
                    completed_iterations=1,
                    time_per_iter_ms=elapsed_ms,
                    reference_verified=True,
                )
                if rank == 0:
                    print(f"rank0 time_per_iter_ms: {elapsed_ms:.9f}", flush=True)
            dist.barrier()
    finally:
        dist.destroy_process_group()


def _validate(
    result_dir: Path,
    *,
    environment: dict[str, str],
    source: str,
    variant: str,
    schedule: str,
    finish_wall_ns: int,
    finish_monotonic_ns: int,
    stdout: str | None = None,
) -> dict[str, Any]:
    if stdout is None:
        stdout = "".join(
            result_dir.with_name(f"{result_dir.name}.rank-{rank}.stdout").read_text(
                encoding="utf-8"
            )
            for rank in range(_WORLD_SIZE)
        )
    return validate_pipeline_child_result_bundle(
        result_dir,
        contract=make_pipeline_child_result_contract(batch_size=_SHAPE[0], parameter_count=1),
        run_id=environment[RUN_ID_ENV],
        source=source,
        variant=variant,
        schedule=schedule,
        world_size=_WORLD_SIZE,
        requested_iterations=1,
        shape=_SHAPE,
        num_layers=_LAYERS,
        launch_wall_ns=int(environment[LAUNCH_WALL_NS_ENV]),
        launch_monotonic_ns=int(environment[LAUNCH_MONOTONIC_NS_ENV]),
        finish_wall_ns=finish_wall_ns,
        finish_monotonic_ns=finish_monotonic_ns,
        stdout=stdout,
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_real_gloo_schedules_transport_timed_full_outputs_and_reject_bad_receipts(
    tmp_path: Path,
) -> None:
    launch_wall_ns = time.time_ns()
    launch_monotonic_ns = time.monotonic_ns()
    baseline_dir = tmp_path / "baseline"
    optimized_dir = tmp_path / "optimized"
    baseline_dir.mkdir()
    optimized_dir.mkdir()
    baseline_environment = _run_environment(
        result_dir=baseline_dir,
        run_id=uuid.uuid4().hex,
        source="ch04.baseline_pipeline_parallel",
        variant="baseline",
        schedule="gpipe",
        launch_wall_ns=launch_wall_ns,
        launch_monotonic_ns=launch_monotonic_ns,
    )
    optimized_environment = _run_environment(
        result_dir=optimized_dir,
        run_id=uuid.uuid4().hex,
        source="ch04.optimized_pipeline_parallel_1f1b",
        variant="optimized",
        schedule="1f1b-paired-p2p",
        launch_wall_ns=launch_wall_ns,
        launch_monotonic_ns=launch_monotonic_ns,
    )
    rendezvous_path = tmp_path / "gloo-rendezvous"
    torch.multiprocessing.spawn(
        _gloo_worker,
        args=(
            f"file://{rendezvous_path}",
            [baseline_environment, optimized_environment],
        ),
        nprocs=_WORLD_SIZE,
        join=True,
        daemon=False,
    )
    finish_monotonic_ns = time.monotonic_ns()
    finish_wall_ns = time.time_ns()

    baseline = _validate(
        baseline_dir,
        environment=baseline_environment,
        source="ch04.baseline_pipeline_parallel",
        variant="baseline",
        schedule="gpipe",
        finish_wall_ns=finish_wall_ns,
        finish_monotonic_ns=finish_monotonic_ns,
    )
    optimized = _validate(
        optimized_dir,
        environment=optimized_environment,
        source="ch04.optimized_pipeline_parallel_1f1b",
        variant="optimized",
        schedule="1f1b-paired-p2p",
        finish_wall_ns=finish_wall_ns,
        finish_monotonic_ns=finish_monotonic_ns,
    )
    assert baseline["input_signature"].matches(optimized["input_signature"])
    assert baseline["verify_inputs"].keys() == optimized["verify_inputs"].keys()
    assert baseline["verify_output"].keys() == optimized["verify_output"].keys()
    for name in baseline["verify_inputs"]:
        torch.testing.assert_close(
            baseline["verify_inputs"][name], optimized["verify_inputs"][name], rtol=0, atol=0
        )
    for name in baseline["verify_output"]:
        torch.testing.assert_close(
            baseline["verify_output"][name], optimized["verify_output"][name], rtol=0, atol=0
        )

    original = torch.arange(16, dtype=torch.float32).reshape(_SHAPE).to(torch.bfloat16)
    torch.testing.assert_close(
        optimized["verify_inputs"]["rank-0:pipeline_input"], original, rtol=0, atol=0
    )
    torch.testing.assert_close(
        optimized["verify_inputs"]["rank-1:pipeline_input"], original + 1, rtol=0, atol=0
    )
    torch.testing.assert_close(
        optimized["verify_output"]["rank-0:pipeline_output"], original + 33, rtol=0, atol=0
    )
    torch.testing.assert_close(
        optimized["verify_output"]["rank-1:pipeline_output"], original + 23, rtol=0, atol=0
    )
    assert len({row["pid"] for row in optimized["metadata_by_rank"].values()}) == 2
    assert all(row["time_per_iter_ms"] > 0 for row in optimized["metadata_by_rank"].values())

    optimized_stdout = "".join(
        optimized_dir.with_name(f"{optimized_dir.name}.rank-{rank}.stdout").read_text(
            encoding="utf-8"
        )
        for rank in range(_WORLD_SIZE)
    )
    receipt_lines = [
        line
        for line in optimized_stdout.splitlines(keepends=True)
        if line.startswith("AISP_PIPELINE_RESULT_RECEIPT:")
    ]
    missing_receipt_stdout = optimized_stdout.replace(receipt_lines[1], "", 1)
    with pytest.raises(RuntimeError, match="stdout receipt rank quorum"):
        _validate(
            optimized_dir,
            environment=optimized_environment,
            source="ch04.optimized_pipeline_parallel_1f1b",
            variant="optimized",
            schedule="1f1b-paired-p2p",
            finish_wall_ns=finish_wall_ns,
            finish_monotonic_ns=finish_monotonic_ns,
            stdout=missing_receipt_stdout,
        )

    receipt_payload = json.loads(
        receipt_lines[0].removeprefix("AISP_PIPELINE_RESULT_RECEIPT:")
    )
    receipt_payload["pid"] += 1
    mismatched_receipt = (
        "AISP_PIPELINE_RESULT_RECEIPT:"
        + json.dumps(receipt_payload, sort_keys=True, separators=(",", ":"))
        + "\n"
    )
    mismatched_receipt_stdout = optimized_stdout.replace(
        receipt_lines[0], mismatched_receipt, 1
    )
    with pytest.raises(RuntimeError, match="metadata differs from its worker stdout receipt"):
        _validate(
            optimized_dir,
            environment=optimized_environment,
            source="ch04.optimized_pipeline_parallel_1f1b",
            variant="optimized",
            schedule="1f1b-paired-p2p",
            finish_wall_ns=finish_wall_ns,
            finish_monotonic_ns=finish_monotonic_ns,
            stdout=mismatched_receipt_stdout,
        )

    missing_dir = tmp_path / "missing"
    shutil.copytree(optimized_dir, missing_dir)
    (missing_dir / "rank-1.meta.json").unlink()
    with pytest.raises(RuntimeError, match="rank quorum"):
        _validate(
            missing_dir,
            environment=optimized_environment,
            source="ch04.optimized_pipeline_parallel_1f1b",
            variant="optimized",
            schedule="1f1b-paired-p2p",
            finish_wall_ns=finish_wall_ns,
            finish_monotonic_ns=finish_monotonic_ns,
            stdout=optimized_stdout,
        )

    stale_dir = tmp_path / "stale"
    shutil.copytree(optimized_dir, stale_dir)
    stale_metadata_path = stale_dir / "rank-0.meta.json"
    stale_metadata = json.loads(stale_metadata_path.read_text(encoding="utf-8"))
    stale_metadata["created_wall_ns"] = launch_wall_ns - 1
    stale_metadata_path.write_text(json.dumps(stale_metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="freshness"):
        _validate(
            stale_dir,
            environment=optimized_environment,
            source="ch04.optimized_pipeline_parallel_1f1b",
            variant="optimized",
            schedule="1f1b-paired-p2p",
            finish_wall_ns=finish_wall_ns,
            finish_monotonic_ns=finish_monotonic_ns,
            stdout=optimized_stdout,
        )

    mismatch_dir = tmp_path / "mismatch"
    shutil.copytree(optimized_dir, mismatch_dir)
    mismatch_metadata_path = mismatch_dir / "rank-0.meta.json"
    mismatch_metadata = json.loads(mismatch_metadata_path.read_text(encoding="utf-8"))
    mismatch_metadata["source"] = "ch04.baseline_pipeline_parallel"
    mismatch_metadata_path.write_text(json.dumps(mismatch_metadata), encoding="utf-8")
    with pytest.raises(RuntimeError, match="source/variant mismatch"):
        _validate(
            mismatch_dir,
            environment=optimized_environment,
            source="ch04.optimized_pipeline_parallel_1f1b",
            variant="optimized",
            schedule="1f1b-paired-p2p",
            finish_wall_ns=finish_wall_ns,
            finish_monotonic_ns=finish_monotonic_ns,
            stdout=optimized_stdout,
        )


def test_post_timing_reference_rejects_a_wrong_actual_microbatch() -> None:
    capture = PipelineIterationCapture.create(2)
    capture.forward_inputs[:] = [torch.ones((1, 2)), torch.full((1, 2), 2.0)]
    capture.backward_inputs[:] = [torch.full((1, 2), 4.0), torch.full((1, 2), 5.0)]
    capture.backward_outputs[:] = [torch.full((1, 2), 5.0), torch.full((1, 2), 999.0)]

    with pytest.raises(AssertionError):
        verify_and_concatenate_pipeline_capture(
            rank=1,
            capture=capture,
            forward_stages=[lambda value: value + 1, lambda value: value + 2],
            backward_stages=[lambda value: value + 10, lambda value: value + 20],
            tolerance=(0.0, 0.0),
        )


def test_default_reference_tolerance_rejects_a_small_bf16_output_error() -> None:
    capture = PipelineIterationCapture.create(1)
    capture.forward_inputs[:] = [torch.ones((1, 2), dtype=torch.bfloat16)]
    capture.backward_inputs[:] = [torch.ones((1, 2), dtype=torch.bfloat16)]
    capture.backward_outputs[:] = [torch.ones((1, 2), dtype=torch.bfloat16)]
    stages = [lambda value: value, lambda value: value]
    _, actual = verify_and_concatenate_pipeline_capture(
        rank=1, capture=capture, forward_stages=stages, backward_stages=stages,
    )
    torch.testing.assert_close(actual, torch.ones_like(actual), rtol=0, atol=0)
    capture.backward_outputs[0][0, 0] += 0.125
    with pytest.raises(AssertionError):
        verify_and_concatenate_pipeline_capture(
            rank=1, capture=capture, forward_stages=stages, backward_stages=stages,
        )


@pytest.mark.parametrize(
    ("module_name", "class_name"),
    (
        ("ch04.baseline_pipeline_parallel", "BaselinePipelineParallelBenchmark"),
        ("ch04.optimized_pipeline_parallel_1f1b", "OptimizedPipelineParallelBenchmark"),
    ),
)
def test_profile_specs_do_not_request_full_tensor_transport(
    module_name: str,
    class_name: str,
    tmp_path: Path,
) -> None:
    module = __import__(module_name, fromlist=[class_name])
    benchmark = getattr(module, class_name)()
    config = BenchmarkConfig(nproc_per_node=2, iterations=1, warmup=5)

    spec = benchmark.get_profile_torchrun_spec(
        profiler="torch",
        config=config,
        output_path=tmp_path / "trace.json",
    )

    assert spec is not None
    assert spec.result_callback is None
    assert RESULT_DIR_ENV not in spec.env
    assert spec.env["AISP_TORCH_PROFILE_OUTPUT"] == str(tmp_path / "trace.json")
    assert benchmark._pipeline_result_context is None
