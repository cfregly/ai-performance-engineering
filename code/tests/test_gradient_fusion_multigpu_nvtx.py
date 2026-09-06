from __future__ import annotations

import ast
import os
import shutil
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from ch04.baseline_gradient_fusion_multigpu import BaselineGradientFusionMultiGPU
from ch04.gradient_fusion_multigpu import (
    BASELINE_PROFILE_NVTX_RANGE,
    OPTIMIZED_PROFILE_NVTX_RANGE,
    PAIR_NUM_TENSORS,
    PAIR_TENSOR_KB,
    PAIR_TIMED_ITERATIONS,
    _run_collectives,
    _stable_average_reduce_op,
)
from ch04.gradient_fusion_result import (
    GRADIENT_FUSION_LAUNCH_WALL_NS_ENV,
    GRADIENT_FUSION_RESULT_CALLBACK,
    GRADIENT_FUSION_RESULT_DIR_ENV,
    GRADIENT_FUSION_RESULT_TOKEN_ENV,
    GRADIENT_FUSION_VARIANT_ENV,
    write_gradient_fusion_child_result,
)
from ch04.optimized_gradient_fusion_multigpu import OptimizedGradientFusionMultiGPU
from core.harness.benchmark_harness import BenchmarkConfig
from core.profiling.profiler_config import build_profiler_config_from_benchmark


@pytest.mark.parametrize(
    ("benchmark_type", "expected_range"),
    [
        (BaselineGradientFusionMultiGPU, BASELINE_PROFILE_NVTX_RANGE),
        (OptimizedGradientFusionMultiGPU, OPTIMIZED_PROFILE_NVTX_RANGE),
    ],
)
def test_direct_torchrun_profile_declares_its_real_nvtx_range(
    benchmark_type: type,
    expected_range: str,
) -> None:
    config = benchmark_type().get_config()
    assert config.profiling.nsys_nvtx_include == (expected_range,)

    profiler = build_profiler_config_from_benchmark(config)
    command = profiler.get_ncu_command_for_target(
        "/tmp/gradient-fusion-profile",
        ["python", "gradient_fusion_multigpu.py"],
    )
    include_index = command.index("--nvtx-include")
    assert command[include_index + 1] == expected_range
    assert command[command.index("--replay-mode") + 1] == "app-range"

    benchmark = benchmark_type()
    config = BenchmarkConfig(nproc_per_node=2)
    ncu_spec = benchmark.get_profile_torchrun_spec(profiler="ncu", config=config)
    assert ncu_spec is not None
    assert ncu_spec.script_args[-2:] == ["--profile-rank", "0"]
    assert ncu_spec.timing_iterations_per_sample == 1
    nsys_spec = benchmark.get_profile_torchrun_spec(profiler="nsys", config=config)
    assert nsys_spec.timing_iterations_per_sample == 1
    torch_spec = benchmark.get_profile_torchrun_spec(
        profiler="torch", config=config, output_path=Path("/tmp/gradient-trace.json")
    )
    assert torch_spec.timing_iterations_per_sample == 1
    assert torch_spec.env["AISP_TORCH_PROFILE_OUTPUT"] == "/tmp/gradient-trace.json"
    ordinary_spec = benchmark.get_torchrun_spec(config)
    assert ordinary_spec.timing_iterations_per_sample == PAIR_TIMED_ITERATIONS


def test_real_target_limits_nvtx_range_to_timed_collectives() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "ch04/gradient_fusion_multigpu.py"
    ).read_text(encoding="utf-8")
    run_body = source.split("def run_benchmark", 1)[1].split("def parse_args", 1)[0]
    warmup_call = run_body.index(
        "_run_collectives(mode, tensors, fused, iterations=5, reduce_op=average_op)"
    )
    start_event = run_body.index("start.record()")
    range_start = run_body.index("with nvtx_range(profile_range, enable=profile_rank is None or rank == profile_rank):")
    timed_call = run_body.index("iterations=iterations", range_start)
    end_event = run_body.index("end.record()")
    result_write = run_body.index("write_gradient_fusion_child_result(")

    assert warmup_call < start_event < range_start < timed_call < end_event < result_write
    assert 'BASELINE_PROFILE_NVTX_RANGE = "compute_kernel:gradient_fusion_many_allreduces"' in source
    assert 'OPTIMIZED_PROFILE_NVTX_RANGE = "compute_kernel:gradient_fusion_fused_allreduce"' in source
    assert 'if mode == "baseline"\n        else OPTIMIZED_PROFILE_NVTX_RANGE' in run_body
    assert 'make_premul_sum(1.0 / world_size)' in source
    assert 'print(f"rank0 time_per_iter_ms: {time_per_iter_ms:.9f}"' in source

    tree = ast.parse(source)
    run_function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "run_benchmark"
    )
    timed_range = next(
        node
        for node in ast.walk(run_function)
        if isinstance(node, ast.With)
        and isinstance(node.items[0].context_expr, ast.Call)
        and ast.unparse(node.items[0].context_expr.func) == "nvtx_range"
    )
    calls_in_range = {
        ast.unparse(node.func)
        for node in ast.walk(timed_range)
        if isinstance(node, ast.Call)
    }
    assert {"_run_collectives", "end.record", "torch.cuda.synchronize"} <= calls_in_range


def _gloo_worker(rank: int, world_size: int, init_method: str, result_dir: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        average_op = _stable_average_reduce_op(world_size)
        tensors = [
            torch.tensor([rank + 0.25 + index], dtype=torch.float16)
            for index in range(3)
        ]
        _run_collectives(
            "baseline",
            tensors,
            None,
            iterations=55,
            reduce_op=average_op,
        )
        for index, tensor in enumerate(tensors):
            torch.testing.assert_close(
                tensor,
                torch.tensor([0.75 + index], dtype=torch.float16),
            )
            assert bool(torch.isfinite(tensor).all())

        fused = torch.tensor(
            [rank + 0.25, rank + 1.25],
            dtype=torch.float16,
        )
        _run_collectives(
            "optimized",
            [],
            fused,
            iterations=55,
            reduce_op=average_op,
        )
        torch.testing.assert_close(
            fused,
            torch.tensor([0.75, 1.75], dtype=torch.float16),
        )
        assert bool(torch.isfinite(fused).all())
        (Path(result_dir) / f"rank-{rank}.passed").write_text("passed\n", encoding="utf-8")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="the real CPU collective control requires the Gloo backend",
)
def test_gradient_fusion_cpu_gloo_average_control_runs_55_times_on_two_real_ranks(
    tmp_path: Path,
) -> None:
    world_size = 2
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    mp.spawn(
        _gloo_worker,
        args=(world_size, (tmp_path / "rendezvous").as_uri(), str(result_dir)),
        nprocs=world_size,
        join=True,
        daemon=False,
    )
    assert sorted(path.name for path in result_dir.iterdir()) == [
        "rank-0.passed",
        "rank-1.passed",
    ]


def _gloo_transport_worker(
    rank: int,
    world_size: int,
    init_method: str,
    launch_wall_ns: int,
    transports: list[tuple[str, dict[str, str]]],
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=30),
    )
    try:
        num_tensors = 2
        tensor_kb = 1
        iterations = 5
        numel = tensor_kb * 1024 // 2
        initial = (
            torch.arange(num_tensors * numel, dtype=torch.float16) / 1024
            + rank
        )
        reference = initial.float()
        dist.all_reduce(reference, op=dist.ReduceOp.SUM)
        reference.div_(world_size)
        reference = reference.half()
        average_op = _stable_average_reduce_op(world_size)

        for variant, transport in transports:
            os.environ.update(transport)
            os.environ[GRADIENT_FUSION_LAUNCH_WALL_NS_ENV] = str(launch_wall_ns)
            working = initial.clone()
            tensors = list(working.view(num_tensors, numel).unbind(0))
            fused = working if variant == "optimized" else None
            _run_collectives(
                variant,
                tensors,
                fused,
                iterations=iterations,
                reduce_op=average_op,
            )
            output = torch.cat(tensors) if variant == "baseline" else fused
            assert output is not None
            assert write_gradient_fusion_child_result(
                variant=variant,
                rank=rank,
                world_size=world_size,
                num_tensors=num_tensors,
                tensor_kb=tensor_kb,
                iterations=iterations,
                initial_gradients=initial,
                reference_average=reference,
                verify_output=output,
            )
            dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="the real CPU child-result control requires the Gloo backend",
)
def test_gradient_fusion_real_rank_outputs_reach_pair_verification(
    tmp_path: Path,
) -> None:
    benchmarks = [BaselineGradientFusionMultiGPU(), OptimizedGradientFusionMultiGPU()]
    transports: list[tuple[str, dict[str, str]]] = []
    for variant, benchmark in zip(("baseline", "optimized"), benchmarks, strict=True):
        transports.append(
            (
                variant,
                benchmark.prepare_gradient_fusion_child_result(
                    variant=variant,
                    world_size=2,
                    num_tensors=2,
                    tensor_kb=1,
                    iterations=5,
                ),
            )
        )

    launch_wall_ns = time.time_ns()
    try:
        mp.spawn(
            _gloo_transport_worker,
            args=(
                2,
                (tmp_path / "transport-rendezvous").as_uri(),
                launch_wall_ns,
                transports,
            ),
            nprocs=2,
            join=True,
            daemon=False,
        )
        finish_wall_ns = time.time_ns()
        for benchmark in benchmarks:
            benchmark.consume_gradient_fusion_child_results(
                launch_wall_ns=launch_wall_ns,
                finish_wall_ns=finish_wall_ns,
                returncode=0,
            )
            benchmark.capture_verification_payload()
            assert benchmark.get_input_signature().collective_algorithm == (
                "premultiplied_sum_average"
            )
            assert benchmark.get_verify_output().numel() == 1024

        torch.testing.assert_close(
            benchmarks[0].get_verify_output(),
            benchmarks[1].get_verify_output(),
        )
    finally:
        for benchmark in benchmarks:
            context = benchmark._gradient_fusion_result_context
            if context is not None:
                shutil.rmtree(context["result_dir"], ignore_errors=True)


@pytest.mark.parametrize(
    ("benchmark_type", "variant"),
    [
        (BaselineGradientFusionMultiGPU, "baseline"),
        (OptimizedGradientFusionMultiGPU, "optimized"),
    ],
)
def test_gradient_fusion_specs_bind_full_output_callback_and_worker_timing(
    benchmark_type: type,
    variant: str,
) -> None:
    benchmark = benchmark_type()
    spec = benchmark.get_torchrun_spec(BenchmarkConfig(nproc_per_node=2))
    try:
        assert spec.result_callback == GRADIENT_FUSION_RESULT_CALLBACK
        assert spec.timing_source == "rank0_time_per_iter_ms"
        assert spec.timing_iterations_per_sample == PAIR_TIMED_ITERATIONS
        assert spec.script_args == [
            "--mode",
            variant,
            "--num-tensors",
            str(PAIR_NUM_TENSORS),
            "--tensor-kb",
            str(PAIR_TENSOR_KB),
            "--iterations",
            str(PAIR_TIMED_ITERATIONS),
        ]
        assert set(spec.env) == {
            GRADIENT_FUSION_RESULT_DIR_ENV,
            GRADIENT_FUSION_RESULT_TOKEN_ENV,
            GRADIENT_FUSION_VARIANT_ENV,
        }
        assert Path(spec.env[GRADIENT_FUSION_RESULT_DIR_ENV]).is_dir()
        with pytest.raises(RuntimeError, match="fresh measured child result"):
            benchmark.capture_verification_payload()
    finally:
        shutil.rmtree(spec.env[GRADIENT_FUSION_RESULT_DIR_ENV], ignore_errors=True)


def test_gradient_fusion_callback_rejects_one_changed_output_element(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = BaselineGradientFusionMultiGPU()
    transport = benchmark.prepare_gradient_fusion_child_result(
        variant="baseline",
        world_size=2,
        num_tensors=2,
        tensor_kb=1,
        iterations=5,
    )
    launch_wall_ns = time.time_ns()
    for name, value in transport.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setenv(GRADIENT_FUSION_LAUNCH_WALL_NS_ENV, str(launch_wall_ns))

    result_dir = Path(transport[GRADIENT_FUSION_RESULT_DIR_ENV])
    try:
        inputs = [
            torch.arange(1024, dtype=torch.float16) / 1024 + rank
            for rank in range(2)
        ]
        reference = ((inputs[0].float() + inputs[1].float()) / 2).half()
        for rank in range(2):
            assert write_gradient_fusion_child_result(
                variant="baseline",
                rank=rank,
                world_size=2,
                num_tensors=2,
                tensor_kb=1,
                iterations=5,
                initial_gradients=inputs[rank],
                reference_average=reference,
                verify_output=reference.clone(),
            )

        rank1_path = result_dir / "rank-1.pt"
        rank1_payload = torch.load(rank1_path, map_location="cpu", weights_only=True)
        rank1_payload["verify_output"][731] += 1
        torch.save(rank1_payload, rank1_path)

        with pytest.raises(RuntimeError, match="full timed output differs"):
            benchmark.consume_gradient_fusion_child_results(
                launch_wall_ns=launch_wall_ns,
                finish_wall_ns=time.time_ns(),
                returncode=0,
            )
        assert result_dir.is_dir()
    finally:
        shutil.rmtree(result_dir, ignore_errors=True)
