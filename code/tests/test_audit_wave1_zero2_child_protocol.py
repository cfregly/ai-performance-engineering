"""Actual child-result and negative controls for the four ZeRO adapters."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from core.harness.benchmark_harness import BenchmarkConfig, BenchmarkHarness, LaunchVia
from labs.train_distributed.training_utils.zero2_torchrun_benchmark import (
    Zero2TorchrunBenchmark,
)
import labs.train_distributed.training_utils.zero2_torchrun_benchmark as zero2_adapter
import labs.train_distributed.zero2_child_protocol as zero2_protocol
from labs.train_distributed.zero2_child_protocol import (
    LAUNCH_MONOTONIC_NS_ENV,
    LAUNCH_WALL_NS_ENV,
    MODE_ENV,
    POST_TIMING_PROFILE_KIND,
    PROFILE_KIND_ENV,
    RESULT_DIR_ENV,
    RUN_ID_ENV,
    VARIANT_ENV,
    VERIFICATION_ONLY_PROFILE_KIND,
    run_zero2_result_profile,
    validate_zero2_result_bundle,
)


CODE_ROOT = Path(__file__).resolve().parents[1]
ZERO2_ENTRYPOINT = CODE_ROOT / "labs" / "train_distributed" / "zero2.py"


def _profile_without_rsag_worker(
    rank: int,
    rendezvous: str,
    result_dir: str,
    launch_wall_ns: int,
    launch_monotonic_ns: int,
) -> None:
    import labs.train_distributed.zero2_common as zero2_common
    from labs.train_distributed.optimized_zero2_multigpu import _build_optimizer

    os.environ.update(
        {
            RESULT_DIR_ENV: result_dir,
            RUN_ID_ENV: "pytest-zero2-missing-rsag",
            MODE_ENV: "optimized",
            VARIANT_ENV: "multigpu",
            PROFILE_KIND_ENV: VERIFICATION_ONLY_PROFILE_KIND,
            LAUNCH_WALL_NS_ENV: str(launch_wall_ns),
            LAUNCH_MONOTONIC_NS_ENV: str(launch_monotonic_ns),
        }
    )
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
    )
    try:
        def build_without_rsag(model, learning_rate, *, optimized, device_ids=None):
            assert optimized
            ddp = DDP(
                model,
                device_ids=device_ids,
                static_graph=True,
                gradient_as_bucket_view=True,
                bucket_cap_mb=25,
            )
            return ddp, _build_optimizer(ddp.parameters(), learning_rate)

        zero2_common.build_training_components = build_without_rsag
        torch.default_generator.manual_seed(42)
        run_zero2_result_profile(
            optimized=True,
            variant="multigpu",
            device=torch.device("cpu"),
        )
    finally:
        dist.destroy_process_group()


def _profile_with_ordinary_allreduce_worker(
    rank: int,
    rendezvous: str,
    result_dir: str,
    launch_wall_ns: int,
    launch_monotonic_ns: int,
) -> None:
    import labs.train_distributed.zero2_common as zero2_common

    os.environ.update(
        {
            RESULT_DIR_ENV: result_dir,
            RUN_ID_ENV: "pytest-zero2-ordinary-allreduce",
            MODE_ENV: "optimized",
            VARIANT_ENV: "multigpu",
            PROFILE_KIND_ENV: VERIFICATION_ONLY_PROFILE_KIND,
            LAUNCH_WALL_NS_ENV: str(launch_wall_ns),
            LAUNCH_MONOTONIC_NS_ENV: str(launch_monotonic_ns),
        }
    )

    def ordinary_allreduce_hook(state, bucket):
        state.hook_invocations += 1
        buffer = bucket.buffer()
        dist.all_reduce(buffer, group=state.process_group)
        buffer.div_(state.process_group.size())
        future = torch.futures.Future()
        future.set_result(buffer)
        return future

    ordinary_allreduce_hook.__annotations__["bucket"] = dist.GradBucket
    ordinary_allreduce_hook.__annotations__["return"] = torch.futures.Future[torch.Tensor]
    zero2_common._tracked_reduce_scatter_allgather_hook = ordinary_allreduce_hook
    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
    )
    try:
        torch.default_generator.manual_seed(42)
        run_zero2_result_profile(
            optimized=True,
            variant="multigpu",
            device=torch.device("cpu"),
        )
    finally:
        dist.destroy_process_group()


def _single_profile_context(tmp_path: Path, monkeypatch, run_id: str) -> dict:
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    launch_wall_ns = time.time_ns()
    launch_monotonic_ns = time.monotonic_ns()
    values = {
        RESULT_DIR_ENV: str(result_dir),
        RUN_ID_ENV: run_id,
        MODE_ENV: "baseline",
        VARIANT_ENV: "single",
        PROFILE_KIND_ENV: VERIFICATION_ONLY_PROFILE_KIND,
        LAUNCH_WALL_NS_ENV: str(launch_wall_ns),
        LAUNCH_MONOTONIC_NS_ENV: str(launch_monotonic_ns),
    }
    for name, value in values.items():
        monkeypatch.setenv(name, value)
    return {
        "result_dir": result_dir,
        "run_id": run_id,
        "mode": "baseline",
        "variant": "single",
        "world_size": 1,
        "launch_wall_ns": launch_wall_ns,
        "launch_monotonic_ns": launch_monotonic_ns,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validation_kwargs(context, *, finish_now=False):
    return {
        "run_id": context["run_id"],
        "mode": context["mode"],
        "variant": context["variant"],
        "world_size": context["world_size"],
        "launch_wall_ns": context["launch_wall_ns"],
        "launch_monotonic_ns": context["launch_monotonic_ns"],
        "finish_wall_ns": time.time_ns() if finish_now else context["finish_wall_ns"],
        "finish_monotonic_ns": (
            time.monotonic_ns() if finish_now else context["finish_monotonic_ns"]
        ),
        "profile_kind": VERIFICATION_ONLY_PROFILE_KIND,
    }


@pytest.fixture(scope="module")
def optimized_gloo_bundle(tmp_path_factory):
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("Gloo unavailable")
    result_dir = tmp_path_factory.mktemp("zero2-child-results")
    context = {
        "result_dir": result_dir,
        "run_id": "pytest-zero2-optimized-gloo-two-rank",
        "mode": "optimized",
        "variant": "multigpu",
        "world_size": 2,
    }
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_ROOT)
    env.update(
        {
            RESULT_DIR_ENV: str(result_dir),
            RUN_ID_ENV: context["run_id"],
            MODE_ENV: context["mode"],
            VARIANT_ENV: context["variant"],
            PROFILE_KIND_ENV: VERIFICATION_ONLY_PROFILE_KIND,
        }
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context["launch_wall_ns"] = time.time_ns()
    context["launch_monotonic_ns"] = time.monotonic_ns()
    env[LAUNCH_WALL_NS_ENV] = str(context["launch_wall_ns"])
    env[LAUNCH_MONOTONIC_NS_ENV] = str(context["launch_monotonic_ns"])
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "2",
        "--nnodes",
        "1",
        "--rdzv_backend",
        "static",
        "--rdzv_endpoint",
        f"127.0.0.1:{port}",
        "-m",
        "core.harness.torchrun_wrapper",
        "--aisp-target-script",
        str(ZERO2_ENTRYPOINT),
        "--aisp-expected-torch-seed",
        "42",
        "--mode",
        "optimized",
        "--variant",
        "multigpu",
        "--verification-only",
        "--verification-backend",
        "gloo",
    ]
    result = subprocess.run(
        command,
        cwd=CODE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    context["finish_monotonic_ns"] = time.monotonic_ns()
    context["finish_wall_ns"] = time.time_ns()
    context["stdout"] = result.stdout
    context["stderr"] = result.stderr
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    context["bundle"] = validate_zero2_result_bundle(
        result_dir,
        **_validation_kwargs(context),
    )
    return context


def _launch_gloo_matrix_case(result_dir: Path, *, mode: str, variant: str) -> dict:
    result_dir.mkdir()
    world_size = 1 if variant == "single" else 2
    run_id = f"pytest-zero2-matrix-{mode}-{variant}"
    context = {
        "result_dir": result_dir,
        "run_id": run_id,
        "mode": mode,
        "variant": variant,
        "world_size": world_size,
    }
    env = os.environ.copy()
    env["PYTHONPATH"] = str(CODE_ROOT)
    env.update(
        {
            RESULT_DIR_ENV: str(result_dir),
            RUN_ID_ENV: run_id,
            MODE_ENV: mode,
            VARIANT_ENV: variant,
            PROFILE_KIND_ENV: VERIFICATION_ONLY_PROFILE_KIND,
        }
    )
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    context["launch_wall_ns"] = time.time_ns()
    context["launch_monotonic_ns"] = time.monotonic_ns()
    env[LAUNCH_WALL_NS_ENV] = str(context["launch_wall_ns"])
    env[LAUNCH_MONOTONIC_NS_ENV] = str(context["launch_monotonic_ns"])
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        str(world_size),
        "--nnodes",
        "1",
        "--rdzv_backend",
        "static",
        "--rdzv_endpoint",
        f"127.0.0.1:{port}",
        "-m",
        "core.harness.torchrun_wrapper",
        "--aisp-target-script",
        str(ZERO2_ENTRYPOINT),
        "--aisp-expected-torch-seed",
        "42",
        "--mode",
        mode,
        "--variant",
        variant,
        "--verification-only",
        "--verification-backend",
        "gloo",
    ]
    result = subprocess.run(
        command,
        cwd=CODE_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    context["finish_monotonic_ns"] = time.monotonic_ns()
    context["finish_wall_ns"] = time.time_ns()
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    context["bundle"] = validate_zero2_result_bundle(
        result_dir,
        **_validation_kwargs(context),
    )
    context["stdout"] = result.stdout
    return context


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo unavailable",
)
@pytest.mark.parametrize("mode", ["baseline", "optimized"])
@pytest.mark.parametrize("variant", ["single", "multigpu"])
def test_fresh_verification_only_cpu_gloo_factory_matrix(tmp_path, mode, variant):
    context = _launch_gloo_matrix_case(
        tmp_path / f"{mode}-{variant}",
        mode=mode,
        variant=variant,
    )
    bundle = context["bundle"]
    assert bundle["mode"] == mode
    assert bundle["variant"] == variant
    assert bundle["world_size"] == (1 if variant == "single" else 2)
    assert "no performance timing was collected" in context["stdout"]
    for manifest in bundle["manifests"]:
        communication = manifest["communication"]
        if mode == "baseline":
            assert communication == {
                "mechanism": "ddp-all-reduce",
                "hook_invocations": 0,
                "reduce_scatter_completions": 0,
                "all_gather_completions": 0,
            }
        elif variant == "single":
            assert communication["hook_invocations"] > 0
            assert communication["reduce_scatter_completions"] == 0
            assert communication["all_gather_completions"] == 0
        else:
            assert communication["hook_invocations"] > 0
            assert communication["reduce_scatter_completions"] == communication["hook_invocations"]
            assert communication["all_gather_completions"] == communication["hook_invocations"]


def _copy_bundle(context, destination: Path) -> Path:
    destination.mkdir()
    for source in Path(context["result_dir"]).iterdir():
        shutil.copy2(source, destination / source.name)
    return destination


def _rewrite_payload(bundle_dir: Path, rank: int, mutate) -> None:
    payload_path = bundle_dir / f"rank-{rank}.pt"
    manifest_path = bundle_dir / f"rank-{rank}.json"
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    manifest = json.loads(manifest_path.read_text())
    mutate(payload, manifest)
    torch.save(payload, payload_path)
    manifest["payload_size"] = payload_path.stat().st_size
    manifest["payload_sha256"] = _sha256(payload_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_actual_two_rank_child_is_fresh_complete_and_not_a_performance_run(optimized_gloo_bundle):
    context = optimized_gloo_bundle
    bundle = context["bundle"]
    assert bundle["mode"] == "optimized"
    assert bundle["world_size"] == 2
    assert len(bundle["parameter_names"]) == 14
    assert bundle["verify_inputs"]["x"].shape == (2, 3, 2, 3, 8)
    assert bundle["verify_output"]["rank_final_microbatch_losses"].shape == (2, 3)
    assert all(
        manifest["communication"]["mechanism"] == "reduce-scatter-all-gather"
        and manifest["communication"]["hook_invocations"] > 0
        and manifest["communication"]["reduce_scatter_completions"]
        == manifest["communication"]["hook_invocations"]
        and manifest["communication"]["all_gather_completions"]
        == manifest["communication"]["hook_invocations"]
        for manifest in bundle["manifests"]
    )
    assert "no performance timing was collected" in context["stdout"]
    assert "training_seconds=" not in context["stdout"]
    assert "samples/s" not in context["stdout"]


def test_performance_adapter_rejects_verification_only_child_results(
    optimized_gloo_bundle,
):
    context = optimized_gloo_bundle
    benchmark = Zero2TorchrunBenchmark(
        mode="optimized",
        variant="multigpu",
        script_path=ZERO2_ENTRYPOINT,
        base_args=["--mode", "optimized", "--variant", "multigpu"],
        multi_gpu_required=True,
        name="optimized_zero2_protocol_test",
    )
    benchmark._zero2_result_context = {
        key: context[key]
        for key in ("result_dir", "run_id", "mode", "variant", "world_size")
    }
    benchmark._zero2_result_context["profile_kind"] = VERIFICATION_ONLY_PROFILE_KIND
    with pytest.raises(RuntimeError, match="only post-timing"):
        benchmark.consume_zero2_child_results(
            launch_wall_ns=context["launch_wall_ns"],
            launch_monotonic_ns=context["launch_monotonic_ns"],
            finish_wall_ns=context["finish_wall_ns"],
            finish_monotonic_ns=context["finish_monotonic_ns"],
            returncode=0,
        )
    assert benchmark.validate_result() is not None


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (
            ["--verification-only", "--verification-backend", "gloo"],
            "cannot satisfy a post-timing",
        ),
        ([], "normal ZeRO performance child requires a post-timing"),
    ],
)
def test_child_execution_route_rejects_the_opposite_profile_kind(
    tmp_path,
    monkeypatch,
    argv,
    message,
):
    import labs.train_distributed.zero2_common as zero2_common

    monkeypatch.setenv(RESULT_DIR_ENV, str(tmp_path))
    monkeypatch.setenv(
        PROFILE_KIND_ENV,
        POST_TIMING_PROFILE_KIND if argv else VERIFICATION_ONLY_PROFILE_KIND,
    )
    args = zero2_common.parse_args(argv)
    with pytest.raises(RuntimeError, match=message):
        zero2_common.run_training(args, optimized=False, multi_gpu=False)


def test_child_execution_route_rejects_compile_before_capability_checks():
    import labs.train_distributed.zero2_common as zero2_common

    args = zero2_common.parse_args(["--compile"])
    with pytest.raises(RuntimeError, match="requires eager FP32 execution"):
        zero2_common.run_training(args, optimized=False, multi_gpu=False)


def test_adapter_cleans_success_artifacts_after_retaining_validated_diagnostics(
    optimized_gloo_bundle,
    tmp_path,
    monkeypatch,
):
    context = optimized_gloo_bundle
    result_dir = _copy_bundle(context, tmp_path / "adapter-success-cleanup")
    benchmark = Zero2TorchrunBenchmark(
        mode="optimized",
        variant="multigpu",
        script_path=ZERO2_ENTRYPOINT,
        base_args=["--mode", "optimized", "--variant", "multigpu"],
        multi_gpu_required=True,
        name="optimized_zero2_cleanup_test",
    )
    benchmark._zero2_result_context = {
        "result_dir": result_dir,
        "run_id": context["run_id"],
        "mode": context["mode"],
        "variant": context["variant"],
        "world_size": context["world_size"],
        "profile_kind": POST_TIMING_PROFILE_KIND,
    }
    monkeypatch.setattr(
        zero2_adapter,
        "validate_zero2_result_bundle",
        lambda *args, **kwargs: dict(context["bundle"]),
    )
    benchmark.consume_zero2_child_results(
        launch_wall_ns=context["launch_wall_ns"],
        launch_monotonic_ns=context["launch_monotonic_ns"],
        finish_wall_ns=context["finish_wall_ns"],
        finish_monotonic_ns=context["finish_monotonic_ns"],
        returncode=0,
    )
    assert benchmark.validate_result() is None
    assert not result_dir.exists()
    assert benchmark._zero2_result_bundle["artifact_retention"]["status"] == "cleaned-after-success"
    assert benchmark.get_verify_output()["rank_final_microbatch_losses"].shape == (2, 3)
    assert benchmark.get_verify_inputs()["x"].shape == (2, 3, 2, 3, 8)
    assert benchmark.get_input_signature().world_size == 2
    assert benchmark.get_output_tolerance() == (1.0e-5, 1.0e-6)


def test_harness_invokes_only_the_explicit_zero2_result_callback(tmp_path, monkeypatch):
    target = tmp_path / "zero2_callback_control.py"
    target.write_text("print('unused fake child')\n")
    benchmark = Zero2TorchrunBenchmark(
        mode="baseline",
        variant="single",
        script_path=target,
        base_args=[],
        multi_gpu_required=False,
        default_nproc_per_node=1,
        name="zero2_callback_control",
    )
    observed = {}

    def callback(**kwargs):
        observed.update(kwargs)

    benchmark.consume_zero2_child_results = callback

    class FakeProcess:
        returncode = 0
        pid = os.getpid()

        def communicate(self, timeout=None):
            return "ZERO2_CALLBACK_CONTROL\n", ""

    def fake_popen(command, **kwargs):
        observed["command"] = command
        observed["env"] = kwargs["env"]
        return FakeProcess()

    monkeypatch.setattr("core.harness.benchmark_harness.subprocess.Popen", fake_popen)
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        iterations=1,
        warmup=5,
        launch_via=LaunchVia.TORCHRUN,
        nproc_per_node=1,
        multi_gpu_required=False,
        use_subprocess=False,
        enable_profiling=False,
        lock_gpu_clocks=False,
        enforce_environment_validation=False,
        measurement_timeout_seconds=30,
        nnodes="1",
        rdzv_backend="static",
        rdzv_endpoint="127.0.0.1:1",
    )
    result = BenchmarkHarness(config=config)._benchmark_with_torchrun(benchmark, config)
    assert not result.errors
    assert observed["returncode"] == 0
    assert observed["launch_wall_ns"] <= observed["finish_wall_ns"]
    assert observed["launch_monotonic_ns"] <= observed["finish_monotonic_ns"]
    assert observed["env"][LAUNCH_WALL_NS_ENV] == str(observed["launch_wall_ns"])
    assert observed["env"][LAUNCH_MONOTONIC_NS_ENV] == str(
        observed["launch_monotonic_ns"]
    )
    shutil.rmtree(benchmark._zero2_result_context["result_dir"])


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo unavailable",
)
def test_fixed_profile_uses_production_training_step_and_rejects_noop(
    tmp_path,
    monkeypatch,
):
    import labs.train_distributed.zero2_common as zero2_common

    context = _single_profile_context(tmp_path, monkeypatch, "pytest-zero2-noop-step")
    rendezvous = tmp_path / "rendezvous"
    torch.default_generator.manual_seed(42)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous.as_uri(),
        rank=0,
        world_size=1,
    )

    def noop_training_step(
        model,
        optimizer,
        x,
        y,
        generator,
        grad_accum,
        *,
        post_clip_callback=None,
        **kwargs,
    ):
        optimizer.zero_grad(set_to_none=True)
        for parameter in model.parameters():
            parameter.grad = torch.zeros_like(parameter)
        if post_clip_callback is not None:
            post_clip_callback(model.parameters())
        return torch.zeros((), dtype=x.dtype, device=x.device)

    monkeypatch.setattr(zero2_common, "training_step", noop_training_step)
    try:
        run_zero2_result_profile(
            optimized=False,
            variant="single",
            device=torch.device("cpu"),
        )
    finally:
        dist.destroy_process_group()
    context["finish_monotonic_ns"] = time.monotonic_ns()
    context["finish_wall_ns"] = time.time_ns()
    with pytest.raises(RuntimeError, match="mismatch|no-op|no local AdamW state"):
        validate_zero2_result_bundle(
            context["result_dir"],
            **_validation_kwargs(context),
        )


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo unavailable",
)
def test_profile_avoids_global_manual_seed_and_preserves_cpu_rng(tmp_path, monkeypatch):
    context = _single_profile_context(tmp_path, monkeypatch, "pytest-zero2-rng-scope")
    rendezvous = tmp_path / "rendezvous"
    torch.default_generator.manual_seed(42)
    before = torch.get_rng_state().clone()

    def forbidden_global_seed(*args, **kwargs):
        raise AssertionError("profile must not globally seed every accelerator")

    monkeypatch.setattr(torch, "manual_seed", forbidden_global_seed)
    dist.init_process_group(
        "gloo",
        init_method=rendezvous.as_uri(),
        rank=0,
        world_size=1,
    )
    try:
        run_zero2_result_profile(
            optimized=False,
            variant="single",
            device=torch.device("cpu"),
        )
    finally:
        dist.destroy_process_group()
    assert torch.equal(torch.get_rng_state(), before)
    context["finish_monotonic_ns"] = time.monotonic_ns()
    context["finish_wall_ns"] = time.time_ns()
    bundle = validate_zero2_result_bundle(
        context["result_dir"],
        **_validation_kwargs(context),
    )
    assert bundle["verify_output"]["rank_final_microbatch_losses"].shape == (1, 3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not dist.is_nccl_available(),
    reason="CUDA/NCCL unavailable",
)
def test_profile_preserves_all_visible_cuda_rng_states(tmp_path, monkeypatch):
    context = _single_profile_context(tmp_path, monkeypatch, "pytest-zero2-cuda-rng-scope")
    rendezvous = tmp_path / "rendezvous"
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.default_generator.manual_seed(42)
    torch.cuda.manual_seed_all(81_230)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    dist.init_process_group(
        "nccl",
        init_method=rendezvous.as_uri(),
        rank=0,
        world_size=1,
        device_id=device,
    )
    try:
        run_zero2_result_profile(
            optimized=False,
            variant="single",
            device=device,
        )
    finally:
        dist.destroy_process_group()
    assert torch.equal(torch.get_rng_state(), cpu_before)
    cuda_after = torch.cuda.get_rng_state_all()
    assert len(cuda_after) == len(cuda_before)
    assert all(torch.equal(after, before) for after, before in zip(cuda_after, cuda_before))
    context["finish_monotonic_ns"] = time.monotonic_ns()
    context["finish_wall_ns"] = time.time_ns()
    bundle = validate_zero2_result_bundle(
        context["result_dir"],
        **_validation_kwargs(context),
    )
    assert bundle["manifests"][0]["config"]["device_type"] == "cuda"


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo unavailable",
)
def test_optimized_profile_rejects_ordinary_ddp_without_rsag(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    launch_wall_ns = time.time_ns()
    launch_monotonic_ns = time.monotonic_ns()
    context = torch.multiprocessing.spawn(
        _profile_without_rsag_worker,
        args=(
            (tmp_path / "rendezvous").as_uri(),
            str(result_dir),
            launch_wall_ns,
            launch_monotonic_ns,
        ),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Two-rank missing-RS/AG negative timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    finish_monotonic_ns = time.monotonic_ns()
    finish_wall_ns = time.time_ns()
    with pytest.raises(RuntimeError, match="communication mechanism mismatch"):
        validate_zero2_result_bundle(
            result_dir,
            run_id="pytest-zero2-missing-rsag",
            mode="optimized",
            variant="multigpu",
            world_size=2,
            launch_wall_ns=launch_wall_ns,
            launch_monotonic_ns=launch_monotonic_ns,
            finish_wall_ns=finish_wall_ns,
            finish_monotonic_ns=finish_monotonic_ns,
            profile_kind=VERIFICATION_ONLY_PROFILE_KIND,
        )


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="Gloo unavailable",
)
def test_optimized_profile_rejects_ordinary_allreduce_inside_tracked_hook(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    launch_wall_ns = time.time_ns()
    launch_monotonic_ns = time.monotonic_ns()
    context = torch.multiprocessing.spawn(
        _profile_with_ordinary_allreduce_worker,
        args=(
            (tmp_path / "rendezvous").as_uri(),
            str(result_dir),
            launch_wall_ns,
            launch_monotonic_ns,
        ),
        nprocs=2,
        join=False,
    )
    deadline = time.monotonic() + 60
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Two-rank ordinary-allreduce negative timed out")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    with pytest.raises(RuntimeError, match="completed RS/AG counts"):
        validate_zero2_result_bundle(
            result_dir,
            run_id="pytest-zero2-ordinary-allreduce",
            mode="optimized",
            variant="multigpu",
            world_size=2,
            launch_wall_ns=launch_wall_ns,
            launch_monotonic_ns=launch_monotonic_ns,
            finish_wall_ns=time.time_ns(),
            finish_monotonic_ns=time.monotonic_ns(),
            profile_kind=VERIFICATION_ONLY_PROFILE_KIND,
        )


def test_child_result_rejects_missing_rank(optimized_gloo_bundle, tmp_path):
    source = Path(optimized_gloo_bundle["result_dir"])
    incomplete = tmp_path / "missing-rank"
    incomplete.mkdir()
    shutil.copy2(source / "rank-0.json", incomplete / "rank-0.json")
    shutil.copy2(source / "rank-0.pt", incomplete / "rank-0.pt")
    with pytest.raises(RuntimeError, match="quorum mismatch"):
        validate_zero2_result_bundle(
            incomplete,
            **_validation_kwargs(optimized_gloo_bundle, finish_now=True),
        )


def test_child_result_rejects_truncated_payload(optimized_gloo_bundle, tmp_path):
    bundle_dir = _copy_bundle(optimized_gloo_bundle, tmp_path / "truncated")
    payload = bundle_dir / "rank-1.pt"
    payload.write_bytes(payload.read_bytes()[:32])
    with pytest.raises(RuntimeError, match="payload (size|checksum) mismatch"):
        validate_zero2_result_bundle(
            bundle_dir,
            **_validation_kwargs(optimized_gloo_bundle, finish_now=True),
        )


def test_child_result_rejects_payload_over_safety_limit(
    optimized_gloo_bundle,
    tmp_path,
    monkeypatch,
):
    bundle_dir = _copy_bundle(optimized_gloo_bundle, tmp_path / "oversized")
    payload_size = (bundle_dir / "rank-0.pt").stat().st_size
    monkeypatch.setattr(zero2_protocol, "MAX_PAYLOAD_BYTES", payload_size - 1)
    with pytest.raises(RuntimeError, match="payload exceeds"):
        validate_zero2_result_bundle(
            bundle_dir,
            **_validation_kwargs(optimized_gloo_bundle, finish_now=True),
        )


def test_post_timing_validator_rejects_a_gloo_cpu_bundle(
    optimized_gloo_bundle,
    tmp_path,
):
    bundle_dir = _copy_bundle(optimized_gloo_bundle, tmp_path / "gloo-as-post-timing")

    def relabel(payload, manifest):
        payload["config"]["profile_kind"] = POST_TIMING_PROFILE_KIND
        manifest["config"] = dict(payload["config"])

    for rank in range(2):
        _rewrite_payload(bundle_dir, rank, relabel)
    validation = _validation_kwargs(optimized_gloo_bundle, finish_now=True)
    validation["profile_kind"] = POST_TIMING_PROFILE_KIND
    with pytest.raises(RuntimeError, match="requires the NCCL/CUDA performance child"):
        validate_zero2_result_bundle(bundle_dir, **validation)


def test_optimized_validator_rejects_a_degenerate_empty_owner_rank(
    optimized_gloo_bundle,
    tmp_path,
):
    bundle_dir = _copy_bundle(optimized_gloo_bundle, tmp_path / "empty-owner")
    payloads = [
        torch.load(bundle_dir / f"rank-{rank}.pt", map_location="cpu", weights_only=True)
        for rank in range(2)
    ]
    payloads[0]["local_optimizer_state"].update(payloads[1]["local_optimizer_state"])
    payloads[1]["local_optimizer_state"] = {}
    for rank, payload in enumerate(payloads):
        payload_path = bundle_dir / f"rank-{rank}.pt"
        manifest_path = bundle_dir / f"rank-{rank}.json"
        torch.save(payload, payload_path)
        manifest = json.loads(manifest_path.read_text())
        manifest["payload_size"] = payload_path.stat().st_size
        manifest["payload_sha256"] = _sha256(payload_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(RuntimeError, match="degenerate owner coverage"):
        validate_zero2_result_bundle(
            bundle_dir,
            **_validation_kwargs(optimized_gloo_bundle, finish_now=True),
        )


def _stale(payload, manifest):
    payload["freshness"]["started_wall_ns"] = payload["freshness"]["launch_wall_ns"] - 1
    manifest["freshness"] = dict(payload["freshness"])


def _wrong_config(payload, manifest):
    payload["config"]["hidden_size"] += 1
    manifest["config"] = dict(payload["config"])


def _nonfinite(payload, manifest):
    first = sorted(payload["final_parameters"])[0]
    payload["final_parameters"][first].view(-1)[0] = float("nan")


def _noop(payload, manifest):
    payload["final_parameters"] = {
        name: tensor.clone() for name, tensor in payload["initial_parameters"].items()
    }
    for state in payload["local_optimizer_state"].values():
        state["exp_avg"].zero_()


def _corrupt_parameter(payload, manifest):
    first = sorted(payload["final_parameters"])[0]
    payload["final_parameters"][first].view(-1)[0].add_(0.25)


def _wrong_update_count(payload, manifest):
    first = sorted(payload["local_optimizer_state"])[0]
    payload["local_optimizer_state"][first]["step"].fill_(99)


def _wrong_input(payload, manifest):
    payload["inputs"]["x"].view(-1)[0].add_(1.0)


def _rng_mutation(payload, manifest):
    payload["rng_after"] = dict(payload["rng_after"])
    payload["rng_after"]["cpu_state_sha256"] = "0" * 64
    manifest["rng_after"] = dict(payload["rng_after"])


def _wrong_seed(payload, manifest):
    payload["rng_before"] = dict(payload["rng_before"])
    payload["rng_after"] = dict(payload["rng_after"])
    payload["rng_before"]["torch_initial_seed"] = 43
    payload["rng_after"]["torch_initial_seed"] = 43
    manifest["rng_before"] = dict(payload["rng_before"])
    manifest["rng_after"] = dict(payload["rng_after"])


def _missing_rsag_evidence(payload, manifest):
    payload["communication"] = {
        "mechanism": "ddp-all-reduce",
        "hook_invocations": 0,
        "reduce_scatter_completions": 0,
        "all_gather_completions": 0,
    }
    manifest["communication"] = dict(payload["communication"])


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        ("stale", _stale, "freshness"),
        ("wrong-config", _wrong_config, "config mismatch"),
        ("nonfinite", _nonfinite, "Non-finite tensor"),
        ("noop", _noop, "mismatch|no-op"),
        ("corruption", _corrupt_parameter, "mismatch"),
        ("wrong-update-count", _wrong_update_count, "update count mismatch"),
        ("wrong-input", _wrong_input, "input x"),
        ("rng-mutation", _rng_mutation, "mutated RNG state"),
        ("wrong-seed", _wrong_seed, "wrapper seed mismatch"),
        ("missing-rsag", _missing_rsag_evidence, "communication mechanism mismatch"),
    ],
)
def test_child_result_negative_controls(
    optimized_gloo_bundle,
    tmp_path,
    name,
    mutate,
    message,
):
    bundle_dir = _copy_bundle(optimized_gloo_bundle, tmp_path / name)
    _rewrite_payload(bundle_dir, 0, mutate)
    with pytest.raises(RuntimeError, match=message):
        validate_zero2_result_bundle(
            bundle_dir,
            **_validation_kwargs(optimized_gloo_bundle, finish_now=True),
        )
