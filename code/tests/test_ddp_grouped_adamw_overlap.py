from __future__ import annotations

import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from core.harness.benchmark_harness import BenchmarkConfig
from core.harness.run_benchmarks import _build_torchrun_profile_command
from labs.train_distributed import baseline_ddp_multigpu as baseline_target
from labs.train_distributed import ddp as dispatcher
from labs.train_distributed import optimized_ddp_multigpu as target
from labs.train_distributed.training_utils.child_result import RESULT_DIR_ENV
from labs.train_distributed.training_utils.overlap_adamw import parameter_groups


def _args(
    *,
    overlap_optimizer: bool,
    grad_accum: int = 1,
    compile_enabled: bool = False,
    learning_rate: float = 2e-4,
) -> SimpleNamespace:
    return SimpleNamespace(
        overlap_optimizer=overlap_optimizer,
        grad_accum=grad_accum,
        compile=compile_enabled,
        learning_rate=learning_rate,
    )


def test_overlap_optimizer_flag_is_explicit_and_disabled_by_default() -> None:
    assert target.parse_args([]).overlap_optimizer is False
    assert target.parse_args(["--overlap-optimizer"]).overlap_optimizer is True
    assert "--overlap-optimizer" not in target.get_benchmark()._base_args


def test_parameter_groups_preserve_every_parameter_in_reverse_bucket_order() -> None:
    parameters = [
        torch.nn.Parameter(torch.zeros(2)),
        torch.nn.Parameter(torch.zeros(3)),
        torch.nn.Parameter(torch.zeros(4)),
    ]

    groups = parameter_groups(parameters, bucket_bytes=16)

    assert groups == [[parameters[2]], [parameters[1], parameters[0]]]
    assert [parameter for group in groups for parameter in group] == list(
        reversed(parameters)
    )


@pytest.mark.parametrize(
    ("world_size", "grad_accum", "compile_enabled"),
    [
        (1, 1, False),
        (3, 1, False),
        (2, 2, False),
        (2, 1, True),
    ],
)
def test_requested_overlap_rejects_unsupported_training_modes(
    world_size: int,
    grad_accum: int,
    compile_enabled: bool,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            r"--overlap-optimizer requires exactly two ranks, --grad-accum 1, "
            r"and compile disabled"
        ),
    ):
        target._validate_overlap_optimizer_request(
            _args(
                overlap_optimizer=True,
                grad_accum=grad_accum,
                compile_enabled=compile_enabled,
            ),
            world_size=world_size,
        )


@pytest.mark.parametrize(
    ("extra_argv", "world_size", "failure_detail"),
    [
        ([], 1, "world_size=1"),
        (["--grad-accum", "2"], 2, "grad_accum=2"),
        (["--compile"], 2, "compile=True"),
    ],
)
def test_unsupported_overlap_fails_before_gpu_or_process_group_initialization(
    extra_argv: list[str],
    world_size: int,
    failure_detail: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        ["optimized_ddp_multigpu.py", "--overlap-optimizer", *extra_argv],
    )
    monkeypatch.setenv("WORLD_SIZE", str(world_size))
    monkeypatch.setattr(target.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(
        target,
        "require_min_gpus",
        lambda *_args, **_kwargs: pytest.fail("GPU preflight ran before overlap validation"),
    )
    monkeypatch.setattr(
        target.dist,
        "init_process_group",
        lambda *_args, **_kwargs: pytest.fail("process group initialized before overlap validation"),
    )

    with pytest.raises(ValueError, match=failure_detail):
        target.main()


@pytest.mark.parametrize("encoded", [None, "two"])
def test_overlap_world_size_preflight_rejects_missing_or_invalid_environment(
    encoded: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(target.dist, "is_initialized", lambda: False)
    if encoded is None:
        monkeypatch.delenv("WORLD_SIZE", raising=False)
    else:
        monkeypatch.setenv("WORLD_SIZE", encoded)

    with pytest.raises(ValueError, match=r"WORLD_SIZE"):
        target._overlap_world_size_before_init()


def test_disabled_overlap_retains_synchronous_optimizer_for_other_modes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parameters = (object(), object())
    ddp_model = SimpleNamespace(parameters=lambda: iter(parameters))
    sentinel = object()
    observed: dict[str, object] = {}

    def fake_make_ddp_adamw(params, learning_rate, *, prefer_fused):
        observed.update(
            params=tuple(params),
            learning_rate=learning_rate,
            prefer_fused=prefer_fused,
        )
        return sentinel

    monkeypatch.setattr(target, "make_ddp_adamw", fake_make_ddp_adamw)
    args = _args(
        overlap_optimizer=False,
        grad_accum=4,
        compile_enabled=True,
        learning_rate=3e-4,
    )

    target._validate_overlap_optimizer_request(args, world_size=8)
    optimizer = target._build_optimizer(object(), ddp_model, args)

    assert optimizer is sentinel
    assert observed == {
        "params": parameters,
        "learning_rate": 3e-4,
        "prefer_fused": True,
    }


def test_supported_overlap_wires_model_ddp_and_tested_bucket_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = object()
    ddp_model = object()
    sentinel = object()
    observed: dict[str, object] = {}

    def fake_overlapped_adamw(
        supplied_model,
        optimizer_factory,
        learning_rate,
        *,
        ddp,
        bucket_bytes,
    ):
        observed.update(
            model=supplied_model,
            optimizer_factory=optimizer_factory,
            learning_rate=learning_rate,
            ddp=ddp,
            bucket_bytes=bucket_bytes,
        )
        return sentinel

    monkeypatch.setattr(target, "OverlappedAdamW", fake_overlapped_adamw)
    args = _args(overlap_optimizer=True, learning_rate=4e-4)
    target._validate_overlap_optimizer_request(args, world_size=2)

    optimizer = target._build_optimizer(model, ddp_model, args)

    assert optimizer is sentinel
    assert observed == {
        "model": model,
        "optimizer_factory": target.make_ddp_adamw,
        "learning_rate": 4e-4,
        "ddp": ddp_model,
        "bucket_bytes": 50 * 1024 * 1024,
    }


@pytest.mark.parametrize("mode", ["baseline", "optimized"])
def test_dispatcher_applies_pair_flag_only_to_optimized_arm(
    mode: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    observed: list[str] = []
    selected = (
        dispatcher.baseline_multi_run
        if mode == "baseline"
        else dispatcher.optimized_multi_run
    )
    monkeypatch.setattr(selected, "main", lambda: observed.extend(sys.argv[1:]))
    monkeypatch.setattr(sys, "argv", ["ddp.py"])

    dispatcher.main(
        [
            "--mode",
            mode,
            "--variant",
            "multigpu",
            "--overlap-optimizer",
            "--steps",
            "3",
        ]
    )

    expected = ["--steps", "3"]
    if mode == "optimized":
        expected.append("--overlap-optimizer")
    assert observed == expected
    output = capsys.readouterr().out
    assert ("baseline arm remains synchronous" in output) is (mode == "baseline")


@pytest.mark.parametrize(
    "argv",
    [
        ["--mode", "optimized", "--variant", "single", "--overlap-optimizer"],
        ["--mode", "optimized_flash", "--variant", "multigpu", "--overlap-optimizer"],
    ],
)
def test_dispatcher_rejects_overlap_outside_plain_multigpu_pair(
    argv: list[str],
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit, match="2"):
        dispatcher.main(argv)
    assert "supported only by the baseline/optimized multigpu comparison" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("benchmark_module", "mode", "base_tail"),
    [
        (baseline_target, "baseline", ["--batch-size", "32"]),
        (target, "optimized", ["--batch-size", "32", "--grad-accum", "1"]),
    ],
)
def test_public_pair_maps_three_iterations_and_shared_overlap_override(
    benchmark_module,
    mode: str,
    base_tail: list[str],
) -> None:
    label = "labs/train_distributed:ddp_multigpu"
    config = BenchmarkConfig(
        iterations=3,
        nproc_per_node=2,
        nnodes="1",
        target_label=label,
        target_extra_args={label: ["--overlap-optimizer"]},
    )
    benchmark = benchmark_module.get_benchmark()
    spec = benchmark.get_torchrun_spec(config)
    result_dir = Path(spec.env[RESULT_DIR_ENV])

    try:
        command, _env = _build_torchrun_profile_command(config, spec=spec)
    finally:
        shutil.rmtree(result_dir)

    expected_child_args = [
        "--mode",
        mode,
        "--variant",
        "multigpu",
        *base_tail,
        "--steps",
        "3",
        "--overlap-optimizer",
    ]
    assert spec.config_arg_map == {"iterations": "--steps"}
    assert command[-len(expected_child_args) :] == expected_child_args
