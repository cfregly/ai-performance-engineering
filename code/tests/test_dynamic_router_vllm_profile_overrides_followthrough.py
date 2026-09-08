from __future__ import annotations

import json
import os
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from core.harness import run_benchmarks
from core.profiling.profiler_wrapper import render_nsys_python_profile_wrapper
from labs.dynamic_router import vllm_runner
from labs.dynamic_router.baseline_dual_pool_vllm import (
    BaselineDualPoolVllmBenchmark,
)
from labs.dynamic_router.baseline_dynamic_router_vllm import (
    BaselineDynamicRouterVllmBenchmark,
)
from labs.dynamic_router.optimized_dual_pool_vllm import (
    OptimizedDualPoolVllmBenchmark,
)
from labs.dynamic_router.optimized_dynamic_router_vllm import (
    OptimizedDynamicRouterVllmBenchmark,
)

TARGET_LABEL = "labs/dynamic_router:dual_pool_vllm"
TARGET_ARGV = [
    "--model",
    "/models/gpt-oss-20b",
    "--prefill-gpus",
    "0",
    "--decode-gpus",
    "1",
    "--attention-backend",
    "TRITON_ATTN",
    "--max-tokens",
    "3",
    "--long-prompt-tokens",
    "32",
    "--short-prompt-tokens",
    "4",
    "--prefill-burst",
    "2",
    "--decode-requests",
    "3",
    "--continue-requests",
    "4",
]


@pytest.mark.parametrize(
    "benchmark_type,expected_requests,expected_tokens",
    [
        (BaselineDualPoolVllmBenchmark, 9, 119),
        (OptimizedDualPoolVllmBenchmark, 9, 119),
    ],
)
def test_dual_pool_target_overrides_are_instance_local_and_refresh_workload(
    benchmark_type, expected_requests: int, expected_tokens: int
) -> None:
    global_args_before = vars(vllm_runner._CLI_ARGS).copy()
    benchmark = benchmark_type()

    benchmark.apply_target_overrides(TARGET_ARGV)

    assert vars(vllm_runner._CLI_ARGS) == global_args_before
    assert benchmark._cli_args.model == "/models/gpt-oss-20b"
    assert benchmark._cli_args.prefill_gpus == "0"
    assert benchmark._cli_args.decode_gpus == "1"
    assert benchmark._cli_args.attention_backend == "TRITON_ATTN"
    assert len(benchmark._prompt_lengths) == expected_requests
    metadata = benchmark.get_workload_metadata()
    assert metadata is not None
    assert metadata.requests_per_iteration == expected_requests
    assert metadata.tokens_per_iteration == expected_tokens
    assert benchmark.profile_require_teardown is True


@pytest.mark.parametrize(
    "benchmark_type",
    [BaselineDynamicRouterVllmBenchmark, OptimizedDynamicRouterVllmBenchmark],
)
def test_router_target_overrides_share_parser_without_global_state(
    benchmark_type,
) -> None:
    benchmark = benchmark_type()
    argv = [
        "--model",
        "/models/gpt-oss-20b",
        "--decode-gpus",
        "0,1",
        "--attention-backend",
        "TRITON_ATTN",
        "--req-count",
        "7",
        "--max-tokens",
        "3",
    ]

    benchmark.apply_target_overrides(argv)

    assert benchmark._cli_args.req_count == 7
    assert benchmark._prompt_lengths == [64] * 7
    metadata = benchmark.get_workload_metadata()
    assert metadata is not None
    assert metadata.requests_per_iteration == 7
    assert metadata.tokens_per_iteration == 7 * (64 + 3)
    assert benchmark.profile_require_teardown is True


@pytest.mark.parametrize(
    "argv,error",
    [
        (["--unknown-vllm-option", "1"], "Unrecognized vLLM target arguments"),
        (["--max-tokens", "0"], "--max-tokens must be positive"),
        (["--decode-gpus", "1,1"], "must not contain duplicate GPU ids"),
        (["--prefill-gpus", "gpu0"], "comma-separated list"),
    ],
)
def test_invalid_target_override_is_retained_for_fail_closed_setup(
    argv: list[str], error: str
) -> None:
    benchmark = BaselineDualPoolVllmBenchmark()

    with pytest.raises(ValueError, match=error):
        benchmark.apply_target_overrides(argv)
    with pytest.raises(ValueError, match="Invalid target override"):
        benchmark.setup()


def test_real_nsys_wrapper_plan_embeds_target_override_and_teardown_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = BaselineDualPoolVllmBenchmark()
    config = replace(
        benchmark.get_config(),
        target_label=TARGET_LABEL,
        target_extra_args={TARGET_LABEL: list(TARGET_ARGV)},
        profile_env_overrides={"VLLM_BATCH_INVARIANT": "1"},
        validity_profile="portable",
        lock_gpu_clocks=True,
    )
    target_argv = run_benchmarks._resolve_target_override_argv(config)
    assert target_argv == TARGET_ARGV
    benchmark_path = (
        Path(__file__).resolve().parents[1]
        / "labs/dynamic_router/baseline_dual_pool_vllm.py"
    )
    source = render_nsys_python_profile_wrapper(
        benchmark_path=benchmark_path,
        nvtx_includes=["compute_kernel:profile"],
        target_label=TARGET_LABEL,
        target_override_argv=target_argv,
        validity_profile=config.validity_profile,
        lock_gpu_clocks_flag=False,
        gpu_sm_clock_mhz=None,
        gpu_mem_clock_mhz=None,
    )
    compile(source, "<dynamic-router-nsys-wrapper>", "exec")
    assert "_apply_overrides(list(_target_override_argv))" in source
    assert "profile_require_teardown" in source

    repo_root = Path(run_benchmarks.__file__).resolve().parents[2]
    monkeypatch.delenv("PYTHONNOUSERSITE", raising=False)
    with run_benchmarks._temporary_python_profile_launch(
        source,
        chapter_dir=benchmark_path.parent,
        repo_root=repo_root,
        config=config,
        benchmark=benchmark,
    ) as (wrapper_path, command, env, use_torchrun):
        assert wrapper_path.is_file()
        assert command == [sys.executable, str(wrapper_path)]
        assert use_torchrun is False
        assert env["VLLM_BATCH_INVARIANT"] == "1"
        assert str(repo_root) in env["PYTHONPATH"].split(os.pathsep)


def test_profile_output_and_lifecycle_receipts_retain_full_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(vllm_runner.VLLM_PROFILE_RECEIPT_DIR_ENV, str(tmp_path))
    args = vllm_runner.parse_vllm_target_overrides(TARGET_ARGV)
    summary = {
        "mode": "shared",
        "requests": 2,
        "completed": 2,
        "ttft_ms_p95": 1.25,
        "_verification_output_token_ids": [2, 11, 12, 1, 13],
    }

    vllm_runner.emit_vllm_profile_output_receipt("sentinel", args, summary)
    output = json.loads(
        (tmp_path / "profile-output.json").read_text(encoding="utf-8")
    )
    assert output["schema"] == vllm_runner.VLLM_PROFILE_OUTPUT_SCHEMA
    assert output["framed_token_ids"] == [2, 11, 12, 1, 13]
    assert output["scalar_metrics"]["ttft_ms_p95"] == 1.25
    assert len(output["framed_token_ids_sha256"]) == 64

    session = vllm_runner.VllmEngineSession.__new__(vllm_runner.VllmEngineSession)
    session._primary_failure = None
    session.workload_kind = "dual_pool"
    session.mode = "shared"
    session.engine_startup_ms = 1.0
    session._phase_durations_ms = {"warmup": [2.0], "steady_state": [3.0]}
    session._failed_runs = []
    session._teardown_ms = 4.0
    session._end_to_end_ms = 10.0
    session._emit_lifecycle("completed", [])
    lifecycle = json.loads(
        (tmp_path / "lifecycle.json").read_text(encoding="utf-8")
    )
    assert lifecycle["schema"] == vllm_runner.VLLM_PROFILE_LIFECYCLE_SCHEMA
    assert lifecycle["disposition"] == "completed"
    assert lifecycle["shutdown_errors"] == []
