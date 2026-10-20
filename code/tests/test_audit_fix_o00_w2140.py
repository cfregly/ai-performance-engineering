"""Regressions for W2-140 and the W2-034 residual in scripts/full_virtualized_rerun.py.

These are queue control-flow tests. They intercept only the worker launch, the
`aisp bench run` subprocess and host probes. The queue state, argument parsing
and worker loop run as written.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.full_virtualized_rerun as rerun

TARGET_A = "ch01:performance"
TARGET_B = "ch02:memory_transfer"


def _args(monkeypatch: pytest.MonkeyPatch, *argv: str):
    monkeypatch.setattr(sys, "argv", ["full_virtualized_rerun.py", *argv])
    return rerun._parse_args()


def _queue_args(tmp_path: Path) -> list[str]:
    return ["--queue-root", str(tmp_path / "queue"), "--run-root", str(tmp_path / "runs")]


def _saved_state(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "queue" / "state.json").read_text(encoding="utf-8"))


def _expected_worker_command(queue_root: Path) -> list[str]:
    return [sys.executable, str(Path(rerun.__file__).resolve()), "run-worker", "--queue-root", str(queue_root.resolve())]


@pytest.fixture
def no_worker_spawn(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    # The worker is a detached subprocess. The real `_spawn_worker` runs, and
    # only the Popen call that would launch the worker is intercepted, so the
    # tests see the exact command line the worker would get. Its pid is this
    # test process, so the queue sees a live worker until a test marks it exited.
    spawned: list[list[str]] = []
    real_popen = subprocess.Popen

    def _fake_popen(cmd: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(cmd, list) and "run-worker" in cmd:
            spawned.append([str(part) for part in cmd])
            return SimpleNamespace(pid=os.getpid())
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(rerun.subprocess, "Popen", _fake_popen)
    return spawned


def _stub_worker_environment(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, Path]]:
    """Stub the bench run and host probes so `_command_run_worker` runs its real queue loop."""

    seen: list[tuple[str, Path]] = []

    def _fake_run_target(*, target: str, run_root: Path, **_: Any) -> dict[str, Any]:
        seen.append((target, run_root))
        return {"target": target, "benchmarks": []}

    monkeypatch.setattr(rerun, "_run_target", _fake_run_target)
    monkeypatch.setattr(rerun, "_git_commit", lambda repo_root: "deadbeef")
    monkeypatch.setattr(rerun, "detect_execution_environment", lambda: SimpleNamespace(kind="virtualized"))
    return seen


def _mark_worker_exited(tmp_path: Path) -> None:
    state_path = tmp_path / "queue" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state["worker_pid"] = None
    state_path.write_text(json.dumps(state), encoding="utf-8")


def _start_configured_queue(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    args = _args(
        monkeypatch,
        "start",
        *_queue_args(tmp_path),
        "--target",
        TARGET_A,
        "--profile",
        "deep_dive",
        "--suite-timeout",
        "900",
        "--gpu-sm-clock-mhz",
        "1500",
        "--gpu-mem-clock-mhz",
        "3200",
        "--allow-portable-expectations-update",
    )
    assert rerun._command_start(args) == 0


def test_new_queue_uses_documented_defaults(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_worker_spawn: list[list[str]]
) -> None:
    args = _args(monkeypatch, "start", *_queue_args(tmp_path), "--target", TARGET_A)
    assert args.suite_timeout is None
    assert args.gpu_sm_clock_mhz is None
    assert args.gpu_mem_clock_mhz is None
    assert args.allow_portable_expectations_update is None

    assert rerun._command_start(args) == 0

    state = _saved_state(tmp_path)
    assert state["run_root"] == str((tmp_path / "runs").resolve())
    assert state["profile"] == "none"
    assert state["suite_timeout"] == 0
    assert state["gpu_sm_clock_mhz"] == 1965
    assert state["gpu_mem_clock_mhz"] == 3996
    assert state["allow_portable_expectations_update"] is False
    # The worker reads the run root from state.json, so its command line does not carry one.
    assert no_worker_spawn == [_expected_worker_command(tmp_path / "queue")]
    assert state["worker_command"] == no_worker_spawn[0]


def test_new_queue_without_run_root_uses_dated_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_worker_spawn: list[list[str]]
) -> None:
    dated_default = tmp_path / "dated_default_runs"
    monkeypatch.setattr(rerun, "_default_run_root", lambda repo_root: dated_default)

    assert rerun._command_start(_args(monkeypatch, "start", "--queue-root", str(tmp_path / "queue"), "--target", TARGET_A)) == 0

    assert _saved_state(tmp_path)["run_root"] == str(dated_default.resolve())
    assert len(no_worker_spawn) == 1
    assert dated_default.is_dir()


def test_restart_with_only_queue_root_keeps_saved_run_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_worker_spawn: list[list[str]]
) -> None:
    configured_run_root = (tmp_path / "my_runs").resolve()
    dated_default = tmp_path / "dated_default_runs"
    monkeypatch.setattr(rerun, "_default_run_root", lambda repo_root: dated_default)
    queue_args = ["--queue-root", str(tmp_path / "queue")]

    first = _args(
        monkeypatch,
        "start",
        *queue_args,
        "--run-root",
        str(configured_run_root),
        "--target",
        TARGET_A,
        "--profile",
        "deep_dive",
        "--suite-timeout",
        "900",
    )
    assert rerun._command_start(first) == 0
    _mark_worker_exited(tmp_path)

    assert rerun._command_start(_args(monkeypatch, "start", *queue_args, "--target", TARGET_B)) == 0

    state = _saved_state(tmp_path)
    assert state["run_root"] == str(configured_run_root)
    assert no_worker_spawn == [_expected_worker_command(tmp_path / "queue")] * 2
    assert state["worker_command"] == _expected_worker_command(tmp_path / "queue")
    assert state["pending_targets"] == [TARGET_A, TARGET_B]
    assert state["profile"] == "deep_dive"
    assert state["suite_timeout"] == 900
    assert not dated_default.exists()


def test_explicit_run_root_replaces_saved_run_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_worker_spawn: list[list[str]]
) -> None:
    _start_configured_queue(monkeypatch, tmp_path)
    queue_args = ["--queue-root", str(tmp_path / "queue")]
    enqueue_root = (tmp_path / "enqueue_runs").resolve()
    restart_root = (tmp_path / "restart_runs").resolve()

    enqueue = _args(monkeypatch, "enqueue", *queue_args, "--run-root", str(enqueue_root), "--target", TARGET_B)
    assert rerun._command_enqueue(enqueue) == 0
    assert _saved_state(tmp_path)["run_root"] == str(enqueue_root)

    _mark_worker_exited(tmp_path)
    assert rerun._command_start(_args(monkeypatch, "start", *queue_args, "--run-root", str(restart_root))) == 0
    assert _saved_state(tmp_path)["run_root"] == str(restart_root)
    assert len(no_worker_spawn) == 2


@pytest.mark.parametrize("change_command", ["start", "enqueue"])
def test_run_root_change_before_spawned_worker_takes_lock_is_kept(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_worker_spawn: list[list[str]], change_command: str
) -> None:
    # `start` spawns the worker and returns. Before the new worker process takes
    # the queue lock, a second command changes the run root. The worker must
    # then use the new root and must not write the launch-time root back.
    queue_args = ["--queue-root", str(tmp_path / "queue")]
    root_a = (tmp_path / "runs_a").resolve()
    root_b = (tmp_path / "runs_b").resolve()

    assert rerun._command_start(_args(monkeypatch, "start", *queue_args, "--run-root", str(root_a), "--target", TARGET_A)) == 0
    assert len(no_worker_spawn) == 1
    worker_command = no_worker_spawn[0]
    assert worker_command[1] == str(Path(rerun.__file__).resolve())

    # worker_pid is this test process, so `start` sees a live worker and spawns none.
    change = _args(monkeypatch, change_command, *queue_args, "--run-root", str(root_b), "--target", TARGET_B)
    command = rerun._command_start if change_command == "start" else rerun._command_enqueue
    assert command(change) == 0
    assert len(no_worker_spawn) == 1
    assert _saved_state(tmp_path)["run_root"] == str(root_b)

    seen = _stub_worker_environment(monkeypatch)
    # Run the worker with the exact arguments the spawn passed after the script path.
    assert rerun._command_run_worker(_args(monkeypatch, *worker_command[2:])) == 0

    assert seen == [(TARGET_A, root_b), (TARGET_B, root_b)]
    assert _saved_state(tmp_path)["run_root"] == str(root_b)


def test_manual_run_worker_run_root_still_overrides_saved_value(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    queue_args = ["--queue-root", str(tmp_path / "queue")]
    saved_root = (tmp_path / "runs_saved").resolve()
    manual_root = (tmp_path / "runs_manual").resolve()
    assert rerun._command_enqueue(_args(monkeypatch, "enqueue", *queue_args, "--run-root", str(saved_root), "--target", TARGET_A)) == 0

    seen = _stub_worker_environment(monkeypatch)
    assert rerun._command_run_worker(_args(monkeypatch, "run-worker", *queue_args, "--run-root", str(manual_root))) == 0

    assert seen == [(TARGET_A, manual_root)]
    assert _saved_state(tmp_path)["run_root"] == str(manual_root)


def test_worker_reads_run_root_from_state_for_each_target(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    first_root = (tmp_path / "runs_a").resolve()
    second_root = (tmp_path / "runs_b").resolve()
    queue_args = ["--queue-root", str(tmp_path / "queue")]
    enqueue = _args(monkeypatch, "enqueue", *queue_args, "--run-root", str(first_root), "--target", TARGET_A, "--target", TARGET_B)
    assert rerun._command_enqueue(enqueue) == 0

    seen: list[tuple[str, Path]] = []

    def _fake_run_target(*, target: str, run_root: Path, **_: Any) -> dict[str, Any]:
        seen.append((target, run_root))
        if target == TARGET_A:
            # A later `enqueue --run-root` while this worker is busy must reach the next target.
            change_root = _args(monkeypatch, "enqueue", *queue_args, "--run-root", str(second_root))
            assert rerun._command_enqueue(change_root) == 0
        return {"target": target, "benchmarks": []}

    monkeypatch.setattr(rerun, "_run_target", _fake_run_target)
    monkeypatch.setattr(rerun, "_git_commit", lambda repo_root: "deadbeef")
    monkeypatch.setattr(rerun, "detect_execution_environment", lambda: SimpleNamespace(kind="virtualized"))

    assert rerun._command_run_worker(_args(monkeypatch, "run-worker", *queue_args)) == 0

    assert seen == [(TARGET_A, first_root), (TARGET_B, second_root)]


@pytest.mark.parametrize(
    "argv",
    [
        ("status", "--allow-portable-expectations-update"),
        ("stop", "--no-allow-portable-expectations-update"),
        ("run-worker", "--allow-portable-expectations-update"),
        ("status", "--run-root", "runs"),
        ("stop", "--run-root", "runs"),
    ],
)
def test_subcommands_reject_settings_they_would_ignore(monkeypatch: pytest.MonkeyPatch, argv: tuple[str, ...]) -> None:
    with pytest.raises(SystemExit):
        _args(monkeypatch, *argv)


def test_restart_without_flags_keeps_every_saved_run_setting(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_worker_spawn: list[list[str]]
) -> None:
    _start_configured_queue(monkeypatch, tmp_path)

    restart = _args(monkeypatch, "start", *_queue_args(tmp_path), "--target", TARGET_B)
    assert rerun._command_start(restart) == 0

    state = _saved_state(tmp_path)
    assert state["pending_targets"] == [TARGET_A, TARGET_B]
    assert state["profile"] == "deep_dive"
    assert state["suite_timeout"] == 900
    assert state["gpu_sm_clock_mhz"] == 1500
    assert state["gpu_mem_clock_mhz"] == 3200
    assert state["allow_portable_expectations_update"] is True


def test_restart_applies_only_the_overrides_it_names(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_worker_spawn: list[list[str]]
) -> None:
    _start_configured_queue(monkeypatch, tmp_path)

    restart = _args(
        monkeypatch,
        "start",
        *_queue_args(tmp_path),
        "--target",
        TARGET_B,
        "--suite-timeout",
        "0",
        "--no-allow-portable-expectations-update",
    )
    assert rerun._command_start(restart) == 0

    state = _saved_state(tmp_path)
    assert state["suite_timeout"] == 0
    assert state["allow_portable_expectations_update"] is False
    assert state["profile"] == "deep_dive"
    assert state["gpu_sm_clock_mhz"] == 1500
    assert state["gpu_mem_clock_mhz"] == 3200


def test_enqueue_and_retry_problems_keep_saved_expectation_write_opt_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_worker_spawn: list[list[str]]
) -> None:
    _start_configured_queue(monkeypatch, tmp_path)

    assert rerun._command_enqueue(_args(monkeypatch, "enqueue", *_queue_args(tmp_path), "--target", TARGET_B)) == 0
    assert _saved_state(tmp_path)["allow_portable_expectations_update"] is True

    assert rerun._command_retry_problems(_args(monkeypatch, "retry-problems", *_queue_args(tmp_path))) == 0
    assert _saved_state(tmp_path)["allow_portable_expectations_update"] is True

    opt_out = _args(monkeypatch, "enqueue", *_queue_args(tmp_path), "--no-allow-portable-expectations-update")
    assert rerun._command_enqueue(opt_out) == 0
    state = _saved_state(tmp_path)
    assert state["allow_portable_expectations_update"] is False
    assert state["suite_timeout"] == 900


def _fake_bench_run(results_payload: dict[str, Any], captured: list[list[str]]):
    def _run_command(cmd: list[str], *, cwd: Path, log_path: Path) -> int:
        captured.append(list(cmd))
        artifacts_dir = Path(cmd[cmd.index("--artifacts-dir") + 1])
        run_id = cmd[cmd.index("--run-id") + 1]
        results_json = artifacts_dir / run_id / "results" / "benchmark_test_results.json"
        results_json.parent.mkdir(parents=True, exist_ok=True)
        results_json.write_text(json.dumps(results_payload), encoding="utf-8")
        return 0

    return _run_command


def _run_target_with_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    state: dict[str, Any],
    bench: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    captured: list[list[str]] = []
    payload = {"results": [{"chapter": "ch01", "benchmarks": [bench]}]}
    monkeypatch.setattr(rerun, "_run_command", _fake_bench_run(payload, captured))
    queue_root = tmp_path / "queue"
    queue_root.mkdir(parents=True, exist_ok=True)
    record = rerun._run_target(
        repo_root=tmp_path / "repo",
        queue_root=queue_root,
        run_root=tmp_path / "runs",
        target=TARGET_A,
        state=state,
        execution_environment="virtualized",
        dmi_product_name=None,
        git_commit="deadbeef",
        paths=rerun._queue_paths(queue_root),
    )
    assert len(captured) == 1
    return record, captured[0]


def _succeeded_bench(**identity: str) -> dict[str, Any]:
    return {
        **identity,
        "status": "succeeded",
        "baseline_time_ms": 2.0,
        "optimizations": [
            {
                "status": "succeeded",
                "speedup": 2.0,
                "time_ms": 1.0,
                "file": "optimized_performance.py",
                "technique": "optimized_performance",
            }
        ],
    }


def test_worker_bench_command_uses_settings_kept_across_restart(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_worker_spawn: list[list[str]]
) -> None:
    _start_configured_queue(monkeypatch, tmp_path)
    assert rerun._command_start(_args(monkeypatch, "start", *_queue_args(tmp_path), "--target", TARGET_B)) == 0

    state = rerun._load_state(tmp_path / "queue" / "state.json")
    _, cmd = _run_target_with_results(
        monkeypatch, tmp_path, state, _succeeded_bench(example="performance", type="python")
    )

    assert cmd[cmd.index("--profile") + 1] == "deep_dive"
    assert cmd[cmd.index("--suite-timeout") + 1] == "900"
    assert cmd[cmd.index("--gpu-sm-clock-mhz") + 1] == "1500"
    assert cmd[cmd.index("--gpu-mem-clock-mhz") + 1] == "3200"


def test_expectation_key_refuses_to_guess_missing_identity_fields() -> None:
    with pytest.raises(ValueError, match="missing type"):
        rerun._expectation_example_key({"example": "double_buffered_pipeline"})
    with pytest.raises(ValueError, match="missing example"):
        rerun._expectation_example_key({"type": "cuda"})
    with pytest.raises(ValueError, match="missing example, type"):
        rerun._expectation_example_key({"example": "  ", "type": ""})


def test_expectation_key_matches_harness_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    # Importing the harness appends -lineinfo to NVCCFLAGS. Restore it after the test.
    monkeypatch.setenv("NVCCFLAGS", os.environ.get("NVCCFLAGS", ""))
    from core.harness.run_benchmarks import expectation_example_key

    for example, bench_type in (
        ("launch_bounds_cuda", "cuda"),
        ("double_buffered_pipeline", "cuda"),
        ("double_buffered_pipeline", "CUDA"),
        ("regional_triton", "python"),
        ("performance", "Python"),
        ("tiny_gemm_triton", "triton"),
    ):
        assert rerun._expectation_example_key({"example": example, "type": bench_type}) == expectation_example_key(
            example, bench_type
        )


def test_run_target_queues_result_without_type_instead_of_guessing_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = rerun._initialize_state(
        run_root=tmp_path / "runs",
        queue_root=tmp_path / "queue",
        profile="none",
        suite_timeout=0,
        gpu_sm_clock_mhz=1965,
        gpu_mem_clock_mhz=3996,
        allow_portable_expectations_update=True,
    )
    state["hardware_key"] = "b200"
    (tmp_path / "repo" / "ch01").mkdir(parents=True)

    record, _ = _run_target_with_results(monkeypatch, tmp_path, state, _succeeded_bench(example="performance"))

    assert record["written_expectation_count"] == 0
    assert record["queued_problem_count"] == 1
    problem = record["benchmarks"][0]
    assert problem["queue_reasons"] == ["missing_benchmark_identity"]
    assert problem["missing_identity_fields"] == ["type"]
    assert not (tmp_path / "repo" / "ch01" / "expectations_b200.json").exists()

    typed_record, _ = _run_target_with_results(
        monkeypatch, tmp_path, state, _succeeded_bench(example="performance", type="python")
    )
    typed_problem = typed_record["benchmarks"][0]
    assert "missing_benchmark_identity" not in typed_problem["queue_reasons"]
    assert typed_problem["queue_reasons"] == ["missing_expectation"]
