"""Control-plane coverage for the explicit two-rank TCP NCU helper.

The fake NCU executable below only exercises argv transport and artifact
receipts. These tests do not represent a GPU or profiler capture.
"""

from __future__ import annotations

import errno
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path

import pytest

from core.profiling.ncu_torchrun_capture import (
    _CAPTURE_OWNER_ENV,
    _cleanup_capture_processes,
    _drain_owned_process_group,
    _marked_pids,
    _proc_cleanup_supported,
    _snapshot_proc_identities,
    build_capture_plan,
)
from core.profiling.ncu_torchrun_rank import (
    RANK_CONFIG_SCHEMA,
    build_rank_command,
    load_rank_config,
)

CODE_ROOT = Path(__file__).resolve().parents[1]
NCCL_RANGES = ("NCCL@ncclGroupEnd/", "NCCL@ncclAllReduce/")


def _free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def _torchrun_command(
    target: Path,
    *,
    nproc: str = "2",
    nnodes: str = "1",
    interpreter: str = sys.executable,
) -> list[str]:
    return [
        interpreter,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        nproc,
        "--nnodes",
        nnodes,
        "--rdzv_backend",
        "c10d",
        "--rdzv_endpoint",
        f"127.0.0.1:{_free_port()}",
        "-m",
        "core.harness.torchrun_wrapper",
        "--aisp-emit-runtime-provenance",
        "--aisp-target-script",
        str(target),
        "--aisp-expected-torch-seed",
        "42",
    ]


def _plan(tmp_path: Path, command: list[str] | None = None, **overrides):
    values = {
        "label": "pipeline-baseline",
        "repo_root": tmp_path,
        "source": "a" * 40,
        "output_dir": tmp_path / "capture",
        "ncu_path": tmp_path / "ncu",
        "tcp_port": _free_port(),
        "timeout_seconds": 30,
        "nccl_nvtx_includes": NCCL_RANGES,
        "torchrun_argv": command or _torchrun_command(tmp_path / "target.py"),
    }
    values.update(overrides)
    return build_capture_plan(**values)


def _make_git_repo(tmp_path: Path) -> tuple[Path, Path, str]:
    repo = tmp_path / "repo"
    code = repo / "code"
    code.mkdir(parents=True)
    target = code / "target.py"
    target.write_text(
        "import os\n"
        "if os.environ['LOCAL_RANK'] == '0':\n"
        "    print('rank0 time_per_iter_ms: 1.0', flush=True)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "code/target.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    source = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    return repo, target, source


def _make_fake_ncu(path: Path, marker: Path | None = None) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "from pathlib import Path\n"
        f"marker = Path({str(marker)!r}) if {marker is not None!r} else None\n"
        "if marker is not None:\n"
        "    marker.write_text('invoked\\n')\n"
        "if sys.argv[1:] == ['--version']:\n"
        "    print('Version 2026.2.1.0 (test control plane)')\n"
        "    raise SystemExit(0)\n"
        "export_index = sys.argv.index('--export')\n"
        "report = Path(sys.argv[export_index + 1])\n"
        "report.write_text('rank=' + os.environ['LOCAL_RANK'] + '\\n')\n"
        "worker = sys.argv[export_index + 2:]\n"
        "os.execvpe(worker[0], worker, os.environ)\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path.resolve()


def _make_fake_python_launcher(path: Path, *, lingering_child_pid_path: Path | None = None) -> Path:
    linger = (
        'child_code = "import signal,time; '
        'signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"\n'
        "child = subprocess.Popen([sys.executable, '-c', child_code])\n"
        f"Path({str(lingering_child_pid_path)!r}).write_text(str(child.pid))\n"
        "os._exit(0)\n"
        if lingering_child_pid_path is not None
        else "raise SystemExit(0)\n"
    )
    indented_linger = "    " + linger.rstrip().replace("\n", "\n    ") + "\n"
    source = (
        "#!/usr/bin/env python3\n"
        "import os, subprocess, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if args[:2] == ['-m', 'torch.distributed.run']:\n"
        "    child = args[args.index('--no-python') + 1:]\n"
        "    for rank in (0, 1):\n"
        "        env = dict(os.environ)\n"
        "        env.update(LOCAL_RANK=str(rank), RANK=str(rank), WORLD_SIZE='2', "
        "LOCAL_WORLD_SIZE='2', GROUP_RANK='0')\n"
        "        result = subprocess.run(child, env=env, check=False)\n"
        "        if result.returncode:\n"
        "            raise SystemExit(result.returncode)\n"
    )
    source += indented_linger
    source += (
        "if args[:2] == ['-m', 'core.harness.torchrun_wrapper']:\n"
        "    print('control-plane worker only', flush=True)\n"
        "    raise SystemExit(0)\n"
        "os.execv(sys.executable, [sys.executable, *args])\n"
    )
    path.write_text(source, encoding="utf-8")
    path.chmod(0o755)
    return path.resolve()


def _capture_environment(repo: Path) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "AISP_LOCK_GPU_CLOCKS": "1",
            "AISP_RAMP_GPU_CLOCKS": "1",
            "CUDA_VISIBLE_DEVICES": "0,1",
            "PYTHONPATH": os.pathsep.join((str(repo / "code"), str(CODE_ROOT))),
        }
    )
    return environment


def test_plan_exposes_exact_rank_commands_and_scoped_lockstep(tmp_path: Path) -> None:
    plan = _plan(tmp_path)

    assert plan.launch_argv[:3] == (sys.executable, "-m", "torch.distributed.run")
    assert plan.launch_argv[-4:-1] == (
        sys.executable,
        "-m",
        "core.profiling.ncu_torchrun_rank",
    )
    assert plan.ncu_prefix.count("--lockstep-nvtx-include") == 2
    assert plan.ncu_prefix.count("--nvtx-include") == 2
    assert "--launch-count" in plan.ncu_prefix
    assert plan.ncu_prefix[plan.ncu_prefix.index("--launch-count") + 1] == "1"
    assert "compute_kernel" not in " ".join(plan.ncu_prefix)
    for rank in (0, 1):
        command = plan.expected_rank_argv(rank)
        assert command[command.index("--devices") + 1] == str(rank)
        assert command[command.index("--export") + 1] == str(plan.report_path(rank))
    assert plan.expected_rank_argv(0) != plan.expected_rank_argv(1)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"nccl_nvtx_includes": ["compute_kernel:profile"]}, "named NCCL push/pop"),
        ({"nccl_nvtx_includes": ["NCCL@ncclGroupEnd"]}, "named NCCL push/pop"),
        ({"nccl_nvtx_includes": []}, "At least one explicit NCCL"),
        ({"tcp_port": 80}, "tcp_port"),
        ({"timeout_seconds": math.nan}, "timeout_seconds"),
        ({"timeout_seconds": math.inf}, "timeout_seconds"),
    ],
)
def test_plan_rejects_unsupported_capture_options(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _plan(tmp_path, **overrides)


@pytest.mark.parametrize(
    ("nproc", "nnodes", "match"),
    [("4", "1", "nproc_per_node 2"), ("2", "2", "exactly one node")],
)
def test_plan_rejects_unsupported_torchrun_shape(
    tmp_path: Path, nproc: str, nnodes: str, match: str
) -> None:
    command = _torchrun_command(tmp_path / "target.py", nproc=nproc, nnodes=nnodes)
    with pytest.raises(ValueError, match=match):
        _plan(tmp_path, command)


def test_all_matching_kernels_omits_only_the_launch_limit(tmp_path: Path) -> None:
    tcp_port = _free_port()
    command = _torchrun_command(tmp_path / "target.py")
    first = _plan(tmp_path, command, tcp_port=tcp_port)
    all_matching = _plan(tmp_path, command, tcp_port=tcp_port, all_matching_kernels=True)

    assert "--launch-count" in first.ncu_prefix
    assert "--launch-count" not in all_matching.ncu_prefix
    first_without_limit = list(first.ncu_prefix)
    limit_index = first_without_limit.index("--launch-count")
    del first_without_limit[limit_index : limit_index + 2]
    assert first_without_limit == list(all_matching.ncu_prefix)
    assert all_matching.all_matching_kernels is True


def test_rank_entrypoint_executes_the_recorded_command(tmp_path: Path) -> None:
    fake_ncu = _make_fake_ncu(tmp_path / "ncu")
    output = tmp_path / "out"
    output.mkdir()
    config_path = tmp_path / "rank-config.json"
    config_path.write_text(
        json.dumps(
            {
                "schema": RANK_CONFIG_SCHEMA,
                "ncu_prefix": [str(fake_ncu), "--replay-mode", "kernel"],
                "worker_argv": [sys.executable, "-c", "pass"],
                "output_dir": str(output),
                "world_size": 2,
            }
        ),
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment.update(
        {
            "LOCAL_RANK": "1",
            "RANK": "1",
            "WORLD_SIZE": "2",
            "LOCAL_WORLD_SIZE": "2",
            "GROUP_RANK": "0",
        }
    )
    config = load_rank_config(config_path)
    expected, report, argv_path = build_rank_command(config, environment)

    result = subprocess.run(
        [sys.executable, "-m", "core.profiling.ncu_torchrun_rank", str(config_path)],
        cwd=CODE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert json.loads(argv_path.read_text(encoding="utf-8")) == expected
    assert report.read_text(encoding="utf-8") == "rank=1\n"


def test_cleanup_drains_child_that_outlives_parent_and_ignores_sigterm(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "child.pid"
    child_code = (
        "import os, signal, time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(child_pid_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    parent_code = (
        "import os, subprocess, sys\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        "os._exit(0)\n"
    )
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        start_new_session=True,
    )
    parent.wait(timeout=10)
    deadline = time.monotonic() + 10
    while not child_pid_path.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert child_pid_path.is_file()
    child_pid = int(child_pid_path.read_text())

    signals_sent, drained, errors = _drain_owned_process_group(parent, grace_seconds=0.2)

    assert signals_sent == ["SIGTERM", "SIGKILL"]
    assert drained is True
    assert errors == []
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_process_group_cleanup_reaps_running_timed_out_leader_after_kill(
    tmp_path: Path,
) -> None:
    ready = tmp_path / "leader-ready"
    code = (
        "import signal,time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(ready)!r}).write_text('ready\\n')\n"
        "time.sleep(60)\n"
    )
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.is_file()
        signals_sent, drained, errors = _drain_owned_process_group(process, grace_seconds=0.1)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)

    assert signals_sent == ["SIGTERM", "SIGKILL"]
    assert drained is True
    assert errors == []
    assert process.returncode == -signal.SIGKILL


def test_marker_scan_matches_only_the_exact_capture_owner(tmp_path: Path) -> None:
    proc_root = tmp_path / "proc"
    for pid, marker in ((4242, "owned"), (4243, "other")):
        process_dir = proc_root / str(pid)
        process_dir.mkdir(parents=True)
        (process_dir / "environ").write_bytes(
            b"PATH=/bin\0" + f"{_CAPTURE_OWNER_ENV}={marker}".encode() + b"\0"
        )

    pids, error = _marked_pids("owned", proc_root)

    assert pids == {4242}
    assert error is None


def test_marker_scan_ignores_unreadable_unrelated_same_user_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    owned_process = proc_root / "4242"
    owned_process.mkdir(parents=True)
    (owned_process / "environ").write_bytes(
        b"PATH=/bin\0" + f"{_CAPTURE_OWNER_ENV}=owned".encode() + b"\0"
    )
    unrelated_process = proc_root / "4243"
    unrelated_process.mkdir()
    unrelated_environment = unrelated_process / "environ"
    unrelated_environment.write_bytes(b"PATH=/bin\0")
    stat_fields = ["S", *(["0"] * 18), "100"]
    (unrelated_process / "stat").write_text(
        f"4243 (unrelated) {' '.join(stat_fields)}\n", encoding="utf-8"
    )
    assert unrelated_process.stat().st_uid == os.geteuid()

    original_read_bytes = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if path == unrelated_environment:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    pids, error = _marked_pids(
        "owned",
        proc_root,
        owned_candidates={4242},
        preexisting_identities=frozenset({(4243, 100)}),
    )

    assert pids == {4242}
    assert error is None


def test_marker_scan_fails_closed_for_unreadable_owned_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    owned_process = proc_root / "4242"
    owned_process.mkdir(parents=True)
    owned_environment = owned_process / "environ"
    owned_environment.write_bytes(b"PATH=/bin\0" + f"{_CAPTURE_OWNER_ENV}=owned".encode() + b"\0")
    original_read_bytes = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if path == owned_environment:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    pids, error = _marked_pids("owned", proc_root, owned_candidates={4242})

    assert pids == set()
    assert error is not None
    assert "owned candidate pid 4242" in error
    assert "Permission denied" in error


def test_marker_scan_fails_closed_for_unreadable_new_marked_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    proc_root = tmp_path / "proc"
    child_process = proc_root / "4243"
    child_process.mkdir(parents=True)
    child_environment = child_process / "environ"
    child_environment.write_bytes(b"PATH=/bin\0" + f"{_CAPTURE_OWNER_ENV}=owned".encode() + b"\0")
    stat_fields = ["S", *(["0"] * 18), "200"]
    (child_process / "stat").write_text(
        f"4243 (new-child) {' '.join(stat_fields)}\n", encoding="utf-8"
    )
    original_read_bytes = Path.read_bytes

    def read_bytes(path: Path) -> bytes:
        if path == child_environment:
            raise PermissionError(errno.EACCES, "Permission denied", str(path))
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)

    pids, error = _marked_pids(
        "owned",
        proc_root,
        preexisting_identities=frozenset({(4243, 100)}),
    )

    assert pids == set()
    assert error is not None
    assert "unclassified post-snapshot pid 4243" in error
    assert "Permission denied" in error


@pytest.mark.skipif(
    not _proc_cleanup_supported(),
    reason="detached-session process-tree cleanup requires Linux procfs",
)
def test_marker_cleanup_drains_detached_child_that_ignores_sigterm(tmp_path: Path) -> None:
    marker = "detached-child-test"
    preexisting_identities = _snapshot_proc_identities()
    child_pid_path = tmp_path / "detached.pid"
    child_code = (
        "import os,signal,time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(child_pid_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    parent_code = (
        "import os,subprocess,sys\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}], "
        "start_new_session=True)\n"
        "os._exit(0)\n"
    )
    environment = dict(os.environ, **{_CAPTURE_OWNER_ENV: marker})
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_code],
        env=environment,
        start_new_session=True,
    )
    child_pid = -1
    try:
        parent.wait(timeout=10)
        deadline = time.monotonic() + 10
        while not child_pid_path.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert child_pid_path.is_file()
        child_pid = int(child_pid_path.read_text())

        cleanup = _cleanup_capture_processes(
            parent,
            marker,
            grace_seconds=0.1,
            preexisting_identities=preexisting_identities,
        )

        assert cleanup.supported is True
        assert cleanup.natural is False
        assert cleanup.drained is True
        assert cleanup.signals_sent == ("SIGTERM", "SIGKILL")
        assert child_pid in cleanup.observed_pids
        assert cleanup.surviving_pids == ()
        assert cleanup.errors == ()
    finally:
        if child_pid > 0:
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)


def test_cli_records_real_control_plane_subprocess_receipts(tmp_path: Path) -> None:
    repo, target, source = _make_git_repo(tmp_path)
    fake_ncu = _make_fake_ncu(tmp_path / "ncu")
    fake_python = _make_fake_python_launcher(tmp_path / "python")
    output = tmp_path / "capture"
    tcp_port = _free_port()
    command = _torchrun_command(target, interpreter=str(fake_python))
    cli = [
        sys.executable,
        "-m",
        "core.profiling.ncu_torchrun_capture",
        "--label",
        "control-plane-test",
        "--repo-root",
        str(repo),
        "--source",
        source,
        "--output-dir",
        str(output),
        "--ncu",
        str(fake_ncu),
        "--tcp-port",
        str(tcp_port),
        "--timeout-seconds",
        "45",
    ]
    for include in NCCL_RANGES:
        cli.extend(["--nvtx-include", include])
    cli.extend(["--", *command])

    result = subprocess.run(
        cli,
        cwd=CODE_ROOT,
        env=_capture_environment(repo),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    expected_status = (
        "CAPTURE_ARTIFACTS_PRESENT" if _proc_cleanup_supported() else "CLEANUP_UNVERIFIED"
    )
    assert result.returncode == (0 if _proc_cleanup_supported() else 1), result.stderr
    assert receipt["status"] == expected_status
    assert receipt["returncode"] == 0
    assert receipt["timed_out"] is False
    assert receipt["source"] == source
    assert receipt["torchrun_argv"] == command
    assert receipt["claim_limit"].startswith("Artifact presence")
    assert receipt["cleanup"]["natural"] is True
    assert [item["local_rank"] for item in receipt["rank_artifacts"]] == [0, 1]
    assert all(item["observed_argv"] == item["expected_argv"] for item in receipt["rank_artifacts"])
    assert len({item["report_sha256"] for item in receipt["rank_artifacts"]}) == 2


def test_zero_return_with_reports_and_forced_cleanup_is_not_success(tmp_path: Path) -> None:
    repo, target, source = _make_git_repo(tmp_path)
    fake_ncu = _make_fake_ncu(tmp_path / "ncu")
    child_pid_path = tmp_path / "lingering.pid"
    fake_python = _make_fake_python_launcher(
        tmp_path / "python", lingering_child_pid_path=child_pid_path
    )
    output = tmp_path / "capture"
    cli = [
        sys.executable,
        "-m",
        "core.profiling.ncu_torchrun_capture",
        "--label",
        "forced-cleanup-test",
        "--repo-root",
        str(repo),
        "--source",
        source,
        "--output-dir",
        str(output),
        "--ncu",
        str(fake_ncu),
        "--tcp-port",
        str(_free_port()),
        "--timeout-seconds",
        "45",
    ]
    for include in NCCL_RANGES:
        cli.extend(["--nvtx-include", include])
    cli.extend(["--", *_torchrun_command(target, interpreter=str(fake_python))])

    try:
        result = subprocess.run(
            cli,
            cwd=CODE_ROOT,
            env=_capture_environment(repo),
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )

        assert result.returncode == 1, result.stderr
        receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
        assert receipt["returncode"] == 0
        assert receipt["status"] != "CAPTURE_ARTIFACTS_PRESENT"
        assert receipt["status"] == (
            "FORCED_CLEANUP" if _proc_cleanup_supported() else "CLEANUP_UNVERIFIED"
        )
        assert receipt["artifact_errors"] == []
        assert receipt["cleanup"]["forced"] is True
        assert receipt["cleanup"]["natural"] is False
        assert receipt["cleanup"]["drained"] is _proc_cleanup_supported()
    finally:
        if child_pid_path.is_file():
            with suppress(ProcessLookupError):
                os.kill(int(child_pid_path.read_text()), signal.SIGKILL)


def test_invalid_cli_fails_before_ncu_or_output_creation(tmp_path: Path) -> None:
    repo, target, source = _make_git_repo(tmp_path)
    marker = tmp_path / "ncu-invoked"
    fake_ncu = _make_fake_ncu(tmp_path / "ncu", marker=marker)
    output = tmp_path / "capture"
    cli = [
        sys.executable,
        "-m",
        "core.profiling.ncu_torchrun_capture",
        "--label",
        "invalid",
        "--repo-root",
        str(repo),
        "--source",
        source,
        "--output-dir",
        str(output),
        "--ncu",
        str(fake_ncu),
        "--tcp-port",
        str(_free_port()),
        "--timeout-seconds",
        "30",
        "--nvtx-include",
        "compute_kernel:profile",
        "--",
        *_torchrun_command(target),
    ]

    result = subprocess.run(
        cli,
        cwd=CODE_ROOT,
        env=_capture_environment(repo),
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 2
    assert "named NCCL push/pop" in result.stderr
    assert not marker.exists()
    assert not output.exists()
