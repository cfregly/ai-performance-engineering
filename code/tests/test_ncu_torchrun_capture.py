"""Control-plane coverage for the explicit two-rank NCU helper.

The fake NCU executable below only exercises argv transport and artifact
receipts. These tests do not represent a GPU or profiler capture.
"""

from __future__ import annotations

import base64
import csv
import errno
import functools
import hashlib
import io
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

from core.benchmark.run_manifest import capture_runtime_provenance
from core.harness.torchrun_runtime_provenance import (
    TORCHRUN_RUNTIME_PROVENANCE_PREFIX,
    format_torchrun_runtime_provenance_frames,
)
from core.profiling.ncu_torchrun_capture import (
    APPLICATION_CAPTURE_RECEIPT_SCHEMA,
    CAPTURE_RECEIPT_SCHEMA,
    _CAPTURE_OWNER_ENV,
    _application_report_artifact,
    _cleanup_capture_processes,
    _drain_owned_process_group,
    _marked_pids,
    _parse_application_replay_log,
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


def test_two_rank_capture_docs_cover_both_rank_local_replay_modes() -> None:
    guide = (CODE_ROOT.parent / "docs" / "tooling-and-profiling.md").read_text(
        encoding="utf-8"
    )
    section = guide.split("### Two-rank NCCL captures", 1)[1].split(
        "## Framework traces", 1
    )[0]

    assert "kernel replay by default or explicit range replay" in section
    assert "--replay-mode range" in section
    assert "Range replay requires" in section
    assert "does not provide per-kernel attribution" in section


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


def _application_plan(tmp_path: Path, command: list[str] | None = None, **overrides):
    values = {
        "tcp_port": None,
        "nccl_nvtx_includes": (),
        "replay_mode": "application",
        "target_visible_device": 1,
        "kernel_name": "regex:ncclDevKernel_SendRecv.*",
        "launch_skip": 22,
        "launch_count": 1,
    }
    values.update(overrides)
    return _plan(tmp_path, command, **values)


@functools.lru_cache(maxsize=1)
def _runtime_snapshot_fixture():
    return capture_runtime_provenance()


def _synthetic_cuda_snapshot(process_id: int, **updates):
    snapshot = _runtime_snapshot_fixture()
    values = {
        "process_id": process_id,
        "cuda_available": True,
        "driver_version": "test-driver",
        "cuda_version": "test-cuda",
        "cudnn_version": "test-cudnn",
        "library_versions_complete": True,
        "collection_warnings": [],
    }
    values.update(updates)
    return snapshot.model_copy(update=values)


def _application_generation(
    pass_number: int,
    *,
    rank0_pid: int,
    rank1_pid: int,
    marker_pass: int | None = None,
    include_ranks: tuple[int, ...] = (0, 1),
    rank0_updates: dict[str, object] | None = None,
    rank1_updates: dict[str, object] | None = None,
) -> bytes:
    chunks = [
        (
            '==PROF== Profiling on device 1 : "ncclDevKernel_SendRecv": '
            f"Application replay pass {pass_number if marker_pass is None else marker_pass}\n"
        ).encode("utf-8")
    ]
    for local_rank, process_id in ((0, rank0_pid), (1, rank1_pid)):
        if local_rank not in include_ranks:
            continue
        updates = (
            rank0_updates
            if local_rank == 0 and rank0_updates is not None
            else rank1_updates
            if local_rank == 1 and rank1_updates is not None
            else {}
        )
        snapshot = _synthetic_cuda_snapshot(process_id, **updates)
        chunks.extend(
            format_torchrun_runtime_provenance_frames(
                snapshot,
                local_rank=local_rank,
                rank=local_rank,
                world_size=2,
                local_world_size=2,
                pipe_buf=4096,
            )
        )
    return b"".join(chunks)


def _application_csv(*, process_id: int, pass_count: int = 2, device: int = 1) -> str:
    metrics = [
        "gpu__time_duration.avg",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    ]
    header = [
        "ID",
        "Process ID",
        "Device",
        "device__attribute_device_index",
        "Kernel Name",
        "profiler__replayer_passes",
        *metrics,
    ]
    units = ["", "", "", "", "", "", "ns", "%", "%", "%", "%"]
    values = [
        "1",
        str(process_id),
        str(device),
        str(device),
        "ncclDevKernel_SendRecv(int)",
        str(pass_count),
        "1000",
        "25",
        "20",
        "15",
        "50",
    ]
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows((header, units, values))
    return stream.getvalue()


def _make_fake_application_ncu(
    path: Path,
    *,
    combined_log: bytes,
    raw_csv: str,
) -> Path:
    encoded_log = base64.b64encode(combined_log).decode("ascii")
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import base64, sys\n"
        "from pathlib import Path\n"
        f"combined_log = base64.b64decode({encoded_log!r})\n"
        f"raw_csv = {raw_csv!r}\n"
        "args = sys.argv[1:]\n"
        "if args == ['--version']:\n"
        "    print('Version 2026.2.1.0 (test control plane)')\n"
        "    raise SystemExit(0)\n"
        "if '--import' in args:\n"
        "    sys.stdout.write(raw_csv)\n"
        "    raise SystemExit(0)\n"
        "export_index = args.index('--export')\n"
        "Path(args[export_index + 1]).write_bytes(b'test application report\\n')\n"
        "sys.stderr.buffer.write(combined_log)\n"
        "sys.stderr.buffer.flush()\n",
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path.resolve()


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
        "if '--replay-mode' in sys.argv and sys.argv[sys.argv.index('--replay-mode') + 1] == 'range' and '--kernel-name' in sys.argv:\n"
        "    print('Option kernel-name is not supported during range replay.', file=sys.stderr)\n"
        "    raise SystemExit(2)\n"
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
    assert plan.ncu_prefix[plan.ncu_prefix.index("--replay-mode") + 1] == "kernel"
    assert plan.replay_mode == "kernel"
    assert plan.ncu_prefix[plan.ncu_prefix.index("--kernel-name") + 1] == "regex:nccl.*"
    assert "compute_kernel" not in " ".join(plan.ncu_prefix)
    for rank in (0, 1):
        command = plan.expected_rank_argv(rank)
        assert command[command.index("--devices") + 1] == str(rank)
        assert command[command.index("--export") + 1] == str(plan.report_path(rank))
    assert plan.expected_rank_argv(0) != plan.expected_rank_argv(1)


def test_application_plan_wraps_the_exact_full_torchrun_command(tmp_path: Path) -> None:
    command = _torchrun_command(tmp_path / "target.py")
    plan = _application_plan(tmp_path, command)

    assert plan.launch_argv[-len(command) :] == tuple(command)
    assert plan.launch_argv[:5] == (
        str(tmp_path / "ncu"),
        "--clock-control",
        "none",
        "--target-processes",
        "all",
    )
    expected_options = {
        "--devices": "1",
        "--replay-mode": "application",
        "--app-replay-mode": "strict",
        "--app-replay-match": "grid",
        "--filter-mode": "per-gpu",
        "--kernel-name": "regex:ncclDevKernel_SendRecv.*",
        "--launch-skip": "22",
        "--launch-count": "1",
        "--kill": "no",
        "--check-exit-code": "yes",
    }
    for option, value in expected_options.items():
        assert plan.launch_argv[plan.launch_argv.index(option) + 1] == value
    assert plan.launch_argv[plan.launch_argv.index("--export") + 1] == str(
        plan.application_report_path
    )
    assert "--communicator" not in plan.launch_argv
    assert "--nvtx" not in plan.launch_argv
    assert "core.profiling.ncu_torchrun_rank" not in plan.launch_argv
    assert plan.nccl_nvtx_includes == ()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"nccl_nvtx_includes": ["compute_kernel:profile"]}, "named NCCL push/pop"),
        ({"nccl_nvtx_includes": ["NCCL@ncclGroupEnd"]}, "named NCCL push/pop"),
        ({"nccl_nvtx_includes": []}, "At least one explicit NCCL"),
        ({"tcp_port": 80}, "tcp_port"),
        ({"timeout_seconds": math.nan}, "timeout_seconds"),
        ({"timeout_seconds": math.inf}, "timeout_seconds"),
        ({"replay_mode": "range"}, "all_matching_kernels=True"),
    ],
)
def test_plan_rejects_unsupported_capture_options(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _plan(tmp_path, **overrides)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"target_visible_device": None}, "target_visible_device"),
        ({"target_visible_device": 2}, "target_visible_device"),
        ({"kernel_name": None}, "kernel_name"),
        ({"kernel_name": "ncclDevKernel_SendRecv"}, "explicit nonempty regex"),
        ({"kernel_name": "regex:["}, "regular expression"),
        ({"launch_skip": -1}, "launch_skip"),
        ({"launch_count": 2}, "exactly 1"),
        ({"tcp_port": _free_port()}, "rejects tcp_port"),
        ({"nccl_nvtx_includes": NCCL_RANGES}, "rejects NCCL NVTX"),
        ({"all_matching_kernels": True}, "rejects all_matching_kernels"),
    ],
)
def test_application_plan_rejects_missing_or_cross_mode_options(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _application_plan(tmp_path, **overrides)


@pytest.mark.parametrize(
    "replay_mode",
    ["kernel", "range"],
)
def test_rank_local_modes_reject_application_selectors(
    tmp_path: Path, replay_mode: str
) -> None:
    with pytest.raises(ValueError, match="rejects application options"):
        _plan(
            tmp_path,
            replay_mode=replay_mode,
            all_matching_kernels=replay_mode == "range",
            target_visible_device=1,
        )


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


def test_range_replay_omits_per_kernel_filter_and_launch_limit(
    tmp_path: Path,
) -> None:
    tcp_port = _free_port()
    command = _torchrun_command(tmp_path / "target.py")
    kernel = _plan(
        tmp_path,
        command,
        tcp_port=tcp_port,
        all_matching_kernels=True,
    )
    range_replay = _plan(
        tmp_path,
        command,
        tcp_port=tcp_port,
        all_matching_kernels=True,
        replay_mode="range",
    )

    kernel_prefix = list(kernel.ncu_prefix)
    range_prefix = list(range_replay.ncu_prefix)
    filter_index = kernel_prefix.index("--kernel-name")
    assert kernel_prefix[filter_index + 1] == "regex:nccl.*"
    del kernel_prefix[filter_index : filter_index + 2]
    replay_value = kernel_prefix.index("--replay-mode") + 1
    assert kernel_prefix[replay_value] == "kernel"
    assert range_prefix[replay_value] == "range"
    range_prefix[replay_value] = "kernel"
    assert range_prefix == kernel_prefix
    assert "--kernel-name" not in range_replay.ncu_prefix
    assert "--launch-count" not in range_replay.ncu_prefix
    assert range_replay.ncu_prefix.count("--nvtx-include") == 2
    assert range_replay.ncu_prefix.count("--lockstep-nvtx-include") == 2
    assert range_replay.replay_mode == "range"


@pytest.mark.parametrize("pass_count", [2, 9])
def test_application_log_retains_each_complete_two_rank_generation(
    tmp_path: Path, pass_count: int
) -> None:
    plan = _application_plan(tmp_path)
    plan.output_dir.mkdir()
    plan.combined_path.write_bytes(
        b"".join(
            _application_generation(
                pass_number,
                rank0_pid=10_000 + pass_number * 2,
                rank1_pid=10_001 + pass_number * 2,
            )
            for pass_number in range(1, pass_count + 1)
        )
    )

    generations, log_artifact = _parse_application_replay_log(plan)

    assert [item["application_replay_pass"] for item in generations] == list(
        range(1, pass_count + 1)
    )
    assert all(item["both_post_main_receipts"] is True for item in generations)
    assert all(item["cuda_runtime_parity"] is True for item in generations)
    assert len(
        {
            pid
            for item in generations
            for pid in item["execution_process_ids"].values()
        }
    ) == pass_count * 2
    assert generations[0]["marker_start_byte"] == 0
    assert generations[-1]["payload_end_byte"] == log_artifact["bytes"]
    assert log_artifact["stream_mode"] == (
        "stdout-and-stderr-shared-open-file-description-at-launch"
    )


@pytest.mark.parametrize(
    ("raw_log", "match"),
    [
        (
            lambda: _application_generation(1, rank0_pid=101, rank1_pid=102)
            + _application_generation(2, marker_pass=3, rank0_pid=103, rank1_pid=104),
            "consecutive from one",
        ),
        (
            lambda: _application_generation(1, rank0_pid=101, rank1_pid=102)
            + _application_generation(2, marker_pass=1, rank0_pid=103, rank1_pid=104),
            "consecutive from one",
        ),
        (
            lambda: _application_generation(1, rank0_pid=101, rank1_pid=102).replace(
                b"on device 1", b"on device 0", 1
            ),
            "different visible device",
        ),
        (
            lambda: _application_generation(1, rank0_pid=101, rank1_pid=102).replace(
                b"ncclDevKernel_SendRecv", b"unselected_kernel", 1
            ),
            "does not match selector",
        ),
        (
            lambda: _application_generation(1, rank0_pid=101, rank1_pid=102).replace(
                b"Application replay pass 1", b"Application replay pass zero", 1
            ),
            "Unexpected application replay marker",
        ),
        (
            lambda: _application_generation(
                1,
                rank0_pid=101,
                rank1_pid=102,
                include_ranks=(0,),
            ),
            "rank set mismatch",
        ),
        (
            lambda: _application_generation(1, rank0_pid=101, rank1_pid=102)
            + _application_generation(2, rank0_pid=101, rank1_pid=104),
            "reused worker process IDs",
        ),
        (
            lambda: _application_generation(1, rank0_pid=101, rank1_pid=102)
            + _application_generation(
                2,
                rank0_pid=103,
                rank1_pid=104,
                rank0_updates={"torch_version": "drifted-test-version"},
                rank1_updates={"torch_version": "drifted-test-version"},
            ),
            "runtime provenance drifted",
        ),
        (
            lambda: _application_generation(1, rank0_pid=101, rank1_pid=102)
            + (TORCHRUN_RUNTIME_PROVENANCE_PREFIX + "{}").encode("utf-8"),
            "Incomplete torchrun runtime provenance frame",
        ),
        (
            lambda: _application_generation(1, rank0_pid=101, rank1_pid=102).split(
                b"\n", 1
            )[1],
            "no recognized pass markers",
        ),
    ],
)
def test_application_log_rejects_ambiguous_replay_generations(
    tmp_path: Path, raw_log, match: str
) -> None:
    plan = _application_plan(tmp_path)
    plan.output_dir.mkdir()
    plan.combined_path.write_bytes(raw_log())

    with pytest.raises(RuntimeError, match=match):
        _parse_application_replay_log(plan)


def test_application_log_rejects_provenance_before_the_first_marker(tmp_path: Path) -> None:
    plan = _application_plan(tmp_path)
    plan.output_dir.mkdir()
    complete = _application_generation(1, rank0_pid=101, rank1_pid=102)
    marker_start = complete.index(b"\n") + 1
    plan.combined_path.write_bytes(complete[marker_start:] + complete[:marker_start])

    with pytest.raises(RuntimeError, match="before application replay pass 1"):
        _parse_application_replay_log(plan)


def test_application_report_import_validates_one_selected_row(tmp_path: Path) -> None:
    fake_ncu = _make_fake_application_ncu(
        tmp_path / "ncu",
        combined_log=b"",
        raw_csv=_application_csv(process_id=102),
    )
    plan = _application_plan(tmp_path, ncu_path=fake_ncu)
    plan.output_dir.mkdir()
    plan.combined_path.write_bytes(
        _application_generation(1, rank0_pid=101, rank1_pid=102)
        + _application_generation(2, rank0_pid=103, rank1_pid=104)
    )
    generations, _ = _parse_application_replay_log(plan)
    start_ns = time.time_ns()
    plan.application_report_path.write_bytes(b"fresh report\n")

    artifact, errors = _application_report_artifact(
        plan,
        start_ns=start_ns,
        generations=generations,
    )

    assert errors == []
    assert artifact["row"]["visible_device"] == 1
    assert artifact["row"]["process_id"] == 102
    assert artifact["row"]["profiler_pass_count"] == 2
    assert set(artifact["row"]["metrics"]) == {
        "gpu__time_duration.avg",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "lts__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    }


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
    assert receipt["schema"] == CAPTURE_RECEIPT_SCHEMA
    assert CAPTURE_RECEIPT_SCHEMA == "aisp.ncu-torchrun-capture.v1"
    assert receipt["returncode"] == 0
    assert receipt["timed_out"] is False
    assert receipt["source"] == source
    assert receipt["torchrun_argv"] == command
    assert receipt["claim_limit"].startswith("Artifact presence")
    assert receipt["cleanup"]["natural"] is True
    assert [item["local_rank"] for item in receipt["rank_artifacts"]] == [0, 1]
    assert all(item["observed_argv"] == item["expected_argv"] for item in receipt["rank_artifacts"])
    assert len({item["report_sha256"] for item in receipt["rank_artifacts"]}) == 2


def test_application_cli_retains_repeated_post_main_receipts_and_one_report(
    tmp_path: Path,
) -> None:
    repo, target, source = _make_git_repo(tmp_path)
    combined_log = (
        _application_generation(1, rank0_pid=201, rank1_pid=202)
        + _application_generation(2, rank0_pid=203, rank1_pid=204)
    )
    fake_ncu = _make_fake_application_ncu(
        tmp_path / "ncu",
        combined_log=combined_log,
        raw_csv=_application_csv(process_id=202),
    )
    output = tmp_path / "capture"
    command = _torchrun_command(target)
    cli = [
        sys.executable,
        "-m",
        "core.profiling.ncu_torchrun_capture",
        "--label",
        "application-control-plane-test",
        "--repo-root",
        str(repo),
        "--source",
        source,
        "--output-dir",
        str(output),
        "--ncu",
        str(fake_ncu),
        "--timeout-seconds",
        "45",
        "--replay-mode",
        "application",
        "--target-visible-device",
        "1",
        "--kernel-name",
        "regex:ncclDevKernel_SendRecv.*",
        "--launch-skip",
        "22",
        "--launch-count",
        "1",
        "--",
        *command,
    ]

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
    assert receipt["schema"] == APPLICATION_CAPTURE_RECEIPT_SCHEMA
    assert (
        APPLICATION_CAPTURE_RECEIPT_SCHEMA
        == "aisp.ncu-torchrun-application-capture.v1"
    )
    assert receipt["capture_topology"] == "parent-application-replay"
    assert receipt["torchrun_argv"] == command
    assert receipt["argv"][-len(command) :] == command
    assert receipt["profiler_pass_count"] == 2
    assert len(receipt["replay_generations"]) == 2
    assert receipt["application_artifact"]["row"]["visible_device"] == 1
    assert receipt["application_artifact"]["row"]["profiler_pass_count"] == 2
    assert receipt["application_artifact"]["report_sha256"]
    assert receipt["combined_log"]["sha256"] == hashlib.sha256(combined_log).hexdigest()
    assert receipt["artifact_errors"] == []
    assert receipt["numerical_output_qualified"] is False
    assert receipt["performance_qualified"] is False
    assert receipt["all_matching_launches_covered"] is False
    assert receipt["full_coverage_qualified"] is False
    assert not (output / "rank-config.json").exists()
    assert list(output.glob("*.ncu-rep")) == [output / "application.ncu-rep"]


def test_range_replay_cli_propagates_exact_argv_and_receipt(tmp_path: Path) -> None:
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
        "range-replay-control-plane-test",
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
        "--all-matching-kernels",
        "--replay-mode",
        "range",
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
    assert receipt["schema"] == CAPTURE_RECEIPT_SCHEMA
    assert receipt["replay_mode"] == "range"
    assert receipt["all_matching_kernels"] is True
    for artifact in receipt["rank_artifacts"]:
        observed = artifact["observed_argv"]
        assert observed == artifact["expected_argv"]
        assert observed[observed.index("--replay-mode") + 1] == "range"
        assert "--kernel-name" not in observed
        assert "--launch-count" not in observed


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
