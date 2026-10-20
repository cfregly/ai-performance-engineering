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
import re
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace

import pytest

import core.profiling.ncu_torchrun_capture as ncu_capture
from core.benchmark.run_manifest import capture_runtime_provenance
from core.harness.torchrun_runtime_provenance import (
    TORCHRUN_RUNTIME_PROVENANCE_PREFIX,
    format_torchrun_runtime_provenance_frames,
)
from core.profiling.ncu_torchrun_capture import (
    APPLICATION_CAPTURE_RECEIPT_SCHEMA,
    CAPTURE_RECEIPT_SCHEMA,
    RANK_APPLICATION_ALL_MATCHING_CAPTURE_RECEIPT_SCHEMA,
    RANK_APPLICATION_CAPTURE_RECEIPT_SCHEMA,
    RANK_APPLICATION_KERNEL_WINDOW_CAPTURE_RECEIPT_SCHEMA,
    _CAPTURE_OWNER_ENV,
    _all_matching_application_rows,
    _application_report_artifact,
    _args,
    _cleanup_capture_processes,
    _drain_owned_process_group,
    _marked_pids,
    _parse_application_replay_log,
    _parse_rank_application_generations,
    _parse_rank_application_log,
    _proc_cleanup_supported,
    _rank_application_evidence,
    _run_all_matching_report_import,
    _snapshot_proc_identities,
    build_capture_plan,
)
from core.profiling.ncu_torchrun_generation import GENERATION_MARKER_PREFIX
from core.profiling.ncu_torchrun_rank import (
    RANK_APPLICATION_CONFIG_SCHEMA,
    RANK_CONFIG_SCHEMA,
    build_rank_command,
    load_rank_config,
)
from core.profiling.profiler_config import MINIMAL_METRICS

CODE_ROOT = Path(__file__).resolve().parents[1]
NCCL_RANGES = ("NCCL@ncclGroupEnd/", "NCCL@ncclAllReduce/")


def test_two_rank_capture_docs_cover_each_explicit_replay_topology() -> None:
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
    assert "--application-topology parent" in section
    assert "--application-topology rank-tcp" in section
    assert "--expected-application-passes" in section
    assert "--application-rdzv-port-base" in section
    assert "--expected-kernel-counts" in section
    assert "--all-matching-kernels" in section
    assert "--kernel-window START COUNT" in section
    assert "NCU_TIMEOUT_SECONDS=1800" in section
    assert 'NCU_REPORT_IMPORT_TIMEOUT_SECONDS=600' in section
    assert '--timeout-seconds "$NCU_TIMEOUT_SECONDS"' in section
    assert (
        '--report-import-timeout-seconds "$NCU_REPORT_IMPORT_TIMEOUT_SECONDS"'
        in section
    )
    assert "--kernel-name \"$KERNEL_NAME\" --launch-skip 0" in section
    assert "TORCHELASTIC_USE_AGENT_STORE=False" in section
    assert "KERNEL_NAME='regex:ncclDevKernel_AllReduce.*'" in section
    assert "LAUNCH_SKIP=2" in section
    assert "AISP_NCU_PROFILE_REQUESTS=4 AISP_NCU_PROFILE_CONTEXT=256" in section
    assert "ch15.optimized_disaggregated_inference_multigpu" in section
    assert "-- --iters 4 --warmup 5 --seed 42" in section
    assert "requested_visible_devices: [0, 1]" in section


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


def _rank_application_plan(
    tmp_path: Path, command: list[str] | None = None, **overrides
):
    command = command or _torchrun_command(tmp_path / "target.py")
    torchrun_port = int(command[command.index("--rdzv_endpoint") + 1].rsplit(":", 1)[1])
    tcp_port = 31_001 if torchrun_port != 31_001 else 31_002
    requested_passes = overrides.get("expected_application_passes", 2)
    port_span = requested_passes if type(requested_passes) is int and requested_passes > 0 else 2
    rdzv_port_base = next(
        base
        for base in (31_100, 31_200)
        if torchrun_port not in range(base, base + port_span)
        and tcp_port not in range(base, base + port_span)
    )
    values = {
        "application_topology": "rank-tcp",
        "tcp_port": tcp_port,
        "nccl_nvtx_includes": (),
        "replay_mode": "application",
        "target_visible_device": None,
        "kernel_name": "regex:ncclDevKernel_AllReduce.*",
        "launch_skip": 22,
        "launch_count": 1,
        "expected_application_passes": 2,
        "application_rdzv_port_base": rdzv_port_base,
    }
    values.update(overrides)
    return _plan(tmp_path, command, **values)


def _all_matching_rank_application_plan(
    tmp_path: Path,
    command: list[str] | None = None,
    **overrides,
):
    values = {
        "all_matching_kernels": True,
        "launch_skip": 0,
        "launch_count": None,
        "expected_application_passes": 9,
        "expected_kernel_counts": (2, 3),
    }
    values.update(overrides)
    return _rank_application_plan(tmp_path, command, **values)


def _kernel_window_rank_application_plan(
    tmp_path: Path,
    command: list[str] | None = None,
    **overrides,
):
    values = {
        "all_matching_kernels": False,
        "launch_skip": None,
        "launch_count": None,
        "expected_application_passes": 9,
        "expected_kernel_counts": (6, 7),
        "kernel_window": (2, 3),
    }
    values.update(overrides)
    return _rank_application_plan(tmp_path, command, **values)


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


def _generation_marker(
    event: str,
    *,
    generation: int,
    master_port: int,
    process_id: int,
    rank: int,
    returncode: int | None = None,
) -> bytes:
    payload = {
        "event": event,
        "generation": generation,
        "master_port": master_port,
        "pid": process_id,
        "rank": rank,
    }
    if returncode is not None:
        payload["returncode"] = returncode
    return (
        GENERATION_MARKER_PREFIX
        + json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")


def _rank_application_generation(
    generation: int,
    *,
    master_port_base: int,
    rank0_bootstrap_pid: int,
    rank1_bootstrap_pid: int,
    rank0_worker_pid: int,
    rank1_worker_pid: int,
    rank0_updates: dict[str, object] | None = None,
    rank1_updates: dict[str, object] | None = None,
) -> bytes:
    master_port = master_port_base + generation
    chunks = [
        _generation_marker(
            "begin",
            generation=generation,
            master_port=master_port,
            process_id=rank0_bootstrap_pid,
            rank=0,
        ),
        _generation_marker(
            "begin",
            generation=generation,
            master_port=master_port,
            process_id=rank1_bootstrap_pid,
            rank=1,
        ),
    ]
    for local_rank, process_id, updates in (
        (0, rank0_worker_pid, rank0_updates or {}),
        (1, rank1_worker_pid, rank1_updates or {}),
    ):
        chunks.extend(
            format_torchrun_runtime_provenance_frames(
                _synthetic_cuda_snapshot(process_id, **updates),
                local_rank=local_rank,
                rank=local_rank,
                world_size=2,
                local_world_size=2,
                pipe_buf=4096,
            )
        )
    chunks.extend(
        [
            _generation_marker(
                "end",
                generation=generation,
                master_port=master_port,
                process_id=rank0_bootstrap_pid,
                rank=0,
                returncode=0,
            ),
            _generation_marker(
                "end",
                generation=generation,
                master_port=master_port,
                process_id=rank1_bootstrap_pid,
                rank=1,
                returncode=0,
            ),
        ]
    )
    return b"".join(chunks)


def _rank_application_log(
    *,
    local_rank: int,
    communicator_rank: int,
    pass_count: int,
    kernel_names: tuple[str, ...] = ("ncclDevKernel_AllReduce_Sum_f32_RING_LL",),
) -> bytes:
    lines = [
        f"==PROF== [Comm rank {communicator_rank}]  Connected to process 1000\n"
    ]
    lines.extend(
        "==PROF== "
        f"[Comm rank {communicator_rank}]  Profiling on device {local_rank} : "
        f'"{kernel_name}": Application replay pass {pass_number}\n'
        for pass_number in range(1, pass_count + 1)
        for kernel_name in kernel_names
    )
    return "".join(lines).encode("utf-8")


def _application_csv(
    *,
    process_id: int,
    pass_count: int = 2,
    device: int = 1,
    kernel_name: str = "ncclDevKernel_SendRecv(int)",
) -> str:
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
        kernel_name,
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


def _all_matching_application_csv(
    *,
    process_id: int,
    row_ids: tuple[int, ...],
    pass_count: int = 9,
    device: int = 0,
    row_updates: dict[int, dict[str, str]] | None = None,
) -> str:
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
    data_rows: list[list[str]] = []
    for data_index, row_id in enumerate(row_ids):
        fields = {
            "ID": str(row_id),
            "Process ID": str(process_id),
            "Device": str(device),
            "device__attribute_device_index": str(device),
            "Kernel Name": f"ncclDevKernel_AllReduce_Sum_f32_RING_LL_{data_index}",
            "profiler__replayer_passes": str(pass_count),
            "gpu__time_duration.avg": "1000",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed": "25",
            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "20",
            "lts__throughput.avg.pct_of_peak_sustained_elapsed": "15",
            "sm__warps_active.avg.pct_of_peak_sustained_active": "50",
        }
        fields.update((row_updates or {}).get(data_index, {}))
        data_rows.append([fields[column] for column in header])
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerows((header, units, *data_rows))
    return stream.getvalue()


def _invalid_all_matching_application_csv(case: str) -> str:
    row_ids = (0, 1, 2)
    row_updates: dict[int, dict[str, str]] = {}
    if case == "missing-row":
        row_ids = (0, 1)
    elif case == "extra-row":
        row_ids = (0, 1, 2, 3)
    elif case == "duplicate-row-id":
        row_ids = (0, 1, 1)
    elif case == "gapped-row-id":
        row_ids = (0, 1, 3)
    elif case == "wrong-process":
        row_updates = {1: {"Process ID": "999"}}
    elif case == "wrong-device":
        row_updates = {1: {"Device": "1"}}
    elif case == "wrong-pass-count":
        row_updates = {1: {"profiler__replayer_passes": "8"}}
    elif case == "nonfinite-metric":
        row_updates = {
            1: {"gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "nan"}
        }
    elif case == "percentage-over-100":
        row_updates = {
            1: {"sm__throughput.avg.pct_of_peak_sustained_elapsed": "101"}
        }
    elif case == "zero-duration":
        row_updates = {1: {"gpu__time_duration.avg": "0"}}
    elif case == "zero-active-warps":
        row_updates = {
            1: {"sm__warps_active.avg.pct_of_peak_sustained_active": "0"}
        }
    else:
        raise AssertionError(f"unsupported all-matching CSV test case: {case}")
    return _all_matching_application_csv(
        process_id=30_000,
        row_ids=row_ids,
        row_updates=row_updates,
    )


def _invalid_application_csv(case: str) -> str:
    rows = list(csv.reader(io.StringIO(_application_csv(process_id=102))))
    header, _, values = rows
    if case == "wrong-device":
        values[header.index("Device")] = "0"
    elif case == "wrong-process":
        values[header.index("Process ID")] = "999"
    elif case == "wrong-pass-count":
        values[header.index("profiler__replayer_passes")] = "3"
    elif case == "missing-metric":
        index = header.index("gpu__time_duration.avg")
        for row in rows:
            del row[index]
    elif case == "nonfinite-metric":
        values[header.index("gpu__time_duration.avg")] = "nan"
    elif case == "zero-metric":
        values[
            header.index("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        ] = "0"
    elif case == "percentage-over-100":
        values[
            header.index("sm__throughput.avg.pct_of_peak_sustained_elapsed")
        ] = "101"
    elif case == "duplicate-header":
        header[header.index("gpu__time_duration.avg")] = "Device"
    elif case == "multiple-data-rows":
        rows.append(list(values))
    elif case != "stale-report":
        raise AssertionError(f"unsupported application CSV test case: {case}")
    stream = io.StringIO()
    csv.writer(stream, lineterminator="\n").writerows(rows)
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


def _make_fake_rank_application_ncu(
    path: Path, *, rank0_csv: str, rank1_csv: str
) -> Path:
    path.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        f"rank_csv = {{0: {rank0_csv!r}, 1: {rank1_csv!r}}}\n"
        "args = sys.argv[1:]\n"
        "if args == ['--version']:\n"
        "    print('Version 2026.2.1.0 (test control plane)')\n"
        "    raise SystemExit(0)\n"
        "report = Path(args[args.index('--import') + 1])\n"
        "rank = int(report.name.removeprefix('rank-').split('.', 1)[0])\n"
        "sys.stdout.write(rank_csv[rank])\n",
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
    assert plan.application_topology == "parent"
    assert plan.expected_application_passes is None
    assert plan.application_rdzv_port_base is None


def test_rank_tcp_application_plan_exposes_exact_rank_commands(tmp_path: Path) -> None:
    command = _torchrun_command(tmp_path / "target.py")
    plan = _rank_application_plan(tmp_path, command)
    expected_prefix = (
        str(tmp_path / "ncu"),
        "--clock-control",
        "none",
        "--target-processes",
        "all",
        "--communicator",
        "tcp",
        "--communicator-tcp-hostname",
        "127.0.0.1",
        "--communicator-tcp-port",
        str(plan.tcp_port),
        "--communicator-tcp-num-peers",
        "2",
        "--lockstep-kernel-launch",
        "--replay-mode",
        "application",
        "--app-replay-mode",
        "strict",
        "--app-replay-match",
        "grid",
        "--filter-mode",
        "per-gpu",
        "--kernel-name",
        "regex:ncclDevKernel_AllReduce.*",
        "--launch-skip",
        "22",
        "--launch-count",
        "1",
        "--kill",
        "no",
        "--check-exit-code",
        "yes",
        "--metrics",
        ",".join(MINIMAL_METRICS),
    )

    assert plan.is_rank_application is True
    assert (
        RANK_APPLICATION_CAPTURE_RECEIPT_SCHEMA
        == "aisp.ncu-torchrun-rank-application-capture.v1"
    )
    assert plan.ncu_prefix == expected_prefix
    assert plan.launch_argv == (
        *plan.launcher_argv,
        "--no-python",
        command[0],
        "-m",
        "core.profiling.ncu_torchrun_rank",
        str(plan.rank_config_path),
    )
    assert plan.rank_config() == {
        "schema": RANK_APPLICATION_CONFIG_SCHEMA,
        "ncu_prefix": list(expected_prefix),
        "worker_argv": [
            command[0],
            "-m",
            "core.profiling.ncu_torchrun_generation",
            str(plan.generation_config_path),
        ],
        "output_dir": str(plan.output_dir),
        "world_size": 2,
    }
    assert plan.generation_config() == {
        "schema": "aisp.ncu-torchrun-generation.v1",
        "worker_argv": list(plan.worker_argv),
        "output_dir": str(plan.output_dir),
        "world_size": 2,
        "expected_passes": 2,
        "master_addr": "127.0.0.1",
        "master_port_base": plan.application_rdzv_port_base,
    }
    for local_rank in (0, 1):
        assert plan.expected_rank_argv(local_rank) == (
            *expected_prefix,
            "--log-file",
            str(plan.rank_log_path(local_rank)),
            "--devices",
            str(local_rank),
            "--export",
            str(plan.report_path(local_rank)),
            command[0],
            "-m",
            "core.profiling.ncu_torchrun_generation",
            str(plan.generation_config_path),
        )
        assert "%q{LOCAL_RANK}" not in " ".join(plan.expected_rank_argv(local_rank))


def test_all_matching_application_control_plane_plan_has_no_launch_cutoff(
    tmp_path: Path,
) -> None:
    plan = _all_matching_rank_application_plan(
        tmp_path,
        report_import_timeout_seconds=600.0,
    )

    assert plan.all_matching_kernels is True
    assert plan.expected_kernel_counts == (2, 3)
    assert plan.launch_skip == 0
    assert plan.launch_count is None
    assert plan.expected_application_passes == 9
    assert plan.report_import_timeout_seconds == 600.0
    assert (
        RANK_APPLICATION_ALL_MATCHING_CAPTURE_RECEIPT_SCHEMA
        == "aisp.ncu-torchrun-rank-application-all-matching-capture.v1"
    )
    assert plan.ncu_prefix[plan.ncu_prefix.index("--launch-skip") + 1] == "0"
    assert "--launch-count" not in plan.ncu_prefix
    assert all(
        "--launch-count" not in plan.expected_rank_argv(local_rank)
        for local_rank in (0, 1)
    )


def test_kernel_window_application_plan_derives_bounded_native_selector(
    tmp_path: Path,
) -> None:
    plan = _kernel_window_rank_application_plan(
        tmp_path,
        report_import_timeout_seconds=600.0,
    )

    assert plan.kernel_window == (2, 3)
    assert plan.expected_kernel_counts == (6, 7)
    assert plan.expected_captured_kernel_counts == (3, 3)
    assert plan.all_matching_kernels is False
    assert plan.launch_skip == 2
    assert plan.launch_count == 3
    assert plan.expected_application_passes == 9
    assert plan.report_import_timeout_seconds == 600.0
    assert (
        RANK_APPLICATION_KERNEL_WINDOW_CAPTURE_RECEIPT_SCHEMA
        == "aisp.ncu-torchrun-rank-application-kernel-window-capture.v1"
    )
    assert plan.ncu_prefix[plan.ncu_prefix.index("--launch-skip") + 1] == "2"
    assert plan.ncu_prefix[plan.ncu_prefix.index("--launch-count") + 1] == "3"
    assert plan.ncu_prefix[plan.ncu_prefix.index("--app-replay-buffer") + 1] == "file"
    assert "--nvtx" in plan.ncu_prefix
    assert "--nvtx-include" not in plan.ncu_prefix
    assert "--nvtx-exclude" not in plan.ncu_prefix
    assert plan.ncu_prefix[plan.ncu_prefix.index("--kill") + 1] == "no"
    for local_rank in (0, 1):
        rank_argv = plan.expected_rank_argv(local_rank)
        assert rank_argv[rank_argv.index("--launch-skip") + 1] == "2"
        assert rank_argv[rank_argv.index("--launch-count") + 1] == "3"


def test_kernel_window_application_cli_parses_start_count_and_full_counts(
    tmp_path: Path,
) -> None:
    args = _args(
        [
            "--label",
            "window-control-plane",
            "--repo-root",
            str(tmp_path),
            "--source",
            "a" * 40,
            "--output-dir",
            str(tmp_path / "capture"),
            "--ncu",
            str(tmp_path / "ncu"),
            "--tcp-port",
            "31001",
            "--timeout-seconds",
            "30",
            "--report-import-timeout-seconds",
            "600",
            "--replay-mode",
            "application",
            "--application-topology",
            "rank-tcp",
            "--expected-application-passes",
            "9",
            "--application-rdzv-port-base",
            "31100",
            "--expected-kernel-counts",
            "481",
            "481",
            "--kernel-window",
            "64",
            "64",
            "--kernel-name",
            "regex:ncclDevKernel.*",
            "--",
            *_torchrun_command(tmp_path / "target.py"),
        ]
    )

    assert args.kernel_window == [64, 64]
    assert args.expected_kernel_counts == [481, 481]
    assert args.launch_skip is None
    assert args.launch_count is None
    assert args.all_matching_kernels is False
    assert args.report_import_timeout_seconds == 600.0


def test_all_matching_application_control_plane_cli_parses_two_rank_counts(
    tmp_path: Path,
) -> None:
    args = _args(
        [
            "--label",
            "all-matching-control-plane",
            "--repo-root",
            str(tmp_path),
            "--source",
            "a" * 40,
            "--output-dir",
            str(tmp_path / "capture"),
            "--ncu",
            str(tmp_path / "ncu"),
            "--tcp-port",
            "31001",
            "--timeout-seconds",
            "30",
            "--report-import-timeout-seconds",
            "600",
            "--all-matching-kernels",
            "--replay-mode",
            "application",
            "--application-topology",
            "rank-tcp",
            "--expected-application-passes",
            "9",
            "--application-rdzv-port-base",
            "31100",
            "--expected-kernel-counts",
            "2",
            "3",
            "--kernel-name",
            "regex:ncclDevKernel.*",
            "--launch-skip",
            "0",
            "--",
            *_torchrun_command(tmp_path / "target.py"),
        ]
    )

    assert args.expected_kernel_counts == [2, 3]
    assert args.launch_count is None
    assert args.all_matching_kernels is True
    assert args.report_import_timeout_seconds == 600.0


def test_report_import_timeout_control_plane_keeps_legacy_default(
    tmp_path: Path,
) -> None:
    plans = (
        _plan(tmp_path),
        _application_plan(tmp_path),
        _rank_application_plan(tmp_path),
        _all_matching_rank_application_plan(tmp_path),
        _kernel_window_rank_application_plan(tmp_path),
    )

    assert all(plan.report_import_timeout_seconds == 60.0 for plan in plans)


def test_report_import_timeout_control_plane_cli_propagates_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CLI wiring only. This does not represent a profiler capture."""

    observed: dict[str, object] = {}
    sentinel_plan = object()

    def fake_build_capture_plan(**kwargs: object) -> object:
        observed.update(kwargs)
        return sentinel_plan

    monkeypatch.setattr(ncu_capture, "build_capture_plan", fake_build_capture_plan)
    monkeypatch.setattr(
        ncu_capture,
        "run_capture",
        lambda _plan: SimpleNamespace(
            status="CAPTURE_ARTIFACTS_PRESENT",
            returncode=0,
            timed_out=False,
            receipt_path=tmp_path / "receipt.json",
        ),
    )
    result = ncu_capture.main(
        [
            "--label",
            "all-matching-control-plane",
            "--repo-root",
            str(tmp_path),
            "--source",
            "a" * 40,
            "--output-dir",
            str(tmp_path / "capture"),
            "--ncu",
            str(tmp_path / "ncu"),
            "--tcp-port",
            "31001",
            "--timeout-seconds",
            "1800",
            "--report-import-timeout-seconds",
            "600",
            "--all-matching-kernels",
            "--replay-mode",
            "application",
            "--application-topology",
            "rank-tcp",
            "--expected-application-passes",
            "9",
            "--application-rdzv-port-base",
            "31100",
            "--expected-kernel-counts",
            "2",
            "3",
            "--kernel-name",
            "regex:ncclDevKernel.*",
            "--launch-skip",
            "0",
            "--",
            *_torchrun_command(tmp_path / "target.py"),
        ]
    )

    assert result == 0
    assert observed["report_import_timeout_seconds"] == 600.0
    assert observed["timeout_seconds"] == 1800.0
    assert observed["expected_kernel_counts"] == (2, 3)


def test_kernel_window_cli_propagates_public_tuple_without_fake_capture_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify CLI wiring only. This does not represent a profiler capture."""

    observed: dict[str, object] = {}
    sentinel_plan = object()

    def fake_build_capture_plan(**kwargs: object) -> object:
        observed.update(kwargs)
        return sentinel_plan

    monkeypatch.setattr(ncu_capture, "build_capture_plan", fake_build_capture_plan)
    monkeypatch.setattr(
        ncu_capture,
        "run_capture",
        lambda plan: SimpleNamespace(
            status="CAPTURE_FAILED",
            returncode=1,
            timed_out=False,
            receipt_path=tmp_path / "receipt.json",
        ),
    )
    result = ncu_capture.main(
        [
            "--label",
            "window-control-plane",
            "--repo-root",
            str(tmp_path),
            "--source",
            "a" * 40,
            "--output-dir",
            str(tmp_path / "capture"),
            "--ncu",
            str(tmp_path / "ncu"),
            "--tcp-port",
            "31001",
            "--timeout-seconds",
            "1800",
            "--report-import-timeout-seconds",
            "600",
            "--replay-mode",
            "application",
            "--application-topology",
            "rank-tcp",
            "--expected-application-passes",
            "9",
            "--application-rdzv-port-base",
            "31100",
            "--expected-kernel-counts",
            "481",
            "481",
            "--kernel-window",
            "64",
            "64",
            "--kernel-name",
            "regex:ncclDevKernel.*",
            "--",
            *_torchrun_command(tmp_path / "target.py"),
        ]
    )

    assert result == 1
    assert observed["kernel_window"] == (64, 64)
    assert observed["expected_kernel_counts"] == (481, 481)
    assert observed["launch_skip"] is None
    assert observed["launch_count"] is None
    assert observed["all_matching_kernels"] is False
    assert observed["report_import_timeout_seconds"] == 600.0


@pytest.mark.parametrize(
    "report_import_timeout_seconds",
    [
        None,
        "600",
        [],
        True,
        False,
        0,
        -1,
        math.nan,
        math.inf,
        -math.inf,
    ],
)
def test_report_import_timeout_control_plane_requires_finite_positive_number(
    tmp_path: Path,
    report_import_timeout_seconds: object,
) -> None:
    with pytest.raises(ValueError, match="report_import_timeout_seconds"):
        _all_matching_rank_application_plan(
            tmp_path,
            report_import_timeout_seconds=report_import_timeout_seconds,
        )


@pytest.mark.parametrize(
    "mode",
    ["kernel", "range", "application-parent", "application-rank-selected"],
)
def test_nondefault_report_import_timeout_rejects_non_multirow_modes(
    tmp_path: Path,
    mode: str,
) -> None:
    message = (
        "nondefault report_import_timeout_seconds requires all-matching or "
        "kernel-window rank-tcp application replay"
    )
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        if mode == "kernel":
            _plan(tmp_path, report_import_timeout_seconds=600.0)
        elif mode == "range":
            _plan(
                tmp_path,
                replay_mode="range",
                all_matching_kernels=True,
                report_import_timeout_seconds=600.0,
            )
        elif mode == "application-parent":
            _application_plan(tmp_path, report_import_timeout_seconds=600.0)
        else:
            _rank_application_plan(
                tmp_path,
                report_import_timeout_seconds=600.0,
            )


def test_nondefault_report_import_timeout_accepts_kernel_window(
    tmp_path: Path,
) -> None:
    plan = _kernel_window_rank_application_plan(
        tmp_path,
        report_import_timeout_seconds=600.0,
    )

    assert plan.report_import_timeout_seconds == 600.0


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
        ({"all_matching_kernels": True}, "requires rank-tcp topology"),
        ({"expected_application_passes": 2}, "rejects expected_application_passes"),
        ({"application_rdzv_port_base": 31_100}, "rejects application_rdzv_port_base"),
    ],
)
def test_application_plan_rejects_missing_or_cross_mode_options(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _application_plan(tmp_path, **overrides)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"tcp_port": None}, "requires tcp_port"),
        ({"target_visible_device": 1}, "rejects target_visible_device"),
        ({"expected_application_passes": None}, "expected_application_passes > 0"),
        ({"expected_application_passes": 0}, "expected_application_passes > 0"),
        ({"application_rdzv_port_base": None}, "ports must stay within"),
        ({"application_rdzv_port_base": 1023}, "ports must stay within"),
        (
            {
                "application_rdzv_port_base": 65_535,
                "expected_application_passes": 2,
            },
            "ports must stay within",
        ),
        ({"nccl_nvtx_includes": NCCL_RANGES}, "rejects NCCL NVTX"),
        ({"all_matching_kernels": True}, "requires launch_skip=0"),
        (
            {"expected_kernel_counts": (1, 1)},
            "selected application replay rejects expected_kernel_counts",
        ),
    ],
)
def test_rank_tcp_application_plan_rejects_missing_or_cross_mode_options(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _rank_application_plan(tmp_path, **overrides)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"expected_kernel_counts": None}, "requires expected_kernel_counts"),
        ({"launch_skip": 1}, "requires launch_skip=0"),
        ({"launch_count": 1}, "rejects launch_count"),
        ({"expected_application_passes": 8}, "requires exactly 9 replay passes"),
    ],
)
def test_all_matching_application_control_plane_rejects_unbounded_contracts(
    tmp_path: Path, overrides: dict[str, object], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _all_matching_rank_application_plan(tmp_path, **overrides)


@pytest.mark.parametrize(
    "expected_kernel_counts",
    [
        [],
        (),
        (1,),
        (1, 2, 3),
        (0, 2),
        (-1, 2),
        (True, 2),
        (1, 2.0),
    ],
)
def test_all_matching_application_control_plane_requires_two_positive_exact_counts(
    tmp_path: Path, expected_kernel_counts: object
) -> None:
    with pytest.raises(ValueError, match="tuple of two positive integers"):
        _all_matching_rank_application_plan(
            tmp_path,
            expected_kernel_counts=expected_kernel_counts,
        )


@pytest.mark.parametrize(
    ("kernel_window", "message"),
    [
        ([], "kernel_window must be a tuple of two integers"),
        ((), "kernel_window must be a tuple of two integers"),
        ((1,), "kernel_window must be a tuple of two integers"),
        ((1, 2, 3), "kernel_window must be a tuple of two integers"),
        ((1.0, 1), "kernel_window start must be a nonnegative integer"),
        ((-1, 1), "kernel_window start must be a nonnegative integer"),
        ((True, 1), "kernel_window start must be a nonnegative integer"),
        ((0, 0), "kernel_window count must be a positive integer"),
        ((0, True), "kernel_window count must be a positive integer"),
        ((0, 1.0), "kernel_window count must be a positive integer"),
        ((0, 65), "kernel_window count must be at most 64"),
    ],
)
def test_kernel_window_requires_strict_bounded_integer_pair(
    tmp_path: Path,
    kernel_window: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _kernel_window_rank_application_plan(
            tmp_path,
            kernel_window=kernel_window,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        (
            {"expected_kernel_counts": None},
            "application kernel-window capture requires expected_kernel_counts",
        ),
        (
            {"launch_skip": 0},
            "application kernel-window capture rejects separately supplied "
            "launch_skip or launch_count",
        ),
        (
            {"launch_count": 1},
            "application kernel-window capture rejects separately supplied "
            "launch_skip or launch_count",
        ),
        (
            {"expected_application_passes": 8},
            "application kernel-window capture requires exactly 9 replay passes",
        ),
        (
            {"kernel_window": (4, 3), "expected_kernel_counts": (6, 7)},
            "kernel_window end 7 exceeds expected kernel count 6 for rank 0",
        ),
        (
            {"kernel_window": (5, 2), "expected_kernel_counts": (7, 6)},
            "kernel_window end 7 exceeds expected kernel count 6 for rank 1",
        ),
        (
            {"all_matching_kernels": True},
            "kernel_window is mutually exclusive with all_matching_kernels=True",
        ),
        (
            {"application_topology": "parent"},
            "application kernel-window capture requires rank-tcp topology",
        ),
    ],
)
def test_kernel_window_rejects_cross_mode_and_incomplete_contracts(
    tmp_path: Path,
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _kernel_window_rank_application_plan(tmp_path, **overrides)


@pytest.mark.parametrize(
    "expected_kernel_counts",
    [
        [],
        (),
        (1,),
        (1, 2, 3),
        (0, 2),
        (-1, 2),
        (True, 2),
        (1, 2.0),
    ],
)
def test_kernel_window_requires_two_positive_full_rank_counts(
    tmp_path: Path,
    expected_kernel_counts: object,
) -> None:
    message = "expected_kernel_counts must be a tuple of two positive integers"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        _kernel_window_rank_application_plan(
            tmp_path,
            expected_kernel_counts=expected_kernel_counts,
        )


def test_non_application_modes_reject_kernel_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="rejects application options"):
        _plan(tmp_path, kernel_window=(0, 1))


def test_rank_tcp_application_plan_rejects_all_port_collisions(tmp_path: Path) -> None:
    command = _torchrun_command(tmp_path / "target.py")
    torchrun_port = int(command[command.index("--rdzv_endpoint") + 1].rsplit(":", 1)[1])

    with pytest.raises(ValueError, match="communicator port must differ"):
        _rank_application_plan(tmp_path, command, tcp_port=torchrun_port)
    with pytest.raises(ValueError, match="generation ports must differ"):
        _rank_application_plan(
            tmp_path,
            command,
            application_rdzv_port_base=torchrun_port,
        )
    plan = _rank_application_plan(tmp_path, command)
    with pytest.raises(ValueError, match="generation ports must differ"):
        _rank_application_plan(
            tmp_path,
            command,
            application_rdzv_port_base=plan.tcp_port,
        )


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


@pytest.mark.parametrize("replay_mode", ["kernel", "range"])
def test_rank_local_modes_reject_rank_application_topology(
    tmp_path: Path, replay_mode: str
) -> None:
    with pytest.raises(ValueError, match="rejects application_topology"):
        _plan(
            tmp_path,
            replay_mode=replay_mode,
            all_matching_kernels=replay_mode == "range",
            application_topology="rank-tcp",
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


def test_kernel_window_keeps_selected_and_full_application_prefixes_stable(
    tmp_path: Path,
) -> None:
    selected = _rank_application_plan(tmp_path)
    full = _all_matching_rank_application_plan(tmp_path)
    window = _kernel_window_rank_application_plan(tmp_path)

    assert selected.kernel_window is None
    assert selected.ncu_prefix[selected.ncu_prefix.index("--launch-skip") + 1] == "22"
    assert selected.ncu_prefix[selected.ncu_prefix.index("--launch-count") + 1] == "1"
    assert "--nvtx" not in selected.ncu_prefix
    assert "--app-replay-buffer" not in selected.ncu_prefix

    assert full.kernel_window is None
    assert full.ncu_prefix[full.ncu_prefix.index("--launch-skip") + 1] == "0"
    assert "--launch-count" not in full.ncu_prefix
    assert "--nvtx" in full.ncu_prefix
    assert "--app-replay-buffer" not in full.ncu_prefix

    assert window.kernel_window == (2, 3)
    assert window.ncu_prefix[window.ncu_prefix.index("--app-replay-buffer") + 1] == "file"


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


@pytest.mark.parametrize("pass_count", [2, 9])
def test_rank_tcp_generations_retain_fresh_processes_and_runtime_parity(
    tmp_path: Path, pass_count: int
) -> None:
    plan = _rank_application_plan(
        tmp_path,
        expected_application_passes=pass_count,
    )
    plan.output_dir.mkdir()
    assert plan.application_rdzv_port_base is not None
    raw = b"".join(
        _rank_application_generation(
            generation,
            master_port_base=plan.application_rdzv_port_base,
            rank0_bootstrap_pid=20_000 + generation * 4,
            rank1_bootstrap_pid=20_001 + generation * 4,
            rank0_worker_pid=30_000 + generation * 4,
            rank1_worker_pid=30_001 + generation * 4,
        )
        for generation in range(pass_count)
    )
    plan.stdout_path.write_bytes(raw)

    generations, stdout_artifact = _parse_rank_application_generations(plan)

    assert [item["generation"] for item in generations] == list(range(pass_count))
    assert [item["master_port"] for item in generations] == [
        plan.application_rdzv_port_base + generation
        for generation in range(pass_count)
    ]
    assert all(item["both_post_main_receipts"] is True for item in generations)
    assert all(item["cuda_runtime_parity"] is True for item in generations)
    assert len(
        {
            process_id
            for item in generations
            for process_id in (
                *item["bootstrap_process_ids"].values(),
                *item["execution_process_ids"].values(),
            )
        }
    ) == pass_count * 4
    assert stdout_artifact["generation_marker_count"] == pass_count * 4
    assert stdout_artifact["generation_pair_count"] == pass_count * 2
    assert stdout_artifact["sha256"] == hashlib.sha256(raw).hexdigest()


def test_rank_tcp_generation_log_rejects_malformed_or_ambiguous_state(
    tmp_path: Path,
) -> None:
    plan = _rank_application_plan(tmp_path)
    plan.output_dir.mkdir()
    assert plan.application_rdzv_port_base is not None
    first = _rank_application_generation(
        0,
        master_port_base=plan.application_rdzv_port_base,
        rank0_bootstrap_pid=201,
        rank1_bootstrap_pid=202,
        rank0_worker_pid=301,
        rank1_worker_pid=302,
    )

    cases = [
        (
            first.replace(
                (GENERATION_MARKER_PREFIX + '{"event":"begin"').encode("utf-8"),
                (GENERATION_MARKER_PREFIX + "{broken").encode("utf-8"),
                1,
            ),
            "Malformed rank application generation JSON",
        ),
        (
            first
            + _rank_application_generation(
                1,
                master_port_base=plan.application_rdzv_port_base,
                rank0_bootstrap_pid=201,
                rank1_bootstrap_pid=204,
                rank0_worker_pid=303,
                rank1_worker_pid=304,
            ),
            "reused generation process IDs",
        ),
        (
            first
            + _rank_application_generation(
                1,
                master_port_base=plan.application_rdzv_port_base,
                rank0_bootstrap_pid=203,
                rank1_bootstrap_pid=204,
                rank0_worker_pid=301,
                rank1_worker_pid=304,
            ),
            "reused worker process IDs",
        ),
        (
            first
            + _rank_application_generation(
                1,
                master_port_base=plan.application_rdzv_port_base,
                rank0_bootstrap_pid=203,
                rank1_bootstrap_pid=204,
                rank0_worker_pid=303,
                rank1_worker_pid=304,
                rank0_updates={"torch_version": "drifted-test-version"},
            ),
            "runtime provenance is incomplete or inconsistent",
        ),
    ]

    for raw, match in cases:
        plan.stdout_path.write_bytes(raw)
        with pytest.raises(RuntimeError, match=match):
            _parse_rank_application_generations(plan)


def test_rank_tcp_logs_accept_real_communicator_prefix_independent_of_local_rank(
    tmp_path: Path,
) -> None:
    plan = _rank_application_plan(tmp_path)
    plan.output_dir.mkdir()
    plan.rank_log_path(0).write_bytes(
        _rank_application_log(local_rank=0, communicator_rank=1, pass_count=2)
    )
    plan.rank_log_path(1).write_bytes(
        _rank_application_log(local_rank=1, communicator_rank=0, pass_count=2)
    )

    rank_logs = [
        _parse_rank_application_log(plan, local_rank) for local_rank in (0, 1)
    ]

    assert [
        [marker["communicator_rank"] for marker in item["markers"]]
        for item in rank_logs
    ] == [[1, 1], [0, 0]]
    assert [
        [marker["visible_device"] for marker in item["markers"]]
        for item in rank_logs
    ] == [[0, 0], [1, 1]]
    assert all(
        marker["kernel_name"] == "ncclDevKernel_AllReduce_Sum_f32_RING_LL"
        for item in rank_logs
        for marker in item["markers"]
    )


@pytest.mark.parametrize(
    ("local_rank", "communicator_rank", "kernel_count"),
    [(0, 1, 2), (1, 0, 3)],
)
def test_all_matching_application_control_plane_log_groups_nine_passes(
    tmp_path: Path,
    local_rank: int,
    communicator_rank: int,
    kernel_count: int,
) -> None:
    plan = _all_matching_rank_application_plan(tmp_path)
    plan.output_dir.mkdir()
    kernel_names = tuple(
        f"ncclDevKernel_AllReduce_Sum_f32_RING_LL_{index}"
        for index in range(kernel_count)
    )
    plan.rank_log_path(local_rank).write_bytes(
        _rank_application_log(
            local_rank=local_rank,
            communicator_rank=communicator_rank,
            pass_count=9,
            kernel_names=kernel_names,
        )
    )

    artifact = _parse_rank_application_log(plan, local_rank)

    assert artifact["marker_policy"] == "one-marker-per-matching-kernel-per-pass"
    assert artifact["markers_per_pass"] == kernel_count
    assert artifact["expected_matching_kernel_count"] == kernel_count
    assert artifact["communicator_rank"] == communicator_rank
    assert artifact["stable_kernel_order"] is True
    assert artifact["kernel_order"] == list(kernel_names)
    assert [marker["application_replay_pass"] for marker in artifact["markers"]] == [
        pass_number
        for pass_number in range(1, 10)
        for _ in range(kernel_count)
    ]


@pytest.mark.parametrize(
    ("local_rank", "communicator_rank"),
    [(0, 1), (1, 0)],
)
def test_kernel_window_log_validates_captured_count_instead_of_full_count(
    tmp_path: Path,
    local_rank: int,
    communicator_rank: int,
) -> None:
    plan = _kernel_window_rank_application_plan(
        tmp_path,
        expected_kernel_counts=(481, 481),
        kernel_window=(64, 3),
    )
    plan.output_dir.mkdir()
    kernel_names = tuple(
        f"ncclDevKernel_AllReduce_Sum_f32_RING_LL_{index}"
        for index in range(3)
    )
    plan.rank_log_path(local_rank).write_bytes(
        _rank_application_log(
            local_rank=local_rank,
            communicator_rank=communicator_rank,
            pass_count=9,
            kernel_names=kernel_names,
        )
    )

    artifact = _parse_rank_application_log(plan, local_rank)

    assert artifact["marker_policy"] == "one-marker-per-matching-kernel-per-pass"
    assert artifact["markers_per_pass"] == 3
    assert artifact["expected_matching_kernel_count"] == 3
    assert artifact["stable_kernel_order"] is True
    assert artifact["kernel_order"] == list(kernel_names)


def test_all_matching_application_control_plane_log_rejects_pass_or_order_drift(
    tmp_path: Path,
) -> None:
    plan = _all_matching_rank_application_plan(tmp_path)
    plan.output_dir.mkdir()
    raw = _rank_application_log(
        local_rank=0,
        communicator_rank=1,
        pass_count=9,
        kernel_names=(
            "ncclDevKernel_AllReduce_Sum_f32_RING_LL_0",
            "ncclDevKernel_AllReduce_Sum_f32_RING_LL_1",
        ),
    )
    wrong_pass_group = raw.replace(
        b"Application replay pass 9",
        b"Application replay pass 8",
        1,
    )
    missing_communicator_rank = raw.replace(
        b"==PROF== [Comm rank 1]  Profiling",
        b"==PROF== Profiling",
        1,
    )
    drifting_communicator_rank = raw.replace(
        b"==PROF== [Comm rank 1]  Profiling",
        b"==PROF== [Comm rank 0]  Profiling",
        1,
    )
    wrong_order_lines = raw.splitlines(keepends=True)
    pass_five_start = 1 + 4 * 2
    wrong_order_lines[pass_five_start], wrong_order_lines[pass_five_start + 1] = (
        wrong_order_lines[pass_five_start + 1],
        wrong_order_lines[pass_five_start],
    )
    wrong_order = b"".join(wrong_order_lines)

    for invalid_log, match in (
        (wrong_pass_group, "replay pass groups differ"),
        (wrong_order, "kernel marker order drifted"),
        (missing_communicator_rank, "missing a communicator rank"),
        (drifting_communicator_rank, "communicator rank changed"),
    ):
        plan.rank_log_path(0).write_bytes(invalid_log)
        with pytest.raises(ValueError, match=match):
            _parse_rank_application_log(plan, 0)


@pytest.mark.parametrize(
    ("raw_log", "local_rank", "match"),
    [
        (
            _rank_application_log(
                local_rank=0, communicator_rank=0, pass_count=1
            ),
            0,
            "replay passes differ",
        ),
        (
            _rank_application_log(
                local_rank=0, communicator_rank=2, pass_count=2
            ),
            0,
            "invalid communicator rank",
        ),
        (
            _rank_application_log(
                local_rank=0, communicator_rank=0, pass_count=2
            ).replace(b"on device 0", b"on device 1"),
            0,
            "selected the wrong visible device",
        ),
    ],
)
def test_rank_tcp_logs_reject_invalid_rank_or_pass_identity(
    tmp_path: Path, raw_log: bytes, local_rank: int, match: str
) -> None:
    plan = _rank_application_plan(tmp_path)
    plan.output_dir.mkdir()
    plan.rank_log_path(local_rank).write_bytes(raw_log)

    with pytest.raises(ValueError, match=match):
        _parse_rank_application_log(plan, local_rank)


def test_rank_tcp_evidence_gates_then_validates_both_native_reports(
    tmp_path: Path,
) -> None:
    pass_count = 9
    kernel_name = (
        "ncclDevKernel_AllReduce_Sum_f32_RING_LL"
        "(ncclDevKernelArgsStorage<4096>)"
    )
    fake_ncu = _make_fake_rank_application_ncu(
        tmp_path / "ncu",
        rank0_csv=_application_csv(
            process_id=30_000,
            pass_count=pass_count,
            device=0,
            kernel_name=kernel_name,
        ),
        rank1_csv=_application_csv(
            process_id=30_001,
            pass_count=pass_count,
            device=1,
            kernel_name=kernel_name,
        ),
    )
    plan = _rank_application_plan(
        tmp_path,
        ncu_path=fake_ncu,
        expected_application_passes=pass_count,
    )
    plan.output_dir.mkdir()
    assert plan.application_rdzv_port_base is not None
    start_ns = time.time_ns()
    plan.stdout_path.write_bytes(
        b"".join(
            _rank_application_generation(
                generation,
                master_port_base=plan.application_rdzv_port_base,
                rank0_bootstrap_pid=20_000 + generation * 4,
                rank1_bootstrap_pid=20_001 + generation * 4,
                rank0_worker_pid=30_000 + generation * 4,
                rank1_worker_pid=30_001 + generation * 4,
            )
            for generation in range(pass_count)
        )
    )
    for local_rank, communicator_rank in ((0, 1), (1, 0)):
        plan.rank_log_path(local_rank).write_bytes(
            _rank_application_log(
                local_rank=local_rank,
                communicator_rank=communicator_rank,
                pass_count=pass_count,
            )
        )
        (plan.output_dir / f"rank-{local_rank}-generation.counter").write_text(
            f"{pass_count}\n", encoding="ascii"
        )
        plan.rank_argv_path(local_rank).write_text(
            json.dumps(list(plan.expected_rank_argv(local_rank))),
            encoding="utf-8",
        )
        plan.report_path(local_rank).write_bytes(
            f"rank {local_rank} native report\n".encode("ascii")
        )
    freshness_paths = [
        path
        for local_rank in (0, 1)
        for path in (
            plan.report_path(local_rank),
            plan.rank_argv_path(local_rank),
            plan.output_dir / f"rank-{local_rank}-generation.counter",
        )
    ]
    fresh_mtime_ns = start_ns + 1_000_000_000
    for path in freshness_paths:
        os.utime(path, ns=(fresh_mtime_ns, fresh_mtime_ns))

    _, blocked_generations, _, blocked_imports, blocked_errors = (
        _rank_application_evidence(
            plan,
            start_ns=start_ns,
            allow_report_import=False,
        )
    )

    assert len(blocked_generations) == pass_count
    assert blocked_imports is False
    assert any("reports were not imported" in error for error in blocked_errors)
    assert not plan.rank_csv_path(0).exists()
    assert not plan.rank_csv_path(1).exists()

    artifacts, generations, _, imports_attempted, errors = _rank_application_evidence(
        plan,
        start_ns=start_ns,
        allow_report_import=True,
    )

    assert errors == []
    assert imports_attempted is True
    assert len(generations) == pass_count
    assert [artifact["local_rank"] for artifact in artifacts] == [0, 1]
    assert [
        artifact["application_report"]["row"]["visible_device"]
        for artifact in artifacts
    ] == [0, 1]
    assert [
        artifact["application_report"]["row"]["process_id"]
        for artifact in artifacts
    ] == [30_000, 30_001]
    assert all(
        artifact["application_report"]["row"]["profiler_pass_count"]
        == pass_count
        for artifact in artifacts
    )
    assert all(
        artifact["application_report"]["row"]["kernel_name"] == kernel_name
        for artifact in artifacts
    )
    assert all(
        len(artifact["application_report"]["row"]["metrics"]) == 5
        and all(
            value > 0
            for value in artifact["application_report"]["row"]["metrics"].values()
        )
        for artifact in artifacts
    )


def test_all_matching_application_control_plane_evidence_validates_exact_rank_rows(
    tmp_path: Path,
) -> None:
    pass_count = 9
    expected_kernel_counts = (2, 3)
    fake_ncu = _make_fake_rank_application_ncu(
        tmp_path / "ncu",
        rank0_csv=_all_matching_application_csv(
            process_id=30_000,
            row_ids=(0, 1),
            device=0,
        ),
        rank1_csv=_all_matching_application_csv(
            process_id=30_001,
            row_ids=(0, 1, 2),
            device=1,
        ),
    )
    plan = _all_matching_rank_application_plan(
        tmp_path,
        ncu_path=fake_ncu,
        expected_kernel_counts=expected_kernel_counts,
        report_import_timeout_seconds=600.0,
    )
    plan.output_dir.mkdir()
    assert plan.application_rdzv_port_base is not None
    start_ns = time.time_ns()
    plan.stdout_path.write_bytes(
        b"".join(
            _rank_application_generation(
                generation,
                master_port_base=plan.application_rdzv_port_base,
                rank0_bootstrap_pid=20_000 + generation * 4,
                rank1_bootstrap_pid=20_001 + generation * 4,
                rank0_worker_pid=30_000 + generation * 4,
                rank1_worker_pid=30_001 + generation * 4,
            )
            for generation in range(pass_count)
        )
    )
    for local_rank, communicator_rank in ((0, 1), (1, 0)):
        kernel_names = tuple(
            f"ncclDevKernel_AllReduce_Sum_f32_RING_LL_{index}"
            for index in range(expected_kernel_counts[local_rank])
        )
        plan.rank_log_path(local_rank).write_bytes(
            _rank_application_log(
                local_rank=local_rank,
                communicator_rank=communicator_rank,
                pass_count=pass_count,
                kernel_names=kernel_names,
            )
        )
        (plan.output_dir / f"rank-{local_rank}-generation.counter").write_text(
            f"{pass_count}\n", encoding="ascii"
        )
        plan.rank_argv_path(local_rank).write_text(
            json.dumps(list(plan.expected_rank_argv(local_rank))),
            encoding="utf-8",
        )
        plan.report_path(local_rank).write_bytes(
            f"rank {local_rank} synthetic control-plane report\n".encode("ascii")
        )
    fresh_mtime_ns = start_ns + 1_000_000_000
    for local_rank in (0, 1):
        for path in (
            plan.report_path(local_rank),
            plan.rank_argv_path(local_rank),
            plan.output_dir / f"rank-{local_rank}-generation.counter",
        ):
            os.utime(path, ns=(fresh_mtime_ns, fresh_mtime_ns))

    artifacts, generations, _, imports_attempted, errors = _rank_application_evidence(
        plan,
        start_ns=start_ns,
        allow_report_import=True,
    )

    assert errors == []
    assert imports_attempted is True
    assert len(generations) == pass_count
    for local_rank, expected_count in enumerate(expected_kernel_counts):
        report = artifacts[local_rank]["application_report"]
        assert "row" not in report
        assert report["expected_row_count"] == expected_count
        assert report["observed_row_count"] == expected_count
        assert report["all_rows_validated"] is True
        report_import = report["report_import"]
        assert report_import["timeout_seconds"] == 600.0
        assert report_import["timed_out"] is False
        assert report_import["stdout"]["partial"] is False
        assert report_import["stderr"]["partial"] is False
        assert report_import["cleanup"] == {
            "mode": "dedicated-process-group",
            "natural": True,
            "forced": False,
            "signals_sent": [],
            "drained": True,
            "errors": [],
        }
        assert [row["id"] for row in report["rows"]] == list(
            range(expected_count)
        )


def test_kernel_window_evidence_validates_exact_captured_rows_and_literal_argv(
    tmp_path: Path,
) -> None:
    pass_count = 9
    captured_count = 3
    fake_ncu = _make_fake_rank_application_ncu(
        tmp_path / "ncu",
        rank0_csv=_all_matching_application_csv(
            process_id=30_000,
            row_ids=(0, 1, 2),
            device=0,
        ),
        rank1_csv=_all_matching_application_csv(
            process_id=30_001,
            row_ids=(0, 1, 2),
            device=1,
        ),
    )
    plan = _kernel_window_rank_application_plan(
        tmp_path,
        ncu_path=fake_ncu,
        expected_kernel_counts=(6, 7),
        kernel_window=(2, captured_count),
        report_import_timeout_seconds=600.0,
    )
    plan.output_dir.mkdir()
    assert plan.application_rdzv_port_base is not None
    start_ns = time.time_ns()
    plan.stdout_path.write_bytes(
        b"".join(
            _rank_application_generation(
                generation,
                master_port_base=plan.application_rdzv_port_base,
                rank0_bootstrap_pid=20_000 + generation * 4,
                rank1_bootstrap_pid=20_001 + generation * 4,
                rank0_worker_pid=30_000 + generation * 4,
                rank1_worker_pid=30_001 + generation * 4,
            )
            for generation in range(pass_count)
        )
    )
    for local_rank, communicator_rank in ((0, 1), (1, 0)):
        kernel_names = tuple(
            f"ncclDevKernel_AllReduce_Sum_f32_RING_LL_{index}"
            for index in range(captured_count)
        )
        plan.rank_log_path(local_rank).write_bytes(
            _rank_application_log(
                local_rank=local_rank,
                communicator_rank=communicator_rank,
                pass_count=pass_count,
                kernel_names=kernel_names,
            )
        )
        (plan.output_dir / f"rank-{local_rank}-generation.counter").write_text(
            f"{pass_count}\n", encoding="ascii"
        )
        plan.rank_argv_path(local_rank).write_text(
            json.dumps(list(plan.expected_rank_argv(local_rank))),
            encoding="utf-8",
        )
        plan.report_path(local_rank).write_bytes(
            f"rank {local_rank} synthetic control-plane report\n".encode("ascii")
        )
    fresh_mtime_ns = start_ns + 1_000_000_000
    for local_rank in (0, 1):
        for path in (
            plan.report_path(local_rank),
            plan.rank_argv_path(local_rank),
            plan.output_dir / f"rank-{local_rank}-generation.counter",
        ):
            os.utime(path, ns=(fresh_mtime_ns, fresh_mtime_ns))

    artifacts, generations, _, imports_attempted, errors = _rank_application_evidence(
        plan,
        start_ns=start_ns,
        allow_report_import=True,
    )

    assert errors == []
    assert imports_attempted is True
    assert len(generations) == pass_count
    for local_rank in (0, 1):
        assert artifacts[local_rank]["observed_argv"] == list(
            plan.expected_rank_argv(local_rank)
        )
        report = artifacts[local_rank]["application_report"]
        assert report["expected_row_count"] == captured_count
        assert report["observed_row_count"] == captured_count
        assert report["all_rows_validated"] is True
        assert report["kernel_window"] == {
            "start": 2,
            "count": captured_count,
            "end_exclusive": 5,
        }
        assert report["requested_global_ordinal_mapping"] == {
            "basis": "kernel_window.start + raw_csv.ID",
            "status": "DECLARED_PENDING_EXTERNAL_NSYS_AUDIT",
            "nsys_reconciled": False,
        }
        assert report["import_argv"] == [
            str(fake_ncu),
            "--import",
            str(plan.report_path(local_rank)),
            "--page",
            "raw",
            "--csv",
        ]
        assert [row["id"] for row in report["rows"]] == [0, 1, 2]
        assert [row["requested_global_ordinal"] for row in report["rows"]] == [
            2,
            3,
            4,
        ]
        assert all(row["profiler_pass_count"] == 9 for row in report["rows"])


def test_kernel_window_launch_failure_receipt_keeps_partial_claims_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inspect the receipt boundary without representing a profiler capture."""

    plan = _kernel_window_rank_application_plan(
        tmp_path,
        expected_kernel_counts=(481, 481),
        kernel_window=(64, 64),
        report_import_timeout_seconds=600.0,
    )

    def refuse_launch(*_args: object, **_kwargs: object) -> None:
        raise OSError("synthetic launch refusal")

    monkeypatch.setattr(ncu_capture, "_preflight", lambda _plan, _env: {})
    monkeypatch.setattr(ncu_capture, "_proc_cleanup_supported", lambda: False)
    monkeypatch.setattr(ncu_capture.subprocess, "Popen", refuse_launch)

    with pytest.raises(OSError, match="synthetic launch refusal"):
        ncu_capture.run_capture(plan, environment={})

    receipt = json.loads(plan.receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == RANK_APPLICATION_KERNEL_WINDOW_CAPTURE_RECEIPT_SCHEMA
    assert receipt["status"] == "LAUNCH_FAILED"
    assert receipt["error"] == "OSError: synthetic launch refusal"
    assert receipt["cleanup"]["mode"] == "not-started"
    assert receipt["cleanup"]["forced"] is False
    assert receipt["cleanup"]["drained"] is True
    assert receipt["capture_scope"] == "kernel-window-requested"
    assert receipt["all_matching_kernels"] is False
    assert receipt["nvtx_capture_enabled"] is True
    assert receipt["kernel_window"] == {
        "start": 64,
        "count": 64,
        "end_exclusive": 128,
    }
    assert receipt["expected_kernel_counts"] == [481, 481]
    assert receipt["captured_expected_kernel_counts"] == [64, 64]
    assert receipt["report_import_timeout_seconds"] == 600.0
    assert receipt["ncu_launch_count_argument_present"] is True
    assert receipt["window_ncu_rows_validated"] is False
    assert receipt["requested_global_ordinal_mapping"] == {
        "basis": "kernel_window.start + raw_csv.ID",
        "status": "DECLARED_PENDING_EXTERNAL_NSYS_AUDIT",
        "nsys_reconciled": False,
    }
    assert receipt["application_replay"]["buffer"] == "file"
    assert receipt["all_matching_launches_covered"] is False
    assert receipt["full_coverage_qualified"] is False
    assert receipt["numerical_output_qualified"] is False
    assert receipt["performance_qualified"] is False


# These synthetic CSV cases exercise the all-matching control-plane parser.
# They do not represent a GPU execution or a successful Nsight Compute capture.
def test_all_matching_application_control_plane_accepts_bounded_zero_metrics(
    tmp_path: Path,
) -> None:
    plan = _all_matching_rank_application_plan(
        tmp_path,
        expected_kernel_counts=(3, 2),
    )
    rows = list(
        csv.reader(
            io.StringIO(
                _all_matching_application_csv(
                    process_id=30_000,
                    row_ids=(0, 1, 2),
                    row_updates={
                        0: {
                            "sm__throughput.avg.pct_of_peak_sustained_elapsed": "0"
                        },
                        1: {
                            "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "0"
                        },
                        2: {
                            "lts__throughput.avg.pct_of_peak_sustained_elapsed": "0"
                        },
                    },
                )
            )
        )
    )

    artifact, errors = _all_matching_application_rows(
        plan,
        rows,
        expected_device=0,
        expected_process_id=30_000,
        expected_pass_count=9,
        pass_count_source="the generation ledger",
    )

    assert errors == []
    assert artifact["expected_row_count"] == 3
    assert artifact["observed_row_count"] == 3
    assert artifact["all_rows_validated"] is True
    assert "row" not in artifact
    assert [row["id"] for row in artifact["rows"]] == [0, 1, 2]
    assert artifact["rows_with_zero_metrics"] == [0, 1, 2]
    assert artifact["zero_metric_counts"] == {
        "gpu__time_duration.avg": 0,
        "sm__throughput.avg.pct_of_peak_sustained_elapsed": 1,
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": 1,
        "lts__throughput.avg.pct_of_peak_sustained_elapsed": 1,
        "sm__warps_active.avg.pct_of_peak_sustained_active": 0,
    }
    assert all(
        row["metrics"]["gpu__time_duration.avg"] > 0
        and row["metrics"]["sm__warps_active.avg.pct_of_peak_sustained_active"]
        > 0
        for row in artifact["rows"]
    )


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing-row", "row count differs"),
        ("extra-row", "row count differs"),
        ("duplicate-row-id", "IDs contain duplicates"),
        ("gapped-row-id", "IDs are not contiguous"),
        ("wrong-process", "Process ID does not match"),
        ("wrong-device", "Device differs"),
        ("wrong-pass-count", "profiler pass count differs"),
        ("nonfinite-metric", "metric is not finite and nonnegative"),
        ("percentage-over-100", "percentage metric exceeds 100"),
        ("zero-duration", "metric must be positive"),
        ("zero-active-warps", "metric must be positive"),
    ],
)
def test_all_matching_application_control_plane_rejects_invalid_rows(
    tmp_path: Path, case: str, match: str
) -> None:
    plan = _all_matching_rank_application_plan(
        tmp_path,
        expected_kernel_counts=(3, 2),
    )
    rows = list(
        csv.reader(io.StringIO(_invalid_all_matching_application_csv(case)))
    )

    artifact, errors = _all_matching_application_rows(
        plan,
        rows,
        expected_device=0,
        expected_process_id=30_000,
        expected_pass_count=9,
        pass_count_source="the generation ledger",
    )

    assert any(match in error for error in errors)
    assert artifact.get("all_rows_validated") is not True


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("missing-row", "row count differs"),
        ("extra-row", "row count differs"),
        ("duplicate-row-id", "IDs contain duplicates"),
        ("gapped-row-id", "IDs are not contiguous"),
        ("wrong-process", "Process ID does not match"),
        ("wrong-device", "Device differs"),
        ("wrong-pass-count", "profiler pass count differs"),
        ("nonfinite-metric", "metric is not finite and nonnegative"),
        ("percentage-over-100", "percentage metric exceeds 100"),
        ("zero-duration", "metric must be positive"),
        ("zero-active-warps", "metric must be positive"),
    ],
)
def test_kernel_window_rows_reuse_strict_multirow_validation(
    tmp_path: Path,
    case: str,
    match: str,
) -> None:
    plan = _kernel_window_rank_application_plan(
        tmp_path,
        expected_kernel_counts=(6, 7),
        kernel_window=(2, 3),
    )
    rows = list(
        csv.reader(io.StringIO(_invalid_all_matching_application_csv(case)))
    )

    artifact, errors = _all_matching_application_rows(
        plan,
        rows,
        expected_device=0,
        expected_process_id=30_000,
        expected_pass_count=9,
        pass_count_source="the generation ledger",
    )

    assert any(match in error for error in errors)
    assert artifact.get("all_rows_validated") is not True


def test_application_report_import_validates_one_selected_row(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_ncu = _make_fake_application_ncu(
        tmp_path / "ncu",
        combined_log=b"",
        raw_csv=_application_csv(process_id=102),
    )
    plan = _application_plan(
        tmp_path,
        ncu_path=fake_ncu,
        timeout_seconds=120.0,
    )
    observed_timeouts: list[object] = []
    real_subprocess_run = subprocess.run

    def record_selected_import(*args: object, **kwargs: object):
        observed_timeouts.append(kwargs.get("timeout"))
        return real_subprocess_run(*args, **kwargs)

    def reject_all_matching_import(*args: object, **kwargs: object) -> None:
        raise AssertionError("selected import used the all-matching subprocess path")

    monkeypatch.setattr(ncu_capture.subprocess, "run", record_selected_import)
    monkeypatch.setattr(
        ncu_capture,
        "_run_all_matching_report_import",
        reject_all_matching_import,
    )
    plan.output_dir.mkdir()
    plan.combined_path.write_bytes(
        _application_generation(1, rank0_pid=101, rank1_pid=102)
        + _application_generation(2, rank0_pid=103, rank1_pid=104)
    )
    generations, _ = _parse_application_replay_log(plan)
    plan.application_report_path.write_bytes(b"fresh report\n")
    start_ns = plan.application_report_path.stat().st_mtime_ns

    artifact, errors = _application_report_artifact(
        plan,
        start_ns=start_ns,
        generations=generations,
    )

    assert errors == []
    assert plan.report_import_timeout_seconds == 60.0
    assert observed_timeouts == [60.0]
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


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("wrong-device", "selected the wrong visible device"),
        ("wrong-process", "process does not match"),
        ("wrong-pass-count", "pass count differs"),
        ("missing-metric", "missing fields"),
        ("nonfinite-metric", "not finite and positive"),
        ("zero-metric", "not finite and positive"),
        ("percentage-over-100", "percentage metric exceeds 100"),
        ("duplicate-header", "header is empty or contains duplicates"),
        ("multiple-data-rows", "must contain one header"),
        ("stale-report", "empty or stale"),
    ],
)
def test_application_report_import_rejects_invalid_artifacts(
    tmp_path: Path, case: str, match: str
) -> None:
    fake_ncu = _make_fake_application_ncu(
        tmp_path / "ncu",
        combined_log=b"",
        raw_csv=_invalid_application_csv(case),
    )
    plan = _application_plan(tmp_path, ncu_path=fake_ncu)
    plan.output_dir.mkdir()
    plan.combined_path.write_bytes(
        _application_generation(1, rank0_pid=101, rank1_pid=102)
        + _application_generation(2, rank0_pid=103, rank1_pid=104)
    )
    generations, _ = _parse_application_replay_log(plan)
    plan.application_report_path.write_bytes(b"report\n")
    report_mtime_ns = plan.application_report_path.stat().st_mtime_ns
    start_ns = report_mtime_ns + 1 if case == "stale-report" else report_mtime_ns

    _, errors = _application_report_artifact(
        plan,
        start_ns=start_ns,
        generations=generations,
    )

    assert any(match in error for error in errors)


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


def test_all_matching_report_import_timeout_preserves_output_and_drains_group(
    tmp_path: Path,
) -> None:
    """Exercise the CPU subprocess control plane without representing NCU output."""

    child_pid_path = tmp_path / "import-child.pid"
    child_code = (
        "import os,signal,time\n"
        "from pathlib import Path\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"Path({str(child_pid_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(60)\n"
    )
    importer_code = (
        "import subprocess,sys,time\n"
        "from pathlib import Path\n"
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}])\n"
        f"child_pid_path = Path({str(child_pid_path)!r})\n"
        "deadline = time.monotonic() + 10\n"
        "while not child_pid_path.exists() and time.monotonic() < deadline:\n"
        "    time.sleep(0.01)\n"
        "sys.stdout.buffer.write(b'partial import stdout\\n')\n"
        "sys.stdout.buffer.flush()\n"
        "sys.stderr.buffer.write(b'partial import stderr\\n')\n"
        "sys.stderr.buffer.flush()\n"
        "time.sleep(60)\n"
    )
    child_pid = -1
    try:
        result = _run_all_matching_report_import(
            [sys.executable, "-c", importer_code],
            timeout_seconds=0.5,
        )
        assert child_pid_path.is_file()
        child_pid = int(child_pid_path.read_text())

        assert result.started is True
        assert result.process_id is not None
        assert result.timed_out is True
        assert result.start_error is None
        assert result.communication_error is None
        assert b"partial import stdout" in result.stdout
        assert b"partial import stderr" in result.stderr
        assert result.signals_sent == ("SIGTERM", "SIGKILL")
        assert result.cleanup_drained is True
        assert result.cleanup_errors == ()
        with pytest.raises(ProcessLookupError):
            os.kill(child_pid, 0)
    finally:
        if child_pid > 0:
            with suppress(ProcessLookupError):
                os.kill(child_pid, signal.SIGKILL)


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
