"""Exercise MPI profiler orchestration without treating test files as GPU evidence."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from ch04 import nccl_symmetric_memory_profile as capture
from ch04 import run_nccl_symmetric_memory as runner


def fake_profiler(path: Path, *, help_text: str = "") -> None:
    path.write_text(
        f"#!{sys.executable}\n"
        "import pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if args == ['--version']:\n"
        "    print('control-plane test tool')\n"
        "elif args == ['--help']:\n"
        f"    print({help_text!r})\n"
        "else:\n"
        "    flag = '--export' if '--export' in args else '--output'\n"
        "    pathlib.Path(args[args.index(flag) + 1]).write_text('control-plane fixture')\n"
        "    print('test profiler stdout')\n"
        "    print('test profiler stderr', file=sys.stderr)\n"
    )
    path.chmod(0o755)


@pytest.mark.parametrize("name", ["nsys", "ncu"])
def test_mpi_profile_rank_receipts_and_freshness(tmp_path, monkeypatch, name):
    tool = tmp_path / name
    help_text = " ".join([*capture.NCU_FLAGS, "--communicator-tcp-num-peers", "kernel range"])
    fake_profiler(tool, help_text=help_text)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    directory = tmp_path / "diagnostics"
    wrapper, config = capture.prepare_profile(
        directory,
        name,
        2,
        ["nvcc", "-DENABLE_NVTX_PROFILING"],
        ["/bin/true"],
    )
    for rank in range(2):
        environment = os.environ | {"OMPI_COMM_WORLD_RANK": str(rank), "OMPI_COMM_WORLD_SIZE": "2"}
        subprocess.run(wrapper, env=environment, check=True)
    summary = capture.collect_profile(directory, config)
    reports = summary["ranks"]
    assert [r["rank"] for r in reports] == [0, 1]
    assert len({r["report"]["path"] for r in reports}) == 2
    assert summary["version"] == "control-plane test tool"
    assert summary["tool_sha256"] == capture.file_sha256(tool)
    for rank in reports:
        command = rank["command"]
        assert command[0] == str(tool)
        assert command[-1] == "/bin/true"
        assert rank["report"]["path"] in command
        for field in ("report", "stdout", "stderr", "command_file"):
            assert rank[field]["sha256"] == capture.file_sha256(Path(rank[field]["path"]))
        if name == "ncu":
            assert command[command.index("--replay-mode") + 1] == "range"
            assert command[command.index("--communicator") + 1] == "tcp"
            assert command[command.index("--communicator-tcp-num-peers") + 1] == "2"
            assert command[command.index("--nvtx-include") + 1] == capture.STEADY_RANGE
            assert command[command.index("--lockstep-nvtx-include") + 1] == capture.STEADY_RANGE
            assert "--lockstep-kernel-launch" in command
        else:
            assert "profile" in command and "--trace=cuda,nvtx" in command
    duplicate = subprocess.run(wrapper, env=environment, capture_output=True)
    assert duplicate.returncode != 0
    with pytest.raises(FileExistsError):
        capture.prepare_profile(directory, name, 2, [], [])
    extra = directory / f"unexpected.{name}-rep"
    extra.touch()
    with pytest.raises(RuntimeError, match="ambiguous"):
        capture.collect_profile(directory, config)
    extra.unlink()
    last = Path(reports[-1]["report"]["path"])
    last.write_bytes(b"")
    with pytest.raises(RuntimeError, match="Empty native"):
        capture.collect_profile(directory, config)
    last.unlink()
    with pytest.raises(RuntimeError, match="Missing"):
        capture.collect_profile(directory, config)


def test_rank_rejects_preload_injected_by_launcher(tmp_path, monkeypatch):
    fake_profiler(tmp_path / "nsys")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    directory = tmp_path / "mpi"
    wrapper, _ = capture.prepare_profile(directory, "nsys", 2, [], ["/bin/true"])
    environment = os.environ | {
        "OMPI_COMM_WORLD_RANK": "0",
        "OMPI_COMM_WORLD_SIZE": "2",
        "LD_PRELOAD": "/untrusted.so",
    }
    assert subprocess.run(wrapper, env=environment, check=False).returncode != 0
    assert "LD_PRELOAD must be empty" in (directory / "rank-0" / "stderr.log").read_text()
    assert not list(directory.rglob("*.nsys-rep"))


def test_default_ncu_without_lockstep_skips_before_launch(tmp_path, monkeypatch):
    fake_profiler(tmp_path / "ncu", help_text="--replay-mode kernel range app-range")
    monkeypatch.setenv("PATH", str(tmp_path))
    directory = tmp_path / "ncu-preflight"
    with pytest.raises(capture.ProfilerUnavailableError, match="--lockstep-kernel-launch"):
        capture.prepare_profile(directory, "ncu", 2, [], ["/bin/true"])
    receipt = json.loads((directory / "preflight.json").read_text())
    assert receipt["probes"]["--version"]["command"] == [str(tmp_path / "ncu"), "--version"]
    assert receipt["probes"]["--help"]["sha256"] == capture.file_sha256(directory / "ncu.help.log")
    assert not list(directory.glob("rank-*"))


@pytest.mark.parametrize(
    "rank_env",
    [
        {},
        {"PMI_RANK": "0", "PMI_SIZE": "1"},
        {"PMI_RANK": "2", "PMI_SIZE": "2"},
        {
            "PMI_RANK": "1",
            "PMI_SIZE": "2",
            "OMPI_COMM_WORLD_RANK": "0",
            "OMPI_COMM_WORLD_SIZE": "2",
        },
    ],
)
def test_rank_rejects_missing_conflicting_or_single_rank_identity(rank_env):
    with pytest.raises(RuntimeError):
        capture.mpi_rank(2, rank_env)


@pytest.mark.skipif(shutil.which("mpirun") is None, reason="local MPI launcher unavailable")
def test_mpirun_launches_one_control_plane_process_per_rank(tmp_path, monkeypatch):
    fake_profiler(tmp_path / "nsys")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    directory = tmp_path / "mpi"
    wrapper, config = capture.prepare_profile(directory, "nsys", 2, [], ["/bin/true"])
    subprocess.run(["mpirun", "-np", "2", *wrapper], check=True)
    assert [row["rank"] for row in capture.collect_profile(directory, config)["ranks"]] == [0, 1]


def test_runner_defaults_to_timing_and_labels_skipped_profile(tmp_path):
    assert runner.build_parser().parse_args([]).profile is None
    receipt = tmp_path / "profile.json"
    rc = runner.main(
        [
            "--profile",
            "nsys",
            "--ranks",
            "2",
            "--binary",
            str(tmp_path / "missing"),
            "--json-out",
            str(receipt),
        ]
    )
    assert rc == 77
    result = json.loads(receipt.read_text())
    assert result["schema"] == capture.PROFILE_SCHEMA and result["status"] == "SKIPPED"
    assert "samples_ms" not in result


@pytest.mark.parametrize("name", [None, "nsys"])
def test_direct_invocation_places_profiler_inside_mpi(tmp_path, monkeypatch, name):
    binary = tmp_path / "native"
    binary.touch()
    fake_profiler(tmp_path / "nsys")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(runner, "validate_build_receipt", lambda _: {"compile_command": ["nvcc"]})
    monkeypatch.setattr(runner, "child_runtime_environment", lambda _: (dict(os.environ), []))
    monkeypatch.setattr(runner, "verify_loader_binding", lambda *args: tmp_path / "libnccl.so.2")
    commands = []

    def capture_failed_launch(command, **kwargs):
        commands.append(command)
        return {"returncode": 1, "stdout": "", "stderr": "control-plane stop before launch"}

    monkeypatch.setattr(runner, "run_recorded_cooperative_process", capture_failed_launch)
    receipt = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="Native MPI run failed"):
        runner.invoke_native(
            runner.NativeWorkload("cuda_malloc", ranks=2),
            binary=binary,
            json_out=receipt,
            dump_output=tmp_path / "output.f32",
            launcher=sys.executable,
            profile=name,
        )
    (command,) = commands
    assert command[1:3] == ["-np", "2"]
    if name is None:
        assert command[3] == str(binary)
        assert "--allow-profiler-preload" not in command
    else:
        assert command[3:5] == [sys.executable, str(Path(capture.__file__).resolve())]
        config = json.loads(Path(command[5]).read_text())
        assert config["native_argv"][0] == str(binary)
        assert str(tmp_path / "result.json.nsys" / "profiled.native.json") in config["native_argv"]
        assert config["native_argv"][-1] == "--allow-profiler-preload"
    assert not receipt.exists()
