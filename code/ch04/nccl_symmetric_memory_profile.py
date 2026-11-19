"""Launch one Nsight instance per MPI rank and verify its native report."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch04.nccl_symmetric_memory_build_receipt import file_sha256  # noqa: E402

PROFILE_SCHEMA = "nccl_symmetric_memory.profile.v1"
# nvtx_utils.cuh classifies the native "allreduce:..." label as a reduction.
STEADY_RANGE = "reduce:allreduce:nccl_symmetric_memory_steady/"
NCU_FLAGS = (
    "--communicator",
    "--communicator-tcp-hostname",
    "--communicator-tcp-port",
    "--lockstep-kernel-launch",
    "--lockstep-nvtx-include",
    "--replay-mode",
    "--nvtx-include",
)


class ProfilerUnavailableError(RuntimeError):
    """The default profiler lacks a capability this capture needs."""


def _write_json(path: Path, content: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(content, stream, indent=2)
        stream.write("\n")


def artifact(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Expected a regular profile artifact at {path}")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def prepare_profile(
    directory: Path,
    name: str,
    ranks: int,
    compile_command: list[str],
    native_argv: list[str],
) -> tuple[list[str], dict[str, Any]]:
    """Create a fresh capture directory and probe the tool selected by PATH."""
    if name not in {"ncu", "nsys"} or ranks < 2:
        raise ValueError("Choose ncu or nsys and at least two MPI ranks")
    directory.mkdir(exist_ok=False)
    tool = shutil.which(name)
    if tool is None:
        raise ProfilerUnavailableError(f"SKIPPED: {name} is unavailable on PATH")
    tool = os.path.abspath(tool)
    probes = {}
    help_text = ""
    for option in ["--version", "--help"] if name == "ncu" else ["--version"]:
        command = [tool, option]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        output = (completed.stdout + "\n" + completed.stderr).strip()
        log = directory / f"{name}.{option[2:]}.log"
        log.write_text(output + "\n", encoding="utf-8")
        probes[option] = {"command": command, "returncode": completed.returncode, **artifact(log)}
        if completed.returncode or not output:
            raise ProfilerUnavailableError(f"SKIPPED: {tool} {option} failed. See {log}")
        if option == "--version":
            version = output
        else:
            help_text = output
    config = {
        "name": name,
        "tool": tool,
        "tool_sha256": file_sha256(Path(tool)),
        "version": version,
        "probes": probes,
        "ranks": ranks,
        "native_argv": native_argv,
    }
    if name == "ncu":
        flags = set(re.findall(r"--[a-z][a-z0-9-]*", help_text))
        # NVIDIA renamed the peer count when it introduced the shmem communicator.
        peer_flag = next(
            (f for f in ("--communicator-tcp-num-peers", "--communicator-num-peers") if f in flags),
            None,
        )
        missing = [flag for flag in NCU_FLAGS if flag not in flags]
        if peer_flag is None:
            missing.append("TCP communicator peer count")
        if not re.search(r"(?<![\w-])range(?![\w-])", help_text):
            missing.append("range replay")
        if missing:
            _write_json(directory / "preflight.json", config | {"missing": missing})
            raise ProfilerUnavailableError(
                f"SKIPPED: {version} lacks required coordinated NCCL range profiling controls: "
                f"{', '.join(missing)}. See {directory}"
            )
        if "-DENABLE_NVTX_PROFILING" not in compile_command:
            raise ProfilerUnavailableError(
                "SKIPPED: NCU needs the steady NVTX range. Rebuild with NVTX_ENABLED=1"
            )
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
        config.update(peer_flag=peer_flag, port=port)
    config_file = directory / "capture.json"
    _write_json(config_file, config)
    wrapper = Path(__file__).resolve()
    return [sys.executable, str(wrapper), str(config_file)], config


def mpi_rank(ranks: int, environment: dict[str, str]) -> int:
    """Use the MPI job's global rank and size, even inside a Slurm allocation."""
    participants = []
    for rank_var, size_var in (
        ("OMPI_COMM_WORLD_RANK", "OMPI_COMM_WORLD_SIZE"),
        ("PMI_RANK", "PMI_SIZE"),
        ("PMIX_RANK", "PMIX_JOB_SIZE"),
    ):
        if rank_var in environment and size_var in environment:
            participants.append((int(environment[rank_var]), int(environment[size_var])))
    if not participants and "SLURM_PROCID" in environment and "SLURM_NTASKS" in environment:
        participants.append((int(environment["SLURM_PROCID"]), int(environment["SLURM_NTASKS"])))
    if len(set(participants)) != 1:
        raise RuntimeError(f"Missing or inconsistent global MPI rank and size: {participants}")
    rank, size = participants[0]
    if size != ranks or ranks < 2 or not 0 <= rank < ranks:
        raise RuntimeError(f"Expected {ranks} MPI ranks, received rank {rank} of {size}")
    return rank


def rank_command(config: dict[str, Any], report: Path) -> list[str]:
    if config["name"] == "nsys":
        options = [
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--output",
            str(report),
        ]
    else:
        options = [
            "--config-file",
            "off",
            "--clock-control",
            "none",
            "--communicator",
            "tcp",
            "--communicator-tcp-hostname",
            "127.0.0.1",
            "--communicator-tcp-port",
            str(config["port"]),
            config["peer_flag"],
            str(config["ranks"]),
            "--lockstep-kernel-launch",
            "--lockstep-nvtx-include",
            STEADY_RANGE,
            "--replay-mode",
            "range",
            "--nvtx",
            "--nvtx-include",
            STEADY_RANGE,
            "--metrics",
            "gpu__time_duration.sum",
            "--check-exit-code",
            "yes",
            "--export",
            str(report),
        ]
    return [config["tool"], *options, *config["native_argv"]]


def execute_rank(config_file: Path) -> None:
    config = json.loads(config_file.read_text(encoding="utf-8"))
    rank = mpi_rank(config["ranks"], dict(os.environ))
    directory = config_file.parent / f"rank-{rank}"
    directory.mkdir(exist_ok=False)
    command = rank_command(config, directory / f"capture.{config['name']}-rep")
    _write_json(directory / "command.json", command)
    with (
        (directory / "stdout.log").open("xb") as stdout,
        (directory / "stderr.log").open("xb") as stderr,
    ):
        os.dup2(stdout.fileno(), 1)
        os.dup2(stderr.fileno(), 2)
    if os.environ.get("LD_PRELOAD"):
        raise RuntimeError("NCCL runtime binding: LD_PRELOAD must be empty before rank profiling")
    # Keep the profiler inside the existing MPI-owned process tree on interruption.
    os.execv(config["tool"], command)


def collect_profile(directory: Path, config: dict[str, Any]) -> dict[str, Any]:
    expected = [
        directory / f"rank-{rank}" / f"capture.{config['name']}-rep"
        for rank in range(config["ranks"])
    ]
    actual = set(directory.rglob("*.nsys-rep")) | set(directory.rglob("*.ncu-rep"))
    if actual != set(expected):
        raise RuntimeError(
            f"Missing or ambiguous per-rank reports: expected {expected}, found {sorted(actual)}"
        )
    ranks = []
    report_ids = set()
    for rank, path in enumerate(expected):
        if path.parent.is_symlink():
            raise RuntimeError(f"Expected an owned directory for rank {rank}")
        identity = path.stat().st_dev, path.stat().st_ino
        if identity in report_ids:
            raise RuntimeError(f"Multiple ranks refer to the same native profiler report: {path}")
        report_ids.add(identity)
        command_path = path.parent / "command.json"
        command = json.loads(command_path.read_text(encoding="utf-8"))
        if command != rank_command(config, path):
            raise RuntimeError(f"Profiler command differs from the planned rank {rank} command")
        report = artifact(path)
        if not report["bytes"]:
            raise RuntimeError(f"Empty native profiler report: {path}")
        ranks.append(
            {
                "rank": rank,
                "command": command,
                "command_file": artifact(command_path),
                "report": report,
                "stdout": artifact(path.parent / "stdout.log"),
                "stderr": artifact(path.parent / "stderr.log"),
            }
        )
    if file_sha256(Path(config["tool"])) != config["tool_sha256"]:
        raise RuntimeError("Default profiler executable changed during capture")
    return {
        "tool": config["tool"],
        "version": config["version"],
        "tool_sha256": config["tool_sha256"],
        "config": artifact(directory / "capture.json"),
        "probes": config["probes"],
        "wrapper": artifact(Path(__file__).resolve()),
        "ranks": ranks,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config_file", type=Path)
    execute_rank(parser.parse_args().config_file)
