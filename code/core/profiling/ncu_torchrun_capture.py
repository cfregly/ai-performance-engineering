"""Explicit two-rank Nsight Compute capture for torchrun.

Kernel and range replay use the existing per-rank TCP/lockstep topology.
Application replay either profiles one caller-selected visible device around
the full torchrun command or runs one coordinated NCU process per rank. The
rank-local path can capture one selected launch or explicitly request every
matching launch. All modes use NCU 2026.2.1 and the repository's five minimal
metrics. Receipts prove source and argument custody plus bounded artifact checks.
They do not qualify numerical output, complete coverage, or performance. Full
process-tree cleanup proof requires Linux procfs and the inherited owner marker.
Unsupported hosts can run only to an explicitly rejected receipt.
"""

from __future__ import annotations

import argparse
import csv
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
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.benchmark.run_manifest import (
    RunManifest,
    RuntimeProvenanceParityError,
    require_runtime_provenance_parity,
)
from core.harness.torchrun_runtime_provenance import (
    TORCHRUN_RUNTIME_PROVENANCE_PREFIX,
    TorchrunRuntimeProvenanceError,
    parse_torchrun_runtime_provenance_stdout,
)
from core.profiling.ncu_torchrun_generation import (
    GENERATION_CONFIG_SCHEMA,
    GENERATION_MARKER_PREFIX,
)
from core.profiling.ncu_torchrun_rank import (
    RANK_APPLICATION_CONFIG_SCHEMA,
    RANK_CONFIG_SCHEMA,
)
from core.profiling.profiler_config import MINIMAL_METRICS

CAPTURE_RECEIPT_SCHEMA = "aisp.ncu-torchrun-capture.v1"
APPLICATION_CAPTURE_RECEIPT_SCHEMA = "aisp.ncu-torchrun-application-capture.v1"
RANK_APPLICATION_CAPTURE_RECEIPT_SCHEMA = (
    "aisp.ncu-torchrun-rank-application-capture.v1"
)
RANK_APPLICATION_ALL_MATCHING_CAPTURE_RECEIPT_SCHEMA = (
    "aisp.ncu-torchrun-rank-application-all-matching-capture.v1"
)
SUPPORTED_NCU_VERSION = "2026.2.1"
WORLD_SIZE = 2
CLAIM_LIMIT = (
    "Artifact presence, hashes, and a clean process return do not validate counter values, "
    "kernel/range attribution, workload correctness, or performance."
)
CLEANUP_LIMIT = (
    "Full-tree drainage is proven only on Linux procfs for processes retaining this "
    "launch's inherited owner marker. Clearing that marker is unsupported."
)
_NCCL_RANGE = re.compile(r"^NCCL@[A-Za-z][A-Za-z0-9_.:-]*/$")
_RANK_MODULE = "core.profiling.ncu_torchrun_rank"
_GENERATION_MODULE = "core.profiling.ncu_torchrun_generation"
_TORCHRUN_MODULE = "torch.distributed.run"
_WRAPPER_MODULE = "core.harness.torchrun_wrapper"
_CAPTURE_OWNER_ENV = "AISP_NCU_TORCHRUN_CAPTURE_ID"
_PROC_ROOT = Path("/proc")
_ProcIdentity = tuple[int, int]
_APPLICATION_MARKER_TEXT = "Application replay pass"
_APPLICATION_MARKER = re.compile(
    rb'^==PROF== Profiling(?: on device (?P<device>[0-9]+) :)? '
    rb'"(?P<kernel>[^"\r\n]+)": Application replay pass '
    rb'(?P<pass_number>[1-9][0-9]*)\r?\n$'
)
_COMM_RANK_PREFIX = re.compile(
    rb"^==PROF== \[Comm rank (?P<comm_rank>[0-9]+)\] (?P<body> .*)$"
)


def _rank(value: int) -> int:
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"local_rank must be 0 or 1, got {value!r}")
    return value


def _positive_kernel_count(value: str) -> int:
    """Parse one positive expected all-matching kernel count."""

    if not isinstance(value, str) or not re.fullmatch(r"[1-9][0-9]*", value):
        raise argparse.ArgumentTypeError("expected kernel counts must be positive integers")
    return int(value)


@dataclass(frozen=True)
class NcuTorchrunCapturePlan:
    """Inspectable launch plan used by the CLI and external validators."""

    label: str
    repo_root: Path
    source: str
    output_dir: Path
    ncu_path: Path
    tcp_port: int | None
    timeout_seconds: float
    nccl_nvtx_includes: tuple[str, ...]
    torchrun_argv: tuple[str, ...]
    launcher_argv: tuple[str, ...]
    worker_argv: tuple[str, ...]
    ncu_prefix: tuple[str, ...]
    all_matching_kernels: bool
    replay_mode: str
    target_visible_device: int | None
    kernel_name: str | None
    launch_skip: int | None
    launch_count: int | None
    application_topology: str = "parent"
    expected_application_passes: int | None = None
    application_rdzv_port_base: int | None = None
    expected_kernel_counts: tuple[int, int] | None = None
    report_import_timeout_seconds: float = 60.0

    @property
    def is_rank_application(self) -> bool:
        return self.replay_mode == "application" and self.application_topology == "rank-tcp"

    @property
    def rank_config_path(self) -> Path:
        return self.output_dir / "rank-config.json"

    @property
    def generation_config_path(self) -> Path:
        return self.output_dir / "generation-config.json"

    @property
    def receipt_path(self) -> Path:
        return self.output_dir / "receipt.json"

    @property
    def stdout_path(self) -> Path:
        return self.output_dir / "stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self.output_dir / "stderr.log"

    @property
    def combined_path(self) -> Path:
        return self.output_dir / "combined.log"

    @property
    def application_report_path(self) -> Path:
        return self.output_dir / "application.ncu-rep"

    @property
    def application_csv_path(self) -> Path:
        return self.output_dir / "application.raw.csv"

    @property
    def application_import_stderr_path(self) -> Path:
        return self.output_dir / "application-import.stderr.log"

    def rank_log_path(self, local_rank: int) -> Path:
        return self.output_dir / f"rank-{_rank(local_rank)}.ncu.log"

    def rank_csv_path(self, local_rank: int) -> Path:
        return self.output_dir / f"rank-{_rank(local_rank)}.raw.csv"

    def rank_import_stderr_path(self, local_rank: int) -> Path:
        return self.output_dir / f"rank-{_rank(local_rank)}-import.stderr.log"

    def report_path(self, local_rank: int) -> Path:
        return self.output_dir / f"rank-{_rank(local_rank)}.ncu-rep"

    def rank_argv_path(self, local_rank: int) -> Path:
        return self.output_dir / f"rank-{_rank(local_rank)}-argv.json"

    @property
    def generation_worker_argv(self) -> tuple[str, ...]:
        if not self.is_rank_application:
            raise ValueError("Generation worker argv requires rank-tcp application replay")
        return (
            self.worker_argv[0],
            "-m",
            _GENERATION_MODULE,
            str(self.generation_config_path),
        )

    def expected_rank_argv(self, local_rank: int) -> tuple[str, ...]:
        if self.replay_mode == "application" and not self.is_rank_application:
            raise ValueError("Application replay has one parent NCU command, not rank commands")
        local_rank = _rank(local_rank)
        return (
            *self.ncu_prefix,
            *(
                ("--log-file", str(self.rank_log_path(local_rank)))
                if self.is_rank_application
                else ()
            ),
            "--devices",
            str(local_rank),
            "--export",
            str(self.report_path(local_rank)),
            *(self.generation_worker_argv if self.is_rank_application else self.worker_argv),
        )

    @property
    def launch_argv(self) -> tuple[str, ...]:
        if self.replay_mode == "application" and not self.is_rank_application:
            return (
                *self.ncu_prefix,
                "--export",
                str(self.application_report_path),
                *self.torchrun_argv,
            )
        return (
            *self.launcher_argv,
            "--no-python",
            self.torchrun_argv[0],
            "-m",
            _RANK_MODULE,
            str(self.rank_config_path),
        )

    def rank_config(self) -> dict[str, Any]:
        if self.replay_mode == "application" and not self.is_rank_application:
            raise ValueError("Application replay does not use the rank-local configuration")
        return {
            "schema": (
                RANK_APPLICATION_CONFIG_SCHEMA if self.is_rank_application else RANK_CONFIG_SCHEMA
            ),
            "ncu_prefix": list(self.ncu_prefix),
            "worker_argv": list(
                self.generation_worker_argv if self.is_rank_application else self.worker_argv
            ),
            "output_dir": str(self.output_dir),
            "world_size": WORLD_SIZE,
        }

    def generation_config(self) -> dict[str, Any]:
        if not self.is_rank_application:
            raise ValueError("Generation config requires rank-tcp application replay")
        assert self.expected_application_passes is not None
        assert self.application_rdzv_port_base is not None
        return {
            "schema": GENERATION_CONFIG_SCHEMA,
            "worker_argv": list(self.worker_argv),
            "output_dir": str(self.output_dir),
            "world_size": WORLD_SIZE,
            "expected_passes": self.expected_application_passes,
            "master_addr": "127.0.0.1",
            "master_port_base": self.application_rdzv_port_base,
        }


@dataclass(frozen=True)
class NcuTorchrunCaptureResult:
    status: str
    returncode: int | None
    timed_out: bool
    receipt_path: Path
    stdout_path: Path
    stderr_path: Path


@dataclass(frozen=True)
class _CleanupResult:
    mode: str
    supported: bool
    natural: bool
    drained: bool
    signals_sent: tuple[str, ...]
    signal_events: tuple[dict[str, Any], ...]
    observed_pids: tuple[int, ...]
    surviving_pids: tuple[int, ...]
    errors: tuple[str, ...]

    def receipt(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "supported": self.supported,
            "natural": self.natural,
            "forced": bool(self.signals_sent),
            "drained": self.drained,
            "signals_sent": list(self.signals_sent),
            "signal_events": list(self.signal_events),
            "observed_pids": list(self.observed_pids),
            "surviving_pids": list(self.surviving_pids),
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class _ReportImportResult:
    started: bool
    process_id: int | None
    returncode: int | None
    timed_out: bool
    stdout: bytes
    stderr: bytes
    wall_seconds: float
    start_error: str | None
    communication_error: str | None
    signals_sent: tuple[str, ...]
    cleanup_drained: bool
    cleanup_errors: tuple[str, ...]


def _option(argv: Sequence[str], *names: str, required: bool = True) -> str | None:
    found: list[str] = []
    for index, item in enumerate(argv):
        for name in names:
            if item == name:
                if index + 1 == len(argv):
                    raise ValueError(f"Torchrun option {name} requires a value")
                found.append(argv[index + 1])
            elif item.startswith(name + "="):
                found.append(item.split("=", 1)[1])
    expected = "exactly one" if required else "at most one"
    if len(found) > 1 or (required and not found):
        raise ValueError(f"Torchrun command requires {expected} of {names}, found {found}")
    return found[0] if found else None


def _split_torchrun(argv: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    command = tuple(str(item) for item in argv)
    if len(command) < 7 or command[1:3] != ("-m", _TORCHRUN_MODULE):
        raise ValueError("Command must start with '<python> -m torch.distributed.run'")
    markers = [
        i for i in range(3, len(command) - 1) if command[i : i + 2] == ("-m", _WRAPPER_MODULE)
    ]
    if len(markers) != 1:
        raise ValueError(f"Command must contain exactly one '-m {_WRAPPER_MODULE}'")
    launcher = command[: markers[0]]
    worker = (command[0], *command[markers[0] :])
    if {"--no-python", "--no_python"}.intersection(launcher):
        raise ValueError("Input torchrun command must not already use --no-python")
    nproc = _option(launcher, "--nproc_per_node", "--nproc-per-node")
    if nproc != "2":
        raise ValueError(f"Two-rank NCU capture requires --nproc_per_node 2, got {nproc!r}")
    nnodes = _option(launcher, "--nnodes", required=False)
    if nnodes not in (None, "1"):
        raise ValueError(f"Two-rank NCU capture supports exactly one node, got {nnodes!r}")
    backend = _option(launcher, "--rdzv_backend", "--rdzv-backend")
    if backend != "c10d":
        raise ValueError(f"Torchrun rendezvous backend must be c10d, got {backend!r}")
    endpoint = _option(launcher, "--rdzv_endpoint", "--rdzv-endpoint")
    try:
        host, port = str(endpoint).rsplit(":", 1)
        valid_endpoint = host in {"127.0.0.1", "localhost"} and 1 <= int(port) <= 65535
    except ValueError:
        valid_endpoint = False
    if not valid_endpoint:
        raise ValueError(f"One-node torchrun requires a loopback rendezvous: {endpoint!r}")
    if "--aisp-emit-runtime-provenance" not in worker:
        raise ValueError("Torchrun worker must request --aisp-emit-runtime-provenance")
    return launcher, worker


def _ranges(values: Sequence[str]) -> tuple[str, ...]:
    includes = tuple(str(value).strip() for value in values)
    if not includes or any(not value for value in includes):
        raise ValueError("At least one explicit NCCL push/pop NVTX include is required")
    if len(set(includes)) != len(includes):
        raise ValueError("NCCL NVTX includes must not contain duplicates")
    invalid = [value for value in includes if not _NCCL_RANGE.fullmatch(value)]
    if invalid:
        raise ValueError(
            "Lockstep filters must be named NCCL push/pop selectors such as "
            f"'NCCL@ncclGroupEnd/'. Invalid={invalid}"
        )
    return includes


def build_capture_plan(
    *,
    label: str,
    repo_root: Path,
    source: str,
    output_dir: Path,
    ncu_path: Path,
    tcp_port: int | None,
    timeout_seconds: float,
    nccl_nvtx_includes: Sequence[str] | None,
    torchrun_argv: Sequence[str],
    all_matching_kernels: bool = False,
    replay_mode: str = "kernel",
    target_visible_device: int | None = None,
    kernel_name: str | None = None,
    launch_skip: int | None = None,
    launch_count: int | None = None,
    application_topology: str = "parent",
    expected_application_passes: int | None = None,
    application_rdzv_port_base: int | None = None,
    expected_kernel_counts: tuple[int, int] | None = None,
    report_import_timeout_seconds: float = 60.0,
) -> NcuTorchrunCapturePlan:
    """Validate the supported command contract and build exact rank argv."""

    label, source = str(label).strip(), str(source).strip().lower()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise ValueError("label must contain only letters, digits, '.', '_' or '-'")
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("source must be a full 40-character lowercase Git commit")
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ValueError(f"timeout_seconds must be positive and finite: {timeout_seconds!r}")
    if type(all_matching_kernels) is not bool:
        raise ValueError("all_matching_kernels must be a boolean")
    if (
        type(report_import_timeout_seconds) not in (int, float)
        or not math.isfinite(float(report_import_timeout_seconds))
        or report_import_timeout_seconds <= 0
    ):
        raise ValueError(
            "report_import_timeout_seconds must be positive and finite: "
            f"{report_import_timeout_seconds!r}"
        )
    if expected_kernel_counts is not None and (
        type(expected_kernel_counts) is not tuple
        or len(expected_kernel_counts) != WORLD_SIZE
        or any(type(value) is not int or value <= 0 for value in expected_kernel_counts)
    ):
        raise ValueError(
            "expected_kernel_counts must be a tuple of two positive integers"
        )
    if type(replay_mode) is not str or replay_mode not in {"application", "kernel", "range"}:
        raise ValueError("replay_mode must be 'application', 'kernel' or 'range'")
    if type(application_topology) is not str or application_topology not in {
        "parent",
        "rank-tcp",
    }:
        raise ValueError("application_topology must be 'parent' or 'rank-tcp'")
    if replay_mode != "application" and application_topology != "parent":
        raise ValueError(f"{replay_mode} replay rejects application_topology")
    if replay_mode == "range" and not all_matching_kernels:
        raise ValueError("range replay requires all_matching_kernels=True")
    if float(report_import_timeout_seconds) != 60.0 and not (
        replay_mode == "application"
        and application_topology == "rank-tcp"
        and all_matching_kernels
    ):
        raise ValueError(
            "nondefault report_import_timeout_seconds requires all-matching "
            "rank-tcp application replay"
        )
    launcher, worker = _split_torchrun(torchrun_argv)
    torchrun_port = int(
        str(_option(launcher, "--rdzv_endpoint", "--rdzv-endpoint")).rsplit(":", 1)[1]
    )
    metrics = tuple(MINIMAL_METRICS)
    if len(metrics) != 5 or len(set(metrics)) != 5:
        raise RuntimeError("Repository minimal NCU contract must contain exactly five metrics")

    application_values = {
        "target_visible_device": target_visible_device,
        "kernel_name": kernel_name,
        "launch_skip": launch_skip,
        "launch_count": launch_count,
        "expected_application_passes": expected_application_passes,
        "application_rdzv_port_base": application_rdzv_port_base,
        "expected_kernel_counts": expected_kernel_counts,
    }
    if replay_mode == "application":
        if nccl_nvtx_includes:
            raise ValueError("application replay rejects NCCL NVTX includes")
        if (
            type(kernel_name) is not str
            or not kernel_name.startswith("regex:")
            or not kernel_name.removeprefix("regex:")
            or kernel_name != kernel_name.strip()
            or "\n" in kernel_name
            or "\r" in kernel_name
        ):
            raise ValueError("kernel_name must be an explicit nonempty regex: selector")
        try:
            re.compile(kernel_name.removeprefix("regex:"))
        except re.error as exc:
            raise ValueError(f"kernel_name contains an invalid regular expression: {exc}") from exc
        if all_matching_kernels:
            if application_topology != "rank-tcp":
                raise ValueError(
                    "application all-matching capture requires rank-tcp topology"
                )
            if type(launch_skip) is not int or launch_skip != 0:
                raise ValueError(
                    "application all-matching capture requires launch_skip=0"
                )
            if launch_count is not None:
                raise ValueError(
                    "application all-matching capture rejects launch_count"
                )
            if expected_kernel_counts is None:
                raise ValueError(
                    "application all-matching capture requires expected_kernel_counts"
                )
            if expected_application_passes != 9:
                raise ValueError(
                    "application all-matching capture requires exactly 9 replay passes"
                )
        else:
            if expected_kernel_counts is not None:
                raise ValueError(
                    "selected application replay rejects expected_kernel_counts"
                )
            if type(launch_skip) is not int or launch_skip < 0:
                raise ValueError("launch_skip must be a nonnegative integer")
            if type(launch_count) is not int or launch_count != 1:
                raise ValueError(
                    "launch_count must be exactly 1 for bounded application replay"
                )
        includes: tuple[str, ...] = ()
        if application_topology == "parent":
            if tcp_port is not None:
                raise ValueError("parent application replay rejects tcp_port")
            if expected_application_passes is not None:
                raise ValueError("parent application replay rejects expected_application_passes")
            if application_rdzv_port_base is not None:
                raise ValueError("parent application replay rejects application_rdzv_port_base")
            if type(target_visible_device) is not int or target_visible_device not in (0, 1):
                raise ValueError("target_visible_device must be the visible ordinal 0 or 1")
            prefix = [
                str(ncu_path),
                "--clock-control",
                "none",
                "--target-processes",
                "all",
                "--devices",
                str(target_visible_device),
                "--replay-mode",
                "application",
                "--app-replay-mode",
                "strict",
                "--app-replay-match",
                "grid",
                "--filter-mode",
                "per-gpu",
                "--kernel-name",
                kernel_name,
                "--launch-skip",
                str(launch_skip),
                "--launch-count",
                str(launch_count),
                "--kill",
                "no",
                "--check-exit-code",
                "yes",
                "--metrics",
                ",".join(metrics),
            ]
        else:
            if target_visible_device is not None:
                raise ValueError("rank-tcp application replay rejects target_visible_device")
            if type(tcp_port) is not int or not 1024 <= tcp_port <= 65535:
                raise ValueError(
                    "rank-tcp application replay requires tcp_port from 1024 through 65535"
                )
            if tcp_port == torchrun_port:
                raise ValueError(
                    "NCU communicator port must differ from the torchrun rendezvous port"
                )
            if type(expected_application_passes) is not int or expected_application_passes <= 0:
                raise ValueError(
                    "rank-tcp application replay requires expected_application_passes > 0"
                )
            if (
                type(application_rdzv_port_base) is not int
                or application_rdzv_port_base < 1024
                or application_rdzv_port_base + expected_application_passes - 1 > 65535
            ):
                raise ValueError(
                    "rank-tcp application rendezvous ports must stay within 1024..65535"
                )
            generation_ports = set(
                range(
                    application_rdzv_port_base,
                    application_rdzv_port_base + expected_application_passes,
                )
            )
            collisions = generation_ports.intersection({tcp_port, torchrun_port})
            if collisions:
                raise ValueError(
                    "Application generation ports must differ from NCU and torchrun ports: "
                    f"{sorted(collisions)}"
                )
            prefix = [
                str(ncu_path),
                "--clock-control",
                "none",
                "--target-processes",
                "all",
                "--communicator",
                "tcp",
                "--communicator-tcp-hostname",
                "127.0.0.1",
                "--communicator-tcp-port",
                str(tcp_port),
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
                kernel_name,
            ]
            prefix.extend(["--launch-skip", str(launch_skip)])
            if all_matching_kernels:
                prefix.append("--nvtx")
            if not all_matching_kernels:
                prefix.extend(["--launch-count", str(launch_count)])
            prefix.extend(
                [
                    "--kill",
                    "no",
                    "--check-exit-code",
                    "yes",
                    "--metrics",
                    ",".join(metrics),
                ]
            )
    else:
        supplied_application_options = [
            name for name, value in application_values.items() if value is not None
        ]
        if supplied_application_options:
            raise ValueError(
                f"{replay_mode} replay rejects application options: "
                f"{supplied_application_options}"
            )
        if type(tcp_port) is not int or not 1024 <= tcp_port <= 65535:
            raise ValueError(
                f"tcp_port must be an integer from 1024 through 65535: {tcp_port!r}"
            )
        if tcp_port == torchrun_port:
            raise ValueError("NCU communicator port must differ from the torchrun rendezvous port")
        includes = _ranges(nccl_nvtx_includes or ())
        prefix = [
            str(ncu_path),
            "--clock-control",
            "none",
            "--target-processes",
            "all",
            "--communicator",
            "tcp",
            "--communicator-tcp-hostname",
            "127.0.0.1",
            "--communicator-tcp-port",
            str(tcp_port),
            "--communicator-tcp-num-peers",
            "2",
            "--lockstep-kernel-launch",
            "--nvtx",
        ]
        for flag in ("--lockstep-nvtx-include", "--nvtx-include"):
            for include in includes:
                prefix.extend([flag, include])
        if replay_mode == "kernel":
            prefix.extend(["--kernel-name", "regex:nccl.*"])
        if not all_matching_kernels:
            prefix.extend(["--launch-count", "1"])
        prefix.extend(["--replay-mode", replay_mode, "--metrics", ",".join(metrics)])
    return NcuTorchrunCapturePlan(
        label,
        Path(repo_root).resolve(),
        source,
        Path(output_dir).resolve(),
        Path(ncu_path),
        tcp_port,
        float(timeout_seconds),
        includes,
        tuple(str(item) for item in torchrun_argv),
        launcher,
        worker,
        tuple(prefix),
        all_matching_kernels,
        replay_mode,
        target_visible_device,
        kernel_name,
        launch_skip,
        launch_count,
        application_topology,
        expected_application_passes,
        application_rdzv_port_base,
        expected_kernel_counts,
        float(report_import_timeout_seconds),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=15,
    ).strip()


def _preflight(plan: NcuTorchrunCapturePlan, env: Mapping[str, str]) -> dict[str, Any]:
    if plan.output_dir.exists() or plan.output_dir.is_symlink():
        raise FileExistsError(f"Output directory must not exist: {plan.output_dir}")
    if plan.repo_root.is_symlink() or not plan.repo_root.is_dir():
        raise ValueError(f"Repository root must be a real directory: {plan.repo_root}")
    if Path(_git(plan.repo_root, "rev-parse", "--show-toplevel")).resolve() != plan.repo_root:
        raise ValueError(f"repo_root is not the Git top level: {plan.repo_root}")
    observed_source = _git(plan.repo_root, "rev-parse", "HEAD")
    if observed_source != plan.source:
        raise ValueError(f"Source mismatch: requested {plan.source}, observed {observed_source}")
    if _git(plan.repo_root, "status", "--porcelain", "--untracked-files=all"):
        raise ValueError("Repository must be clean before capture")
    ncu = plan.ncu_path
    if (
        not ncu.is_absolute()
        or ncu.is_symlink()
        or not ncu.is_file()
        or not os.access(ncu, os.X_OK)
    ):
        raise ValueError(f"ncu_path must be an absolute, real executable: {ncu}")
    devices = tuple(part.strip() for part in env.get("CUDA_VISIBLE_DEVICES", "").split(","))
    if len(devices) != 2 or any(not item for item in devices) or len(set(devices)) != 2:
        raise ValueError("CUDA_VISIBLE_DEVICES must name exactly two distinct devices")
    if plan.replay_mode == "application" and not plan.is_rank_application:
        assert plan.target_visible_device is not None
        if plan.target_visible_device >= len(devices):
            raise ValueError(
                "target_visible_device is outside the CUDA_VISIBLE_DEVICES ordinal range"
            )
    for name in ("AISP_LOCK_GPU_CLOCKS", "AISP_RAMP_GPU_CLOCKS"):
        if env.get(name) != "1":
            raise ValueError(f"{name}=1 is required")
    code_root = (plan.repo_root / "code").resolve()
    python_paths = {
        str(Path(item).resolve()) for item in env.get("PYTHONPATH", "").split(os.pathsep) if item
    }
    if str(code_root) not in python_paths:
        raise ValueError(f"PYTHONPATH must include {code_root}")
    if plan.replay_mode != "application" or plan.is_rank_application:
        assert plan.tcp_port is not None
        with socket.socket() as listener:
            try:
                listener.bind(("127.0.0.1", plan.tcp_port))
            except OSError as exc:
                raise ValueError(f"NCU TCP port is unavailable: {plan.tcp_port}") from exc
    if plan.is_rank_application:
        assert plan.application_rdzv_port_base is not None
        assert plan.expected_application_passes is not None
        for port in range(
            plan.application_rdzv_port_base,
            plan.application_rdzv_port_base + plan.expected_application_passes,
        ):
            with socket.socket() as listener:
                try:
                    listener.bind(("127.0.0.1", port))
                except OSError as exc:
                    raise ValueError(
                        f"Application generation rendezvous port is unavailable: {port}"
                    ) from exc
    version = subprocess.check_output(
        [str(ncu), "--version"], text=True, stderr=subprocess.STDOUT, timeout=15
    ).strip()
    if f"Version {SUPPORTED_NCU_VERSION}" not in version:
        raise ValueError(f"Qualified only for NCU {SUPPORTED_NCU_VERSION}. Observed {version!r}")
    result = {
        "observed_source": observed_source,
        "ncu_version": version,
        "ncu_binary_sha256": _sha256(ncu),
        "driver_sha256": _sha256(Path(__file__)),
        "rank_helper_sha256": _sha256(Path(__file__).with_name("ncu_torchrun_rank.py")),
        "process_tree_cleanup_supported": _proc_cleanup_supported(),
        "visible_devices": list(devices),
        "clock_env": {
            key: env.get(key)
            for key in (
                "AISP_LOCK_GPU_CLOCKS",
                "AISP_RAMP_GPU_CLOCKS",
                "AISP_GPU_SM_CLOCK_MHZ",
                "AISP_GPU_MEM_CLOCK_MHZ",
            )
        },
    }
    if plan.is_rank_application:
        result["generation_helper_sha256"] = _sha256(
            Path(__file__).with_name("ncu_torchrun_generation.py")
        )
    return result


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _group_exists(pgid: int) -> tuple[bool, str | None]:
    try:
        os.killpg(pgid, 0)
        return True, None
    except ProcessLookupError:
        return False, None
    except (PermissionError, OSError) as exc:
        return True, f"process-group inspection failed: {exc}"


def _wait_group(process: subprocess.Popen[Any], timeout: float) -> tuple[bool, str | None]:
    deadline = time.monotonic() + timeout
    while True:
        process.poll()  # Reap an exited leader while independently checking descendants.
        exists, error = _group_exists(process.pid)
        if error or not exists or time.monotonic() >= deadline:
            return not exists, error
        time.sleep(0.05)


def _drain_owned_process_group(
    process: subprocess.Popen[Any], *, grace_seconds: float = 2.0
) -> tuple[list[str], bool, list[str]]:
    """Drain the dedicated PGID even when its leader has already exited."""

    sent: list[str] = []
    errors: list[str] = []
    drained, error = _wait_group(process, grace_seconds)
    if error:
        return sent, False, [error]
    exists = not drained
    for sig, label, wait in (
        (signal.SIGTERM, "SIGTERM", grace_seconds),
        (signal.SIGKILL, "SIGKILL", 2.0),
    ):
        if not exists:
            break
        try:
            os.killpg(process.pid, sig)
            sent.append(label)
        except ProcessLookupError:
            exists = False
            break
        except (PermissionError, OSError) as exc:
            errors.append(f"{label} failed: {exc}")
            break
        drained, error = _wait_group(process, wait)
        if error:
            errors.append(error)
            break
        exists = not drained
    try:
        process.wait(timeout=max(grace_seconds, 0.1))
    except subprocess.TimeoutExpired:
        errors.append(f"process-group leader {process.pid} could not be reaped")
    drained, error = _wait_group(process, 0.0)
    if error:
        errors.append(error)
    if not drained and not errors:
        errors.append(f"process group {process.pid} survived SIGKILL")
    return sent, drained and not errors, errors


def _captured_subprocess_bytes(value: bytes | None, *, label: str) -> bytes:
    if value is None:
        return b""
    if not isinstance(value, bytes):
        raise RuntimeError(f"{label} was not captured as bytes")
    return value


def _longer_capture(current: bytes, candidate: bytes | None, *, label: str) -> bytes:
    captured = _captured_subprocess_bytes(candidate, label=label)
    return captured if len(captured) >= len(current) else current


def _run_all_matching_report_import(
    argv: Sequence[str], *, timeout_seconds: float
) -> _ReportImportResult:
    """Run one report import in a dedicated group and drain every descendant."""

    start_ns = time.time_ns()
    try:
        process = subprocess.Popen(
            tuple(str(item) for item in argv),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return _ReportImportResult(
            False,
            None,
            None,
            False,
            b"",
            b"",
            (time.time_ns() - start_ns) / 1e9,
            f"{type(exc).__name__}: {exc}",
            None,
            (),
            True,
            (),
        )

    stdout = b""
    stderr = b""
    timed_out = False
    communication_error: str | None = None
    signals_sent: list[str] = []
    cleanup_drained = False
    cleanup_errors: list[str] = []
    try:
        try:
            stdout_value, stderr_value = process.communicate(timeout=timeout_seconds)
            stdout = _captured_subprocess_bytes(
                stdout_value, label="report import stdout"
            )
            stderr = _captured_subprocess_bytes(
                stderr_value, label="report import stderr"
            )
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _longer_capture(
                stdout, exc.output, label="timed-out report import stdout"
            )
            stderr = _longer_capture(
                stderr, exc.stderr, label="timed-out report import stderr"
            )
        except (OSError, subprocess.SubprocessError) as exc:
            communication_error = f"{type(exc).__name__}: {exc}"
    finally:
        signals_sent, cleanup_drained, cleanup_errors = (
            _drain_owned_process_group(process)
        )
    if timed_out or communication_error is not None:
        try:
            stdout_value, stderr_value = process.communicate(timeout=2.0)
            stdout = _longer_capture(
                stdout, stdout_value, label="final report import stdout"
            )
            stderr = _longer_capture(
                stderr, stderr_value, label="final report import stderr"
            )
        except subprocess.TimeoutExpired as exc:
            stdout = _longer_capture(
                stdout, exc.output, label="final timed-out report import stdout"
            )
            stderr = _longer_capture(
                stderr, exc.stderr, label="final timed-out report import stderr"
            )
            cleanup_errors.append("report import pipes remained open after cleanup")
            cleanup_drained = False
        except (OSError, subprocess.SubprocessError) as exc:
            cleanup_errors.append(
                f"final report import communication failed: {type(exc).__name__}: {exc}"
            )
            cleanup_drained = False

    return _ReportImportResult(
        True,
        process.pid,
        process.returncode,
        timed_out,
        stdout,
        stderr,
        (time.time_ns() - start_ns) / 1e9,
        None,
        communication_error,
        tuple(signals_sent),
        cleanup_drained,
        tuple(cleanup_errors),
    )


def _proc_cleanup_supported(proc_root: Path = _PROC_ROOT) -> bool:
    """Return whether inherited owner markers can be inspected on this host."""

    return sys.platform.startswith("linux") and proc_root.is_dir()


def _proc_identity(entry: Path) -> _ProcIdentity:
    """Return a PID/start-time identity that is stable across PID reuse."""

    stat = (entry / "stat").read_bytes()
    fields_after_comm = stat.rsplit(b")", 1)[1].split()
    return int(entry.name), int(fields_after_comm[19])


def _snapshot_proc_identities(proc_root: Path = _PROC_ROOT) -> frozenset[_ProcIdentity]:
    """Snapshot identities that are proven to exist before capture launch."""

    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return frozenset()
    identities: set[_ProcIdentity] = set()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            identities.add(_proc_identity(entry))
        except (
            FileNotFoundError,
            ProcessLookupError,
            PermissionError,
            OSError,
            ValueError,
            IndexError,
        ):
            continue
    return frozenset(identities)


def _marked_pids(
    marker: str,
    proc_root: Path = _PROC_ROOT,
    *,
    owned_candidates: set[int] | frozenset[int] = frozenset(),
    preexisting_identities: frozenset[_ProcIdentity] = frozenset(),
) -> tuple[set[int], str | None]:
    """Find live processes carrying this launch's exact inherited marker.

    An unreadable environment is ignored only when its PID/start-time identity
    was observed before capture launch. The launcher, a previously marked PID,
    or any unclassified post-snapshot process fails closed without being
    signaled.
    """

    token = f"{_CAPTURE_OWNER_ENV}={marker}".encode()
    try:
        entries = tuple(proc_root.iterdir())
    except OSError as exc:
        return set(), f"process-tree inspection failed: {exc}"
    found: set[int] = set()
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            environment = (entry / "environ").read_bytes()
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, OSError) as exc:
            pid = int(entry.name)
            if pid in owned_candidates:
                return (
                    found,
                    f"process-tree inspection failed for owned candidate pid {pid}: {exc}",
                )
            try:
                identity = _proc_identity(entry)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (PermissionError, OSError, ValueError, IndexError) as identity_exc:
                return (
                    found,
                    f"process-tree identity inspection failed for unclassified pid {pid}: "
                    f"{identity_exc}. Environment error: {exc}",
                )
            if identity in preexisting_identities:
                continue
            return (
                found,
                f"process-tree inspection failed for unclassified post-snapshot pid {pid}: {exc}",
            )
        if token in environment.split(b"\0"):
            pid = int(entry.name)
            if pid != os.getpid():
                found.add(pid)
    return found, None


def _wait_marked_processes(
    process: subprocess.Popen[Any],
    marker: str,
    timeout: float,
    proc_root: Path,
    owned_candidates: set[int],
    preexisting_identities: frozenset[_ProcIdentity],
) -> tuple[set[int], set[int], str | None]:
    observed: set[int] = set()
    deadline = time.monotonic() + timeout
    while True:
        launcher_returncode = process.poll()  # Reap a zombie before scanning.
        remaining, error = _marked_pids(
            marker,
            proc_root,
            owned_candidates=owned_candidates,
            preexisting_identities=preexisting_identities,
        )
        observed.update(remaining)
        owned_candidates.update(remaining)
        if error is None and launcher_returncode is None and process.pid not in remaining:
            error = f"live launcher {process.pid} owner marker is unreadable or absent"
        if error or not remaining or time.monotonic() >= deadline:
            return remaining, observed, error
        time.sleep(0.05)


def _signal_marked_processes(
    marker: str,
    pids: set[int],
    sig: signal.Signals,
    proc_root: Path,
) -> tuple[list[int], list[str]]:
    """Signal only PIDs that still carry this capture's unique owner marker."""

    sent: list[int] = []
    errors: list[str] = []
    token = f"{_CAPTURE_OWNER_ENV}={marker}".encode()
    for pid in sorted(pids):
        try:
            environment = (proc_root / str(pid) / "environ").read_bytes()
            if token not in environment.split(b"\0"):
                continue
            os.kill(pid, sig)
            sent.append(pid)
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, OSError) as exc:
            errors.append(f"{sig.name} pid {pid} failed: {exc}")
    return sent, errors


def _cleanup_capture_processes(
    process: subprocess.Popen[Any],
    marker: str,
    *,
    proc_root: Path = _PROC_ROOT,
    grace_seconds: float = 2.0,
    preexisting_identities: frozenset[_ProcIdentity] = frozenset(),
) -> _CleanupResult:
    """Drain all marked launch processes, including detached torchrun ranks.

    Linux procfs is required to prove full-tree drainage. Other hosts still drain
    the launcher's dedicated process group, but the result remains unsupported.
    """

    if not _proc_cleanup_supported(proc_root):
        sent, _, group_errors = _drain_owned_process_group(process, grace_seconds=grace_seconds)
        errors = [
            "full process-tree cleanup is unverified without Linux procfs owner-marker inspection",
            *group_errors,
        ]
        return _CleanupResult(
            "process-group-fallback-unverified",
            False,
            not sent,
            False,
            tuple(sent),
            (),
            (),
            (),
            tuple(errors),
        )

    owned_candidates = {process.pid}
    remaining, observed, scan_error = _wait_marked_processes(
        process,
        marker,
        grace_seconds,
        proc_root,
        owned_candidates,
        preexisting_identities,
    )
    errors = [scan_error] if scan_error else []
    events: list[dict[str, Any]] = []
    labels: list[str] = []
    for sig, wait in ((signal.SIGTERM, grace_seconds), (signal.SIGKILL, 2.0)):
        if not remaining or errors:
            break
        signaled, signal_errors = _signal_marked_processes(marker, remaining, sig, proc_root)
        if signaled:
            labels.append(sig.name)
            events.append({"signal": sig.name, "pids": signaled})
        errors.extend(signal_errors)
        remaining, newly_observed, scan_error = _wait_marked_processes(
            process,
            marker,
            wait,
            proc_root,
            owned_candidates,
            preexisting_identities,
        )
        observed.update(newly_observed)
        if scan_error:
            errors.append(scan_error)
    if errors and process.returncode is None:
        fallback_sent, _, fallback_errors = _drain_owned_process_group(
            process, grace_seconds=grace_seconds
        )
        for label in fallback_sent:
            if label not in labels:
                labels.append(label)
            events.append({"signal": label, "process_group": process.pid})
        errors.extend(fallback_errors)
    try:
        process.wait(timeout=max(grace_seconds, 0.1))
    except subprocess.TimeoutExpired:
        errors.append(f"launcher {process.pid} could not be reaped")
    remaining, newly_observed, scan_error = _wait_marked_processes(
        process,
        marker,
        0.0,
        proc_root,
        owned_candidates,
        preexisting_identities,
    )
    observed.update(newly_observed)
    if scan_error:
        errors.append(scan_error)
    if remaining and not errors:
        errors.append(f"marked processes survived cleanup: {sorted(remaining)}")
    return _CleanupResult(
        "linux-proc-environ-marker",
        True,
        not labels and not remaining and not errors,
        not remaining and process.returncode is not None and not errors,
        tuple(labels),
        tuple(events),
        tuple(sorted(observed)),
        tuple(sorted(remaining)),
        tuple(errors),
    )


def _artifacts(
    plan: NcuTorchrunCapturePlan, start_ns: int
) -> tuple[list[dict[str, Any]], list[str]]:
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    for rank in range(WORLD_SIZE):
        report, argv_path = plan.report_path(rank), plan.rank_argv_path(rank)
        item: dict[str, Any] = {
            "local_rank": rank,
            "report_path": str(report),
            "argv_path": str(argv_path),
            "expected_argv": list(plan.expected_rank_argv(rank)),
        }
        if argv_path.is_symlink() or not argv_path.is_file():
            errors.append(f"rank {rank} argv receipt is missing or symlinked")
        else:
            try:
                item["observed_argv"] = json.loads(argv_path.read_text())
            except (json.JSONDecodeError, OSError) as exc:
                errors.append(f"rank {rank} argv receipt is unreadable: {exc}")
            if item.get("observed_argv") != item["expected_argv"]:
                errors.append(f"rank {rank} argv differs from plan")
        if report.is_symlink() or not report.is_file():
            errors.append(f"rank {rank} report is missing or symlinked")
        else:
            stat = report.stat()
            item.update(
                report_bytes=stat.st_size,
                report_mtime_ns=stat.st_mtime_ns,
                report_sha256=_sha256(report),
            )
            if not stat.st_size or stat.st_mtime_ns < start_ns:
                errors.append(f"rank {rank} report is empty or stale")
        items.append(item)
    hashes = [item.get("report_sha256") for item in items]
    if all(hashes) and len(set(hashes)) != WORLD_SIZE:
        errors.append("rank reports have identical SHA-256 digests")
    return items, errors


def _kernel_selector_matches(selector: str, kernel_name: str) -> bool:
    if selector.startswith("regex:"):
        return re.fullmatch(selector.removeprefix("regex:"), kernel_name) is not None
    return selector == kernel_name


def _parse_application_replay_log(
    plan: NcuTorchrunCapturePlan,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate one complete two-rank provenance generation per replay pass."""

    if plan.replay_mode != "application":
        raise ValueError("Application replay log parsing requires application mode")
    assert plan.kernel_name is not None
    assert plan.target_visible_device is not None
    raw = plan.combined_path.read_bytes()
    markers: list[dict[str, Any]] = []
    offset = 0
    provenance_prefix = TORCHRUN_RUNTIME_PROVENANCE_PREFIX.encode("ascii")
    marker_text = _APPLICATION_MARKER_TEXT.encode("ascii")
    for line in raw.splitlines(keepends=True):
        if marker_text in line:
            marker = _APPLICATION_MARKER.fullmatch(line)
            if marker is None:
                raise TorchrunRuntimeProvenanceError(
                    f"Unexpected application replay marker at combined-log byte {offset}"
                )
            kernel_name = marker.group("kernel").decode("utf-8")
            if not _kernel_selector_matches(plan.kernel_name, kernel_name):
                raise TorchrunRuntimeProvenanceError(
                    f"Application replay marker kernel {kernel_name!r} does not match "
                    f"selector {plan.kernel_name!r}"
                )
            raw_device = marker.group("device")
            if raw_device is not None and int(raw_device) != plan.target_visible_device:
                raise TorchrunRuntimeProvenanceError(
                    "Application replay marker selected a different visible device: "
                    f"expected {plan.target_visible_device}, observed {int(raw_device)}"
                )
            markers.append(
                {
                    "application_replay_pass": int(marker.group("pass_number")),
                    "kernel_name": kernel_name,
                    "marker_device": int(raw_device) if raw_device is not None else None,
                    "marker_start_byte": offset,
                    "payload_start_byte": offset + len(line),
                }
            )
        offset += len(line)

    if not markers:
        raise TorchrunRuntimeProvenanceError(
            "Application replay log contains no recognized pass markers"
        )
    if provenance_prefix in raw[: markers[0]["marker_start_byte"]]:
        raise TorchrunRuntimeProvenanceError(
            "Torchrun runtime provenance appeared before application replay pass 1"
        )
    observed_passes = [item["application_replay_pass"] for item in markers]
    expected_passes = list(range(1, len(markers) + 1))
    if observed_passes != expected_passes:
        raise TorchrunRuntimeProvenanceError(
            "Application replay pass numbers must be consecutive from one: "
            f"expected {expected_passes}, observed {observed_passes}"
        )

    generations: list[dict[str, Any]] = []
    all_process_ids: set[int] = set()
    reference_manifest: RunManifest | None = None
    for index, marker in enumerate(markers):
        payload_end = (
            markers[index + 1]["marker_start_byte"] if index + 1 < len(markers) else len(raw)
        )
        payload = raw[marker["payload_start_byte"] : payload_end]
        try:
            payload_text = payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TorchrunRuntimeProvenanceError(
                "Application replay combined log is not valid UTF-8 in pass "
                f"{marker['application_replay_pass']}"
            ) from exc
        parsed = parse_torchrun_runtime_provenance_stdout(
            payload_text,
            expected_local_ranks=(0, 1),
            target="cuda",
        )
        process_ids = parsed.execution_process_ids
        duplicate_process_ids = all_process_ids.intersection(process_ids.values())
        if duplicate_process_ids:
            raise TorchrunRuntimeProvenanceError(
                "Application replay reused worker process IDs across passes: "
                f"{sorted(duplicate_process_ids)}"
            )
        all_process_ids.update(process_ids.values())

        runtime_by_local_rank: dict[str, Any] = {}
        for local_rank, snapshot in sorted(parsed.snapshots_by_local_rank.items()):
            if snapshot.cuda_available is not True:
                raise TorchrunRuntimeProvenanceError(
                    "Application replay runtime provenance does not report CUDA available "
                    f"for pass {marker['application_replay_pass']}, local rank {local_rank}"
                )
            manifest = RunManifest.model_construct(runtime_provenance=snapshot)
            if reference_manifest is None:
                reference_manifest = manifest
            try:
                require_runtime_provenance_parity(
                    reference_manifest,
                    manifest,
                    target="cuda",
                )
            except RuntimeProvenanceParityError as exc:
                raise TorchrunRuntimeProvenanceError(
                    "Application replay runtime provenance drifted from pass 1, local rank 0 "
                    f"at pass {marker['application_replay_pass']}, local rank {local_rank}: {exc}"
                ) from exc
            runtime_by_local_rank[str(local_rank)] = snapshot.model_dump(mode="json")

        generations.append(
            {
                **marker,
                "payload_end_byte": payload_end,
                "execution_process_ids": process_ids,
                "runtime_by_local_rank": runtime_by_local_rank,
                "both_post_main_receipts": True,
                "cuda_runtime_parity": True,
            }
        )

    return generations, {
        "path": str(plan.combined_path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "stream_mode": "stdout-and-stderr-shared-open-file-description-at-launch",
    }


def _parse_rank_application_generations(
    plan: NcuTorchrunCapturePlan,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate every rank-local generation and its post-main provenance."""

    if not plan.is_rank_application:
        raise ValueError("Rank application generation parsing requires rank-tcp topology")
    assert plan.expected_application_passes is not None
    assert plan.application_rdzv_port_base is not None
    raw = plan.stdout_path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise TorchrunRuntimeProvenanceError(
            "Rank application stdout is not valid UTF-8"
        ) from exc

    begin: dict[tuple[int, int], dict[str, Any]] = {}
    end: dict[tuple[int, int], dict[str, Any]] = {}
    marker_count = 0
    offset = 0
    for line_number, line in enumerate(text.splitlines(keepends=True), start=1):
        content = line.rstrip("\r\n")
        if GENERATION_MARKER_PREFIX not in content:
            offset += len(line)
            continue
        if not line.endswith("\n") or not content.startswith(GENERATION_MARKER_PREFIX):
            raise TorchrunRuntimeProvenanceError(
                f"Malformed rank application generation marker on stdout line {line_number}"
            )
        try:
            payload = json.loads(content.removeprefix(GENERATION_MARKER_PREFIX))
        except json.JSONDecodeError as exc:
            raise TorchrunRuntimeProvenanceError(
                f"Malformed rank application generation JSON on stdout line {line_number}"
            ) from exc
        if not isinstance(payload, dict):
            raise TorchrunRuntimeProvenanceError(
                f"Rank application generation line {line_number} must contain a JSON object"
            )
        event = payload.get("event")
        expected_fields = {
            "event",
            "generation",
            "master_port",
            "pid",
            "rank",
        }
        if event == "end":
            expected_fields.add("returncode")
        if event not in {"begin", "end"} or set(payload) != expected_fields:
            raise TorchrunRuntimeProvenanceError(
                "Rank application generation fields differ from schema on stdout line "
                f"{line_number}"
            )
        rank = payload["rank"]
        generation = payload["generation"]
        process_id = payload["pid"]
        master_port = payload["master_port"]
        if type(rank) is not int or rank not in (0, 1):
            raise TorchrunRuntimeProvenanceError(
                f"Rank application marker has invalid rank on stdout line {line_number}"
            )
        if (
            type(generation) is not int
            or not 0 <= generation < plan.expected_application_passes
        ):
            raise TorchrunRuntimeProvenanceError(
                f"Rank application marker has invalid generation on stdout line {line_number}"
            )
        if type(process_id) is not int or process_id <= 1:
            raise TorchrunRuntimeProvenanceError(
                f"Rank application marker has invalid process ID on stdout line {line_number}"
            )
        expected_port = plan.application_rdzv_port_base + generation
        if type(master_port) is not int or master_port != expected_port:
            raise TorchrunRuntimeProvenanceError(
                "Rank application marker has the wrong rendezvous port on stdout line "
                f"{line_number}: expected {expected_port}, observed {master_port!r}"
            )
        if event == "end" and (
            type(payload["returncode"]) is not int or payload["returncode"] != 0
        ):
            raise TorchrunRuntimeProvenanceError(
                f"Rank application generation did not return zero on stdout line {line_number}"
            )
        record = {**payload, "start_offset": offset, "end_offset": offset + len(line)}
        target = begin if event == "begin" else end
        key = (rank, generation)
        if key in target:
            raise TorchrunRuntimeProvenanceError(
                f"Rank application generation marker is duplicated: {event} {key}"
            )
        target[key] = record
        marker_count += 1
        offset += len(line)

    expected_keys = {
        (rank, generation)
        for rank in range(WORLD_SIZE)
        for generation in range(plan.expected_application_passes)
    }
    if set(begin) != expected_keys or set(end) != expected_keys:
        raise TorchrunRuntimeProvenanceError(
            "Rank application generation marker set differs. "
            f"missing_begin={sorted(expected_keys - set(begin))}, "
            f"unexpected_begin={sorted(set(begin) - expected_keys)}, "
            f"missing_end={sorted(expected_keys - set(end))}, "
            f"unexpected_end={sorted(set(end) - expected_keys)}"
        )
    if marker_count != 2 * len(expected_keys):
        raise TorchrunRuntimeProvenanceError(
            "Rank application marker count differs from the expected begin/end pairs"
        )

    bootstrap_process_ids = [int(begin[key]["pid"]) for key in sorted(expected_keys)]
    if len(set(bootstrap_process_ids)) != len(bootstrap_process_ids):
        raise TorchrunRuntimeProvenanceError(
            "Rank application reused generation process IDs"
        )
    for key in sorted(expected_keys):
        if begin[key]["pid"] != end[key]["pid"]:
            raise TorchrunRuntimeProvenanceError(
                f"Rank application generation process ID changed between markers: {key}"
            )
        if begin[key]["start_offset"] >= end[key]["start_offset"]:
            raise TorchrunRuntimeProvenanceError(
                f"Rank application generation ended before it began: {key}"
            )

    generations: list[dict[str, Any]] = []
    all_worker_process_ids: set[int] = set()
    reference_manifest: RunManifest | None = None
    prior_end = -1
    provenance_offsets = [
        match.start()
        for match in re.finditer(re.escape(TORCHRUN_RUNTIME_PROVENANCE_PREFIX), text)
    ]
    covered_provenance_offsets: set[int] = set()
    for generation in range(plan.expected_application_passes):
        keys = [(rank, generation) for rank in range(WORLD_SIZE)]
        generation_start = min(begin[key]["start_offset"] for key in keys)
        generation_end = max(end[key]["end_offset"] for key in keys)
        if generation_start < prior_end:
            raise TorchrunRuntimeProvenanceError(
                "Rank application generation windows overlap or are out of order at "
                f"generation {generation}"
            )
        prior_end = generation_end
        payload = text[generation_start:generation_end]
        parsed = parse_torchrun_runtime_provenance_stdout(
            payload,
            expected_local_ranks=(0, 1),
            target="cuda",
        )
        process_ids = parsed.execution_process_ids
        duplicate_process_ids = all_worker_process_ids.intersection(process_ids.values())
        if duplicate_process_ids:
            raise TorchrunRuntimeProvenanceError(
                "Rank application reused worker process IDs across generations: "
                f"{sorted(duplicate_process_ids)}"
            )
        if set(process_ids.values()).intersection(bootstrap_process_ids):
            raise TorchrunRuntimeProvenanceError(
                "Rank application reused a generation process ID for a worker"
            )
        all_worker_process_ids.update(process_ids.values())
        covered_provenance_offsets.update(
            item
            for item in provenance_offsets
            if generation_start <= item < generation_end
        )

        runtime_by_local_rank: dict[str, Any] = {}
        for local_rank, snapshot in sorted(parsed.snapshots_by_local_rank.items()):
            if snapshot.cuda_available is not True:
                raise TorchrunRuntimeProvenanceError(
                    "Rank application runtime provenance does not report CUDA available for "
                    f"generation {generation}, local rank {local_rank}"
                )
            manifest = RunManifest.model_construct(runtime_provenance=snapshot)
            if reference_manifest is None:
                reference_manifest = manifest
            try:
                require_runtime_provenance_parity(
                    reference_manifest,
                    manifest,
                    target="cuda",
                )
            except RuntimeProvenanceParityError as exc:
                raise TorchrunRuntimeProvenanceError(
                    "Rank application runtime provenance drifted from generation 0, local "
                    f"rank 0 at generation {generation}, local rank {local_rank}: {exc}"
                ) from exc
            runtime_by_local_rank[str(local_rank)] = snapshot.model_dump(mode="json")

        generations.append(
            {
                "generation": generation,
                "master_port": plan.application_rdzv_port_base + generation,
                "bootstrap_process_ids": {
                    str(rank): begin[(rank, generation)]["pid"] for rank in range(WORLD_SIZE)
                },
                "execution_process_ids": process_ids,
                "runtime_by_local_rank": runtime_by_local_rank,
                "both_post_main_receipts": True,
                "cuda_runtime_parity": True,
            }
        )

    unexpected_provenance = set(provenance_offsets) - covered_provenance_offsets
    if unexpected_provenance:
        raise TorchrunRuntimeProvenanceError(
            "Torchrun runtime provenance appeared outside a rank application generation"
        )
    return generations, {
        "path": str(plan.stdout_path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "generation_marker_count": marker_count,
        "generation_pair_count": len(expected_keys),
    }


def _parse_rank_application_log(
    plan: NcuTorchrunCapturePlan, local_rank: int
) -> dict[str, Any]:
    """Validate one local rank's literal NCU log and replay pass markers."""

    local_rank = _rank(local_rank)
    if not plan.is_rank_application:
        raise ValueError("Rank application log parsing requires rank-tcp topology")
    assert plan.kernel_name is not None
    assert plan.expected_application_passes is not None
    path = plan.rank_log_path(local_rank)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Rank {local_rank} NCU log is missing or symlinked")
    raw = path.read_bytes()
    markers: list[dict[str, Any]] = []
    marker_text = _APPLICATION_MARKER_TEXT.encode("ascii")
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if marker_text not in line:
            continue
        communicator_rank: int | None = None
        normalized = line
        comm_match = _COMM_RANK_PREFIX.fullmatch(line)
        if comm_match is not None:
            communicator_rank = int(comm_match.group("comm_rank"))
            if communicator_rank not in (0, 1):
                raise ValueError(
                    f"Rank {local_rank} NCU log has invalid communicator rank on line {line_number}"
                )
            normalized = b"==PROF==" + comm_match.group("body")
        marker = _APPLICATION_MARKER.fullmatch(normalized + b"\n")
        if marker is None:
            raise ValueError(
                f"Rank {local_rank} NCU log has an unexpected replay marker on line {line_number}"
            )
        kernel_name = marker.group("kernel").decode("utf-8")
        if not _kernel_selector_matches(plan.kernel_name, kernel_name):
            raise ValueError(
                f"Rank {local_rank} NCU log kernel {kernel_name!r} does not match "
                f"selector {plan.kernel_name!r}"
            )
        raw_device = marker.group("device")
        if raw_device is None or int(raw_device) != local_rank:
            raise ValueError(
                f"Rank {local_rank} NCU log selected the wrong visible device"
            )
        markers.append(
            {
                "application_replay_pass": int(marker.group("pass_number")),
                "communicator_rank": communicator_rank,
                "visible_device": int(raw_device),
                "kernel_name": kernel_name,
                "line_number": line_number,
            }
        )
    expected_passes = list(range(1, plan.expected_application_passes + 1))
    observed_passes = [marker["application_replay_pass"] for marker in markers]
    if not plan.all_matching_kernels:
        if observed_passes != expected_passes:
            raise ValueError(
                f"Rank {local_rank} NCU replay passes differ: "
                f"expected {expected_passes}, observed {observed_passes}"
            )
        return {
            "path": str(path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "markers": markers,
        }

    assert plan.expected_kernel_counts is not None
    expected_kernel_count = plan.expected_kernel_counts[local_rank]
    communicator_ranks = [marker["communicator_rank"] for marker in markers]
    if any(communicator_rank is None for communicator_rank in communicator_ranks):
        raise ValueError(
            f"Rank {local_rank} NCU replay marker is missing a communicator rank"
        )
    if len(set(communicator_ranks)) != 1:
        raise ValueError(
            f"Rank {local_rank} NCU communicator rank changed across replay markers"
        )
    if len(markers) == plan.expected_application_passes:
        markers_per_pass = 1
        marker_policy = "one-global-marker-per-pass"
    elif len(markers) == plan.expected_application_passes * expected_kernel_count:
        markers_per_pass = expected_kernel_count
        marker_policy = "one-marker-per-matching-kernel-per-pass"
    else:
        raise ValueError(
            f"Rank {local_rank} NCU replay marker count differs: expected either "
            f"{plan.expected_application_passes} global markers or "
            f"{plan.expected_application_passes * expected_kernel_count} per-kernel "
            f"markers, observed {len(markers)}"
        )

    expected_observed_passes = [
        pass_number
        for pass_number in expected_passes
        for _ in range(markers_per_pass)
    ]
    if observed_passes != expected_observed_passes:
        raise ValueError(
            f"Rank {local_rank} NCU replay pass groups differ: "
            f"expected {expected_observed_passes}, observed {observed_passes}"
        )
    first_kernel_order = [
        marker["kernel_name"] for marker in markers[:markers_per_pass]
    ]
    for pass_index in range(plan.expected_application_passes):
        start = pass_index * markers_per_pass
        observed_kernel_order = [
            marker["kernel_name"]
            for marker in markers[start : start + markers_per_pass]
        ]
        if observed_kernel_order != first_kernel_order:
            raise ValueError(
                f"Rank {local_rank} NCU kernel marker order drifted in replay pass "
                f"{pass_index + 1}"
            )
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "markers": markers,
        "marker_policy": marker_policy,
        "markers_per_pass": markers_per_pass,
        "expected_matching_kernel_count": expected_kernel_count,
        "communicator_rank": communicator_ranks[0],
        "stable_kernel_order": True,
        "kernel_order": first_kernel_order,
    }


def _all_matching_application_rows(
    plan: NcuTorchrunCapturePlan,
    rows: Sequence[Sequence[str]],
    *,
    expected_device: int,
    expected_process_id: int,
    expected_pass_count: int,
    pass_count_source: str,
) -> tuple[dict[str, Any], list[str]]:
    """Validate every native row in one rank's all-matching report."""

    assert plan.is_rank_application
    assert plan.all_matching_kernels
    assert plan.kernel_name is not None
    assert plan.expected_kernel_counts is not None
    expected_row_count = plan.expected_kernel_counts[expected_device]
    artifact: dict[str, Any] = {
        "expected_row_count": expected_row_count,
        "observed_row_count": max(0, len(rows) - 2),
    }
    errors: list[str] = []
    if len(rows) != expected_row_count + 2:
        errors.append(
            "application replay raw CSV row count differs: "
            f"expected {expected_row_count} data rows, observed {max(0, len(rows) - 2)}"
        )
    if len(rows) < 2 or len({len(row) for row in rows}) != 1:
        errors.append(
            "application replay raw CSV must contain one header, one units row, "
            "and equal-width data rows"
        )
        return artifact, errors

    header, units, *data_rows = rows
    if not header or len(set(header)) != len(header):
        errors.append("application replay raw CSV header is empty or contains duplicates")
        return artifact, errors
    required_fields = {
        "ID",
        "Device",
        "Kernel Name",
        "Process ID",
        "profiler__replayer_passes",
        *MINIMAL_METRICS,
    }
    missing_fields = sorted(required_fields - set(header))
    if missing_fields:
        errors.append(f"application replay raw CSV is missing fields: {missing_fields}")
        return artifact, errors

    unit_fields = dict(zip(header, units, strict=True))
    parsed_rows: list[dict[str, Any]] = []
    observed_ids: list[int] = []
    zero_metric_counts = {metric: 0 for metric in MINIMAL_METRICS}
    rows_with_zero_metrics: list[int] = []
    positive_metrics = {
        "gpu__time_duration.avg",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
    }
    percentage_metrics = {
        metric
        for metric in MINIMAL_METRICS
        if metric.endswith(
            ("pct_of_peak_sustained_elapsed", "pct_of_peak_sustained_active")
        )
    }
    geometry_names = (
        "Grid Size",
        "Block Size",
        "Grid X",
        "Grid Y",
        "Grid Z",
        "Block X",
        "Block Y",
        "Block Z",
    )
    for data_index, values in enumerate(data_rows):
        fields = dict(zip(header, values, strict=True))
        row_errors: list[str] = []
        row_id: int | None = None
        observed_device: int | None = None
        process_id: int | None = None
        profiler_pass_count: int | None = None
        try:
            row_id = int(fields["ID"])
            observed_ids.append(row_id)
        except ValueError:
            row_errors.append("ID is not an integer")
        try:
            observed_device = int(fields["Device"])
        except ValueError:
            row_errors.append("Device is not an integer")
        else:
            if observed_device != expected_device:
                row_errors.append(
                    f"Device differs: expected {expected_device}, observed {observed_device}"
                )
        try:
            process_id = int(fields["Process ID"])
        except ValueError:
            row_errors.append("Process ID is not an integer")
        else:
            if process_id != expected_process_id:
                row_errors.append(
                    "Process ID does not match the selected local rank in generation 0"
                )

        device_attribute = fields.get("device__attribute_device_index")
        if device_attribute not in (None, ""):
            try:
                parsed_device_attribute = int(device_attribute)
            except ValueError:
                row_errors.append("device attribute is not an integer")
            else:
                if parsed_device_attribute != expected_device:
                    row_errors.append("device attribute differs from the selector")

        kernel_name = fields["Kernel Name"].strip()
        if not _kernel_selector_matches(plan.kernel_name, kernel_name):
            row_errors.append(
                f"kernel {kernel_name!r} does not match selector {plan.kernel_name!r}"
            )

        try:
            raw_pass_count = float(
                fields["profiler__replayer_passes"].replace(",", "")
            )
        except ValueError:
            row_errors.append("profiler pass count is not numeric")
        else:
            if (
                not math.isfinite(raw_pass_count)
                or raw_pass_count < 1
                or not raw_pass_count.is_integer()
            ):
                row_errors.append("profiler pass count is not a positive integer")
            else:
                profiler_pass_count = int(raw_pass_count)
                if profiler_pass_count != expected_pass_count:
                    row_errors.append(
                        "profiler pass count differs from "
                        f"{pass_count_source}: CSV={profiler_pass_count}, "
                        f"expected={expected_pass_count}"
                    )

        metric_values: dict[str, float] = {}
        zero_metrics: list[str] = []
        for metric in MINIMAL_METRICS:
            try:
                value = float(fields[metric].replace(",", ""))
            except ValueError:
                row_errors.append(f"metric is not numeric: {metric}")
                continue
            if not math.isfinite(value) or value < 0:
                row_errors.append(f"metric is not finite and nonnegative: {metric}")
                continue
            if metric in positive_metrics and value <= 0:
                row_errors.append(f"metric must be positive: {metric}")
                continue
            if metric in percentage_metrics and value > 100:
                row_errors.append(f"percentage metric exceeds 100: {metric}")
                continue
            metric_values[metric] = value
            if value == 0:
                zero_metrics.append(metric)
                zero_metric_counts[metric] += 1
        launch_geometry = {
            name: fields[name]
            for name in geometry_names
            if name in fields and fields[name] != ""
        }
        if zero_metrics:
            rows_with_zero_metrics.append(data_index)
        parsed_rows.append(
            {
                "id": row_id,
                "visible_device": observed_device,
                "kernel_name": kernel_name,
                "process_id": process_id,
                "profiler_pass_count": profiler_pass_count,
                "launch_geometry": launch_geometry,
                "metrics": metric_values,
                "zero_metrics": zero_metrics,
            }
        )
        errors.extend(
            f"application replay raw CSV data row {data_index}: {error}"
            for error in row_errors
        )

    if len(observed_ids) != len(data_rows):
        errors.append("application replay raw CSV does not have an integer ID for every row")
    elif len(set(observed_ids)) != len(observed_ids):
        errors.append("application replay raw CSV IDs contain duplicates")
    elif sorted(observed_ids) != list(range(expected_row_count)):
        errors.append(
            "application replay raw CSV IDs are not contiguous from 0 through "
            f"{expected_row_count - 1}"
        )

    artifact.update(
        rows=parsed_rows,
        metric_units={metric: unit_fields[metric] for metric in MINIMAL_METRICS},
        zero_metric_counts=zero_metric_counts,
        rows_with_zero_metrics=rows_with_zero_metrics,
        zero_metric_row_count=len(rows_with_zero_metrics),
        all_rows_validated=not errors,
    )
    return artifact, errors


def _application_report_artifact_for(
    plan: NcuTorchrunCapturePlan,
    *,
    report: Path,
    csv_path: Path,
    import_stderr_path: Path,
    start_ns: int,
    expected_device: int,
    expected_process_id: int,
    expected_pass_count: int,
    pass_count_source: str,
) -> tuple[dict[str, Any], list[str]]:
    """Import and validate the bounded application-replay report."""

    errors: list[str] = []
    artifact: dict[str, Any] = {
        "report_path": str(report),
        "raw_csv_path": str(csv_path),
        "import_stderr_path": str(import_stderr_path),
    }
    if report.is_symlink() or not report.is_file():
        return artifact, ["application replay report is missing or symlinked"]
    stat = report.stat()
    artifact.update(
        report_bytes=stat.st_size,
        report_mtime_ns=stat.st_mtime_ns,
        report_sha256=_sha256(report),
    )
    if not stat.st_size or stat.st_mtime_ns < start_ns:
        return artifact, ["application replay report is empty or stale"]

    import_argv = (
        str(plan.ncu_path),
        "--import",
        str(report),
        "--page",
        "raw",
        "--csv",
    )
    artifact["import_argv"] = list(import_argv)
    if plan.all_matching_kernels:
        completed_import = _run_all_matching_report_import(
            import_argv,
            timeout_seconds=plan.report_import_timeout_seconds,
        )
        import_stdout = completed_import.stdout
        import_stderr = completed_import.stderr
        csv_path.write_bytes(import_stdout)
        import_stderr_path.write_bytes(import_stderr)
        partial_output = bool(
            completed_import.timed_out
            or completed_import.communication_error is not None
            or completed_import.signals_sent
            or not completed_import.cleanup_drained
        )
        artifact["report_import"] = {
            "argv": list(import_argv),
            "timeout_seconds": plan.report_import_timeout_seconds,
            "process_started": completed_import.started,
            "process_id": completed_import.process_id,
            "returncode": completed_import.returncode,
            "timed_out": completed_import.timed_out,
            "wall_seconds": completed_import.wall_seconds,
            "start_error": completed_import.start_error,
            "communication_error": completed_import.communication_error,
            "stdout": {
                "path": str(csv_path),
                "bytes": len(import_stdout),
                "sha256": hashlib.sha256(import_stdout).hexdigest(),
                "partial": partial_output,
            },
            "stderr": {
                "path": str(import_stderr_path),
                "bytes": len(import_stderr),
                "sha256": hashlib.sha256(import_stderr).hexdigest(),
                "partial": partial_output,
            },
            "cleanup": {
                "mode": "dedicated-process-group",
                "natural": (
                    completed_import.started
                    and not completed_import.signals_sent
                    and completed_import.cleanup_drained
                ),
                "forced": bool(completed_import.signals_sent),
                "signals_sent": list(completed_import.signals_sent),
                "drained": completed_import.cleanup_drained,
                "errors": list(completed_import.cleanup_errors),
            },
        }
        import_errors: list[str] = []
        if completed_import.start_error is not None:
            import_errors.append(
                "application replay report import failed to start: "
                f"{completed_import.start_error}"
            )
        if completed_import.communication_error is not None:
            import_errors.append(
                "application replay report import communication failed: "
                f"{completed_import.communication_error}"
            )
        if completed_import.timed_out:
            import_errors.append(
                "application replay report import timed out after "
                f"{plan.report_import_timeout_seconds} seconds"
            )
        if completed_import.signals_sent and not completed_import.timed_out:
            import_errors.append(
                "application replay report import required process-group cleanup"
            )
        if not completed_import.cleanup_drained or completed_import.cleanup_errors:
            import_errors.append(
                "application replay report import process-group cleanup failed: "
                f"{list(completed_import.cleanup_errors)}"
            )
        if (
            completed_import.started
            and not completed_import.timed_out
            and completed_import.returncode != 0
        ):
            import_errors.append(
                "application replay report import exited "
                f"{completed_import.returncode}"
            )
        import_returncode = completed_import.returncode
    else:
        try:
            completed = subprocess.run(
                import_argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=min(60.0, plan.timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return artifact, ["application replay report import timed out"]
        except (OSError, subprocess.SubprocessError) as exc:
            return artifact, [f"application replay report import failed: {exc}"]
        import_stdout = completed.stdout
        import_stderr = completed.stderr
        import_returncode = completed.returncode
        import_errors = []
        csv_path.write_bytes(import_stdout)
        import_stderr_path.write_bytes(import_stderr)
    artifact.update(
        import_returncode=import_returncode,
        raw_csv_bytes=len(import_stdout),
        raw_csv_sha256=hashlib.sha256(import_stdout).hexdigest(),
        import_stderr_bytes=len(import_stderr),
        import_stderr_sha256=hashlib.sha256(import_stderr).hexdigest(),
    )
    if import_errors:
        return artifact, import_errors
    if import_returncode:
        return artifact, [f"application replay report import exited {import_returncode}"]
    if _sha256(report) != artifact["report_sha256"]:
        return artifact, ["application replay report changed while it was imported"]
    try:
        csv_text = import_stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        return artifact, [f"application replay raw CSV is not valid UTF-8: {exc}"]
    rows = list(csv.reader(io.StringIO(csv_text)))
    if plan.all_matching_kernels:
        row_artifact, row_errors = _all_matching_application_rows(
            plan,
            rows,
            expected_device=expected_device,
            expected_process_id=expected_process_id,
            expected_pass_count=expected_pass_count,
            pass_count_source=pass_count_source,
        )
        artifact.update(row_artifact)
        return artifact, row_errors
    if len(rows) != 3 or len({len(row) for row in rows}) != 1:
        return artifact, [
            "application replay raw CSV must contain one header, one units row, "
            "and one data row"
        ]
    header, units, values = rows
    if not header or len(set(header)) != len(header):
        return artifact, ["application replay raw CSV header is empty or contains duplicates"]
    fields = dict(zip(header, values, strict=True))
    unit_fields = dict(zip(header, units, strict=True))
    required_fields = {
        "Device",
        "Kernel Name",
        "Process ID",
        "profiler__replayer_passes",
        *MINIMAL_METRICS,
    }
    missing_fields = sorted(required_fields - fields.keys())
    if missing_fields:
        return artifact, [
            f"application replay raw CSV is missing fields: {missing_fields}"
        ]
    assert plan.kernel_name is not None
    try:
        observed_device = int(fields["Device"])
        process_id = int(fields["Process ID"])
        raw_pass_count = float(fields["profiler__replayer_passes"].replace(",", ""))
    except ValueError as exc:
        return artifact, [f"application replay raw CSV identity is invalid: {exc}"]
    if observed_device != expected_device:
        errors.append(
            "application replay raw CSV selected the wrong visible device: "
            f"expected {expected_device}, observed {observed_device}"
        )
    device_attribute = fields.get("device__attribute_device_index")
    if device_attribute not in (None, ""):
        try:
            if int(device_attribute) != expected_device:
                errors.append("application replay device attribute differs from the selector")
        except ValueError:
            errors.append("application replay device attribute is not an integer")
    kernel_name = fields["Kernel Name"].strip()
    if not _kernel_selector_matches(plan.kernel_name, kernel_name):
        errors.append(
            f"application replay CSV kernel {kernel_name!r} does not match "
            f"selector {plan.kernel_name!r}"
        )
    if not math.isfinite(raw_pass_count) or raw_pass_count < 1 or not raw_pass_count.is_integer():
        errors.append("application replay profiler pass count is not a positive integer")
        profiler_pass_count = None
    else:
        profiler_pass_count = int(raw_pass_count)
        if profiler_pass_count != expected_pass_count:
            errors.append(
                "application replay profiler pass count differs from "
                f"{pass_count_source}: CSV={profiler_pass_count}, "
                f"expected={expected_pass_count}"
            )
    if process_id != expected_process_id:
        errors.append(
            "application replay CSV process does not match the selected local rank in pass 1"
        )

    metric_values: dict[str, float] = {}
    for metric in MINIMAL_METRICS:
        try:
            value = float(fields[metric].replace(",", ""))
        except ValueError:
            errors.append(f"application replay metric is not numeric: {metric}")
            continue
        if not math.isfinite(value) or value <= 0:
            errors.append(f"application replay metric is not finite and positive: {metric}")
            continue
        if metric.endswith(("pct_of_peak_sustained_elapsed", "pct_of_peak_sustained_active")):
            if value > 100:
                errors.append(f"application replay percentage metric exceeds 100: {metric}")
                continue
        metric_values[metric] = value

    artifact["row"] = {
        "visible_device": observed_device,
        "kernel_name": kernel_name,
        "process_id": process_id,
        "profiler_pass_count": profiler_pass_count,
        "metrics": metric_values,
        "metric_units": {metric: unit_fields[metric] for metric in MINIMAL_METRICS},
    }
    return artifact, errors


def _application_report_artifact(
    plan: NcuTorchrunCapturePlan,
    *,
    start_ns: int,
    generations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    assert plan.target_visible_device is not None
    return _application_report_artifact_for(
        plan,
        report=plan.application_report_path,
        csv_path=plan.application_csv_path,
        import_stderr_path=plan.application_import_stderr_path,
        start_ns=start_ns,
        expected_device=plan.target_visible_device,
        expected_process_id=generations[0]["execution_process_ids"][
            plan.target_visible_device
        ],
        expected_pass_count=len(generations),
        pass_count_source="the merged log",
    )


def _rank_application_evidence(
    plan: NcuTorchrunCapturePlan, *, start_ns: int, allow_report_import: bool
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    bool,
    list[str],
]:
    """Validate paired generations, rank logs, and both native reports."""

    assert plan.is_rank_application
    assert plan.expected_application_passes is not None
    rank_artifacts, errors = _artifacts(plan, start_ns)
    generations: list[dict[str, Any]] = []
    stdout_artifact: dict[str, Any] = {"path": str(plan.stdout_path)}
    try:
        generations, stdout_artifact = _parse_rank_application_generations(plan)
    except (OSError, UnicodeError, ValueError, TorchrunRuntimeProvenanceError) as exc:
        errors.append(f"rank application generation validation failed: {exc}")

    for local_rank, artifact in enumerate(rank_artifacts):
        try:
            artifact["ncu_log"] = _parse_rank_application_log(plan, local_rank)
        except (OSError, UnicodeError, ValueError) as exc:
            artifact["ncu_log"] = {"path": str(plan.rank_log_path(local_rank))}
            errors.append(f"rank {local_rank} application replay log validation failed: {exc}")

        counter_path = plan.output_dir / f"rank-{local_rank}-generation.counter"
        counter_artifact: dict[str, Any] = {"path": str(counter_path)}
        if counter_path.is_symlink() or not counter_path.is_file():
            errors.append(f"rank {local_rank} generation counter is missing or symlinked")
        else:
            counter_raw = counter_path.read_bytes()
            counter_stat = counter_path.stat()
            counter_artifact.update(
                bytes=len(counter_raw),
                sha256=hashlib.sha256(counter_raw).hexdigest(),
                mtime_ns=counter_stat.st_mtime_ns,
            )
            expected_counter = f"{plan.expected_application_passes}\n".encode("ascii")
            if counter_raw != expected_counter or counter_stat.st_mtime_ns < start_ns:
                errors.append(
                    f"rank {local_rank} generation counter is stale or differs from "
                    f"{plan.expected_application_passes}"
                )
        artifact["generation_counter"] = counter_artifact

    import_reports = (
        allow_report_import
        and not errors
        and len(generations) == plan.expected_application_passes
    )
    if import_reports:
        for local_rank, artifact in enumerate(rank_artifacts):
            try:
                report_artifact, report_errors = _application_report_artifact_for(
                    plan,
                    report=plan.report_path(local_rank),
                    csv_path=plan.rank_csv_path(local_rank),
                    import_stderr_path=plan.rank_import_stderr_path(local_rank),
                    start_ns=start_ns,
                    expected_device=local_rank,
                    expected_process_id=generations[0]["execution_process_ids"][local_rank],
                    expected_pass_count=plan.expected_application_passes,
                    pass_count_source="the expected rank application generations",
                )
                artifact["application_report"] = report_artifact
                errors.extend(f"rank {local_rank}: {error}" for error in report_errors)
            except (OSError, UnicodeError, ValueError) as exc:
                artifact["application_report"] = {
                    "report_path": str(plan.report_path(local_rank)),
                    "raw_csv_path": str(plan.rank_csv_path(local_rank)),
                }
                errors.append(
                    f"rank {local_rank} application report validation failed: {exc}"
                )
    else:
        errors.append(
            "rank application reports were not imported because capture completion or "
            "pre-import evidence was invalid"
        )
    return rank_artifacts, generations, stdout_artifact, import_reports, errors


def _capture_status(
    *,
    cleanup: _CleanupResult,
    returncode: int | None,
    timed_out: bool,
    artifact_errors: Sequence[str],
) -> str:
    if not cleanup.supported:
        return "CLEANUP_UNVERIFIED"
    if not cleanup.drained or returncode is None:
        return "CLEANUP_FAILED"
    if timed_out:
        return "TIMEOUT"
    if returncode:
        return "CAPTURE_FAILED"
    if not cleanup.natural:
        return "FORCED_CLEANUP"
    if artifact_errors:
        return "CAPTURE_REJECTED"
    return "CAPTURE_ARTIFACTS_PRESENT"


def _run_application_capture(
    plan: NcuTorchrunCapturePlan,
    *,
    env: dict[str, str],
    preflight: Mapping[str, Any],
) -> NcuTorchrunCaptureResult:
    plan.output_dir.mkdir(parents=True, exist_ok=False)
    capture_marker = uuid.uuid4().hex
    env.update(
        PYTHONNOUSERSITE="1",
        AISP_PROFILE_NO_USER_SITE="1",
        **{_CAPTURE_OWNER_ENV: capture_marker},
    )
    assert plan.target_visible_device is not None
    assert plan.kernel_name is not None
    assert plan.launch_skip is not None
    assert plan.launch_count == 1
    receipt: dict[str, Any] = {
        "schema": APPLICATION_CAPTURE_RECEIPT_SCHEMA,
        "status": "RUNNING",
        "label": plan.label,
        "source": plan.source,
        "cwd": str((plan.repo_root / "code").resolve()),
        "capture_topology": "parent-application-replay",
        "torchrun_argv": list(plan.torchrun_argv),
        "argv": list(plan.launch_argv),
        "world_size": WORLD_SIZE,
        "nnodes": 1,
        "replay_mode": "application",
        "selector": {
            "target_visible_device": plan.target_visible_device,
            "kernel_name": plan.kernel_name,
            "launch_skip": plan.launch_skip,
            "launch_count": plan.launch_count,
        },
        "application_replay": {
            "mode": "strict",
            "match": "grid",
            "filter_mode": "per-gpu",
        },
        "metrics": list(MINIMAL_METRICS),
        "selected_matching_launches_per_device": 1,
        "all_matching_launches_covered": False,
        "full_coverage_qualified": False,
        "both_devices_covered_by_this_receipt": False,
        "pipeline_role_inferred": False,
        "numerical_output_qualified": False,
        "performance_qualified": False,
        "claim_limit": CLAIM_LIMIT,
        "cleanup_limit": CLEANUP_LIMIT,
        "capture_owner": {"environment": _CAPTURE_OWNER_ENV, "id": capture_marker},
        "log_stream": {
            "mode": "stdout-and-stderr-shared-open-file-description-at-launch",
            "path": str(plan.combined_path),
        },
        **preflight,
    }
    _write_json(plan.receipt_path, receipt)
    preexisting_identities = (
        _snapshot_proc_identities() if _proc_cleanup_supported() else frozenset()
    )
    process: subprocess.Popen[bytes] | None = None
    cleanup: _CleanupResult | None = None
    start_ns = time.time_ns()
    timed_out = False

    try:
        with plan.combined_path.open("wb") as combined:
            process = subprocess.Popen(
                plan.launch_argv,
                cwd=plan.repo_root / "code",
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=combined,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            receipt["process_id"] = process.pid
            _write_json(plan.receipt_path, receipt)
            try:
                process.wait(timeout=plan.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
            cleanup = _cleanup_capture_processes(
                process,
                capture_marker,
                preexisting_identities=preexisting_identities,
            )
    except BaseException as exc:
        if process is not None and cleanup is None:
            cleanup = _cleanup_capture_processes(
                process,
                capture_marker,
                preexisting_identities=preexisting_identities,
            )
        cleanup = cleanup or _CleanupResult(
            "not-started", True, True, True, (), (), (), (), ()
        )
        receipt.update(
            status="LAUNCH_FAILED",
            error=f"{type(exc).__name__}: {exc}",
            timed_out=timed_out,
            cleanup=cleanup.receipt(),
            wall_seconds=(time.time_ns() - start_ns) / 1e9,
        )
        _write_json(plan.receipt_path, receipt)
        raise

    assert process is not None
    assert cleanup is not None
    application_errors: list[str] = []
    generations: list[dict[str, Any]] = []
    combined_artifact: dict[str, Any] = {
        "path": str(plan.combined_path),
        "stream_mode": "stdout-and-stderr-shared-open-file-description-at-launch",
    }
    try:
        generations, combined_artifact = _parse_application_replay_log(plan)
    except (OSError, UnicodeError, ValueError, TorchrunRuntimeProvenanceError) as exc:
        application_errors.append(f"application replay log validation failed: {exc}")

    application_artifact: dict[str, Any] = {
        "report_path": str(plan.application_report_path),
        "raw_csv_path": str(plan.application_csv_path),
    }
    if generations:
        try:
            application_artifact, report_errors = _application_report_artifact(
                plan,
                start_ns=start_ns,
                generations=generations,
            )
            application_errors.extend(report_errors)
        except (OSError, UnicodeError, ValueError) as exc:
            application_errors.append(
                f"application replay report validation failed: {exc}"
            )
    else:
        application_errors.append(
            "application replay report was not imported because no complete pass generation exists"
        )
    status = _capture_status(
        cleanup=cleanup,
        returncode=process.returncode,
        timed_out=timed_out,
        artifact_errors=application_errors,
    )
    receipt.update(
        status=status,
        returncode=process.returncode,
        timed_out=timed_out,
        cleanup=cleanup.receipt(),
        wall_seconds=(time.time_ns() - start_ns) / 1e9,
        combined_log=combined_artifact,
        profiler_pass_count=len(generations),
        replay_generations=generations,
        application_artifact=application_artifact,
        artifact_errors=application_errors,
    )
    _write_json(plan.receipt_path, receipt)
    return NcuTorchrunCaptureResult(
        status,
        process.returncode,
        timed_out,
        plan.receipt_path,
        plan.combined_path,
        plan.combined_path,
    )


def run_capture(
    plan: NcuTorchrunCapturePlan, *, environment: Mapping[str, str] | None = None
) -> NcuTorchrunCaptureResult:
    """Execute a plan, drain its owned process group, and retain a receipt."""

    env = dict(os.environ if environment is None else environment)
    preflight = _preflight(plan, env)
    if plan.replay_mode == "application" and not plan.is_rank_application:
        return _run_application_capture(plan, env=env, preflight=preflight)
    plan.output_dir.mkdir(parents=True, exist_ok=False)
    capture_marker = uuid.uuid4().hex
    env.update(
        PYTHONNOUSERSITE="1",
        AISP_PROFILE_NO_USER_SITE="1",
        **{_CAPTURE_OWNER_ENV: capture_marker},
    )
    if plan.is_rank_application:
        _write_json(plan.generation_config_path, plan.generation_config())
    _write_json(plan.rank_config_path, plan.rank_config())
    if plan.is_rank_application:
        assert plan.kernel_name is not None
        if plan.all_matching_kernels:
            assert plan.launch_skip == 0
            assert plan.launch_count is None
            assert plan.expected_kernel_counts is not None
        else:
            assert plan.launch_skip is not None
            assert plan.launch_count == 1
        assert plan.expected_application_passes is not None
        assert plan.application_rdzv_port_base is not None
        receipt = {
            "schema": RANK_APPLICATION_CAPTURE_RECEIPT_SCHEMA,
            "status": "RUNNING",
            "label": plan.label,
            "source": plan.source,
            "cwd": str((plan.repo_root / "code").resolve()),
            "capture_topology": "per-rank-tcp-application-replay",
            "torchrun_argv": list(plan.torchrun_argv),
            "argv": list(plan.launch_argv),
            "rank_config_path": str(plan.rank_config_path),
            "rank_config_sha256": _sha256(plan.rank_config_path),
            "generation_config_path": str(plan.generation_config_path),
            "generation_config_sha256": _sha256(plan.generation_config_path),
            "world_size": WORLD_SIZE,
            "nnodes": 1,
            "tcp_communicator": {
                "hostname": "127.0.0.1",
                "port": plan.tcp_port,
                "num_peers": WORLD_SIZE,
                "lockstep_kernel_launch": True,
            },
            "replay_mode": "application",
            "application_topology": "rank-tcp",
            "selector": {
                "kernel_name": plan.kernel_name,
                "launch_skip": plan.launch_skip,
                "launch_count": plan.launch_count,
            },
            "application_replay": {
                "mode": "strict",
                "match": "grid",
                "filter_mode": "per-gpu",
                "expected_passes": plan.expected_application_passes,
                "fresh_store_ports": list(
                    range(
                        plan.application_rdzv_port_base,
                        plan.application_rdzv_port_base
                        + plan.expected_application_passes,
                    )
                ),
            },
            "metrics": list(MINIMAL_METRICS),
            "requested_visible_devices": [0, 1],
            "selected_matching_launches_per_device": 1,
            "all_matching_launches_covered": False,
            "full_coverage_qualified": False,
            "both_devices_covered_by_this_receipt": False,
            "pipeline_role_inferred": False,
            "numerical_output_qualified": False,
            "performance_qualified": False,
            "claim_limit": CLAIM_LIMIT,
            "cleanup_limit": CLEANUP_LIMIT,
            "capture_owner": {"environment": _CAPTURE_OWNER_ENV, "id": capture_marker},
            **preflight,
        }
        if plan.all_matching_kernels:
            receipt["schema"] = RANK_APPLICATION_ALL_MATCHING_CAPTURE_RECEIPT_SCHEMA
            receipt.pop("selected_matching_launches_per_device")
            receipt.update(
                capture_scope="all-matching-kernels-requested",
                all_matching_kernels=True,
                nvtx_capture_enabled=True,
                expected_kernel_counts=list(plan.expected_kernel_counts),
                report_import_timeout_seconds=(
                    plan.report_import_timeout_seconds
                ),
                ncu_launch_count_argument_present=(
                    "--launch-count" in plan.ncu_prefix
                ),
                all_matching_ncu_rows_validated=False,
            )
    else:
        receipt = {
            "schema": CAPTURE_RECEIPT_SCHEMA,
            "status": "RUNNING",
            "label": plan.label,
            "source": plan.source,
            "cwd": str((plan.repo_root / "code").resolve()),
            "capture_topology": "per-rank-tcp-lockstep",
            "torchrun_argv": list(plan.torchrun_argv),
            "argv": list(plan.launch_argv),
            "rank_config_path": str(plan.rank_config_path),
            "rank_config_sha256": _sha256(plan.rank_config_path),
            "world_size": WORLD_SIZE,
            "nnodes": 1,
            "tcp_communicator": {
                "hostname": "127.0.0.1",
                "port": plan.tcp_port,
                "num_peers": 2,
            },
            "nccl_nvtx_includes": list(plan.nccl_nvtx_includes),
            "replay_mode": plan.replay_mode,
            "all_matching_kernels": plan.all_matching_kernels,
            "metrics": list(MINIMAL_METRICS),
            "claim_limit": CLAIM_LIMIT,
            "cleanup_limit": CLEANUP_LIMIT,
            "capture_owner": {"environment": _CAPTURE_OWNER_ENV, "id": capture_marker},
            **preflight,
        }
    _write_json(plan.receipt_path, receipt)
    preexisting_identities = (
        _snapshot_proc_identities() if _proc_cleanup_supported() else frozenset()
    )
    start_ns, process = time.time_ns(), None
    timed_out = False
    try:
        with plan.stdout_path.open("wb") as stdout, plan.stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                plan.launch_argv,
                cwd=plan.repo_root / "code",
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            receipt["process_id"] = process.pid
            _write_json(plan.receipt_path, receipt)
            try:
                process.wait(timeout=plan.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
    except BaseException as exc:
        cleanup = (
            _cleanup_capture_processes(
                process,
                capture_marker,
                preexisting_identities=preexisting_identities,
            )
            if process
            else _CleanupResult("not-started", True, True, True, (), (), (), (), ())
        )
        receipt.update(
            status="LAUNCH_FAILED",
            error=f"{type(exc).__name__}: {exc}",
            timed_out=timed_out,
            cleanup=cleanup.receipt(),
            wall_seconds=(time.time_ns() - start_ns) / 1e9,
        )
        _write_json(plan.receipt_path, receipt)
        raise
    assert process is not None
    cleanup = _cleanup_capture_processes(
        process,
        capture_marker,
        preexisting_identities=preexisting_identities,
    )
    if plan.is_rank_application:
        artifacts, generations, stdout_artifact, imports_attempted, artifact_errors = (
            _rank_application_evidence(
                plan,
                start_ns=start_ns,
                allow_report_import=(
                    process.returncode == 0
                    and not timed_out
                    and cleanup.supported
                    and cleanup.natural
                    and cleanup.drained
                ),
            )
        )
    else:
        artifacts, artifact_errors = _artifacts(plan, start_ns)
        generations = []
        stdout_artifact = {"path": str(plan.stdout_path)}
        imports_attempted = False
    status = _capture_status(
        cleanup=cleanup,
        returncode=process.returncode,
        timed_out=timed_out,
        artifact_errors=artifact_errors,
    )
    receipt.update(
        status=status,
        returncode=process.returncode,
        timed_out=timed_out,
        cleanup=cleanup.receipt(),
        wall_seconds=(time.time_ns() - start_ns) / 1e9,
        stdout_path=str(plan.stdout_path),
        stderr_path=str(plan.stderr_path),
        rank_artifacts=artifacts,
        artifact_errors=artifact_errors,
    )
    if plan.is_rank_application:
        stderr_raw = plan.stderr_path.read_bytes()
        receipt.update(
            stdout=stdout_artifact,
            stderr={
                "path": str(plan.stderr_path),
                "bytes": len(stderr_raw),
                "sha256": hashlib.sha256(stderr_raw).hexdigest(),
            },
            replay_generation_count=len(generations),
            replay_generations=generations,
            native_report_imports_attempted=imports_attempted,
            native_reports_validated=imports_attempted and not artifact_errors,
            both_devices_covered_by_this_receipt=(
                imports_attempted and not artifact_errors
            ),
            profiler_pass_count_by_rank={
                str(artifact["local_rank"]): artifact.get("application_report", {}).get(
                    "row", {}
                ).get("profiler_pass_count")
                for artifact in artifacts
            },
        )
        if plan.all_matching_kernels:
            receipt.update(
                all_matching_ncu_rows_validated=(
                    imports_attempted and not artifact_errors
                ),
                observed_kernel_counts=[
                    artifact.get("application_report", {}).get(
                        "observed_row_count"
                    )
                    for artifact in artifacts
                ],
                profiler_pass_count_by_rank={
                    str(artifact["local_rank"]): (
                        artifact.get("application_report", {}).get("rows", [{}])[0].get(
                            "profiler_pass_count"
                        )
                        if artifact.get("application_report", {}).get("rows")
                        else None
                    )
                    for artifact in artifacts
                },
                log_marker_policy_by_rank={
                    str(artifact["local_rank"]): artifact.get("ncu_log", {}).get(
                        "marker_policy"
                    )
                    for artifact in artifacts
                },
                zero_metric_counts_by_rank={
                    str(artifact["local_rank"]): artifact.get(
                        "application_report", {}
                    ).get("zero_metric_counts")
                    for artifact in artifacts
                },
            )
    _write_json(plan.receipt_path, receipt)
    return NcuTorchrunCaptureResult(
        status,
        process.returncode,
        timed_out,
        plan.receipt_path,
        plan.stdout_path,
        plan.stderr_path,
    )


def _args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Two-rank NCU capture with TCP/lockstep kernel or range replay, or bounded "
            "parent or rank-tcp application replay. Pass a torchrun-wrapper command after --."
        )
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ncu", required=True, type=Path)
    parser.add_argument("--tcp-port", type=int)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    parser.add_argument("--all-matching-kernels", action="store_true")
    parser.add_argument(
        "--replay-mode", choices=("application", "kernel", "range"), default="kernel"
    )
    parser.add_argument(
        "--application-topology", choices=("parent", "rank-tcp"), default="parent"
    )
    parser.add_argument("--expected-application-passes", type=int)
    parser.add_argument("--application-rdzv-port-base", type=int)
    parser.add_argument(
        "--expected-kernel-counts",
        nargs=WORLD_SIZE,
        type=_positive_kernel_count,
        metavar=("RANK0", "RANK1"),
    )
    parser.add_argument(
        "--report-import-timeout-seconds",
        type=float,
        default=60.0,
        help=(
            "Per-report import timeout for all-matching rank-tcp application replay. "
            "The default is 60 seconds."
        ),
    )
    parser.add_argument("--nvtx-include", action="append", dest="includes")
    parser.add_argument("--target-visible-device", type=int)
    parser.add_argument("--kernel-name")
    parser.add_argument("--launch-skip", type=int)
    parser.add_argument("--launch-count", type=int)
    parser.add_argument("torchrun_argv", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.torchrun_argv[:1] == ["--"]:
        args.torchrun_argv.pop(0)
    if not args.torchrun_argv:
        parser.error("a torchrun-wrapper command is required after --")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _args(argv)
    try:
        plan = build_capture_plan(
            label=args.label,
            repo_root=args.repo_root,
            source=args.source,
            output_dir=args.output_dir,
            ncu_path=args.ncu,
            tcp_port=args.tcp_port,
            timeout_seconds=args.timeout_seconds,
            nccl_nvtx_includes=args.includes,
            torchrun_argv=args.torchrun_argv,
            all_matching_kernels=args.all_matching_kernels,
            replay_mode=args.replay_mode,
            target_visible_device=args.target_visible_device,
            kernel_name=args.kernel_name,
            launch_skip=args.launch_skip,
            launch_count=args.launch_count,
            application_topology=args.application_topology,
            expected_application_passes=args.expected_application_passes,
            application_rdzv_port_base=args.application_rdzv_port_base,
            expected_kernel_counts=(
                tuple(args.expected_kernel_counts)
                if args.expected_kernel_counts is not None
                else None
            ),
            report_import_timeout_seconds=args.report_import_timeout_seconds,
        )
        result = run_capture(plan)
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"ncu-torchrun capture error: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": result.status,
                "returncode": result.returncode,
                "timed_out": result.timed_out,
                "receipt": str(result.receipt_path),
            },
            sort_keys=True,
        )
    )
    return 0 if result.status == "CAPTURE_ARTIFACTS_PRESENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
