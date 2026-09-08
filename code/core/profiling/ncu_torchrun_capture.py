"""Explicit two-rank TCP/lockstep Nsight Compute capture for torchrun.

Supported scope: one node, two ranks, NCU 2026.2.1, kernel replay, five minimal
metrics, ``regex:nccl.*``, and named NCCL push/pop NVTX filters. Receipts prove
source/argv custody and fresh rank artifacts, not counter or performance validity.
Full process-tree cleanup proof requires Linux procfs and the inherited owner
marker; unsupported hosts can run only to an explicitly rejected receipt.
"""

from __future__ import annotations

import argparse
import hashlib
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

from core.profiling.ncu_torchrun_rank import RANK_CONFIG_SCHEMA
from core.profiling.profiler_config import MINIMAL_METRICS

CAPTURE_RECEIPT_SCHEMA = "aisp.ncu-torchrun-capture.v1"
SUPPORTED_NCU_VERSION = "2026.2.1"
WORLD_SIZE = 2
CLAIM_LIMIT = (
    "Artifact presence, hashes, and a clean process return do not validate counter values, "
    "kernel/range attribution, workload correctness, or performance."
)
CLEANUP_LIMIT = (
    "Full-tree drainage is proven only on Linux procfs for processes retaining this "
    "launch's inherited owner marker; clearing that marker is unsupported."
)
_NCCL_RANGE = re.compile(r"^NCCL@[A-Za-z][A-Za-z0-9_.:-]*/$")
_RANK_MODULE = "core.profiling.ncu_torchrun_rank"
_TORCHRUN_MODULE = "torch.distributed.run"
_WRAPPER_MODULE = "core.harness.torchrun_wrapper"
_CAPTURE_OWNER_ENV = "AISP_NCU_TORCHRUN_CAPTURE_ID"
_PROC_ROOT = Path("/proc")


def _rank(value: int) -> int:
    if type(value) is not int or value not in (0, 1):
        raise ValueError(f"local_rank must be 0 or 1, got {value!r}")
    return value


@dataclass(frozen=True)
class NcuTorchrunCapturePlan:
    """Inspectable launch plan used by the CLI and external validators."""

    label: str
    repo_root: Path
    source: str
    output_dir: Path
    ncu_path: Path
    tcp_port: int
    timeout_seconds: float
    nccl_nvtx_includes: tuple[str, ...]
    torchrun_argv: tuple[str, ...]
    launcher_argv: tuple[str, ...]
    worker_argv: tuple[str, ...]
    ncu_prefix: tuple[str, ...]
    all_matching_kernels: bool

    @property
    def rank_config_path(self) -> Path:
        return self.output_dir / "rank-config.json"

    @property
    def receipt_path(self) -> Path:
        return self.output_dir / "receipt.json"

    @property
    def stdout_path(self) -> Path:
        return self.output_dir / "stdout.log"

    @property
    def stderr_path(self) -> Path:
        return self.output_dir / "stderr.log"

    def report_path(self, local_rank: int) -> Path:
        return self.output_dir / f"rank-{_rank(local_rank)}.ncu-rep"

    def rank_argv_path(self, local_rank: int) -> Path:
        return self.output_dir / f"rank-{_rank(local_rank)}-argv.json"

    def expected_rank_argv(self, local_rank: int) -> tuple[str, ...]:
        local_rank = _rank(local_rank)
        return (
            *self.ncu_prefix,
            "--devices",
            str(local_rank),
            "--export",
            str(self.report_path(local_rank)),
            *self.worker_argv,
        )

    @property
    def launch_argv(self) -> tuple[str, ...]:
        return (
            *self.launcher_argv,
            "--no-python",
            self.torchrun_argv[0],
            "-m",
            _RANK_MODULE,
            str(self.rank_config_path),
        )

    def rank_config(self) -> dict[str, Any]:
        return {
            "schema": RANK_CONFIG_SCHEMA,
            "ncu_prefix": list(self.ncu_prefix),
            "worker_argv": list(self.worker_argv),
            "output_dir": str(self.output_dir),
            "world_size": WORLD_SIZE,
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
        raise ValueError(f"Per-rank TCP NCU requires --nproc_per_node 2, got {nproc!r}")
    nnodes = _option(launcher, "--nnodes", required=False)
    if nnodes not in (None, "1"):
        raise ValueError(f"Per-rank TCP NCU supports exactly one node, got {nnodes!r}")
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
            f"'NCCL@ncclGroupEnd/'; invalid={invalid}"
        )
    return includes


def build_capture_plan(
    *,
    label: str,
    repo_root: Path,
    source: str,
    output_dir: Path,
    ncu_path: Path,
    tcp_port: int,
    timeout_seconds: float,
    nccl_nvtx_includes: Sequence[str],
    torchrun_argv: Sequence[str],
    all_matching_kernels: bool = False,
) -> NcuTorchrunCapturePlan:
    """Validate the supported command contract and build exact rank argv."""

    label, source = str(label).strip(), str(source).strip().lower()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", label):
        raise ValueError("label must contain only letters, digits, '.', '_' or '-'")
    if not re.fullmatch(r"[0-9a-f]{40}", source):
        raise ValueError("source must be a full 40-character lowercase Git commit")
    if type(tcp_port) is not int or not 1024 <= tcp_port <= 65535:
        raise ValueError(f"tcp_port must be an integer from 1024 through 65535: {tcp_port!r}")
    if (
        type(timeout_seconds) not in (int, float)
        or not math.isfinite(float(timeout_seconds))
        or timeout_seconds <= 0
    ):
        raise ValueError(f"timeout_seconds must be positive and finite: {timeout_seconds!r}")
    if type(all_matching_kernels) is not bool:
        raise ValueError("all_matching_kernels must be a boolean")
    launcher, worker = _split_torchrun(torchrun_argv)
    if str(_option(launcher, "--rdzv_endpoint", "--rdzv-endpoint")).endswith(f":{tcp_port}"):
        raise ValueError("NCU communicator port must differ from the torchrun rendezvous port")
    includes, metrics = _ranges(nccl_nvtx_includes), tuple(MINIMAL_METRICS)
    if len(metrics) != 5 or len(set(metrics)) != 5:
        raise RuntimeError("Repository minimal NCU contract must contain exactly five metrics")
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
    prefix.extend(["--kernel-name", "regex:nccl.*"])
    if not all_matching_kernels:
        prefix.extend(["--launch-count", "1"])
    prefix.extend(["--replay-mode", "kernel", "--metrics", ",".join(metrics)])
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
    for name in ("AISP_LOCK_GPU_CLOCKS", "AISP_RAMP_GPU_CLOCKS"):
        if env.get(name) != "1":
            raise ValueError(f"{name}=1 is required")
    code_root = (plan.repo_root / "code").resolve()
    python_paths = {
        str(Path(item).resolve()) for item in env.get("PYTHONPATH", "").split(os.pathsep) if item
    }
    if str(code_root) not in python_paths:
        raise ValueError(f"PYTHONPATH must include {code_root}")
    with socket.socket() as listener:
        try:
            listener.bind(("127.0.0.1", plan.tcp_port))
        except OSError as exc:
            raise ValueError(f"NCU TCP port is unavailable: {plan.tcp_port}") from exc
    version = subprocess.check_output(
        [str(ncu), "--version"], text=True, stderr=subprocess.STDOUT, timeout=15
    ).strip()
    if f"Version {SUPPORTED_NCU_VERSION}" not in version:
        raise ValueError(f"Qualified only for NCU {SUPPORTED_NCU_VERSION}; observed {version!r}")
    return {
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


def _proc_cleanup_supported(proc_root: Path = _PROC_ROOT) -> bool:
    """Return whether inherited owner markers can be inspected on this host."""

    return sys.platform.startswith("linux") and proc_root.is_dir()


def _marked_pids(marker: str, proc_root: Path = _PROC_ROOT) -> tuple[set[int], str | None]:
    """Find live processes carrying this launch's exact inherited marker."""

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
            try:
                same_user = entry.stat().st_uid == os.geteuid()
            except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
                same_user = False
            if same_user:
                return (
                    found,
                    f"process-tree inspection failed for same-user pid {entry.name}: {exc}",
                )
            continue
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
) -> tuple[set[int], set[int], str | None]:
    observed: set[int] = set()
    deadline = time.monotonic() + timeout
    while True:
        launcher_returncode = process.poll()  # Reap a zombie before scanning.
        remaining, error = _marked_pids(marker, proc_root)
        observed.update(remaining)
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

    remaining, observed, scan_error = _wait_marked_processes(
        process, marker, grace_seconds, proc_root
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
            process, marker, wait, proc_root
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
    remaining, newly_observed, scan_error = _wait_marked_processes(process, marker, 0.0, proc_root)
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


def run_capture(
    plan: NcuTorchrunCapturePlan, *, environment: Mapping[str, str] | None = None
) -> NcuTorchrunCaptureResult:
    """Execute a plan, drain its owned process group, and retain a receipt."""

    env = dict(os.environ if environment is None else environment)
    preflight = _preflight(plan, env)
    plan.output_dir.mkdir(parents=True, exist_ok=False)
    capture_marker = uuid.uuid4().hex
    env.update(
        PYTHONNOUSERSITE="1",
        AISP_PROFILE_NO_USER_SITE="1",
        **{_CAPTURE_OWNER_ENV: capture_marker},
    )
    _write_json(plan.rank_config_path, plan.rank_config())
    receipt: dict[str, Any] = {
        "schema": CAPTURE_RECEIPT_SCHEMA,
        "status": "RUNNING",
        "label": plan.label,
        "source": plan.source,
        "cwd": str((plan.repo_root / "code").resolve()),
        "torchrun_argv": list(plan.torchrun_argv),
        "argv": list(plan.launch_argv),
        "rank_config_path": str(plan.rank_config_path),
        "rank_config_sha256": _sha256(plan.rank_config_path),
        "world_size": WORLD_SIZE,
        "nnodes": 1,
        "tcp_communicator": {"hostname": "127.0.0.1", "port": plan.tcp_port, "num_peers": 2},
        "nccl_nvtx_includes": list(plan.nccl_nvtx_includes),
        "replay_mode": "kernel",
        "all_matching_kernels": plan.all_matching_kernels,
        "metrics": list(MINIMAL_METRICS),
        "claim_limit": CLAIM_LIMIT,
        "cleanup_limit": CLEANUP_LIMIT,
        "capture_owner": {"environment": _CAPTURE_OWNER_ENV, "id": capture_marker},
        **preflight,
    }
    _write_json(plan.receipt_path, receipt)
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
            _cleanup_capture_processes(process, capture_marker)
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
    cleanup = _cleanup_capture_processes(process, capture_marker)
    artifacts, artifact_errors = _artifacts(plan, start_ns)
    if not cleanup.supported:
        status = "CLEANUP_UNVERIFIED"
    elif not cleanup.drained or process.returncode is None:
        status = "CLEANUP_FAILED"
    elif timed_out:
        status = "TIMEOUT"
    elif process.returncode:
        status = "CAPTURE_FAILED"
    elif not cleanup.natural:
        status = "FORCED_CLEANUP"
    elif artifact_errors:
        status = "CAPTURE_REJECTED"
    else:
        status = "CAPTURE_ARTIFACTS_PRESENT"
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
        description="Two-rank TCP/lockstep NCU capture; pass a torchrun-wrapper command after --."
    )
    parser.add_argument("--label", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ncu", required=True, type=Path)
    parser.add_argument("--tcp-port", required=True, type=int)
    parser.add_argument("--timeout-seconds", required=True, type=float)
    parser.add_argument("--all-matching-kernels", action="store_true")
    parser.add_argument("--nvtx-include", required=True, action="append", dest="includes")
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
