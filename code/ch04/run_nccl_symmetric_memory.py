"""Build-provenance-aware launcher for the native NCCL symmetric-memory study."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import stat
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch04.nccl_symmetric_memory_build_receipt import (  # noqa: E402
    BUILD_SCHEMA,
    RuntimeBindingError,
    expected_nccl_library,
    file_sha256,
    nccl_library_directories,
    serialized_build,
)
from ch04.nccl_symmetric_memory_profile import (  # noqa: E402
    PROFILE_SCHEMA,
    STEADY_RANGE,
    ProfilerUnavailableError,
    artifact,
    collect_profile,
    prepare_profile,
)
from core.utils.cooperative_process_artifacts import (  # noqa: E402
    run_recorded_cooperative_process,
)
from core.utils.cooperative_subprocess import CooperativeCleanupIncomplete  # noqa: E402

CHAPTER_DIR = Path(__file__).resolve().parent
SCHEMA = "nccl_symmetric_memory.v1"
SUITE_SCHEMA = "nccl_symmetric_memory.suite.v1"
MINIMUM_NCCL_VERSION_CODE = 22703
ARMS = ("cuda_malloc", "nccl_mem_alloc", "comm_register", "window_register")
BUILD_SOURCE_FILES = (
    "nccl_symmetric_memory_native.cu",
    "Makefile",
    "nccl_symmetric_memory_build_receipt.py",
    "../core/common/cuda_arch.mk",
    "../core/common/nvtx_utils.cuh",
    "../core/common/nvtx_label_utils.cuh",
)
CONTROL_PLANE_FILES = {
    "runner": Path(__file__).resolve(),
    "benchmark_wrapper": CHAPTER_DIR / "nccl_symmetric_memory_benchmark.py",
    "build_receipt": CHAPTER_DIR / "nccl_symmetric_memory_build_receipt.py",
    "profile": CHAPTER_DIR / "nccl_symmetric_memory_profile.py",
    "cooperative_process_artifacts": (
        CHAPTER_DIR.parent / "core" / "utils" / "cooperative_process_artifacts.py"
    ),
    "cooperative_subprocess": CHAPTER_DIR.parent / "core" / "utils" / "cooperative_subprocess.py",
}
MECHANISM_ENVIRONMENT_PREFIXES = ("NCCL_", "OMPI_MCA_", "UCX_")
SENSITIVE_ENVIRONMENT_NAME = re.compile(
    r"(?:^|_)(?:API_?KEY|AUTH|CREDENTIAL|COOKIE|PASS(?:WORD|WD)?|SECRET|TOKEN)(?:_|$)",
    re.IGNORECASE,
)
NCCL_KERNEL_NAME = re.compile(r"(?<![A-Za-z0-9_])nccl(?:dev)?kernel(?:_|\b)", re.IGNORECASE)


class NativeUnavailableError(RuntimeError):
    """A required native capability is unavailable on the selected host."""


def _error_reason(error: BaseException) -> str:
    if isinstance(error, KeyboardInterrupt):
        return "Interrupted by user before completion"
    return str(error) or error.__class__.__name__


class JsonResultReservation:
    """Own a fresh result path from before launch through final JSON retention."""

    def __init__(self, path: Path, stream: Any, identity: tuple[int, int]) -> None:
        self.path = path
        self._stream = stream
        self._identity = identity
        self.finalized = False

    @classmethod
    def acquire(cls, path: Path) -> JsonResultReservation:
        resolved = path.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            descriptor = os.open(resolved, flags, 0o600)
        except FileExistsError as error:
            raise FileExistsError(
                f"Refusing to reuse an existing result artifact: {resolved}"
            ) from error
        metadata = os.fstat(descriptor)
        stream = os.fdopen(descriptor, "w", encoding="utf-8")
        return cls(resolved, stream, (metadata.st_dev, metadata.st_ino))

    def _require_owned_path(self) -> None:
        try:
            metadata = self.path.stat()
        except OSError as error:
            raise RuntimeError(f"Reserved result path disappeared: {self.path}") from error
        if self.path.is_symlink() or (metadata.st_dev, metadata.st_ino) != self._identity:
            raise RuntimeError(f"Reserved result path changed ownership: {self.path}")

    def write_json(self, content: dict[str, Any]) -> None:
        if self.finalized:
            raise RuntimeError(f"Result reservation was already finalized: {self.path}")
        serialized = json.dumps(content, indent=2) + "\n"
        self._require_owned_path()
        self._stream.write(serialized)
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        self.finalized = True

    def abandon(self) -> None:
        """Release an internal reservation that never became a retained result."""
        if self.finalized:
            return
        if not self._stream.closed:
            self._stream.close()
        try:
            metadata = self.path.stat()
        except FileNotFoundError:
            return
        if not self.path.is_symlink() and (metadata.st_dev, metadata.st_ino) == self._identity:
            self.path.unlink()


@dataclass(frozen=True)
class RunBuildCustody:
    """A read-only binary and build receipt owned by one reserved result."""

    source_binary: Path
    source_receipt: Path
    directory: Path
    directory_identity: tuple[int, int]
    binary: Path
    binary_identity: tuple[int, int]
    build_receipt: Path
    build_receipt_identity: tuple[int, int]
    build: dict[str, Any]
    binary_sha256: str
    build_receipt_sha256: str
    source_sha256: dict[str, str]


@dataclass(frozen=True)
class NativeWorkload:
    arm: str
    ranks: int
    elements: int = 16 * 1024 * 1024
    iterations: int = 20
    warmup: int = 5

    def validate(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        values = (self.ranks, self.elements, self.iterations, self.warmup)
        if any(type(value) is not int for value in values):
            raise ValueError("Ranks, elements, iterations, and warmup must be integers")
        if self.ranks < 2:
            raise ValueError("At least two ranks are required")
        if self.elements < 1 or self.iterations < 1 or self.warmup < 0:
            raise ValueError("Require elements >= 1, iterations >= 1, and warmup >= 0")
        if self.ranks > 2**31 - 1 or self.iterations > 2**31 - 1 or self.warmup > 2**31 - 1:
            raise ValueError("Ranks and iteration counts must fit signed 32-bit integers")
        if self.elements > (2**63 - 1) // 4:
            raise ValueError("The float32 allocation size must fit a signed 64-bit byte count")

    def arguments(self) -> list[str]:
        self.validate()
        return [
            "--arm",
            self.arm,
            "--elements",
            str(self.elements),
            "--iterations",
            str(self.iterations),
            "--warmup",
            str(self.warmup),
        ]


def build_source_hashes(directory: Path = CHAPTER_DIR) -> dict[str, str]:
    return {name: file_sha256(directory / name) for name in BUILD_SOURCE_FILES}


def validate_build_receipt(binary: Path, directory: Path = CHAPTER_DIR) -> dict[str, Any]:
    if not binary.is_file():
        raise NativeUnavailableError(
            f"SKIPPED: Native binary is missing. Build {directory}/nccl_symmetric_memory_native"
        )
    receipt_path = binary.with_name(binary.name + ".build.json")
    if not receipt_path.is_file():
        raise RuntimeError("Native build receipt is missing. Rebuild with the Chapter 4 Makefile")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not isinstance(receipt, dict) or receipt.get("schema") != BUILD_SCHEMA:
        raise RuntimeError("Native build receipt has an unsupported schema")
    if receipt.get("binary_sha256") != file_sha256(binary):
        raise RuntimeError("Native binary differs from its build receipt")
    if receipt.get("source_sha256") != build_source_hashes(directory):
        raise RuntimeError("Native source or build inputs changed after compilation")
    header_code = receipt.get("nccl_header_version_code")
    if type(header_code) is not int or header_code < MINIMUM_NCCL_VERSION_CODE:
        raise RuntimeError("Native receipt does not prove NCCL 2.27.3 or newer headers")
    if receipt.get("minimum_nccl_version_code") != MINIMUM_NCCL_VERSION_CODE:
        raise RuntimeError("Native receipt records an unexpected NCCL minimum version")
    if not isinstance(receipt.get("compile_command"), list) or not receipt["compile_command"]:
        raise RuntimeError("Native receipt is missing its compile command")
    if not all(isinstance(value, str) for value in receipt["compile_command"]):
        raise RuntimeError("Native receipt compile arguments must be strings")
    compiler_versions = receipt.get("compiler_versions")
    if not isinstance(compiler_versions, dict) or not all(
        compiler_versions.get(name) for name in ("nvcc", "mpicxx")
    ):
        raise RuntimeError("Native receipt is missing compiler versions")
    build_options = receipt.get("build_options")
    if not isinstance(build_options, dict) or not isinstance(build_options.get("nccl_libs"), str):
        raise RuntimeBindingError("the build receipt is missing its NCCL linker search path")
    runtime_path = receipt.get("nccl_runtime_library_path")
    runtime_sha256 = receipt.get("nccl_runtime_library_sha256")
    if not isinstance(runtime_path, str) or not runtime_path:
        raise RuntimeBindingError("the build receipt is missing its NCCL runtime library path")
    if not isinstance(runtime_sha256, str) or re.fullmatch(r"[0-9a-f]{64}", runtime_sha256) is None:
        raise RuntimeBindingError("the build receipt is missing its NCCL runtime library SHA256")
    return receipt


def _copy_exclusive(source: Path, destination: Path, mode: int) -> tuple[int, int]:
    """Copy one stable regular file into a path that must not already exist."""
    read_flags = os.O_RDONLY
    write_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    for name in ("O_CLOEXEC", "O_NOFOLLOW"):
        flag = getattr(os, name, 0)
        read_flags |= flag
        write_flags |= flag
    source_descriptor = os.open(source, read_flags)
    destination_descriptor: int | None = None
    try:
        source_before = os.fstat(source_descriptor)
        if not stat.S_ISREG(source_before.st_mode):
            raise RuntimeError(f"Run build input is not a regular file: {source}")
        destination_descriptor = os.open(destination, write_flags, 0o600)
        while True:
            block = os.read(source_descriptor, 1024 * 1024)
            if not block:
                break
            remaining = memoryview(block)
            while remaining:
                written = os.write(destination_descriptor, remaining)
                if written <= 0:
                    raise OSError(f"Could not copy {source} to {destination}")
                remaining = remaining[written:]
        os.fsync(destination_descriptor)
        os.fchmod(destination_descriptor, mode)
        destination_metadata = os.fstat(destination_descriptor)
        source_after = os.fstat(source_descriptor)
    finally:
        os.close(source_descriptor)
        if destination_descriptor is not None:
            os.close(destination_descriptor)

    try:
        source_path_metadata = source.lstat()
    except OSError as error:
        raise RuntimeError(f"Run build input disappeared while it was copied: {source}") from error
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if source.is_symlink() or any(
        getattr(source_before, field) != getattr(source_after, field)
        or getattr(source_after, field) != getattr(source_path_metadata, field)
        for field in stable_fields
    ):
        raise RuntimeError(f"Run build input changed while it was copied: {source}")
    return destination_metadata.st_dev, destination_metadata.st_ino


def stage_run_build(binary: Path, reservation: JsonResultReservation) -> RunBuildCustody:
    """Validate and freeze one build while excluding the cooperating builder."""
    if reservation.finalized:
        raise RuntimeError("Cannot stage a build for a finalized result reservation")
    return prepare_run_cohort(
        binary,
        reservation.path.with_name(reservation.path.name + ".run"),
    )


def prepare_run_cohort(binary: Path, directory: Path) -> RunBuildCustody:
    """Freeze one build for reuse by every invocation in a measured cohort."""
    with serialized_build(CHAPTER_DIR):
        return _stage_run_build_locked(binary, directory.resolve())


def _stage_run_build_locked(binary: Path, directory: Path) -> RunBuildCustody:
    """Copy the executable and receipt while the Chapter 4 build lock is held."""
    source_binary = binary.resolve()
    source_receipt = source_binary.with_name(source_binary.name + ".build.json")
    initial_build = validate_build_receipt(source_binary)
    initial_binary_sha256 = file_sha256(source_binary)
    initial_receipt_sha256 = file_sha256(source_receipt)
    if initial_binary_sha256 != initial_build["binary_sha256"]:
        raise RuntimeError("Native binary changed after its build receipt was validated")

    directory.mkdir(mode=0o700, exist_ok=False)
    directory_metadata = directory.stat()
    run_binary = directory / source_binary.name
    run_receipt = run_binary.with_name(run_binary.name + ".build.json")
    try:
        binary_identity = _copy_exclusive(source_binary, run_binary, 0o500)
        receipt_identity = _copy_exclusive(source_receipt, run_receipt, 0o400)
        copied_build = validate_build_receipt(run_binary)
        copied_binary_sha256 = file_sha256(run_binary)
        copied_receipt_sha256 = file_sha256(run_receipt)
        prelaunch_sources = build_source_hashes()
        if copied_build != initial_build:
            raise RuntimeError("Native build receipt changed while the run copy was staged")
        if copied_binary_sha256 != initial_binary_sha256:
            raise RuntimeError("Native binary changed while the run copy was staged")
        if copied_receipt_sha256 != initial_receipt_sha256:
            raise RuntimeError("Native build receipt bytes changed while the run copy was staged")
        if prelaunch_sources != initial_build["source_sha256"]:
            raise RuntimeError("Native source changed before the immutable run copy was launched")
    finally:
        os.chmod(directory, 0o500)

    return RunBuildCustody(
        source_binary=source_binary,
        source_receipt=source_receipt,
        directory=directory,
        directory_identity=(directory_metadata.st_dev, directory_metadata.st_ino),
        binary=run_binary,
        binary_identity=binary_identity,
        build_receipt=run_receipt,
        build_receipt_identity=receipt_identity,
        build=copied_build,
        binary_sha256=copied_binary_sha256,
        build_receipt_sha256=copied_receipt_sha256,
        source_sha256=prelaunch_sources,
    )


def run_cohort_identity(custody: RunBuildCustody) -> dict[str, Any]:
    """Return the immutable build, source, and NCCL identity for one run cohort."""
    build = custody.build
    return {
        "binary_sha256": custody.binary_sha256,
        "build_receipt_sha256": custody.build_receipt_sha256,
        "source_sha256": dict(custody.source_sha256),
        "nccl_header_version_code": build["nccl_header_version_code"],
        "minimum_nccl_version_code": build["minimum_nccl_version_code"],
        "nccl_runtime_library_path": build["nccl_runtime_library_path"],
        "nccl_runtime_library_sha256": build["nccl_runtime_library_sha256"],
    }


def _require_cohort_for_binary(binary: Path, custody: RunBuildCustody) -> None:
    requested = binary.resolve()
    if requested != custody.source_binary:
        raise RuntimeError(
            f"Mixed-build cohort rejected: requested {requested}, cohort owns "
            f"{custody.source_binary}"
        )
    verify_run_build_custody(custody)


def verify_run_build_custody(custody: RunBuildCustody) -> dict[str, Any]:
    """Recheck the staged identities, modes, bytes, receipt, and source contract."""
    directory_metadata = custody.directory.lstat()
    if (
        custody.directory.is_symlink()
        or not stat.S_ISDIR(directory_metadata.st_mode)
        or (directory_metadata.st_dev, directory_metadata.st_ino) != custody.directory_identity
        or directory_metadata.st_mode & 0o222
    ):
        raise RuntimeError(f"Immutable run build directory changed: {custody.directory}")

    artifacts = {}
    for label, path, identity, expected_sha256 in (
        ("binary", custody.binary, custody.binary_identity, custody.binary_sha256),
        (
            "build_receipt",
            custody.build_receipt,
            custody.build_receipt_identity,
            custody.build_receipt_sha256,
        ),
    ):
        metadata = path.lstat()
        if (
            path.is_symlink()
            or not stat.S_ISREG(metadata.st_mode)
            or (metadata.st_dev, metadata.st_ino) != identity
            or metadata.st_mode & 0o222
        ):
            raise RuntimeError(f"Immutable run {label.replace('_', ' ')} changed: {path}")
        digest = file_sha256(path)
        if digest != expected_sha256:
            raise RuntimeError(f"Immutable run {label.replace('_', ' ')} bytes changed: {path}")
        artifacts[label] = {
            "path": str(path),
            "bytes": metadata.st_size,
            "sha256": digest,
            "mode": oct(stat.S_IMODE(metadata.st_mode)),
        }

    retained_build = json.loads(custody.build_receipt.read_text(encoding="utf-8"))
    if retained_build != custody.build:
        raise RuntimeError("Immutable run build receipt no longer matches the validated receipt")
    if custody.binary_sha256 != custody.build.get("binary_sha256"):
        raise RuntimeError("Immutable run binary no longer matches the validated receipt")
    if custody.source_sha256 != custody.build.get("source_sha256"):
        raise RuntimeError("Immutable run source hashes no longer match the validated receipt")
    return {
        "directory": str(custody.directory),
        "source_binary_path": str(custody.source_binary),
        "source_build_receipt_path": str(custody.source_receipt),
        "source_sha256": custody.source_sha256,
        **artifacts,
    }


def _regular_file_snapshot(path: Path) -> dict[str, Any]:
    """Hash one regular file and reject replacement while it is read."""
    try:
        before = path.lstat()
    except OSError as error:
        raise RuntimeError(f"Required retained file is unavailable: {path}") from error
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"Required retained file is not a regular file: {path}")
    digest = file_sha256(path)
    try:
        after = path.lstat()
    except OSError as error:
        raise RuntimeError(f"Required retained file disappeared while hashing: {path}") from error
    stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(getattr(before, field) != getattr(after, field) for field in stable_fields):
        raise RuntimeError(f"Required retained file changed while hashing: {path}")
    return {
        "path": str(path),
        "bytes": after.st_size,
        "sha256": digest,
        "device": after.st_dev,
        "inode": after.st_ino,
        "mtime_ns": after.st_mtime_ns,
        "ctime_ns": after.st_ctime_ns,
    }


def control_plane_identity() -> dict[str, dict[str, Any]]:
    """Snapshot every Python file that governs one native launch."""
    return {name: _regular_file_snapshot(path) for name, path in CONTROL_PLANE_FILES.items()}


def _require_unchanged_identity(
    label: str,
    before: dict[str, Any],
    after: dict[str, Any],
) -> None:
    changed = sorted(
        name for name in before.keys() | after.keys() if before.get(name) != after.get(name)
    )
    if changed:
        raise RuntimeError(f"{label} changed during execution: {', '.join(changed)}")


def retained_native_artifacts(paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
    """Hash every native and process artifact retained by a successful invocation."""
    return {name: _regular_file_snapshot(path) for name, path in paths.items() if name != "result"}


def mechanism_environment(environment: dict[str, str]) -> dict[str, Any]:
    """Retain mechanism settings while redacting values with secret-bearing names."""
    names = sorted(
        name
        for name in environment
        if name == "CUDA_VISIBLE_DEVICES" or name.startswith(MECHANISM_ENVIRONMENT_PREFIXES)
    )
    variables = {}
    redacted = []
    for name in names:
        if SENSITIVE_ENVIRONMENT_NAME.search(name):
            variables[name] = "<redacted>"
            redacted.append(name)
        else:
            variables[name] = environment[name]
    visibility = environment.get("CUDA_VISIBLE_DEVICES", "<unset>")
    variables.setdefault("CUDA_VISIBLE_DEVICES", visibility)
    nccl_overrides = sorted(name for name in names if name.startswith("NCCL_"))
    return {
        "cuda_visible_devices": visibility,
        "variables": variables,
        "redacted_variables": redacted,
        "nccl_environment_overrides": nccl_overrides,
        "nccl_attribution": "forced_environment" if nccl_overrides else "library_default",
    }


def _write_inspection_log(path: Path, content: str) -> dict[str, Any]:
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
        if content and not content.endswith("\n"):
            stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return dict(artifact(path))


def _profile_scope_markers(output: str) -> dict[str, bool]:
    lowered = output.lower()
    steady = STEADY_RANGE.rstrip("/").lower() in lowered
    # A range label contains both "allreduce" and "nccl". Require the shape of
    # an NCCL kernel symbol so the range label cannot satisfy the kernel gate.
    nccl_kernel = NCCL_KERNEL_NAME.search(output) is not None
    return {"steady_range": steady, "nccl_kernel": nccl_kernel}


def inspect_profile_scope(summary: dict[str, Any], config: dict[str, Any]) -> None:
    """Require every inspectable rank report to contain the requested NCCL work."""
    tool = Path(config["tool"])
    if file_sha256(tool) != config["tool_sha256"]:
        raise RuntimeError("Default profiler executable changed before report inspection")
    name = config["name"]
    if name not in {"ncu", "nsys"}:
        raise RuntimeError(f"Cannot inspect unsupported profiler {name!r}")

    for row in summary["ranks"]:
        rank = row["rank"]
        report = Path(row["report"]["path"])
        expected_report_sha256 = row["report"]["sha256"]
        if file_sha256(report) != expected_report_sha256:
            raise RuntimeError(f"Rank {rank} profiler report changed before inspection")
        if name == "ncu":
            command_options = [("raw", ["--csv", "--page", "raw", "--import"])]
        else:
            command_options = [
                (
                    "kernels",
                    [
                        "stats",
                        "--force-export=true",
                        "--report",
                        "cuda_gpu_kern_sum",
                        "--format",
                        "csv",
                    ],
                ),
                ("nvtx", ["stats", "--report", "nvtx_sum", "--format", "csv"]),
            ]

        inspections = []
        matched = {"steady_range": False, "nccl_kernel": False}
        with tempfile.TemporaryDirectory(prefix=f"nccl-profile-rank-{rank}-") as temporary:
            temporary_directory = Path(temporary)
            for index, (label, options) in enumerate(command_options):
                temporary_report = temporary_directory / f"{index}-{report.name}"
                shutil.copyfile(report, temporary_report)
                command = [str(tool), *options, str(temporary_report)]
                completed = subprocess.run(command, capture_output=True, text=True, check=False)
                markers = (
                    _profile_scope_markers(completed.stdout)
                    if completed.returncode == 0
                    else {
                        "steady_range": False,
                        "nccl_kernel": False,
                    }
                )
                for marker, present in markers.items():
                    matched[marker] = matched[marker] or present
                prefix = report.parent / f"scope-inspection-{label}"
                inspections.append(
                    {
                        "command": command,
                        "returncode": completed.returncode,
                        "markers": markers,
                        "stdout": _write_inspection_log(
                            prefix.with_suffix(".stdout.log"), completed.stdout
                        ),
                        "stderr": _write_inspection_log(
                            prefix.with_suffix(".stderr.log"), completed.stderr
                        ),
                    }
                )
        if file_sha256(report) != expected_report_sha256:
            raise RuntimeError(f"Rank {rank} profiler report changed during inspection")
        if not any(item["returncode"] == 0 for item in inspections):
            raise RuntimeError(f"Default {name} could not inspect the rank {rank} report")
        row["scope_inspection"] = {
            "requested_steady_range": STEADY_RANGE,
            "matched": matched,
            "attempts": inspections,
        }
        if not all(matched.values()):
            missing = ", ".join(name for name, present in matched.items() if not present)
            raise RuntimeError(
                f"Rank {rank} {name} report does not prove both the requested steady range "
                f"and an NCCL kernel. Missing: {missing}"
            )

    if file_sha256(tool) != config["tool_sha256"]:
        raise RuntimeError("Default profiler executable changed during report inspection")


def child_runtime_environment(build: dict[str, Any]) -> tuple[dict[str, str], list[str]]:
    """Prepend the build's library search directories to the child's environment."""
    environment = os.environ.copy()
    if environment.get("LD_PRELOAD"):
        raise RuntimeBindingError("LD_PRELOAD must be empty for this study")
    build_options = build.get("build_options")
    nccl_libs = build_options.get("nccl_libs") if isinstance(build_options, dict) else None
    if not isinstance(nccl_libs, str):
        raise RuntimeBindingError("the build receipt is missing NCCL linker options")
    library_directories = nccl_library_directories(nccl_libs, CHAPTER_DIR)

    inherited = environment.get("LD_LIBRARY_PATH")
    values = [*library_directories]
    if inherited:
        values.append(inherited)
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(values)
    return environment, library_directories


def verify_loader_binding(
    binary: Path, build: dict[str, Any], directories: list[str], environment: dict[str, str]
) -> Path:
    """Require the loader to select the exact NCCL library recorded at build time."""
    library = expected_nccl_library(directories)
    if str(library) != build.get("nccl_runtime_library_path"):
        raise RuntimeBindingError(
            f"the build search path selects {library}, receipt path {build.get('nccl_runtime_library_path')!r}"
        )
    if file_sha256(library) != build.get("nccl_runtime_library_sha256"):
        raise RuntimeBindingError(f"SHA256 for {library} differs from the build receipt")
    try:
        completed = subprocess.run(
            ["ldd", str(binary)], env=environment, capture_output=True, text=True, check=False
        )
    except OSError as error:
        raise RuntimeBindingError(f"could not run ldd for {binary}: {error}") from error
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RuntimeBindingError(f"ldd exited with {completed.returncode}: {detail}")
    rows = []
    for line in completed.stdout.splitlines():
        fields = line.split(maxsplit=1)
        if fields and Path(fields[0]).name == "libnccl.so.2":
            rows.append(line.strip())
    if len(rows) != 1:
        raise RuntimeBindingError(f"ldd must report one libnccl.so.2 dependency, found {rows}")
    match = re.fullmatch(r"libnccl\.so\.2\s+=>\s+(/.+?)\s+\(0x[0-9a-fA-F]+\)", rows[0])
    if not match:
        raise RuntimeBindingError(f"ldd did not resolve libnccl.so.2: {rows[0]}")
    resolved = Path(match[1]).resolve()
    if resolved != library:
        raise RuntimeBindingError(f"ldd resolved libnccl.so.2 to {resolved}, expected {library}")
    return resolved


def verify_rank_binding(
    result: dict[str, Any], build: dict[str, Any], library: Path, ranks: int
) -> str:
    """Check the ranks' observed library identity and recheck the file after MPI exits."""
    if result.get("nccl_runtime_version_code") != build["nccl_header_version_code"]:
        raise RuntimeBindingError(
            f"native NCCL runtime version {result.get('nccl_runtime_version_code')!r} differs "
            f"from build receipt header version {build['nccl_header_version_code']}"
        )
    verified_ranks = result.get("nccl_runtime_library_verified_ranks")
    if (
        result.get("nccl_runtime_library_path") != str(library)
        or type(verified_ranks) is not int
        or verified_ranks != ranks
    ):
        raise RuntimeBindingError(
            f"expected {ranks} ranks to confirm {library}, received path "
            f"{result.get('nccl_runtime_library_path')!r} and rank count {verified_ranks!r}"
        )
    try:
        library_hash = file_sha256(library)
    except OSError as error:
        raise RuntimeBindingError(f"could not read {library} after execution: {error}") from error
    if library_hash != build["nccl_runtime_library_sha256"]:
        raise RuntimeBindingError(f"SHA256 for {library} changed during execution")
    return library_hash


def _finite_nonnegative(result: dict[str, Any], names: tuple[str, ...]) -> None:
    for name in names:
        value = result.get(name)
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise RuntimeError(f"Native result field {name!r} must be finite and nonnegative")


def _p95(samples: list[float]) -> float:
    ordered = sorted(samples)
    return ordered[max(1, math.ceil(0.95 * len(ordered))) - 1]


def expected_statistics(samples: list[float]) -> dict[str, float]:
    return {
        "min": min(samples),
        "max": max(samples),
        "mean": math.fsum(samples) / len(samples),
        "median": statistics.median(samples),
        "p95": _p95(samples),
        "stddev": statistics.pstdev(samples),
    }


def validate_result(
    result: dict[str, Any],
    workload: NativeWorkload,
    *,
    profiler_preload_allowed: bool = False,
    expected_cuda_visibility: str | None = None,
) -> None:
    expected = {
        "schema": SCHEMA,
        "status": "passed",
        "arm": workload.arm,
        "dtype": "float32",
        "reduction": "sum",
        "out_of_place": True,
        "single_node": True,
        "p2p_gate_passed": True,
        "elements": workload.elements,
        "bytes_per_buffer": workload.elements * 4,
        "ranks": workload.ranks,
        "iterations": workload.iterations,
        "warmup": workload.warmup,
        "minimum_nccl_version_code": MINIMUM_NCCL_VERSION_CODE,
        "profiler_preload_allowed": profiler_preload_allowed,
        "timing_scope": "MPI_MAX CUDA event elapsed around one ncclAllReduce",
        "verified_ranks": workload.ranks,
        "checked_elements_per_rank": workload.elements,
        "checked_elements_total": workload.elements * workload.ranks,
    }
    for name, value in expected.items():
        if type(result.get(name)) is not type(value) or result.get(name) != value:
            raise RuntimeError(f"Native result field {name!r} differs from the requested contract")
    registration_ranks = (
        workload.ranks if workload.arm in {"comm_register", "window_register"} else 0
    )
    for buffer in ("send", "receive"):
        verified = result.get(f"{buffer}_registration_handle_verified_ranks")
        if type(verified) is not int or verified != registration_ranks:
            raise RuntimeError(
                f"Expected {registration_ranks} ranks to verify the {buffer} registration handle "
                f"for {workload.arm}, received {verified!r}"
            )
    for version_field in ("nccl_header_version_code", "nccl_runtime_version_code"):
        value = result.get(version_field)
        if type(value) is not int or value < MINIMUM_NCCL_VERSION_CODE:
            raise RuntimeBindingError(
                f"native result field {version_field!r} must prove NCCL >= 2.27.3"
            )
    if result["nccl_runtime_version_code"] != result["nccl_header_version_code"]:
        raise RuntimeBindingError(
            f"native NCCL runtime version {result['nccl_runtime_version_code']} differs from "
            f"native header version {result['nccl_header_version_code']}"
        )

    _finite_nonnegative(
        result,
        (
            "setup_ms",
            "topology_gate_ms",
            "communicator_ms",
            "allocation_ms",
            "registration_ms",
            "initialization_ms",
            "warmup_ms",
            "time_per_iteration_ms",
            "verification_ms",
            "output_write_ms",
            "cleanup_ms",
            "rtol",
            "atol",
            "max_abs_error",
        ),
    )
    if result["rtol"] != 0.0 or result["atol"] != 0.0:
        raise RuntimeError("Native integer-pattern verification must use exact comparison")
    if result["time_per_iteration_ms"] <= 0:
        raise RuntimeError("Native timing must be positive")

    raw_samples = result.get("samples_ms")
    if not isinstance(raw_samples, list) or len(raw_samples) != workload.iterations:
        raise RuntimeError("Native result must retain one MPI_MAX sample per iteration")
    samples: list[float] = []
    for value in raw_samples:
        if (
            not isinstance(value, int | float)
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise RuntimeError("Native MPI_MAX timing samples must be finite and positive")
        samples.append(float(value))
    reported_statistics = result.get("statistics_ms")
    if not isinstance(reported_statistics, dict):
        raise RuntimeError("Native result is missing timing statistics")
    for name, value in expected_statistics(samples).items():
        reported = reported_statistics.get(name)
        if not isinstance(reported, int | float) or isinstance(reported, bool):
            raise RuntimeError(f"Native timing statistic {name!r} is missing")
        if not math.isclose(float(reported), value, rel_tol=2e-9, abs_tol=1e-9):
            raise RuntimeError(f"Native timing statistic {name!r} disagrees with raw samples")
    if not math.isclose(
        result["time_per_iteration_ms"],
        expected_statistics(samples)["mean"],
        rel_tol=2e-9,
        abs_tol=1e-9,
    ):
        raise RuntimeError("Native time per iteration disagrees with the raw-sample mean")

    rank_verification = result.get("rank_verification")
    if not isinstance(rank_verification, list) or len(rank_verification) != workload.ranks:
        raise RuntimeError("Native result is missing full rank verification coverage")
    if {row.get("rank") for row in rank_verification if isinstance(row, dict)} != set(
        range(workload.ranks)
    ):
        raise RuntimeError("Native rank verification has duplicate or missing ranks")
    for row in rank_verification:
        if (
            row.get("checked_elements") != workload.elements
            or row.get("bad_elements") != 0
            or not isinstance(row.get("max_abs_error"), int | float)
        ):
            raise RuntimeError("Native rank verification did not check every output element")

    mapping = result.get("rank_mapping")
    if not isinstance(mapping, list) or len(mapping) != workload.ranks:
        raise RuntimeError("Native result is missing one-rank-per-GPU mapping evidence")
    if {row.get("rank") for row in mapping if isinstance(row, dict)} != set(range(workload.ranks)):
        raise RuntimeError("Native rank mapping has duplicate or missing ranks")
    device_ordinals = [row.get("device_ordinal") for row in mapping]
    if (
        any(type(device) is not int for device in device_ordinals)
        or len(set(device_ordinals)) != workload.ranks
    ):
        raise RuntimeError("Native rank mapping does not assign one distinct GPU per rank")
    gpu_uuids = [row.get("gpu_uuid") for row in mapping]
    pci_bus_ids = [row.get("pci_bus_id") for row in mapping]
    uuid_pattern = re.compile(r"GPU-[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")
    bus_pattern = re.compile(r"(?:[0-9a-f]{4}|[0-9a-f]{8}):[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]")
    if any(type(value) is not str or uuid_pattern.fullmatch(value) is None for value in gpu_uuids):
        raise RuntimeError("Native rank mapping is missing canonical per-rank GPU UUID evidence")
    if any(
        type(value) is not str or bus_pattern.fullmatch(value.lower()) is None
        for value in pci_bus_ids
    ):
        raise RuntimeError("Native rank mapping is missing canonical per-rank PCI bus evidence")
    if len(set(gpu_uuids)) != workload.ranks or len(set(pci_bus_ids)) != workload.ranks:
        raise RuntimeError("Native physical GPU evidence does not prove one distinct GPU per rank")
    visibility = result.get("cuda_visible_devices")
    if not isinstance(visibility, str) or not visibility:
        raise RuntimeError(
            "Native result is missing the ordered CUDA visibility observed by every rank"
        )
    if expected_cuda_visibility is not None and visibility != expected_cuda_visibility:
        raise RuntimeError("Native ordered CUDA visibility differs from the launcher environment")


def _input_value(rank: int, index: int) -> int:
    """Mirror the native rank-keyed SplitMix64 integer input pattern."""
    mask = (1 << 64) - 1
    rank_key = (0x9E3779B97F4A7C15 * (rank + 1)) & mask
    mixed = (index + rank_key) & mask
    mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask
    mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask
    return 1 + ((mixed ^ (mixed >> 31)) & 0xFF)


def _expected_value(ranks: int, index: int) -> float:
    # The bounded integer sum is exactly representable for practical GPU rank counts.
    total = sum(_input_value(rank, index) for rank in range(ranks))
    return float(struct.unpack("f", struct.pack("f", total))[0])


def _expected_output_chunk(
    ranks: int,
    start: int,
    count: int,
) -> npt.NDArray[np.float32]:
    """Vectorize the independent SplitMix64 oracle for one bounded chunk."""
    mask = (1 << 64) - 1
    indices: npt.NDArray[np.uint64] = np.arange(
        start,
        start + count,
        dtype=np.uint64,
    )
    total: npt.NDArray[np.uint64] = np.zeros(count, dtype=np.uint64)
    for rank in range(ranks):
        rank_key = (0x9E3779B97F4A7C15 * (rank + 1)) & mask
        mixed = indices + np.uint64(rank_key)
        mixed = (mixed ^ (mixed >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        mixed = (mixed ^ (mixed >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        total += np.uint64(1) + ((mixed ^ (mixed >> np.uint64(31))) & np.uint64(0xFF))
    return total.astype(np.float32)


def validate_full_output(path: Path, workload: NativeWorkload) -> dict[str, int | float]:
    """Check every retained value with a chunked, independent NumPy oracle."""
    expected_bytes = workload.elements * 4
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise RuntimeError("Native process did not retain one complete float32 output")
    chunk_elements = 1024 * 1024
    with path.open("rb") as stream:
        for start in range(0, workload.elements, chunk_elements):
            count = min(chunk_elements, workload.elements - start)
            payload = stream.read(count * 4)
            if len(payload) != count * 4:
                raise RuntimeError("Retained float32 output has the wrong element count")
            actual: npt.NDArray[np.float32] = np.frombuffer(
                payload,
                dtype=np.dtype("=f4"),
                count=count,
            )
            expected = _expected_output_chunk(workload.ranks, start, count)
            invalid = np.flatnonzero(~np.isfinite(actual) | (actual != expected))
            if invalid.size:
                raise RuntimeError(f"Retained output differs at element {start + int(invalid[0])}")
    return {"checked_elements": workload.elements, "max_abs_error": 0.0}


def _artifact_paths(json_out: Path, dump_output: Path) -> dict[str, Path]:
    native_json = json_out.with_name(json_out.stem + ".native.json")
    return {
        "result": json_out,
        "native_result": native_json,
        "output": dump_output,
        "stdout": json_out.with_name(json_out.name + ".stdout.log"),
        "stderr": json_out.with_name(json_out.name + ".stderr.log"),
        "execution": json_out.with_name(json_out.name + ".execution.json"),
    }


def invoke_native(
    workload: NativeWorkload,
    *,
    binary: Path,
    json_out: Path,
    dump_output: Path,
    launcher: str = "mpirun",
    launcher_args: tuple[str, ...] = (),
    profile: str | None = None,
    cohort: RunBuildCustody | None = None,
    _result_reservation: JsonResultReservation | None = None,
) -> dict[str, Any]:
    """Reserve one result path, then launch and retain one MPI arm."""
    json_out = json_out.resolve()
    owns_reservation = _result_reservation is None
    reservation = _result_reservation or JsonResultReservation.acquire(json_out)
    if reservation.path != json_out:
        raise ValueError("The result reservation does not own the requested JSON path")
    try:
        return _invoke_native_reserved(
            workload,
            binary=binary,
            json_out=json_out,
            dump_output=dump_output,
            launcher=launcher,
            launcher_args=launcher_args,
            profile=profile,
            cohort=cohort,
            result_reservation=reservation,
        )
    except BaseException:
        if owns_reservation:
            reservation.abandon()
        raise


def _invoke_native_reserved(
    workload: NativeWorkload,
    *,
    binary: Path,
    json_out: Path,
    dump_output: Path,
    launcher: str,
    launcher_args: tuple[str, ...],
    profile: str | None,
    cohort: RunBuildCustody | None,
    result_reservation: JsonResultReservation,
) -> dict[str, Any]:
    """Launch one MPI arm and retain its complete output and process diagnostics."""
    workload.validate()
    binary = binary.resolve()
    dump_output = dump_output.resolve()
    launcher_path = shutil.which(launcher)
    if launcher_path is None:
        raise NativeUnavailableError(f"SKIPPED: MPI launcher {launcher!r} is unavailable")

    paths = _artifact_paths(json_out, dump_output)
    profile_directory = json_out.with_name(f"{json_out.name}.{profile}")
    if profile is not None:
        paths["native_result"] = profile_directory / "profiled.native.json"
    if len(set(paths.values())) != len(paths):
        raise ValueError("Result, native, output, and process artifacts need distinct paths")
    for name, path in paths.items():
        if name == "result":
            continue
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to reuse an existing native artifact: {path}")
    json_out.parent.mkdir(parents=True, exist_ok=True)
    dump_output.parent.mkdir(parents=True, exist_ok=True)

    custody = cohort or stage_run_build(binary, result_reservation)
    _require_cohort_for_binary(binary, custody)
    build = custody.build
    cohort_identity = run_cohort_identity(custody)
    source_snapshot = custody.source_sha256
    binary_hash = custody.binary_sha256
    child_environment, runtime_library_directories = child_runtime_environment(build)
    retained_environment = mechanism_environment(child_environment)
    library = verify_loader_binding(
        custody.binary, build, runtime_library_directories, child_environment
    )
    native_command = [
        str(custody.binary),
        *workload.arguments(),
        "--expected-nccl-library",
        str(library),
        "--json-out",
        str(paths["native_result"]),
        "--dump-output",
        str(dump_output),
    ]
    config = None
    rank_command = native_command
    if profile is not None:
        native_command.append("--allow-profiler-preload")
        try:
            rank_command, config = prepare_profile(
                profile_directory,
                profile,
                workload.ranks,
                build["compile_command"],
                native_command,
            )
        except ProfilerUnavailableError as error:
            raise NativeUnavailableError(str(error)) from error
    command = [launcher_path, *launcher_args, "-np", str(workload.ranks), *rank_command]
    if build_source_hashes() != source_snapshot:
        raise RuntimeError("Native source changed before the immutable run copy was launched")
    prelaunch_custody = verify_run_build_custody(custody)
    prelaunch_control_plane = control_plane_identity()
    completed = run_recorded_cooperative_process(
        command,
        output_prefix=json_out,
        env=child_environment,
    )
    post_run_control_plane = control_plane_identity()
    _require_unchanged_identity(
        "Runner control plane",
        prelaunch_control_plane,
        post_run_control_plane,
    )
    post_run_custody = verify_run_build_custody(custody)
    if completed["returncode"] != 0:
        lines = (completed["stdout"] + "\n" + completed["stderr"]).splitlines()
        if config is not None:
            for log in sorted(profile_directory.glob("rank-*/*.log")):
                with log.open(encoding="utf-8", errors="replace") as stream:
                    lines.extend(
                        line
                        for line in stream
                        if "SKIPPED:" in line or "NCCL runtime binding:" in line
                    )
        for line in lines:
            if "NCCL runtime binding:" in line:
                raise RuntimeBindingError(line.split("NCCL runtime binding:", 1)[1].strip())
        diagnostic = next(
            (line[line.index("SKIPPED:") :] for line in lines if "SKIPPED:" in line),
            None,
        )
        if diagnostic is not None:
            raise NativeUnavailableError(diagnostic)
        raise RuntimeError(
            f"Native MPI run failed with exit {completed['returncode']}. "
            f"See {paths['stdout']} and {paths['stderr']}"
            + (f" and rank logs in {profile_directory}" if config is not None else "")
        )
    if build_source_hashes() != source_snapshot:
        raise RuntimeError("Native source changed during execution")
    if not paths["native_result"].is_file():
        raise RuntimeError("Native MPI run exited without its requested raw result")

    result: dict[str, Any] = json.loads(paths["native_result"].read_text(encoding="utf-8"))
    validate_result(
        result,
        workload,
        profiler_preload_allowed=profile is not None,
        expected_cuda_visibility=retained_environment["cuda_visible_devices"],
    )
    library_hash = verify_rank_binding(result, build, library, workload.ranks)
    python_verification_started_ns = time.perf_counter_ns()
    python_verification = validate_full_output(dump_output, workload)
    python_verification_ms = (time.perf_counter_ns() - python_verification_started_ns) / 1_000_000.0
    native_artifacts = retained_native_artifacts(paths)
    if config is not None:
        profile_summary = collect_profile(profile_directory, config)
        inspect_profile_scope(profile_summary, config)
        result = {
            "schema": PROFILE_SCHEMA,
            "status": "passed",
            "purpose": "profile_diagnostics",
            "profiler": profile,
            "workload": asdict(workload),
            "cuda_visible_devices": result["cuda_visible_devices"],
            "rank_mapping": result["rank_mapping"],
            "rank_verification": result["rank_verification"],
            "send_registration_handle_verified_ranks": result[
                "send_registration_handle_verified_ranks"
            ],
            "receive_registration_handle_verified_ranks": result[
                "receive_registration_handle_verified_ranks"
            ],
            "profiler_preload_allowed": result["profiler_preload_allowed"],
            "profile": profile_summary,
            "native_artifacts": native_artifacts,
        }
    result.update(
        {
            "python_verification": python_verification,
            "python_verification_ms": python_verification_ms,
            "build": build,
            "source_sha256": source_snapshot,
            "binary_sha256": binary_hash,
            "binary_path": str(custody.binary),
            "run_build_custody": {
                "identity": cohort_identity,
                "prelaunch": prelaunch_custody,
                "post_run": post_run_custody,
            },
            "control_plane_custody": {
                "prelaunch": prelaunch_control_plane,
                "post_run": post_run_control_plane,
            },
            "mechanism_environment": retained_environment,
            "runner_sha256": prelaunch_control_plane["runner"]["sha256"],
            "command": command,
            "native_result_path": str(paths["native_result"]),
            "output_path": str(dump_output),
            "stdout_path": str(paths["stdout"]),
            "stderr_path": str(paths["stderr"]),
            "execution_path": str(paths["execution"]),
            "process_exit_code": completed["returncode"],
            "runtime_library_directories": runtime_library_directories,
            "nccl_runtime_library_sha256": library_hash,
        }
    )
    result.setdefault("native_artifacts", native_artifacts)
    result_reservation.write_json(result)
    return result


def invoke_all_arms(
    workload: NativeWorkload,
    *,
    binary: Path,
    json_out: Path,
    launcher: str,
    launcher_args: tuple[str, ...],
    profile: str | None = None,
    _result_reservation: JsonResultReservation | None = None,
) -> dict[str, Any]:
    """Reserve the suite result before starting any attribution arm."""
    json_out = json_out.resolve()
    owns_reservation = _result_reservation is None
    reservation = _result_reservation or JsonResultReservation.acquire(json_out)
    if reservation.path != json_out:
        raise ValueError("The suite reservation does not own the requested JSON path")
    try:
        return _invoke_all_arms_reserved(
            workload,
            binary=binary,
            json_out=json_out,
            launcher=launcher,
            launcher_args=launcher_args,
            profile=profile,
            result_reservation=reservation,
        )
    except BaseException:
        if owns_reservation:
            reservation.abandon()
        raise


def _invoke_all_arms_reserved(
    workload: NativeWorkload,
    *,
    binary: Path,
    json_out: Path,
    launcher: str,
    launcher_args: tuple[str, ...],
    profile: str | None,
    result_reservation: JsonResultReservation,
) -> dict[str, Any]:
    """Run every arm against one immutable build and NCCL cohort."""
    suite_schema = PROFILE_SCHEMA + ".suite" if profile else SUITE_SCHEMA
    completed_arms: dict[str, dict[str, Any]] = {}
    cohort = stage_run_build(binary, result_reservation)
    cohort_identity = run_cohort_identity(cohort)
    for arm in ARMS:
        arm_workload = replace(workload, arm=arm)
        arm_result = json_out.with_name(f"{json_out.stem}.{arm}.json")
        arm_output = json_out.with_name(f"{json_out.stem}.{arm}.f32")
        try:
            completed = invoke_native(
                arm_workload,
                binary=binary,
                json_out=arm_result,
                dump_output=arm_output,
                launcher=launcher,
                launcher_args=launcher_args,
                profile=profile,
                cohort=cohort,
            )
            observed_identity = completed.get("run_build_custody", {}).get("identity")
            if observed_identity != cohort_identity:
                raise RuntimeError(f"Mixed-build cohort rejected for arm {arm}")
            completed_arms[arm] = completed
        except BaseException as error:
            status = "SKIPPED" if isinstance(error, NativeUnavailableError) else "FAILED"
            suite = {
                "schema": suite_schema,
                "status": status,
                "reason": _error_reason(error),
                "failed_arm": arm,
                "workload": asdict(replace(workload, arm="cuda_malloc")) | {"arm": "all"},
                "run_cohort_identity": cohort_identity,
                "completed_arms": completed_arms,
            }
            if isinstance(error, CooperativeCleanupIncomplete):
                suite["cleanup_incomplete_receipt"] = error.result
            result_reservation.write_json(suite)
            raise
    suite = {
        "schema": suite_schema,
        "status": "passed",
        "workload": asdict(replace(workload, arm="cuda_malloc")) | {"arm": "all"},
        "run_cohort_identity": cohort_identity,
        "arm_order": list(ARMS),
        "arms": completed_arms,
    }
    result_reservation.write_json(suite)
    return suite


def visible_gpu_count() -> int:
    try:
        import torch
    except ImportError as error:
        raise NativeUnavailableError(
            "SKIPPED: Install the repository Python environment or pass --ranks explicitly"
        ) from error
    count = int(torch.cuda.device_count())
    if count < 2:
        raise NativeUnavailableError("SKIPPED: At least two visible CUDA devices are required")
    return count


def _default_result_path(label: str) -> Path:
    root = CHAPTER_DIR.parent / "artifacts" / "nccl_symmetric_memory"
    root.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=label + "_", dir=root))
    return directory / "result.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--arm", choices=ARMS, default="window_register")
    selection.add_argument(
        "--all-arms",
        action="store_true",
        help="Launch four sequential MPI runs and retain one result per arm",
    )
    parser.add_argument(
        "--ranks",
        type=int,
        help="MPI ranks and GPUs to use. The default is every visible CUDA device",
    )
    parser.add_argument("--elements", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--binary", type=Path, default=CHAPTER_DIR / "nccl_symmetric_memory_native")
    parser.add_argument(
        "--profile",
        choices=("nsys", "ncu"),
        help="Collect a separate diagnostic report on each MPI rank using the profiler on PATH",
    )
    parser.add_argument("--launcher", default="mpirun")
    parser.add_argument("--launcher-arg", action="append", default=[])
    parser.add_argument("--json-out", type=Path)
    parser.add_argument(
        "--dump-output",
        type=Path,
        help="Complete float32 output for one arm. All-arm runs choose one path per arm",
    )
    parser.add_argument("--json", action="store_true", help="Print the retained JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.all_arms and args.dump_output is not None:
        parser.error("--dump-output applies to one --arm and cannot be combined with --all-arms")
    result_path = args.json_out.resolve() if args.json_out is not None else None
    result_reservation: JsonResultReservation | None = None
    try:
        if result_path is not None:
            result_reservation = JsonResultReservation.acquire(result_path)
        else:
            label = "all_arms" if args.all_arms else args.arm
            result_path = _default_result_path(f"{label}_{args.profile}" if args.profile else label)
            result_reservation = JsonResultReservation.acquire(result_path)
        ranks = args.ranks if args.ranks is not None else visible_gpu_count()
        workload = NativeWorkload(
            arm=args.arm,
            ranks=ranks,
            elements=args.elements,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        workload.validate()
        if result_reservation is None:
            raise RuntimeError("Result ownership was not established before launch")
        if args.all_arms:
            result = invoke_all_arms(
                workload,
                binary=args.binary,
                json_out=result_path,
                launcher=args.launcher,
                launcher_args=tuple(args.launcher_arg),
                profile=args.profile,
                _result_reservation=result_reservation,
            )
        else:
            output_path = args.dump_output or result_path.with_suffix(".f32")
            result = invoke_native(
                workload,
                binary=args.binary,
                json_out=result_path,
                dump_output=output_path,
                launcher=args.launcher,
                launcher_args=tuple(args.launcher_arg),
                profile=args.profile,
                _result_reservation=result_reservation,
            )
    except CooperativeCleanupIncomplete as error:
        result = {
            "schema": (PROFILE_SCHEMA + ".suite" if args.all_arms else PROFILE_SCHEMA)
            if args.profile
            else (SUITE_SCHEMA if args.all_arms else SCHEMA),
            "status": "FAILED",
            "reason": _error_reason(error),
            "cleanup_incomplete_receipt": error.result,
        }
        try:
            if result_reservation is not None and not result_reservation.finalized:
                result_reservation.write_json(result)
        except (OSError, RuntimeError, ValueError) as record_error:
            result["failure_record_error"] = str(record_error)
            if not args.json:
                print(f"Could not save the cleanup failure result: {record_error}", file=sys.stderr)
        print(json.dumps(result, indent=2) if args.json else result["reason"])
        if not args.json and result_reservation is not None:
            print(f"Result: {result_reservation.path}")
        return 1
    except KeyboardInterrupt as error:
        result = {
            "schema": (PROFILE_SCHEMA + ".suite" if args.all_arms else PROFILE_SCHEMA)
            if args.profile
            else (SUITE_SCHEMA if args.all_arms else SCHEMA),
            "status": "FAILED",
            "reason": _error_reason(error),
        }
        try:
            if result_reservation is not None and not result_reservation.finalized:
                result_reservation.write_json(result)
        except (OSError, RuntimeError, ValueError) as record_error:
            result["failure_record_error"] = str(record_error)
            if not args.json:
                print(f"Could not save the interrupted result: {record_error}", file=sys.stderr)
        print(json.dumps(result, indent=2) if args.json else result["reason"])
        if not args.json and result_reservation is not None:
            print(f"Result: {result_reservation.path}")
        return 130
    except (NativeUnavailableError, ValueError, RuntimeError, OSError) as error:
        unavailable = isinstance(error, NativeUnavailableError)
        result = {
            "schema": (PROFILE_SCHEMA + ".suite" if args.all_arms else PROFILE_SCHEMA)
            if args.profile
            else (SUITE_SCHEMA if args.all_arms else SCHEMA),
            "status": "SKIPPED" if unavailable else "FAILED",
            "reason": _error_reason(error),
        }
        try:
            if result_reservation is not None and not result_reservation.finalized:
                result_reservation.write_json(result)
        except (OSError, RuntimeError, ValueError) as record_error:
            result["failure_record_error"] = str(record_error)
            if not args.json:
                print(f"Could not save the failure result: {record_error}", file=sys.stderr)
        print(json.dumps(result, indent=2) if args.json else str(error))
        if not args.json and result_reservation is not None:
            print(f"Result: {result_reservation.path}")
        return 77 if unavailable else 1
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        label = "all four arms" if args.all_arms else args.arm
        if args.profile:
            print(f"CAPTURED: {label}, {ranks} {args.profile} reports per arm")
        else:
            print(f"VERIFIED: {label}, {ranks} ranks, {args.iterations} timed samples")
        print(f"Result: {result_path}")
        print("Performance qualification requires locked-clock and profiler evidence")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
