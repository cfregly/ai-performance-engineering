"""Build-provenance-aware launcher for the native NCCL symmetric-memory study."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
from array import array
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ch04.nccl_symmetric_memory_build_receipt import (  # noqa: E402
    BUILD_SCHEMA,
    RuntimeBindingError,
    expected_nccl_library,
    file_sha256,
    nccl_library_directories,
)
from ch04.nccl_symmetric_memory_profile import (  # noqa: E402
    PROFILE_SCHEMA,
    ProfilerUnavailableError,
    artifact,
    collect_profile,
    prepare_profile,
)
from core.utils.cooperative_process_artifacts import (  # noqa: E402
    run_recorded_cooperative_process,
)

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


class NativeUnavailableError(RuntimeError):
    """A required native capability is unavailable on the selected host."""


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
    return receipt


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
    return struct.unpack("f", struct.pack("f", total))[0]


def validate_full_output(path: Path, workload: NativeWorkload) -> dict[str, int | float]:
    expected_bytes = workload.elements * 4
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise RuntimeError("Native process did not retain one complete float32 output")
    values = array("f")
    with path.open("rb") as stream:
        values.fromfile(stream, workload.elements)
    if sys.byteorder != "little":
        values.byteswap()
    if len(values) != workload.elements:
        raise RuntimeError("Retained float32 output has the wrong element count")
    maximum_error = 0.0
    for index, actual in enumerate(values):
        expected = _expected_value(workload.ranks, index)
        error = abs(float(actual) - expected)
        if not math.isfinite(actual) or error != 0.0:
            raise RuntimeError(f"Retained output differs at element {index}")
        maximum_error = max(maximum_error, error)
    return {"checked_elements": workload.elements, "max_abs_error": maximum_error}


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
    result_reservation: JsonResultReservation,
) -> dict[str, Any]:
    """Launch one MPI arm and retain its complete output and process diagnostics."""
    workload.validate()
    binary = binary.resolve()
    dump_output = dump_output.resolve()
    build = validate_build_receipt(binary)
    child_environment, runtime_library_directories = child_runtime_environment(build)
    library = verify_loader_binding(binary, build, runtime_library_directories, child_environment)
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

    source_snapshot = build_source_hashes()
    binary_hash = file_sha256(binary)
    native_command = [
        str(binary),
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
    completed = run_recorded_cooperative_process(
        command,
        output_prefix=json_out,
        env=child_environment,
    )
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
    if build_source_hashes() != source_snapshot or file_sha256(binary) != binary_hash:
        raise RuntimeError("Native source or binary changed during execution")
    if not paths["native_result"].is_file():
        raise RuntimeError("Native MPI run exited without its requested raw result")

    result = json.loads(paths["native_result"].read_text(encoding="utf-8"))
    validate_result(result, workload, profiler_preload_allowed=profile is not None)
    library_hash = verify_rank_binding(result, build, library, workload.ranks)
    python_verification = validate_full_output(dump_output, workload)
    if config is not None:
        result = {
            "schema": PROFILE_SCHEMA,
            "status": "passed",
            "purpose": "profile_diagnostics",
            "profiler": profile,
            "workload": asdict(workload),
            "rank_mapping": result["rank_mapping"],
            "rank_verification": result["rank_verification"],
            "send_registration_handle_verified_ranks": result[
                "send_registration_handle_verified_ranks"
            ],
            "receive_registration_handle_verified_ranks": result[
                "receive_registration_handle_verified_ranks"
            ],
            "profiler_preload_allowed": result["profiler_preload_allowed"],
            "profile": collect_profile(profile_directory, config),
            "native_artifacts": {
                name: artifact(path) for name, path in paths.items() if name != "result"
            },
        }
    result.update(
        {
            "python_verification": python_verification,
            "build": build,
            "source_sha256": source_snapshot,
            "binary_sha256": binary_hash,
            "runner_sha256": file_sha256(Path(__file__)),
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
    """Run all attribution arms sequentially with fresh retained artifacts."""
    suite_schema = PROFILE_SCHEMA + ".suite" if profile else SUITE_SCHEMA
    completed_arms: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        arm_workload = replace(workload, arm=arm)
        arm_result = json_out.with_name(f"{json_out.stem}.{arm}.json")
        arm_output = json_out.with_name(f"{json_out.stem}.{arm}.f32")
        try:
            completed_arms[arm] = invoke_native(
                arm_workload,
                binary=binary,
                json_out=arm_result,
                dump_output=arm_output,
                launcher=launcher,
                launcher_args=launcher_args,
                profile=profile,
            )
        except BaseException as error:
            status = "SKIPPED" if isinstance(error, NativeUnavailableError) else "FAILED"
            suite = {
                "schema": suite_schema,
                "status": status,
                "reason": str(error),
                "failed_arm": arm,
                "workload": asdict(replace(workload, arm="cuda_malloc")) | {"arm": "all"},
                "completed_arms": completed_arms,
            }
            result_reservation.write_json(suite)
            raise
    suite = {
        "schema": suite_schema,
        "status": "passed",
        "workload": asdict(replace(workload, arm="cuda_malloc")) | {"arm": "all"},
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
    count = torch.cuda.device_count()
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
        ranks = args.ranks if args.ranks is not None else visible_gpu_count()
        workload = NativeWorkload(
            arm=args.arm,
            ranks=ranks,
            elements=args.elements,
            iterations=args.iterations,
            warmup=args.warmup,
        )
        workload.validate()
        if result_path is None:
            label = "all_arms" if args.all_arms else args.arm
            result_path = _default_result_path(f"{label}_{args.profile}" if args.profile else label)
            result_reservation = JsonResultReservation.acquire(result_path)
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
    except (NativeUnavailableError, ValueError, RuntimeError, OSError) as error:
        unavailable = isinstance(error, NativeUnavailableError)
        result = {
            "schema": (PROFILE_SCHEMA + ".suite" if args.all_arms else PROFILE_SCHEMA)
            if args.profile
            else (SUITE_SCHEMA if args.all_arms else SCHEMA),
            "status": "SKIPPED" if unavailable else "FAILED",
            "reason": str(error),
        }
        try:
            if result_reservation is not None and not result_reservation.finalized:
                result_reservation.write_json(result)
        except (OSError, RuntimeError, ValueError) as record_error:
            result["failure_record_error"] = str(record_error)
            if not args.json:
                print(f"Could not save the failure result: {record_error}", file=sys.stderr)
        print(json.dumps(result, indent=2) if args.json else str(error))
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
