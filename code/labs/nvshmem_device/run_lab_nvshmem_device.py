"""Run native NVSHMEM device communication and independently check its full output."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import struct
import sys
import tempfile
from array import array
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.utils.cooperative_process_artifacts import run_recorded_cooperative_process

LAB_DIR = Path(__file__).resolve().parent
SCHEMA = "nvshmem_device.v1"
BUILD_SCHEMA = "nvshmem_device.build.v1"
HALO_MODES = ("scalar-halo", "persistent-halo")
MODES = (*HALO_MODES, "block-signal", "persistent-signal", "persistent-atomic")
SOURCE_FILES = ("nvshmem_device_native.cu", "Makefile", "build_receipt.py",
                "../multi_gpu_communication/jacobi_contract.hpp")


class NativeUnavailable(RuntimeError):
    """A requested native capability is unavailable."""


@dataclass(frozen=True)
class NativeWorkload:
    mode: str = "scalar-halo"
    nx: int = 17
    ny: int = 7
    elements: int = 32
    iterations: int = 4
    warmup: int = 1
    ranks: int = 2

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        values = (self.nx, self.ny, self.elements, self.iterations, self.warmup, self.ranks)
        if any(not isinstance(value, int) or isinstance(value, bool) for value in values):
            raise ValueError("Workload dimensions and counts must be integers")
        if any(value > 2**31 - 1 for value in values):
            raise ValueError("Native dimensions and counts must fit signed 32-bit integers")
        if self.nx < 3 or self.ny < 3 or self.elements < 1 or self.iterations < 1 or self.warmup < 0:
            raise ValueError("Require nx >= 3, ny >= 3, elements >= 1, iterations >= 1, and warmup >= 0")
        if self.ranks < 2:
            raise ValueError("At least two ranks are required")
        if self.mode in HALO_MODES and self.ranks > self.ny - 2:
            raise ValueError("Each halo rank must own at least one row")
        if self.mode not in HALO_MODES and 2 * max(self.iterations, self.warmup) * self.ranks * self.elements > 2**24:
            raise ValueError("Tile values must remain exactly representable float32 integers (maximum 2^24)")
        if self.expected_output_elements * 4 > sys.maxsize:
            raise ValueError("Complete output exceeds the addressable file size")

    @property
    def expected_output_elements(self) -> int:
        if self.mode in HALO_MODES:
            return self.nx * self.ny
        return self.ranks * self.iterations * 2 * self.elements

    @property
    def output_layout(self) -> str:
        return "ny,nx" if self.mode in HALO_MODES else "rank,epoch,incoming_edge,element"

    def arguments(self) -> list[str]:
        self.validate()
        return ["--mode", self.mode, "--nx", str(self.nx), "--ny", str(self.ny),
                "--elements", str(self.elements), "--iterations", str(self.iterations),
                "--warmup", str(self.warmup)]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes(directory: Path = LAB_DIR) -> dict[str, str]:
    return {name: sha256(directory / name) for name in SOURCE_FILES}


def _f32(value: float) -> float:
    return struct.unpack("=f", struct.pack("=f", value))[0]


def scalar_reference(nx: int, ny: int, iterations: int) -> array:
    """Compute the Jacobi contract in Python without invoking the C++ reference."""
    if nx < 3 or ny < 3 or iterations < 0:
        raise ValueError("Reference requires nx >= 3, ny >= 3, and nonnegative iterations")
    current = array("f", [0.0]) * (nx * ny)
    for y in range(ny):
        periodic_y = ny - 2 if y == 0 else (1 if y == ny - 1 else y)
        phase = 2 * math.pi * (periodic_y - 1) / (ny - 2)
        for x in range(nx):
            value = math.sin(phase) if x in (0, nx - 1) else (
                0.05 * math.sin(math.pi * x / (nx - 1)) * math.cos(phase))
            current[y * nx + x] = value
    following = array("f", current)
    for _ in range(iterations):
        for y in range(1, ny - 1):
            for x in range(1, nx - 1):
                at = y * nx + x
                total = _f32(current[at - 1] + current[at + 1])
                total = _f32(total + current[at - nx])
                total = _f32(total + current[at + nx])
                following[at] = _f32(0.25 * total)
        following[:nx] = following[(ny - 2) * nx:(ny - 1) * nx]
        following[(ny - 1) * nx:] = following[nx:2 * nx]
        current, following = following, current
    return current


def expected_output(workload: NativeWorkload) -> Iterator[float]:
    """Yield the complete scalar mesh or every received tile in the retained layout."""
    workload.validate()
    if workload.mode in HALO_MODES:
        yield from scalar_reference(workload.nx, workload.ny, workload.iterations)
        return
    for rank in range(workload.ranks):
        for epoch in range(1, workload.iterations + 1):
            for incoming_edge in range(2):
                source = (rank - 1) % workload.ranks if incoming_edge == 0 else (rank + 1) % workload.ranks
                source_edge = 1 - incoming_edge
                first = 1 + (((epoch - 1) * workload.ranks + source) * 2 + source_edge) * workload.elements
                for index in range(workload.elements):
                    yield float(first + index)


def validate_full_output(path: Path, workload: NativeWorkload) -> dict:
    workload.validate()
    if not path.is_file() or path.stat().st_size != workload.expected_output_elements * 4:
        raise RuntimeError("Native output is missing or does not contain the complete float32 array")
    expected = expected_output(workload)
    tolerance = 2e-6 if workload.mode in HALO_MODES else 0.0
    checked = 0
    maximum = 0.0
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 65536):
            for (actual,) in struct.iter_unpack("=f", chunk):
                reference = next(expected)
                error = abs(actual - reference)
                if not math.isfinite(actual) or error > tolerance:
                    raise RuntimeError(f"Independent full-output check failed at element {checked}: {actual} versus {reference}")
                maximum = max(maximum, error)
                checked += 1
    if checked != workload.expected_output_elements or next(expected, None) is not None:
        raise RuntimeError("Independent reference and native output sizes disagree")
    return {"method": "independent_python_all_elements", "checked_elements": checked,
            "max_abs_error": maximum, "absolute_tolerance": tolerance,
            "output_sha256": sha256(path)}


def validate_result(result: dict, workload: NativeWorkload) -> None:
    workload.validate()
    if not isinstance(result, dict):
        raise RuntimeError("Native JSON result must be an object")
    expected = {"schema": SCHEMA, "status": "VERIFIED", "canonical": False, "verified": True,
                "mode": workload.mode, "nx": workload.nx, "ny": workload.ny,
                "elements": workload.elements, "iterations": workload.iterations,
                "warmup": workload.warmup, "ranks": workload.ranks,
                "output_dtype": "float32", "output_layout": workload.output_layout,
                "output_elements": workload.expected_output_elements,
                "checked_output_elements": workload.expected_output_elements,
                "expected_ready_epoch": 0 if workload.mode == "scalar-halo" else workload.iterations,
                "expected_consumed_epoch": (0 if workload.mode == "scalar-halo" else
                                            workload.iterations - int(workload.mode == "persistent-halo"))}
    for name, value in expected.items():
        if type(result.get(name)) is not type(value) or result.get(name) != value:
            raise RuntimeError(f"Native result field {name!r} differs from the requested contract")
    for name, components in (("nvshmem_version", 3), ("nvshmem_header_version", 3),
                             ("openshmem_api_version", 2)):
        value = result.get(name)
        parts = value.split(".") if isinstance(value, str) else []
        if len(parts) != components or any(not part.isascii() or not part.isdecimal() for part in parts):
            raise RuntimeError(
                f"Native result field {name!r} must contain {components} numeric version components"
            )
    for name in ("setup_ms", "allocation_ms", "warmup_ms", "reset_ms", "steady_ms",
                 "time_per_iteration_ms", "verification_ms", "cleanup_ms", "max_abs_error"):
        value = result.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise RuntimeError(f"Native result field {name!r} must be finite and nonnegative")
    if result["steady_ms"] <= 0 or result["time_per_iteration_ms"] <= 0:
        raise RuntimeError("Native timing must be positive")
    if not math.isclose(result["steady_ms"], result["time_per_iteration_ms"] * workload.iterations,
                        rel_tol=1e-8, abs_tol=1e-9):
        raise RuntimeError("Native solve and per-iteration timing disagree")
    if result["max_abs_error"] > (2e-6 if workload.mode in HALO_MODES else 0.0):
        raise RuntimeError("Native full-output error exceeds the workload tolerance")


def validate_build_receipt(binary: Path, directory: Path = LAB_DIR) -> dict:
    if not binary.is_file():
        raise NativeUnavailable(f"SKIPPED: Native binary is missing. Build {LAB_DIR}/nvshmem_device_native first")
    receipt_path = binary.with_name(binary.name + ".build.json")
    if not receipt_path.is_file():
        raise RuntimeError("Native build receipt is missing. Rebuild with the lab Makefile")
    receipt = json.loads(receipt_path.read_text())
    if not isinstance(receipt, dict) or receipt.get("schema") != BUILD_SCHEMA:
        raise RuntimeError("Native build receipt has an unsupported schema")
    if receipt.get("binary_sha256") != sha256(binary):
        raise RuntimeError("Native binary differs from its build receipt")
    if receipt.get("source_sha256") != source_hashes(directory):
        raise RuntimeError("Native source or Jacobi dependency changed after the build. Rebuild before running")
    if not receipt.get("compile_command") or not isinstance(receipt["compile_command"], list):
        raise RuntimeError("Native receipt does not retain its compile command")
    if not all(isinstance(value, str) for value in receipt["compile_command"]):
        raise RuntimeError("Native compile command must contain string arguments")
    if not isinstance(receipt.get("compiler_versions"), dict) or not all(
            receipt["compiler_versions"].get(name) for name in ("nvcc", "mpicxx")):
        raise RuntimeError("Native receipt is missing compiler versions")
    return receipt


def nvshmem_child_environment() -> tuple[dict[str, str], Path]:
    """Select the build's NVSHMEM library directory for one child process tree."""
    if "NVSHMEM_LIBDIR" in os.environ:
        configured = os.environ["NVSHMEM_LIBDIR"]
        if not configured:
            raise RuntimeError("NVSHMEM_LIBDIR must not be empty")
        runtime_libdir = Path(configured)
    else:
        home = os.environ.get("NVSHMEM_HOME")
        if home is None:
            raise NativeUnavailable(
                "SKIPPED: Set NVSHMEM_LIBDIR or NVSHMEM_HOME for the NVSHMEM runtime libraries"
            )
        if not home:
            raise RuntimeError("NVSHMEM_HOME must not be empty")
        runtime_libdir = Path(home) / "lib"
    if not runtime_libdir.is_absolute():
        # The Makefile runs from LAB_DIR, so resolve relative build inputs there too.
        runtime_libdir = LAB_DIR / runtime_libdir
    runtime_libdir = Path(os.path.abspath(runtime_libdir))
    if not runtime_libdir.is_dir():
        raise RuntimeError(f"NVSHMEM runtime library directory does not exist: {runtime_libdir}")

    child_env = dict(os.environ)
    inherited = child_env.get("LD_LIBRARY_PATH")
    child_env["LD_LIBRARY_PATH"] = (
        str(runtime_libdir)
        if not inherited
        else str(runtime_libdir) + os.pathsep + inherited
    )
    return child_env, runtime_libdir


def invoke_native(workload: NativeWorkload, *, binary: Path, json_out: Path, dump_output: Path,
                  launcher: str = "mpirun", launcher_args: tuple[str, ...] = ()) -> dict:
    """Launch real MPI and retain native evidence plus independent full-output checks."""
    arguments = workload.arguments()
    binary, json_out, dump_output = binary.resolve(), json_out.resolve(), dump_output.resolve()
    native_json = json_out.with_name(json_out.name + ".native.json")
    stdout_path = json_out.with_name(json_out.name + ".stdout.log")
    stderr_path = json_out.with_name(json_out.name + ".stderr.log")
    execution_path = json_out.with_name(json_out.name + ".execution.json")
    outputs = (json_out, dump_output, native_json, stdout_path, stderr_path, execution_path)
    if len(set(outputs)) != len(outputs):
        raise ValueError(
            "Final JSON, native JSON, full output, stream logs, and execution receipt "
            "must use different paths"
        )
    for path in outputs:
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to reuse an existing artifact: {path}")
    build = validate_build_receipt(binary)
    snapshot = source_hashes()
    binary_hash = sha256(binary)
    executable = shutil.which(launcher)
    if executable is None:
        raise NativeUnavailable(f"SKIPPED: MPI launcher {launcher!r} is unavailable")
    child_env, runtime_libdir = nvshmem_child_environment()
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    command = [executable, *launcher_args, "-np", str(workload.ranks), str(binary), *arguments,
               "--json-out", str(native_json), "--dump-output", str(dump_output)]
    completed = run_recorded_cooperative_process(
        command, output_prefix=json_out, env=child_env,
    )
    if completed["returncode"] != 0:
        diagnostic = next((line[line.index("SKIPPED:"):] for line in (completed["stdout"] + "\n" + completed["stderr"]).splitlines()
                           if "SKIPPED:" in line), None)
        if diagnostic is not None:
            raise NativeUnavailable(diagnostic)
        raise RuntimeError(f"Native MPI run failed with exit {completed['returncode']}. See {stdout_path} and {stderr_path}")
    if source_hashes() != snapshot or sha256(binary) != binary_hash:
        raise RuntimeError("Native source or binary changed during execution")
    if not native_json.is_file():
        raise RuntimeError("Native MPI exited without the requested raw JSON result")
    result = json.loads(native_json.read_text())
    validate_result(result, workload)
    independent = validate_full_output(dump_output, workload)
    result.update({"python_verification": independent, "build": build,
                   "source_sha256": snapshot, "binary_sha256": binary_hash,
                   "runner_sha256": sha256(Path(__file__)), "command": command,
                   "nvshmem_runtime_libdir": str(runtime_libdir),
                   "native_result_path": str(native_json), "output_path": str(dump_output),
                   "stdout_path": str(stdout_path), "stderr_path": str(stderr_path),
                   "process_exit_code": completed["returncode"],
                   "execution_path": completed["artifact_paths"]["execution"]})
    with json_out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="scalar-halo")
    for name, default in (("nx", 17), ("ny", 7), ("elements", 32), ("iterations", 4), ("warmup", 1), ("ranks", 2)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--binary", type=Path, default=LAB_DIR / "nvshmem_device_native")
    parser.add_argument("--launcher", default="mpirun")
    parser.add_argument("--launcher-arg", action="append", default=[])
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--dump-output", type=Path)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result_path = args.json_out
    try:
        workload = NativeWorkload(**{name: getattr(args, name) for name in (
            "mode", "nx", "ny", "elements", "iterations", "warmup", "ranks")})
        workload.validate()
        if result_path is None:
            root = LAB_DIR.parents[1] / "artifacts" / "nvshmem_device"
            root.mkdir(parents=True, exist_ok=True)
            result_path = Path(tempfile.mkdtemp(prefix=args.mode + "_", dir=root)) / "result.json"
        output_path = args.dump_output or result_path.with_suffix(".f32")
        result = invoke_native(workload, binary=args.binary, json_out=result_path,
                               dump_output=output_path, launcher=args.launcher,
                               launcher_args=tuple(args.launcher_arg))
    except (NativeUnavailable, ValueError, RuntimeError, OSError) as error:
        unavailable = isinstance(error, NativeUnavailable)
        result = {"schema": SCHEMA, "status": "SKIPPED" if unavailable else "FAILED", "reason": str(error)}
        try:
            if result_path is not None and not result_path.exists() and not result_path.is_symlink():
                result_path.parent.mkdir(parents=True, exist_ok=True)
                with result_path.open("x") as stream:
                    json.dump(result, stream, indent=2)
                    stream.write("\n")
        except OSError as record_error:
            result["failure_record_error"] = str(record_error)
            if not args.json:
                print(f"Could not save the failure result: {record_error}", file=sys.stderr)
        print(json.dumps(result) if args.json else str(error))
        return 77 if unavailable else 1
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(f"VERIFIED: {workload.mode}, {workload.ranks} ranks, {workload.iterations} iterations")
        print(f"Independent check: {result['python_verification']['checked_elements']} float32 values")
        print(f"Result: {result_path}")
        print(f"Full output: {result['output_path']}")
        print("Diagnostic, non-canonical timing. Performance qualification requires separate evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
