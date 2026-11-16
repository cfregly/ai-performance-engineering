"""Launch the native lab and validate its retained result without substituting paths."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
from typing import Any


LAB_DIR = Path(__file__).resolve().parent
SCHEMA = "multi_gpu_communication.v1"


class NativeUnavailable(RuntimeError):
    """An explicitly requested native capability is unavailable."""


def require_unprofiled_native_config(config: Any) -> None:
    """Reject the external-profiler selection used by the harness CLI."""
    if (getattr(config, "enable_profiling", False)
            or getattr(config, "profile_mode", "none") not in {None, "none"}
            or getattr(config, "profile_type", "none") not in {None, "none"}):
        raise NativeUnavailable(
            "SKIPPED: The generic harness profiler cannot launch this native MPI child correctly. "
            "Use --profile none for the pair and an explicit per-rank native profiler launch for diagnostics"
        )


@dataclass(frozen=True)
class NativeWorkload:
    backend: str = "nccl"
    schedule: str = "overlap"
    nx: int = 256
    ny: int = 257
    iterations: int = 20
    warmup: int = 2
    ranks: int = 2
    registered_buffers: bool = False
    graph: bool = False
    assert_cuda_aware_mpi: bool = False

    def validate(self) -> None:
        if self.backend not in {"mpi", "nccl", "nvshmem"}:
            raise ValueError("backend must be mpi, nccl, or nvshmem")
        if self.schedule not in {"serialized", "overlap"}:
            raise ValueError("schedule must be serialized or overlap")
        if self.nx < 3 or self.ny < 3 or self.iterations < 1 or self.warmup < 0:
            raise ValueError("Require nx >= 3, ny >= 3, iterations >= 1, and warmup >= 0")
        if self.ranks < 2 or self.ranks > self.ny - 2:
            raise ValueError("Require at least two ranks and at least one owned row per rank")
        if max(self.nx, self.ny, self.iterations, self.warmup, self.ranks) > 2**31 - 1:
            raise ValueError("Native dimensions and iteration counts must fit a signed 32-bit integer")
        if self.backend != "nccl" and (self.registered_buffers or self.graph):
            raise NativeUnavailable("SKIPPED: Registered buffers and graph replay require NCCL")

    def arguments(self) -> list[str]:
        self.validate()
        result = [
            "--backend", self.backend, "--schedule", self.schedule,
            "--nx", str(self.nx), "--ny", str(self.ny),
            "--iterations", str(self.iterations), "--warmup", str(self.warmup),
        ]
        if self.registered_buffers:
            result.append("--registered-buffers")
        if self.graph:
            result.append("--graph")
        if self.assert_cuda_aware_mpi:
            result.append("--assert-cuda-aware-mpi")
        return result


def source_hashes() -> dict[str, str]:
    names = ("jacobi_contract.hpp", "jacobi_native.cu", "Makefile", "native_common.py")
    return {name: hashlib.sha256((LAB_DIR / name).read_bytes()).hexdigest() for name in names}


def validate_result(result: dict[str, Any], workload: NativeWorkload) -> None:
    expected = {
        "schema": SCHEMA, "status": "passed", "stopping_policy": "fixed_iterations",
        "backend": workload.backend, "schedule": workload.schedule,
        "registered_buffers": workload.registered_buffers, "graph": workload.graph,
        "nx": workload.nx, "ny": workload.ny, "iterations": workload.iterations,
        "warmup": workload.warmup, "ranks": workload.ranks,
        "checked_owned_elements": workload.nx * (workload.ny - 2),
        "checked_halo_elements": 2 * workload.ranks * workload.nx,
    }
    for name, value in expected.items():
        if result.get(name) != value:
            raise RuntimeError(f"Native result field {name!r} differs from the requested workload")
    for name in ("setup_ms", "allocation_registration_ms", "capture_ms", "warmup_ms",
                 "steady_ms", "time_per_iteration_ms", "verification_ms", "cleanup_ms", "max_abs_error"):
        value = result.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0:
            raise RuntimeError(f"Native result field {name!r} must be finite and nonnegative")
    if result["steady_ms"] <= 0 or result["time_per_iteration_ms"] <= 0:
        raise RuntimeError("Native timing must be positive")
    if not math.isclose(result["time_per_iteration_ms"] * workload.iterations,
                        result["steady_ms"], rel_tol=1e-8, abs_tol=1e-9):
        raise RuntimeError("Native per-iteration time disagrees with the completed solve")
    if result["max_abs_error"] > 2e-6:
        raise RuntimeError("Native complete-output error exceeds the declared tolerance")
    mapping = result.get("rank_mapping")
    if not isinstance(mapping, list) or len(mapping) != workload.ranks:
        raise RuntimeError("Native result is missing rank coverage")
    if {row.get("rank") for row in mapping} != set(range(workload.ranks)):
        raise RuntimeError("Native result has incomplete or duplicate ranks")
    if sum(row.get("owned_rows", 0) for row in mapping) != workload.ny - 2:
        raise RuntimeError("Native rank decomposition does not cover the mesh")


def invoke_native(
    workload: NativeWorkload,
    *,
    binary: Path,
    json_out: Path,
    dump_output: Path | None = None,
    launcher: str = "mpirun",
    launcher_args: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Run to completion and require a fresh result and the requested complete output."""
    arguments = workload.arguments()
    binary = binary.resolve()
    json_out = json_out.resolve()
    if dump_output is not None and dump_output.resolve() == json_out:
        raise ValueError("JSON and complete-output artifacts must use different paths")
    if not binary.is_file():
        raise NativeUnavailable(f"SKIPPED: Native binary is missing. Build {LAB_DIR}/jacobi_native first")
    receipt_path = binary.with_name(binary.name + ".build.json")
    if not receipt_path.is_file():
        raise RuntimeError("Native build provenance is missing. Rebuild with the lab Makefile")
    build = json.loads(receipt_path.read_text())
    binary_hash = hashlib.sha256(binary.read_bytes()).hexdigest()
    if build.get("schema") != "multi_gpu_communication.build.v1" or build.get("binary_sha256") != binary_hash:
        raise RuntimeError("Native binary differs from its build receipt. Rebuild before running")
    for name in ("jacobi_contract.hpp", "jacobi_native.cu", "Makefile"):
        current_hash = hashlib.sha256((LAB_DIR / name).read_bytes()).hexdigest()
        if build.get("source_sha256", {}).get(name) != current_hash:
            raise RuntimeError(f"Native source {name} changed after the binary was built. Rebuild before running")
    if workload.backend == "nccl" and not build.get("with_nccl"):
        raise NativeUnavailable("SKIPPED: The native build does not include NCCL")
    if workload.backend == "nvshmem" and not build.get("with_nvshmem"):
        raise NativeUnavailable("SKIPPED: The native build does not include NVSHMEM")
    executable = shutil.which(launcher)
    if executable is None:
        raise NativeUnavailable(f"SKIPPED: MPI launcher {launcher!r} is unavailable")
    paths = [json_out] + ([dump_output.resolve()] if dump_output is not None else [])
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Refusing to reuse an existing native artifact: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    command = [executable, *launcher_args, "-np", str(workload.ranks), str(binary),
               *arguments, "--json-out", str(json_out)]
    if dump_output is not None:
        command.extend(["--dump-output", str(dump_output.resolve())])
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        lines = (completed.stdout + "\n" + completed.stderr).splitlines()
        skip = next((line[line.index("SKIPPED:"):] for line in lines if "SKIPPED:" in line), None)
        if skip is not None:
            raise NativeUnavailable(skip)
        raise RuntimeError(f"Native run failed with exit {completed.returncode}\n{completed.stdout}\n{completed.stderr}")
    if not json_out.is_file():
        raise RuntimeError("The native process exited without its requested result artifact")
    result = json.loads(json_out.read_text())
    validate_result(result, workload)
    if dump_output is not None:
        expected_bytes = workload.nx * workload.ny * 4
        if not dump_output.is_file() or dump_output.stat().st_size != expected_bytes:
            raise RuntimeError("The native process did not retain the complete timed float32 output")
    result["source_sha256"] = source_hashes()
    result["build"] = build
    result["binary_sha256"] = binary_hash
    result["command"] = command
    result["process_exit_code"] = completed.returncode
    json_out.write_text(json.dumps(result, indent=2) + "\n")
    return result
