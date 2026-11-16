"""Real CPU contract checks and opt-in native multi-GPU integration tests."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


CODE_ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = CODE_ROOT / "labs" / "multi_gpu_communication"

_CPP_DRIVER = r"""
#include "jacobi_contract.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    using namespace multi_gpu_communication;
    try {
        if (argc < 2) throw std::invalid_argument("A mode is required");
        const std::string mode = argv[1];
        std::cout << std::setprecision(9);
        if (mode == "domains" && argc == 4) {
            const int ny = std::stoi(argv[2]);
            const int ranks = std::stoi(argv[3]);
            if (ranks < 1) throw std::invalid_argument("Positive ranks are required");
            std::cout << '[';
            for (int rank = 0; rank < ranks; ++rank) {
                const auto domain = decompose(ny, ranks, rank);
                if (rank) std::cout << ',';
                std::cout << "{\"rank\":" << domain.rank
                          << ",\"ranks\":" << domain.ranks
                          << ",\"rows\":" << domain.rows
                          << ",\"global_start\":" << domain.global_start << '}';
            }
            std::cout << "]\n";
        } else if (mode == "domain" && argc == 5) {
            const auto domain = decompose(std::stoi(argv[2]), std::stoi(argv[3]),
                                          std::stoi(argv[4]));
            std::cout << domain.rows << '\n';
        } else if (mode == "mesh" && argc == 5) {
            const auto values = reference_jacobi(std::stoi(argv[2]), std::stoi(argv[3]),
                                                  std::stoi(argv[4]));
            std::cout << '[';
            for (std::size_t index = 0; index < values.size(); ++index) {
                if (index) std::cout << ',';
                std::cout << values[index];
            }
            std::cout << "]\n";
        } else if (mode == "initial" && argc == 6) {
            std::cout << initial_value(std::stoi(argv[2]), std::stoi(argv[3]),
                                      std::stoi(argv[4]), std::stoi(argv[5])) << '\n';
        } else {
            throw std::invalid_argument("Unknown mode or argument count");
        }
        return 0;
    } catch (const std::invalid_argument& error) {
        std::cerr << error.what() << '\n';
        return 2;
    }
}
"""


@pytest.fixture(scope="module")
def contract_driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("A real C++17 compiler is required for the native CPU contract")
    build_dir = tmp_path_factory.mktemp("multi-gpu-contract")
    source = build_dir / "contract_driver.cpp"
    binary = build_dir / "contract_driver"
    source.write_text(_CPP_DRIVER, encoding="utf-8")
    result = subprocess.run(
        [
            *compiler,
            "-std=c++17",
            "-O0",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(LAB_ROOT),
            str(source),
            "-o",
            str(binary),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


def _contract_output(driver: Path, *arguments: object) -> object:
    result = subprocess.run(
        [str(driver), *(str(argument) for argument in arguments)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize("ny,ranks", [(3, 1), (5, 3), (6, 2), (9, 3), (10, 4), (19, 4)])
def test_decomposition_covers_every_interior_row_once(
    contract_driver: Path, ny: int, ranks: int
) -> None:
    domains = _contract_output(contract_driver, "domains", ny, ranks)
    covered_rows = []
    row_counts = []
    for rank, domain in enumerate(domains):
        assert domain["rank"] == rank
        assert domain["ranks"] == ranks
        assert domain["rows"] >= 1
        row_counts.append(domain["rows"])
        covered_rows.extend(
            range(domain["global_start"], domain["global_start"] + domain["rows"])
        )
    assert covered_rows == list(range(1, ny - 1))
    assert max(row_counts) - min(row_counts) <= 1


@pytest.mark.parametrize("ny,ranks,rank", [(2, 1, 0), (5, 0, 0), (5, 4, 0), (5, 2, -1), (5, 2, 2)])
def test_decomposition_rejects_invalid_or_empty_partitions(
    contract_driver: Path, ny: int, ranks: int, rank: int
) -> None:
    result = subprocess.run(
        [str(contract_driver), "domain", str(ny), str(ranks), str(rank)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "at least one" in result.stderr


def test_initial_mesh_has_periodic_halos_and_fixed_x_boundaries(contract_driver: Path) -> None:
    nx, ny = 7, 10
    values = _contract_output(contract_driver, "mesh", nx, ny, 0)
    rows = [values[y * nx : (y + 1) * nx] for y in range(ny)]
    assert rows[0] == rows[-2]
    assert rows[-1] == rows[1]
    for y in range(1, ny - 1):
        expected = math.sin(2.0 * math.pi * (y - 1) / (ny - 2))
        assert rows[y][0] == pytest.approx(expected, abs=1e-7)
        assert rows[y][-1] == pytest.approx(expected, abs=1e-7)
    assert rows[1][nx // 2] == pytest.approx(0.05, abs=1e-8)


def _periodic_stencil_oracle(initial: list[float], nx: int, ny: int, steps: int) -> list[float]:
    # Store only owned rows. Periodic neighbors use modular indices without halos.
    interior = [initial[y * nx : (y + 1) * nx] for y in range(1, ny - 1)]
    for _ in range(steps):
        updated = [row[:] for row in interior]
        for row_index, row in enumerate(interior):
            above = interior[(row_index - 1) % len(interior)]
            below = interior[(row_index + 1) % len(interior)]
            for x in range(1, nx - 1):
                updated[row_index][x] = (sum((row[x - 1], row[x + 1], above[x], below[x]))) / 4.0
        interior = updated
    complete = [interior[-1], *interior, interior[0]]
    return [value for row in complete for value in row]


@pytest.mark.parametrize("nx,ny,steps", [(3, 3, 1), (5, 4, 2), (7, 9, 3), (11, 8, 6), (9, 7, 7)])
def test_cpu_reference_matches_independent_periodic_stencil(
    contract_driver: Path, nx: int, ny: int, steps: int
) -> None:
    initial = _contract_output(contract_driver, "mesh", nx, ny, 0)
    expected = _periodic_stencil_oracle(initial, nx, ny, steps)
    actual = _contract_output(contract_driver, "mesh", nx, ny, steps)
    assert len(actual) == nx * ny
    assert all(math.isfinite(value) for value in actual)
    assert actual == pytest.approx(expected, abs=3e-7, rel=3e-7)
    for y in range(ny):
        assert actual[y * nx] == initial[y * nx]
        assert actual[(y + 1) * nx - 1] == initial[(y + 1) * nx - 1]


@pytest.mark.parametrize("nx,ny,steps", [(2, 5, 0), (5, 2, 0), (5, 5, -1)])
def test_cpu_reference_rejects_invalid_shapes_and_steps(
    contract_driver: Path, nx: int, ny: int, steps: int
) -> None:
    result = subprocess.run(
        [str(contract_driver), "mesh", str(nx), str(ny), str(steps)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "Require nx >= 3" in result.stderr


@pytest.mark.parametrize("x,y", [(-1, 1), (5, 1), (1, -1), (1, 6)])
def test_initial_value_rejects_coordinates_outside_the_mesh(
    contract_driver: Path, x: int, y: int
) -> None:
    result = subprocess.run(
        [str(contract_driver), "initial", "5", "6", str(x), str(y)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "out of range" in result.stderr


def test_runner_help_describes_real_execution_modes() -> None:
    result = subprocess.run(
        [sys.executable, str(LAB_ROOT / "run_lab_multi_gpu_communication.py"), "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for option in ("--backend", "--schedule", "--registered-buffers", "--graph", "--nx", "--ny"):
        assert option in result.stdout


@pytest.mark.parametrize(
    "arguments,exit_code,status,reason",
    [
        (["--ranks", "1"], 1, "failed", "at least two ranks"),
        (["--nx", "2"], 1, "failed", "nx >= 3"),
        (["--backend", "mpi", "--graph"], 3, "skipped", "require NCCL"),
        (["--backend", "nvshmem", "--registered-buffers"], 3, "skipped", "require NCCL"),
    ],
)
def test_runner_rejects_invalid_work_before_launch(
    arguments: list[str], exit_code: int, status: str, reason: str
) -> None:
    result = subprocess.run(
        [sys.executable, str(LAB_ROOT / "run_lab_multi_gpu_communication.py"), *arguments, "--json"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema"] == "multi_gpu_communication.v1"
    assert payload["status"] == status
    assert reason in payload["reason"]


def test_missing_native_binary_reports_an_explicit_skip(tmp_path: Path) -> None:
    artifact = tmp_path / "result.json"
    result = subprocess.run(
        [
            sys.executable,
            str(LAB_ROOT / "run_lab_multi_gpu_communication.py"),
            "--binary",
            str(tmp_path / "absent-native-binary"),
            "--json-out",
            str(artifact),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "skipped"
    assert "Native binary is missing" in payload["reason"]
    assert json.loads(artifact.read_text(encoding="utf-8")) == payload


@pytest.mark.parametrize(
    "arguments,reason",
    [
        (["--unrecognized-option"], "unrecognized arguments"),
        (["--backend", "unknown"], "invalid choice"),
        (["--nx"], "expected one argument"),
    ],
)
def test_runner_argparse_errors_do_not_launch_native_work(
    arguments: list[str], reason: str
) -> None:
    result = subprocess.run(
        [sys.executable, str(LAB_ROOT / "run_lab_multi_gpu_communication.py"), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert reason in result.stderr


@pytest.fixture
def cpu_provenance_input(contract_driver: Path, tmp_path: Path) -> Path:
    # This real CPU executable is deliberately not a CUDA lab executable.
    # Receipt tests below assert metadata or pre-launch rejection only.
    binary = tmp_path / "cpu-rejection-input"
    shutil.copy2(contract_driver, binary)
    return binary


def _record_cpu_receipt(binary: Path) -> Path:
    compiler = shlex.split(os.environ.get("CXX", "c++"))[0]
    result = subprocess.run(
        [
            sys.executable,
            str(LAB_ROOT / "build_receipt.py"),
            "--binary",
            str(binary),
            "--arch",
            "cpu-rejection-test",
            "--with-nccl",
            "0",
            "--with-nvshmem",
            "0",
            "--nvcc",
            compiler,
            "--mpicxx",
            compiler,
            "--build-option",
            "purpose=cpu-rejection-test",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary.with_name(binary.name + ".build.json")


def _rejected_runner_output(binary: Path, backend: str = "mpi") -> tuple[int, dict]:
    result = subprocess.run(
        [
            sys.executable,
            str(LAB_ROOT / "run_lab_multi_gpu_communication.py"),
            "--binary",
            str(binary),
            "--backend",
            backend,
            "--launcher",
            str(binary.parent / "absent-mpi-launcher"),
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.stdout, result.stderr
    return result.returncode, json.loads(result.stdout)


def test_build_receipt_records_supplied_file_hashes_without_gpu_claims(
    cpu_provenance_input: Path,
) -> None:
    receipt_path = _record_cpu_receipt(cpu_provenance_input)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema"] == "multi_gpu_communication.build.v1"
    assert receipt["binary_sha256"] == hashlib.sha256(cpu_provenance_input.read_bytes()).hexdigest()
    assert receipt["architecture"] == "cpu-rejection-test"
    assert receipt["with_nccl"] is False
    assert receipt["with_nvshmem"] is False
    assert receipt["build_options"]["purpose"] == "cpu-rejection-test"
    assert receipt["compiler_versions"]["nvcc"]
    assert receipt["compiler_versions"]["mpicxx"]
    for name in ("jacobi_contract.hpp", "jacobi_native.cu", "Makefile"):
        assert receipt["source_sha256"][name] == hashlib.sha256((LAB_ROOT / name).read_bytes()).hexdigest()


def test_runner_rejects_missing_build_receipt(cpu_provenance_input: Path) -> None:
    exit_code, payload = _rejected_runner_output(cpu_provenance_input)
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert "build provenance is missing" in payload["reason"]


def test_runner_rejects_binary_changed_after_receipt(cpu_provenance_input: Path) -> None:
    _record_cpu_receipt(cpu_provenance_input)
    with cpu_provenance_input.open("ab") as binary:
        binary.write(b"changed-after-receipt")
    exit_code, payload = _rejected_runner_output(cpu_provenance_input)
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert "differs from its build receipt" in payload["reason"]


@pytest.mark.parametrize("source_name", ["jacobi_contract.hpp", "jacobi_native.cu", "Makefile"])
def test_runner_rejects_stale_source_hash(
    cpu_provenance_input: Path, source_name: str
) -> None:
    receipt_path = _record_cpu_receipt(cpu_provenance_input)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["source_sha256"][source_name] = hashlib.sha256(b"previous-source-version").hexdigest()
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    exit_code, payload = _rejected_runner_output(cpu_provenance_input)
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert f"Native source {source_name} changed" in payload["reason"]


def test_runner_rejects_unknown_build_receipt_schema(cpu_provenance_input: Path) -> None:
    receipt_path = _record_cpu_receipt(cpu_provenance_input)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["schema"] = "unsupported-build-receipt"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    exit_code, payload = _rejected_runner_output(cpu_provenance_input)
    assert exit_code == 1
    assert payload["status"] == "failed"
    assert "differs from its build receipt" in payload["reason"]


@pytest.mark.parametrize("backend", ["nccl", "nvshmem"])
def test_runner_rejects_backend_missing_from_build(
    cpu_provenance_input: Path, backend: str
) -> None:
    _record_cpu_receipt(cpu_provenance_input)
    exit_code, payload = _rejected_runner_output(cpu_provenance_input, backend=backend)
    assert exit_code == 3
    assert payload["status"] == "skipped"
    assert f"does not include {backend.upper()}" in payload["reason"]


@pytest.fixture(scope="module")
def native_controls():
    # Load the real stdlib control module without importing the Torch harness.
    module_name = "_test_multi_gpu_communication_native_controls"
    specification = importlib.util.spec_from_file_location(module_name, LAB_ROOT / "native_common.py")
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
        yield module
    finally:
        sys.modules.pop(module_name, None)


def _result_metadata(workload) -> dict:
    """Create contract-parser input without claiming native execution."""
    rows, remainder = divmod(workload.ny - 2, workload.ranks)
    versions = {
        "nvshmem_version": "3.7.2",
        "nvshmem_header_version": "3.7.2",
        "openshmem_api_version": "1.3",
    }
    return {
        "schema": "multi_gpu_communication.v1", "status": "passed",
        "stopping_policy": "fixed_iterations", "backend": workload.backend,
        "schedule": workload.schedule, "registered_buffers": workload.registered_buffers,
        "graph": workload.graph, "nx": workload.nx, "ny": workload.ny,
        "iterations": workload.iterations, "warmup": workload.warmup, "ranks": workload.ranks,
        "checked_owned_elements": workload.nx * (workload.ny - 2),
        "checked_halo_elements": 2 * workload.ranks * workload.nx,
        "setup_ms": 1.0, "allocation_registration_ms": 0.1, "capture_ms": 0.0,
        "warmup_ms": 0.2, "steady_ms": float(workload.iterations),
        "time_per_iteration_ms": 1.0, "verification_ms": 0.2, "cleanup_ms": 0.1,
        "max_abs_error": 0.0,
        "rank_mapping": [{"rank": rank, "owned_rows": rows + int(rank < remainder)}
                         for rank in range(workload.ranks)],
        **{name: value if workload.backend == "nvshmem" else None
           for name, value in versions.items()},
    }


@pytest.mark.parametrize("field", ["nvshmem_version", "nvshmem_header_version", "openshmem_api_version"])
@pytest.mark.parametrize("missing", [False, True], ids=["null", "missing"])
def test_nvshmem_result_requires_runtime_header_and_api_versions(native_controls, field: str, missing: bool) -> None:
    workload = native_controls.NativeWorkload(backend="nvshmem")
    result = _result_metadata(workload)
    if missing:
        del result[field]
    else:
        result[field] = None
    with pytest.raises(RuntimeError, match=field):
        native_controls.validate_result(result, workload)


@pytest.mark.parametrize("field,value", [
    ("nvshmem_version", "1.3"), ("nvshmem_header_version", "1.3"),
    ("openshmem_api_version", "3.7.2"), ("nvshmem_version", "3.7.2-dev"),
    ("nvshmem_header_version", "3..2"), ("openshmem_api_version", "1.-3"),
    ("nvshmem_version", 3.7), ("nvshmem_version", "٣.٧.٢"),
    ("openshmem_api_version", True),
])
def test_nvshmem_result_rejects_mislabeled_or_malformed_versions(native_controls, field: str, value) -> None:
    workload = native_controls.NativeWorkload(backend="nvshmem")
    result = _result_metadata(workload)
    result[field] = value
    with pytest.raises(RuntimeError, match=field):
        native_controls.validate_result(result, workload)


def test_nvshmem_result_accepts_distinct_numeric_versions_without_pinning_a_release(native_controls) -> None:
    workload = native_controls.NativeWorkload(backend="nvshmem")
    result = _result_metadata(workload)
    native_controls.validate_result(result, workload)
    result.update(nvshmem_version="4.0.0", nvshmem_header_version="3.8.1", openshmem_api_version="1.4")
    native_controls.validate_result(result, workload)


@pytest.mark.parametrize("backend", ["mpi", "nccl"])
@pytest.mark.parametrize("versions", ["null", "absent", "inactive"])
def test_result_version_checks_apply_only_to_nvshmem(native_controls, backend: str, versions: str) -> None:
    workload = native_controls.NativeWorkload(backend=backend)
    result = _result_metadata(workload)
    for field in ("nvshmem_version", "nvshmem_header_version", "openshmem_api_version"):
        if versions == "absent":
            del result[field]
        elif versions == "inactive":
            result[field] = "not used by this backend"
    native_controls.validate_result(result, workload)


@pytest.mark.parametrize(
    "configuration",
    [
        SimpleNamespace(enable_profiling=False, profile_mode="none", profile_type="minimal"),
        SimpleNamespace(enable_profiling=False, profile_mode="none", profile_type="deep_dive"),
        SimpleNamespace(enable_profiling=True, profile_mode="none", profile_type="none"),
        SimpleNamespace(enable_profiling=False, profile_mode="minimal", profile_type="none"),
        SimpleNamespace(enable_profiling=False, profile_mode="deep_dive", profile_type="none"),
    ],
    ids=["cli-minimal", "cli-deep-dive", "enabled", "mode-minimal", "mode-deep-dive"],
)
def test_native_profiler_guard_rejects_every_active_profile_field(
    native_controls, configuration: SimpleNamespace
) -> None:
    with pytest.raises(native_controls.NativeUnavailable, match="Use --profile none"):
        native_controls.require_unprofiled_native_config(configuration)


@pytest.mark.parametrize(
    "configuration",
    [
        None,
        SimpleNamespace(),
        SimpleNamespace(enable_profiling=False, profile_mode="none", profile_type="none"),
        SimpleNamespace(enable_profiling=False, profile_mode=None, profile_type=None),
    ],
    ids=["no-config", "missing-fields", "cli-none", "optional-none"],
)
def test_native_profiler_guard_accepts_unprofiled_configuration(
    native_controls, configuration: SimpleNamespace | None
) -> None:
    assert native_controls.require_unprofiled_native_config(configuration) is None


@pytest.fixture(scope="module")
def native_binary() -> tuple[Path, str]:
    if os.environ.get("AISP_RUN_MULTI_GPU_COMMUNICATION_TESTS") != "1":
        pytest.skip("Multi-GPU execution is opt-in after reserving at least two GPUs")
    for tool in ("make", "nvcc", "mpicxx"):
        if shutil.which(tool) is None:
            pytest.skip(f"Native multi-GPU integration requires real {tool}")
    launcher = shutil.which("mpiexec") or shutil.which("mpirun")
    if launcher is None:
        pytest.skip("Native multi-GPU integration requires an MPI launcher")
    result = subprocess.run(
        [
            "make",
            "--no-print-directory",
            "WITH_NCCL=1",
            f"WITH_NVSHMEM={int(os.environ.get('AISP_MULTI_GPU_WITH_NVSHMEM') == '1')}",
        ],
        cwd=LAB_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return LAB_ROOT / "jacobi_native", launcher


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.parametrize(
    "backend,schedule,registered,graph,ny,steps",
    [
        ("nccl", "serialized", False, False, 5, 3),
        ("nccl", "overlap", False, False, 4, 4),
        ("nccl", "overlap", False, True, 5, 3),
        ("nccl", "overlap", True, True, 6, 4),
        ("mpi", "overlap", False, False, 5, 3),
        pytest.param(
            "nvshmem", "overlap", False, False, 5, 3,
            marks=pytest.mark.skipif(
                os.environ.get("AISP_MULTI_GPU_WITH_NVSHMEM") != "1",
                reason="NVSHMEM integration requires AISP_MULTI_GPU_WITH_NVSHMEM=1 and a real installation",
            ),
        ),
    ],
)
def test_native_timed_output_matches_independent_stencil(
    native_binary: tuple[Path, str],
    native_controls,
    contract_driver: Path,
    tmp_path: Path,
    backend: str,
    schedule: str,
    registered: bool,
    graph: bool,
    ny: int,
    steps: int,
) -> None:
    import struct

    binary, launcher = native_binary
    nx = 17
    result_path = tmp_path / "result.json"
    output_path = tmp_path / "timed-output.f32"
    command = [
        launcher,
        "-np",
        "2",
        str(binary),
        "--backend",
        backend,
        "--schedule",
        schedule,
        "--nx",
        str(nx),
        "--ny",
        str(ny),
        "--iterations",
        str(steps),
        "--warmup",
        "1",
        "--json-out",
        str(result_path),
        "--dump-output",
        str(output_path),
    ]
    if registered:
        command.append("--registered-buffers")
    if graph:
        command.append("--graph")
    result = subprocess.run(command, cwd=LAB_ROOT, capture_output=True, text=True)
    output = result.stdout + result.stderr
    if result.returncode != 0 and "SKIPPED:" in output:
        pytest.skip(output)
    assert result.returncode == 0, output
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    workload = native_controls.NativeWorkload(
        backend=backend, schedule=schedule, nx=nx, ny=ny, iterations=steps,
        warmup=1, ranks=2, registered_buffers=registered, graph=graph,
    )
    native_controls.validate_result(payload, workload)
    if backend != "nvshmem":
        for field in ("nvshmem_version", "nvshmem_header_version", "openshmem_api_version"):
            assert payload[field] is None
    assert payload["schema"] == "multi_gpu_communication.v1"
    assert payload["status"] == "passed"
    assert payload["canonical"] is False
    assert payload["stopping_policy"] == "fixed_iterations"
    assert payload["backend"] == backend
    assert payload["schedule"] == schedule
    assert payload["registered_buffers"] is registered
    assert payload["graph"] is graph
    assert payload["ranks"] == 2
    assert payload["iterations"] == steps
    assert payload["checked_owned_elements"] == nx * (ny - 2)
    assert payload["checked_halo_elements"] == 4 * nx
    assert math.isfinite(payload["steady_ms"]) and payload["steady_ms"] > 0
    assert payload["max_abs_error"] <= 2e-6
    data = output_path.read_bytes()
    assert len(data) == nx * ny * 4
    actual = struct.unpack(f"={nx * ny}f", data)
    initial = _contract_output(contract_driver, "mesh", nx, ny, 0)
    expected = _periodic_stencil_oracle(initial, nx, ny, steps)
    assert all(math.isfinite(value) for value in actual)
    assert actual == pytest.approx(expected, abs=2e-6, rel=2e-6)
