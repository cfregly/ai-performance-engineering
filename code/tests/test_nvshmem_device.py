"""Independent output oracles, real CLI checks, and opt-in NVSHMEM GPU execution."""
from __future__ import annotations

import importlib.util
import json
import math
import os
import shlex
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import pytest

CODE_ROOT = Path(__file__).resolve().parents[1]
LAB_ROOT = CODE_ROOT / "labs" / "nvshmem_device"
SPEC = importlib.util.spec_from_file_location("nvshmem_device_runner", LAB_ROOT / "run_lab_nvshmem_device.py")
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


@pytest.fixture(scope="module")
def cpu_reference_driver(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler or shutil.which(compiler[0]) is None:
        pytest.skip("A C++17 compiler is required for the independent CPU comparison")
    directory = tmp_path_factory.mktemp("nvshmem-device-cpu-reference")
    source = directory / "reference.cpp"
    binary = directory / "reference"
    source.write_text(
        '#include "jacobi_contract.hpp"\n#include <iostream>\n#include <string>\n'
        'int main(int argc, char** argv) {\n'
        '  if (argc != 4) return 2;\n'
        '  const auto output = multi_gpu_communication::reference_jacobi(\n'
        '      std::stoi(argv[1]), std::stoi(argv[2]), std::stoi(argv[3]));\n'
        '  std::cout.write(reinterpret_cast<const char*>(output.data()), output.size()*sizeof(float));\n'
        '  return std::cout ? 0 : 1;\n}\n'
    )
    result = subprocess.run(
        [*compiler, "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror", "-I",
         str(CODE_ROOT / "labs" / "multi_gpu_communication"), str(source), "-o", str(binary)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return binary


@pytest.mark.parametrize("nx,ny,steps", [(3, 3, 0), (3, 3, 2), (17, 4, 4), (17, 5, 3), (19, 8, 4), (9, 9, 7)])
def test_python_scalar_oracle_matches_real_cpp_reference(
    cpu_reference_driver: Path, nx: int, ny: int, steps: int,
) -> None:
    result = subprocess.run([str(cpu_reference_driver), str(nx), str(ny), str(steps)], capture_output=True, check=False)
    assert result.returncode == 0, result.stderr.decode()
    actual = struct.unpack(f"={nx * ny}f", result.stdout)
    expected = list(runner.scalar_reference(nx, ny, steps))
    assert len(actual) == len(expected) == nx * ny
    assert all(math.isfinite(value) for value in actual)
    assert actual == pytest.approx(expected, rel=0, abs=2e-6)


def test_scalar_oracle_has_a_hand_computed_periodic_case() -> None:
    expected = [0.0, 0.0125, 0.0] * 3
    assert list(runner.scalar_reference(3, 3, 2)) == pytest.approx(expected, rel=0, abs=1e-9)


@pytest.mark.parametrize("mode", ["block-signal", "persistent-signal", "persistent-atomic"])
def test_tile_oracle_checks_both_edges_and_every_epoch(mode: str) -> None:
    workload = runner.NativeWorkload(mode=mode, elements=3, iterations=2)
    expected = [10, 11, 12, 7, 8, 9, 22, 23, 24, 19, 20, 21,
                4, 5, 6, 1, 2, 3, 16, 17, 18, 13, 14, 15]
    assert list(runner.expected_output(workload)) == expected


@pytest.mark.parametrize("mode", runner.HALO_MODES)
def test_both_halo_modes_use_the_same_complete_mesh(mode: str) -> None:
    workload = runner.NativeWorkload(mode=mode, nx=17, ny=5, iterations=3)
    assert workload.output_layout == "ny,nx"
    assert workload.expected_output_elements == 85
    assert list(runner.expected_output(workload)) == list(runner.scalar_reference(17, 5, 3))


@pytest.mark.parametrize("changes,reason", [
    ({"mode": "unknown"}, "mode must"), ({"nx": 2}, "nx >= 3"),
    ({"ny": 3}, "at least one row"), ({"elements": 0}, "elements >= 1"),
    ({"iterations": 0}, "iterations >= 1"), ({"warmup": -1}, "warmup >= 0"),
    ({"ranks": 1}, "two ranks"), ({"nx": 2**31}, "32-bit"),
    ({"iterations": True}, "integers"),
    ({"mode": "block-signal", "elements": 2**22 + 1, "iterations": 1}, r"2\^24"),
])
def test_invalid_workloads_fail_before_launch(changes: dict, reason: str) -> None:
    with pytest.raises(ValueError, match=reason):
        runner.NativeWorkload(**changes).arguments()


def _metadata(workload) -> dict:
    """Create parser input, not evidence of native execution."""
    return {
        "schema": runner.SCHEMA, "status": "VERIFIED", "canonical": False, "verified": True,
        "mode": workload.mode, "nx": workload.nx, "ny": workload.ny,
        "elements": workload.elements, "iterations": workload.iterations,
        "warmup": workload.warmup, "ranks": workload.ranks, "output_dtype": "float32",
        "output_layout": workload.output_layout, "output_elements": workload.expected_output_elements,
        "checked_output_elements": workload.expected_output_elements,
        "expected_ready_epoch": 0 if workload.mode == "scalar-halo" else workload.iterations,
        "expected_consumed_epoch": 0 if workload.mode == "scalar-halo" else workload.iterations - int(workload.mode == "persistent-halo"),
        "setup_ms": 1.0, "allocation_ms": 0.1, "warmup_ms": 0.2, "reset_ms": 0.1,
        "steady_ms": float(workload.iterations), "time_per_iteration_ms": 1.0,
        "verification_ms": 0.2, "cleanup_ms": 0.1, "max_abs_error": 0.0,
        "nvshmem_version": "3.7.2", "nvshmem_header_version": "3.7.2",
        "openshmem_api_version": "1.3",
    }


@pytest.mark.parametrize("field,value", [
    ("schema", "other"), ("status", "FAILED"), ("canonical", True),
    ("verified", False), ("ranks", True), ("output_layout", "sampled"),
    ("checked_output_elements", 1), ("steady_ms", 0),
    ("time_per_iteration_ms", 0.5), ("max_abs_error", math.nan),
    ("expected_consumed_epoch", 1),
])
def test_result_parser_rejects_mismatched_contract(field: str, value) -> None:
    workload = runner.NativeWorkload(mode="persistent-halo", iterations=4)
    data = _metadata(workload)
    data[field] = value
    with pytest.raises(RuntimeError):
        runner.validate_result(data, workload)


@pytest.mark.parametrize("field", ["nvshmem_version", "nvshmem_header_version", "openshmem_api_version"])
def test_result_parser_requires_separate_runtime_header_and_api_versions(field: str) -> None:
    workload = runner.NativeWorkload()
    data = _metadata(workload)
    del data[field]
    with pytest.raises(RuntimeError, match=field):
        runner.validate_result(data, workload)


@pytest.mark.parametrize("field,value", [
    ("nvshmem_version", "1.3"), ("nvshmem_header_version", "1.3"),
    ("openshmem_api_version", "3.7.2"), ("nvshmem_version", "3.7.2-dev"),
    ("nvshmem_header_version", "3..2"), ("openshmem_api_version", "1.-3"),
    ("nvshmem_version", 3.7), ("nvshmem_version", "٣.٧.٢"),
])
def test_result_parser_rejects_mislabeled_or_malformed_versions(field: str, value) -> None:
    workload = runner.NativeWorkload()
    data = _metadata(workload)
    data[field] = value
    with pytest.raises(RuntimeError, match=field):
        runner.validate_result(data, workload)


def test_result_parser_accepts_numeric_versions_without_pinning_a_release() -> None:
    workload = runner.NativeWorkload()
    data = _metadata(workload)
    data.update(nvshmem_version="4.0.0", nvshmem_header_version="3.8.1", openshmem_api_version="1.4")
    runner.validate_result(data, workload)


@pytest.mark.parametrize("mode", ["scalar-halo", "persistent-halo", "persistent-signal"])
@pytest.mark.parametrize("corruption", ["last", "nan", "short"])
def test_full_output_check_rejects_corrupted_retained_values(tmp_path: Path, mode: str, corruption: str) -> None:
    workload = runner.NativeWorkload(mode=mode)
    values = list(runner.expected_output(workload))
    if corruption == "last":
        values[-1] += 1.0
    elif corruption == "nan":
        values[len(values) // 2] = math.nan
    else:
        values.pop()
    path = tmp_path / "output.f32"
    path.write_bytes(struct.pack(f"={len(values)}f", *values))
    with pytest.raises(RuntimeError):
        runner.validate_full_output(path, workload)


def _receipt(binary: Path, directory: Path = LAB_ROOT) -> dict:
    """Prepare provenance-parser input without launching the placeholder binary."""
    value = {"schema": runner.BUILD_SCHEMA, "source_sha256": runner.source_hashes(directory),
             "binary_sha256": runner.sha256(binary), "compile_command": ["nvcc", "source.cu"],
             "compiler_versions": {"nvcc": "control fixture", "mpicxx": "control fixture"}}
    binary.with_name(binary.name + ".build.json").write_text(json.dumps(value))
    return value


def _write_successful_cpu_launcher(path: Path) -> None:
    path.write_text(
        f"#!{sys.executable}\n"
        + """
import os
from pathlib import Path
import shutil
import sys

selected = os.environ["AISP_EXPECTED_NVSHMEM_LIBDIR"]
inherited = os.environ["AISP_EXPECTED_INHERITED_LD_LIBRARY_PATH"]
expected = selected + os.pathsep + inherited
if os.environ.get("LD_LIBRARY_PATH") != expected:
    print(f"unexpected LD_LIBRARY_PATH: {os.environ.get('LD_LIBRARY_PATH')!r}", file=sys.stderr)
    raise SystemExit(9)

def argument(name):
    return Path(sys.argv[sys.argv.index(name) + 1])

shutil.copyfile(os.environ["AISP_NATIVE_JSON_FIXTURE"], argument("--json-out"))
shutil.copyfile(os.environ["AISP_NATIVE_OUTPUT_FIXTURE"], argument("--dump-output"))
"""
    )
    path.chmod(0o700)


@pytest.mark.parametrize("selection", ["explicit-libdir", "home-fallback"])
@pytest.mark.skipif(
    not all(hasattr(os, name) for name in ("waitid", "P_PID", "WEXITED", "WNOWAIT", "WNOHANG")),
    reason="Owned launchers require POSIX waitid with WNOWAIT",
)
def test_real_cli_pins_the_build_nvshmem_library_directory(
    tmp_path: Path, selection: str,
) -> None:
    workload = runner.NativeWorkload()
    binary = tmp_path / "never-executed-native-binary"
    binary.write_bytes(b"CPU launcher fixture, not a GPU executable\n")
    _receipt(binary)
    launcher = tmp_path / "cpu-launcher"
    _write_successful_cpu_launcher(launcher)

    native_json_fixture = tmp_path / "native-result-fixture.json"
    native_json_fixture.write_text(json.dumps(_metadata(workload)))
    output_fixture = tmp_path / "native-output-fixture.f32"
    values = list(runner.expected_output(workload))
    output_fixture.write_bytes(struct.pack(f"={len(values)}f", *values))

    home = tmp_path / "nvshmem-home"
    fallback = home / "lib"
    explicit = tmp_path / "explicit-nvshmem-lib"
    fallback.mkdir(parents=True)
    explicit.mkdir()
    selected = explicit if selection == "explicit-libdir" else fallback
    inherited = tmp_path / "older-runtime-lib"
    environment = dict(os.environ)
    environment["NVSHMEM_HOME"] = str(home)
    if selection == "explicit-libdir":
        environment["NVSHMEM_LIBDIR"] = str(explicit)
    else:
        environment.pop("NVSHMEM_LIBDIR", None)
    environment["LD_LIBRARY_PATH"] = str(inherited)
    environment["AISP_EXPECTED_NVSHMEM_LIBDIR"] = str(selected)
    environment["AISP_EXPECTED_INHERITED_LD_LIBRARY_PATH"] = str(inherited)
    environment["AISP_NATIVE_JSON_FIXTURE"] = str(native_json_fixture)
    environment["AISP_NATIVE_OUTPUT_FIXTURE"] = str(output_fixture)

    result_path = tmp_path / "result.json"
    output_path = tmp_path / "output.f32"
    completed = subprocess.run(
        [sys.executable, str(LAB_ROOT / "run_lab_nvshmem_device.py"),
         "--binary", str(binary), "--launcher", str(launcher),
         "--json-out", str(result_path), "--dump-output", str(output_path), "--json"],
        capture_output=True, text=True, check=False, env=environment,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = json.loads(result_path.read_text())
    assert result["nvshmem_runtime_libdir"] == str(selected)
    assert json.loads(completed.stdout) == result
    assert environment["LD_LIBRARY_PATH"] == str(inherited)


def test_nvshmem_child_environment_is_fail_closed_and_does_not_mutate_parent(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.delenv("NVSHMEM_LIBDIR", raising=False)
    monkeypatch.delenv("NVSHMEM_HOME", raising=False)
    with pytest.raises(runner.NativeUnavailable, match="NVSHMEM_LIBDIR or NVSHMEM_HOME"):
        runner.nvshmem_child_environment()

    home = tmp_path / "home"
    (home / "lib").mkdir(parents=True)
    explicit = tmp_path / "explicit"
    explicit.mkdir()
    inherited = str(tmp_path / "inherited")
    monkeypatch.setenv("NVSHMEM_HOME", str(home))
    monkeypatch.setenv("NVSHMEM_LIBDIR", str(explicit))
    monkeypatch.setenv("LD_LIBRARY_PATH", inherited)
    child_env, selected = runner.nvshmem_child_environment()
    assert selected == explicit
    assert child_env["LD_LIBRARY_PATH"] == str(explicit) + os.pathsep + inherited
    assert os.environ["LD_LIBRARY_PATH"] == inherited

    monkeypatch.delenv("NVSHMEM_LIBDIR")
    child_env, selected = runner.nvshmem_child_environment()
    assert selected == home / "lib"
    assert child_env["LD_LIBRARY_PATH"] == str(home / "lib") + os.pathsep + inherited
    assert os.environ["LD_LIBRARY_PATH"] == inherited

    monkeypatch.setenv("NVSHMEM_LIBDIR", "")
    with pytest.raises(RuntimeError, match="must not be empty"):
        runner.nvshmem_child_environment()
    monkeypatch.setenv("NVSHMEM_LIBDIR", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="does not exist"):
        runner.nvshmem_child_environment()


@pytest.mark.parametrize("change", ["binary", "source", "dependency", "schema", "command"])
def test_stale_or_incomplete_provenance_is_rejected(tmp_path: Path, change: str) -> None:
    directory = tmp_path / "labs" / "nvshmem_device"
    directory.mkdir(parents=True)
    for name in runner.SOURCE_FILES:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("provenance parser fixture\n")
    binary = directory / "native"
    binary.write_bytes(b"not an executable\n")
    receipt = _receipt(binary, directory)
    if change == "binary":
        binary.write_bytes(b"changed\n")
    elif change in {"source", "dependency"}:
        name = runner.SOURCE_FILES[0 if change == "source" else -1]
        (directory / name).write_text("changed\n")
    else:
        receipt["schema" if change == "schema" else "compile_command"] = "invalid" if change == "schema" else []
        binary.with_name(binary.name + ".build.json").write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError):
        runner.validate_build_receipt(binary, directory)


def test_missing_launcher_is_detected_without_executing_a_binary(tmp_path: Path) -> None:
    binary = tmp_path / "not-executable"
    binary.write_bytes(b"provenance parser fixture\n")
    _receipt(binary)
    with pytest.raises(runner.NativeUnavailable, match="launcher"):
        runner.invoke_native(runner.NativeWorkload(), binary=binary,
                             json_out=tmp_path / "result.json", dump_output=tmp_path / "output.f32",
                             launcher="nonexistent-nvshmem-device-launcher")


def test_missing_build_receipt_is_rejected(tmp_path: Path) -> None:
    binary = tmp_path / "native"
    binary.write_bytes(b"provenance parser fixture\n")
    with pytest.raises(RuntimeError, match="receipt is missing"):
        runner.validate_build_receipt(binary)


def test_output_paths_cannot_collide_or_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    with pytest.raises(ValueError, match="different paths"):
        runner.invoke_native(runner.NativeWorkload(), binary=tmp_path / "absent",
                             json_out=path, dump_output=path)
    path.write_text("owner evidence\n")
    with pytest.raises(FileExistsError, match="existing artifact"):
        runner.invoke_native(runner.NativeWorkload(), binary=tmp_path / "absent",
                             json_out=path, dump_output=tmp_path / "output.f32")
    assert path.read_text() == "owner evidence\n"


def test_real_cli_help_has_all_five_native_modes() -> None:
    result = subprocess.run([sys.executable, str(LAB_ROOT / "run_lab_nvshmem_device.py"), "--help"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    for mode in runner.MODES:
        assert mode in result.stdout
    assert "--dump-output" in result.stdout


def test_real_cli_missing_binary_reports_an_explicit_skip(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    result = subprocess.run([sys.executable, str(LAB_ROOT / "run_lab_nvshmem_device.py"),
                             "--binary", str(tmp_path / "absent"), "--json-out", str(path), "--json"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 77, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "SKIPPED"
    assert "binary is missing" in payload["reason"]
    assert json.loads(path.read_text()) == payload


@pytest.mark.parametrize("failure", ["invalid_workload", "missing_binary"])
def test_real_cli_invalid_result_parent_preserves_the_original_failure(tmp_path: Path, failure: str) -> None:
    parent = tmp_path / "not-a-directory"
    parent.write_text("preserve this file\n")
    path = parent / "result.json"
    arguments = ["--nx", "2"] if failure == "invalid_workload" else ["--binary", str(tmp_path / "absent")]
    result = subprocess.run(
        [sys.executable, str(LAB_ROOT / "run_lab_nvshmem_device.py"),
         *arguments, "--json-out", str(path), "--json"],
        capture_output=True, text=True, check=False,
    )
    payload = json.loads(result.stdout)
    assert result.returncode == (1 if failure == "invalid_workload" else 77)
    assert payload["status"] == ("FAILED" if failure == "invalid_workload" else "SKIPPED")
    assert ("nx >= 3" if failure == "invalid_workload" else "binary is missing") in payload["reason"]
    assert str(parent) in payload["failure_record_error"]
    assert "Traceback" not in result.stderr
    assert parent.read_text() == "preserve this file\n"


@pytest.fixture(scope="module")
def native_binary() -> tuple[Path, str, tuple[str, ...]]:
    if os.environ.get("AISP_RUN_NVSHMEM_DEVICE_TESTS") != "1":
        pytest.skip("NVSHMEM GPU execution is opt-in after reserving at least two GPUs")
    for tool in ("make", "nvcc", "mpicxx"):
        if shutil.which(tool) is None:
            pytest.skip(f"Native NVSHMEM integration requires real {tool}")
    if not os.environ.get("NVSHMEM_HOME"):
        pytest.skip("Native NVSHMEM integration requires explicit NVSHMEM_HOME")
    launcher = os.environ.get("NVSHMEM_DEVICE_MPI_LAUNCHER", "mpirun")
    if shutil.which(launcher) is None:
        pytest.skip("Native NVSHMEM integration requires a real MPI launcher")
    architecture = os.environ.get("AISP_NVSHMEM_DEVICE_ARCH")
    if not architecture:
        pytest.skip("Set AISP_NVSHMEM_DEVICE_ARCH explicitly for the allocated GPU")
    result = subprocess.run(["make", f"ARCH={architecture}"], cwd=LAB_ROOT,
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return LAB_ROOT / "nvshmem_device_native", launcher, tuple(shlex.split(os.environ.get("NVSHMEM_DEVICE_MPI_ARGS", "")))


@pytest.mark.cuda
@pytest.mark.integration
@pytest.mark.parametrize("mode,ny,steps,elements", [
    ("scalar-halo", 4, 4, 32), ("scalar-halo", 5, 3, 32), ("scalar-halo", 8, 6, 32),
    ("persistent-halo", 4, 4, 32), ("persistent-halo", 5, 3, 32), ("persistent-halo", 8, 6, 32),
    ("block-signal", 7, 17, 263), ("persistent-signal", 7, 17, 263), ("persistent-atomic", 7, 17, 263),
])
def test_real_native_output_matches_independent_python_oracle(
    native_binary: tuple[Path, str, tuple[str, ...]], tmp_path: Path,
    mode: str, ny: int, steps: int, elements: int,
) -> None:
    binary, launcher, launcher_args = native_binary
    workload = runner.NativeWorkload(mode=mode, nx=17, ny=ny, iterations=steps, elements=elements, warmup=2)
    output = tmp_path / "complete-output.f32"
    result = runner.invoke_native(workload, binary=binary, json_out=tmp_path / "result.json",
                                  dump_output=output, launcher=launcher, launcher_args=launcher_args)
    actual = struct.unpack(f"={workload.expected_output_elements}f", output.read_bytes())
    expected = list(runner.expected_output(workload))
    assert all(math.isfinite(value) for value in actual)
    assert actual == pytest.approx(expected, rel=0, abs=2e-6 if mode in runner.HALO_MODES else 0)
    assert result["status"] == "VERIFIED"
    assert result["canonical"] is False
    assert result["python_verification"]["checked_elements"] == workload.expected_output_elements
