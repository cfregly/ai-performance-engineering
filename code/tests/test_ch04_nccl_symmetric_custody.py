"""Real file-custody regressions for the Chapter 4 NCCL launcher."""

from __future__ import annotations

import json
import stat
import sys
from array import array
from pathlib import Path

import pytest

from ch04 import run_nccl_symmetric_memory as runner


def _write_valid_build(binary: Path) -> dict:
    binary.write_bytes(b"native-build-v1")
    binary.chmod(0o755)
    library = binary.parent / "libnccl.so.2"
    library.write_bytes(b"nccl-runtime-v1")
    receipt = {
        "schema": runner.BUILD_SCHEMA,
        "source_sha256": runner.build_source_hashes(),
        "binary_sha256": runner.file_sha256(binary),
        "nccl_header_version_code": runner.MINIMUM_NCCL_VERSION_CODE,
        "minimum_nccl_version_code": runner.MINIMUM_NCCL_VERSION_CODE,
        "compile_command": ["nvcc", "fixture.cu"],
        "compiler_versions": {"nvcc": "fixture", "mpicxx": "fixture"},
        "build_options": {"nccl_libs": "-L/fixture/lib -lnccl"},
        "nccl_runtime_library_path": str(library),
        "nccl_runtime_library_sha256": runner.file_sha256(library),
    }
    binary.with_name(binary.name + ".build.json").write_text(
        json.dumps(receipt) + "\n", encoding="utf-8"
    )
    return receipt


def test_complete_output_uses_native_endian_without_a_second_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workload = runner.NativeWorkload(
        arm="window_register", ranks=2, elements=8, iterations=1, warmup=0
    )
    output = tmp_path / "native-endian.f32"
    values = array(
        "f", (runner._expected_value(workload.ranks, index) for index in range(workload.elements))
    )
    with output.open("wb") as stream:
        values.tofile(stream)

    monkeypatch.setattr(runner.sys, "byteorder", "big")
    assert runner.validate_full_output(output, workload)["checked_elements"] == workload.elements


def test_staged_run_build_survives_source_replacement_and_detects_copy_tampering(
    tmp_path: Path,
) -> None:
    binary = tmp_path / "build" / "native"
    binary.parent.mkdir()
    receipt = _write_valid_build(binary)
    result = tmp_path / "run" / "result.json"
    reservation = runner.JsonResultReservation.acquire(result)

    custody = runner.stage_run_build(binary, reservation)
    prelaunch = runner.verify_run_build_custody(custody)

    assert custody.binary != binary
    assert custody.binary.parent == result.with_name(result.name + ".run")
    assert prelaunch["binary"]["sha256"] == receipt["binary_sha256"]
    assert prelaunch["build_receipt"]["sha256"] == runner.file_sha256(custody.build_receipt)
    assert stat.S_IMODE(custody.directory.stat().st_mode) == 0o500
    assert stat.S_IMODE(custody.binary.stat().st_mode) == 0o500
    assert stat.S_IMODE(custody.build_receipt.stat().st_mode) == 0o400

    binary.write_bytes(b"replacement-build")
    binary.with_name(binary.name + ".build.json").write_text(
        '{"replacement": true}\n', encoding="utf-8"
    )
    assert runner.verify_run_build_custody(custody) == prelaunch

    custody.directory.chmod(0o700)
    custody.binary.chmod(0o700)
    custody.binary.write_bytes(b"tampered-run-copy")
    custody.binary.chmod(0o500)
    custody.directory.chmod(0o500)
    with pytest.raises(RuntimeError, match="binary bytes changed"):
        runner.verify_run_build_custody(custody)
    reservation.write_json({"status": "FAILED", "reason": "test completed"})


def test_staging_rejects_source_replacement_between_validation_and_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "native"
    _write_valid_build(binary)
    reservation = runner.JsonResultReservation.acquire(tmp_path / "result.json")
    copy_exclusive = runner._copy_exclusive
    replaced = False

    def replace_before_copy(source: Path, destination: Path, mode: int) -> tuple[int, int]:
        nonlocal replaced
        if source == binary and not replaced:
            replaced = True
            binary.write_bytes(b"concurrent-replacement")
        return copy_exclusive(source, destination, mode)

    monkeypatch.setattr(runner, "_copy_exclusive", replace_before_copy)
    with pytest.raises(RuntimeError, match="binary differs|binary changed"):
        runner.stage_run_build(binary, reservation)
    custody_directory = reservation.path.with_name(reservation.path.name + ".run")
    assert custody_directory.is_dir()
    assert stat.S_IMODE(custody_directory.stat().st_mode) == 0o500
    reservation.write_json({"status": "FAILED", "reason": "concurrent replacement rejected"})


def test_ldd_and_mpi_receive_the_reserved_run_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "native"
    _write_valid_build(binary)
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        runner,
        "child_runtime_environment",
        lambda _build: ({"CUDA_VISIBLE_DEVICES": "0,1"}, []),
    )

    def verify_loader(staged_binary, _build, _directories, _environment):
        observed["ldd_binary"] = staged_binary
        return tmp_path / "libnccl.so.2"

    def stop_after_launch(command, **_kwargs):
        observed["command"] = command
        return {"returncode": 1, "stdout": "", "stderr": "stopped after path check"}

    monkeypatch.setattr(runner, "verify_loader_binding", verify_loader)
    monkeypatch.setattr(runner, "run_recorded_cooperative_process", stop_after_launch)
    result = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="Native MPI run failed"):
        runner.invoke_native(
            runner.NativeWorkload("cuda_malloc", ranks=2, elements=8, iterations=1, warmup=0),
            binary=binary,
            json_out=result,
            dump_output=tmp_path / "output.f32",
            launcher=sys.executable,
        )

    command = observed["command"]
    assert isinstance(command, list)
    staged_binary = Path(command[3])
    assert staged_binary == observed["ldd_binary"]
    assert staged_binary != binary
    assert staged_binary.parent == result.with_name(result.name + ".run")
    assert runner.file_sha256(staged_binary) == _write_valid_build_digest(binary)


def _write_valid_build_digest(binary: Path) -> str:
    receipt = json.loads(binary.with_name(binary.name + ".build.json").read_text(encoding="utf-8"))
    return receipt["binary_sha256"]


def _valid_native_result(workload: runner.NativeWorkload, library: Path) -> dict:
    samples = [1.0 + index / 4 for index in range(workload.iterations)]
    registration_ranks = (
        workload.ranks if workload.arm in {"comm_register", "window_register"} else 0
    )
    return {
        "schema": runner.SCHEMA,
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
        "nccl_header_version_code": runner.MINIMUM_NCCL_VERSION_CODE,
        "nccl_runtime_version_code": runner.MINIMUM_NCCL_VERSION_CODE,
        "nccl_runtime_library_path": str(library),
        "nccl_runtime_library_verified_ranks": workload.ranks,
        "profiler_preload_allowed": False,
        "cuda_visible_devices": ",".join(str(rank) for rank in range(workload.ranks)),
        "minimum_nccl_version_code": runner.MINIMUM_NCCL_VERSION_CODE,
        "timing_scope": "MPI_MAX CUDA event elapsed around one ncclAllReduce",
        "setup_ms": 1.0,
        "topology_gate_ms": 1.0,
        "communicator_ms": 1.0,
        "allocation_ms": 1.0,
        "registration_ms": 1.0,
        "send_registration_handle_verified_ranks": registration_ranks,
        "receive_registration_handle_verified_ranks": registration_ranks,
        "initialization_ms": 1.0,
        "warmup_ms": 1.0,
        "time_per_iteration_ms": runner.expected_statistics(samples)["mean"],
        "verification_ms": 1.0,
        "output_write_ms": 1.0,
        "cleanup_ms": 1.0,
        "rtol": 0.0,
        "atol": 0.0,
        "samples_ms": samples,
        "statistics_ms": runner.expected_statistics(samples),
        "verified_ranks": workload.ranks,
        "checked_elements_per_rank": workload.elements,
        "checked_elements_total": workload.elements * workload.ranks,
        "max_abs_error": 0.0,
        "rank_verification": [
            {
                "rank": rank,
                "checked_elements": workload.elements,
                "bad_elements": 0,
                "max_abs_error": 0.0,
            }
            for rank in range(workload.ranks)
        ],
        "rank_mapping": [
            {
                "rank": rank,
                "device_ordinal": rank,
                "gpu_uuid": f"GPU-00000000-0000-0000-0000-{rank + 1:012x}",
                "pci_bus_id": f"0000:{rank + 1:02x}:00.0",
            }
            for rank in range(workload.ranks)
        ],
    }


def _install_successful_native_process(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commands: list[list[str]],
) -> None:
    library = tmp_path / "libnccl.so.2"
    monkeypatch.setattr(
        runner,
        "child_runtime_environment",
        lambda _build: ({"CUDA_VISIBLE_DEVICES": "0,1"}, []),
    )
    monkeypatch.setattr(runner, "verify_loader_binding", lambda *_args: library)

    def complete(command, *, output_prefix, env):
        commands.append(command)
        workload = runner.NativeWorkload(
            arm=command[command.index("--arm") + 1],
            ranks=int(command[command.index("-np") + 1]),
            elements=int(command[command.index("--elements") + 1]),
            iterations=int(command[command.index("--iterations") + 1]),
            warmup=int(command[command.index("--warmup") + 1]),
        )
        native_result = Path(command[command.index("--json-out") + 1])
        output = Path(command[command.index("--dump-output") + 1])
        native_result.parent.mkdir(parents=True, exist_ok=True)
        native_result.write_text(
            json.dumps(_valid_native_result(workload, library)) + "\n",
            encoding="utf-8",
        )
        values = array(
            "f",
            (runner._expected_value(workload.ranks, index) for index in range(workload.elements)),
        )
        with output.open("wb") as stream:
            values.tofile(stream)
        stdout = "fixture stdout\n"
        stderr = ""
        output_prefix.with_name(output_prefix.name + ".stdout.log").write_text(
            stdout, encoding="utf-8"
        )
        output_prefix.with_name(output_prefix.name + ".stderr.log").write_text(
            stderr, encoding="utf-8"
        )
        output_prefix.with_name(output_prefix.name + ".execution.json").write_text(
            json.dumps({"returncode": 0, "command": command}) + "\n",
            encoding="utf-8",
        )
        return {"returncode": 0, "stdout": stdout, "stderr": stderr}

    monkeypatch.setattr(runner, "run_recorded_cooperative_process", complete)


def test_normal_result_hashes_every_retained_native_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "native"
    _write_valid_build(binary)
    commands: list[list[str]] = []
    _install_successful_native_process(tmp_path, monkeypatch, commands)
    result_path = tmp_path / "result.json"

    result = runner.invoke_native(
        runner.NativeWorkload("cuda_malloc", ranks=2, elements=8, iterations=2, warmup=0),
        binary=binary,
        json_out=result_path,
        dump_output=tmp_path / "output.f32",
        launcher=sys.executable,
    )

    assert len(commands) == 1
    assert set(result["native_artifacts"]) == {
        "native_result",
        "output",
        "stdout",
        "stderr",
        "execution",
    }
    for retained in result["native_artifacts"].values():
        path = Path(retained["path"])
        assert retained["sha256"] == runner.file_sha256(path)
        assert retained["bytes"] == path.stat().st_size
    assert set(result["control_plane_custody"]["prelaunch"]) == {
        "runner",
        "benchmark_wrapper",
        "build_receipt",
        "profile",
        "cooperative_process_artifacts",
        "cooperative_subprocess",
    }
    assert (
        result["control_plane_custody"]["prelaunch"] == result["control_plane_custody"]["post_run"]
    )


def test_all_arms_reuses_one_immutable_run_cohort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "native"
    _write_valid_build(binary)
    commands: list[list[str]] = []
    _install_successful_native_process(tmp_path, monkeypatch, commands)

    suite = runner.invoke_all_arms(
        runner.NativeWorkload("cuda_malloc", ranks=2, elements=8, iterations=1, warmup=0),
        binary=binary,
        json_out=tmp_path / "suite.json",
        launcher=sys.executable,
        launcher_args=(),
    )

    staged_binaries = {command[3] for command in commands}
    assert len(commands) == len(runner.ARMS)
    assert len(staged_binaries) == 1
    identities = {
        json.dumps(result["run_build_custody"]["identity"], sort_keys=True)
        for result in suite["arms"].values()
    }
    assert identities == {json.dumps(suite["run_cohort_identity"], sort_keys=True)}


def test_reused_cohort_rejects_a_different_source_build(tmp_path: Path) -> None:
    first = tmp_path / "first" / "native"
    second = tmp_path / "second" / "native"
    first.parent.mkdir()
    second.parent.mkdir()
    _write_valid_build(first)
    _write_valid_build(second)
    cohort = runner.prepare_run_cohort(first, tmp_path / "cohort")

    with pytest.raises(RuntimeError, match="Mixed-build cohort rejected"):
        runner.invoke_native(
            runner.NativeWorkload("cuda_malloc", ranks=2, elements=8, iterations=1, warmup=0),
            binary=second,
            json_out=tmp_path / "mixed.json",
            dump_output=tmp_path / "mixed.f32",
            launcher=sys.executable,
            cohort=cohort,
        )


def test_all_arms_rejects_a_mixed_result_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "native"
    _write_valid_build(binary)
    calls = 0

    def mixed_result(_workload, *, cohort, **_kwargs):
        nonlocal calls
        calls += 1
        identity = runner.run_cohort_identity(cohort)
        if calls == 2:
            identity = {**identity, "binary_sha256": "0" * 64}
        return {"run_build_custody": {"identity": identity}}

    monkeypatch.setattr(runner, "invoke_native", mixed_result)
    result_path = tmp_path / "suite.json"
    with pytest.raises(RuntimeError, match="Mixed-build cohort rejected for arm"):
        runner.invoke_all_arms(
            runner.NativeWorkload("cuda_malloc", ranks=2, elements=8, iterations=1, warmup=0),
            binary=binary,
            json_out=result_path,
            launcher=sys.executable,
            launcher_args=(),
        )

    failure = json.loads(result_path.read_text(encoding="utf-8"))
    assert failure["status"] == "FAILED"
    assert failure["failed_arm"] == runner.ARMS[1]
    assert list(failure["completed_arms"]) == [runner.ARMS[0]]


def test_runner_rejects_control_plane_replacement_during_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "native"
    _write_valid_build(binary)
    control = tmp_path / "control.py"
    control.write_text("version = 1\n", encoding="utf-8")
    monkeypatch.setattr(runner, "CONTROL_PLANE_FILES", {"fixture": control})
    monkeypatch.setattr(
        runner,
        "child_runtime_environment",
        lambda _build: ({"CUDA_VISIBLE_DEVICES": "0,1"}, []),
    )
    monkeypatch.setattr(
        runner,
        "verify_loader_binding",
        lambda *_args: tmp_path / "libnccl.so.2",
    )

    def replace_control(_command, **_kwargs):
        replacement = tmp_path / "control-replacement.py"
        replacement.write_text("version = 2\n", encoding="utf-8")
        replacement.replace(control)
        return {"returncode": 1, "stdout": "", "stderr": "fixture stop"}

    monkeypatch.setattr(runner, "run_recorded_cooperative_process", replace_control)
    with pytest.raises(RuntimeError, match="Runner control plane changed.*fixture"):
        runner.invoke_native(
            runner.NativeWorkload("cuda_malloc", ranks=2, elements=8, iterations=1, warmup=0),
            binary=binary,
            json_out=tmp_path / "control-change.json",
            dump_output=tmp_path / "control-change.f32",
            launcher=sys.executable,
        )


def test_keyboard_interrupt_retains_terminal_failure_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def interrupt(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "invoke_native", interrupt)
    result = tmp_path / "interrupted.json"
    returncode = runner.main(
        [
            "--ranks",
            "2",
            "--binary",
            str(tmp_path / "unused-native"),
            "--json-out",
            str(result),
        ]
    )

    assert returncode == 130
    retained = json.loads(result.read_text(encoding="utf-8"))
    assert retained == {
        "schema": runner.SCHEMA,
        "status": "FAILED",
        "reason": "Interrupted by user before completion",
    }
    assert f"Result: {result.resolve()}" in capsys.readouterr().out


def test_cleanup_incomplete_retains_exact_receipt_and_nonzero_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cleanup_receipt = {
        "execution": {
            "process_group_id": 4242,
            "completion_scope": "cleanup_incomplete",
            "live_process_group_members": None,
        },
        "capture_errors": ["fixture could not prove cleanup"],
    }

    def cleanup_incomplete(*_args, **_kwargs):
        raise runner.CooperativeCleanupIncomplete(cleanup_receipt)

    monkeypatch.setattr(runner, "invoke_native", cleanup_incomplete)
    result = tmp_path / "cleanup-incomplete.json"
    returncode = runner.main(
        [
            "--ranks",
            "2",
            "--binary",
            str(tmp_path / "unused-native"),
            "--json-out",
            str(result),
        ]
    )

    assert returncode == 1
    retained = json.loads(result.read_text(encoding="utf-8"))
    assert retained["status"] == "FAILED"
    assert retained["cleanup_incomplete_receipt"] == cleanup_receipt
    assert f"Result: {result.resolve()}" in capsys.readouterr().out


def test_cleanup_incomplete_json_mode_prints_only_the_failure_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    cleanup_receipt = {
        "execution": {
            "process_group_id": 4242,
            "completion_scope": "cleanup_incomplete",
            "live_process_group_members": None,
        }
    }

    def cleanup_incomplete(*_args, **_kwargs):
        raise runner.CooperativeCleanupIncomplete(cleanup_receipt)

    monkeypatch.setattr(runner, "invoke_native", cleanup_incomplete)
    result = tmp_path / "cleanup-incomplete-json.json"
    assert (
        runner.main(
            [
                "--json",
                "--ranks",
                "2",
                "--binary",
                str(tmp_path / "unused-native"),
                "--json-out",
                str(result),
            ]
        )
        == 1
    )

    printed = capsys.readouterr().out
    assert json.loads(printed)["cleanup_incomplete_receipt"] == cleanup_receipt
    assert "Result:" not in printed


def test_all_arms_cleanup_failure_preserves_the_exact_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = tmp_path / "native"
    _write_valid_build(binary)
    cleanup_receipt = {
        "execution": {
            "process_group_id": 5150,
            "completion_scope": "cleanup_incomplete",
            "live_process_group_members": None,
        },
        "capture_errors": ["fixture ownership observation failed"],
    }

    def cleanup_incomplete(*_args, **_kwargs):
        raise runner.CooperativeCleanupIncomplete(cleanup_receipt)

    monkeypatch.setattr(runner, "invoke_native", cleanup_incomplete)
    result = tmp_path / "all-arms-cleanup.json"
    assert (
        runner.main(
            [
                "--all-arms",
                "--ranks",
                "2",
                "--binary",
                str(binary),
                "--json-out",
                str(result),
            ]
        )
        == 1
    )

    retained = json.loads(result.read_text(encoding="utf-8"))
    assert retained["status"] == "FAILED"
    assert retained["failed_arm"] == runner.ARMS[0]
    assert retained["cleanup_incomplete_receipt"] == cleanup_receipt


def test_default_text_failure_prints_the_retained_result_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(*_args, **_kwargs):
        raise RuntimeError("fixture failure")

    monkeypatch.setattr(runner, "invoke_native", fail)
    result = tmp_path / "failed.json"
    assert (
        runner.main(
            [
                "--ranks",
                "2",
                "--binary",
                str(tmp_path / "unused-native"),
                "--json-out",
                str(result),
            ]
        )
        == 1
    )

    assert json.loads(result.read_text(encoding="utf-8"))["status"] == "FAILED"
    assert f"Result: {result.resolve()}" in capsys.readouterr().out


def test_default_result_is_reserved_before_gpu_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = tmp_path / "default-interrupted.json"
    monkeypatch.setattr(runner, "_default_result_path", lambda _label: result)

    def interrupt_discovery():
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "visible_gpu_count", interrupt_discovery)
    assert runner.main([]) == 130
    retained = json.loads(result.read_text(encoding="utf-8"))
    assert retained["status"] == "FAILED"
    assert retained["reason"] == "Interrupted by user before completion"
