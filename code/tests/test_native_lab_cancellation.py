"""Exercise launcher ownership with CPU processes, without executing a GPU binary."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import sys
import threading
from types import SimpleNamespace

import pytest

from core.utils.cooperative_subprocess import CooperativeProcessInterrupted
from core.utils.cooperative_process_artifacts import run_recorded_cooperative_process
from labs.multi_gpu_communication import native_common as jacobi
from labs.multi_gpu_communication.native_benchmark import NativeJacobiBenchmark
from labs.nvshmem_device import run_lab_nvshmem_device as nvshmem
from labs.nvshmem_device.native_benchmark import NvshmemDeviceBenchmark


pytestmark = pytest.mark.skipif(
    os.name != "posix" or not hasattr(os, "waitid"),
    reason="Owned launchers require POSIX waitid with WNOWAIT",
)


@pytest.mark.parametrize("value", [None, 7, False, 0.0])
@pytest.mark.parametrize(
    ("benchmark_type", "arguments"),
    [
        (NativeJacobiBenchmark, {"overlapped": True}),
        (NvshmemDeviceBenchmark, {"persistent": True}),
    ],
    ids=["jacobi", "nvshmem-device"],
)
@pytest.mark.parametrize("entrypoint", ["setup", "_run_once"])
def test_native_harness_rejects_unowned_finite_launcher_before_work(
    benchmark_type,
    arguments: dict,
    value,
    entrypoint: str,
) -> None:
    benchmark = benchmark_type(**arguments)
    benchmark._config = SimpleNamespace(execution_timeout_seconds=value)
    with pytest.raises(RuntimeError, match="requires explicit execution_timeout_seconds=0"):
        getattr(benchmark, entrypoint)()


@pytest.mark.parametrize("launcher_name", ["Jacobi", "NVSHMEM device"])
def test_native_harness_accepts_exact_owned_uncapped_selector(launcher_name: str) -> None:
    from core.utils.cooperative_process_artifacts import require_owned_uncapped_native_config

    require_owned_uncapped_native_config(
        SimpleNamespace(execution_timeout_seconds=0), launcher_name=launcher_name
    )


@pytest.mark.parametrize("lab", ["jacobi", "nvshmem"])
def test_native_setup_build_waits_for_interrupt_cleanup_and_retains_streams(
    tmp_path: Path,
    monkeypatch,
    lab: str,
) -> None:
    toolchain = tmp_path / "toolchain"
    toolchain.mkdir()
    for tool in ("nvcc", "mpicxx", "mpirun"):
        executable = toolchain / tool
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o700)

    make = toolchain / "make"
    make.write_text(
        f"#!{sys.executable}\n"
        + """
import os, signal, socket, sys
control = socket.create_connection(("127.0.0.1", int(os.environ["AISP_BUILD_TEST_PORT"])))
interrupts = 0
def interrupted(signum, frame):
    global interrupts
    interrupts += 1
    print("build stdout before cleanup release", flush=True)
    print("build stderr before cleanup release", file=sys.stderr, flush=True)
    control.sendall(b"cleanup ready\\n")
    control.recv(1024)
    print("build stdout after cleanup release", flush=True)
    print("build stderr after cleanup release", file=sys.stderr, flush=True)
    control.sendall((str(interrupts) + "\\n").encode())
    control.close()
    raise SystemExit(0)
signal.signal(signal.SIGINT, interrupted)
control.sendall((str(os.getpid()) + " " + str(os.getpgrp()) + "\\n").encode())
while True:
    signal.pause()
"""
    )
    make.chmod(0o700)
    monkeypatch.setenv("PATH", str(toolchain) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("NVSHMEM_HOME", str(tmp_path / "nvshmem"))

    if lab == "jacobi":

        class CpuBuildBenchmark(NativeJacobiBenchmark):
            def _build_architecture(self) -> str:
                return "sm_80"

        benchmark = CpuBuildBenchmark(overlapped=True)
    else:

        class CpuBuildBenchmark(NvshmemDeviceBenchmark):
            def _build_architecture(self) -> str:
                return "sm_80"

        benchmark = CpuBuildBenchmark(persistent=True)

    benchmark._config = SimpleNamespace(
        execution_timeout_seconds=0,
        enable_profiling=False,
        profile_mode="none",
        profile_type="none",
        profiling_output_dir=str(tmp_path / "artifacts"),
    )
    failures = []
    observed = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        monkeypatch.setenv("AISP_BUILD_TEST_PORT", str(listener.getsockname()[1]))

        def cancel_and_release() -> None:
            connection = None
            try:
                connection, _ = listener.accept()
                with connection.makefile("rb") as messages:
                    pid, group = map(int, messages.readline().split())
                    observed.append(pid)
                    assert group == pid == os.getpgid(pid)
                    os.kill(os.getpid(), signal.SIGINT)
                    assert messages.readline() == b"cleanup ready\n"
                    assert os.getpgid(pid) == group
                    connection.sendall(b"release\n")
                    assert messages.readline() == b"1\n"
            except BaseException as error:
                failures.append(error)
            finally:
                if connection is not None:
                    try:
                        connection.sendall(b"release\n")
                    except OSError:
                        pass
                    connection.close()

        controller = threading.Thread(target=cancel_and_release)
        controller.start()
        try:
            with pytest.raises(KeyboardInterrupt) as interrupted:
                benchmark.setup()
        finally:
            if controller.is_alive():
                try:
                    with socket.create_connection(listener.getsockname()) as wake:
                        wake.sendall(b"0 0\n")
                except OSError:
                    pass
            controller.join()

    assert not failures
    assert isinstance(interrupted.value, CooperativeProcessInterrupted)
    receipt = interrupted.value.result
    assert receipt["returncode"] == 0
    assert receipt["cancellation"]["status"] == "cleanup_completed"
    assert receipt["execution"]["live_process_group_members"] == []
    assert receipt["stdout"] == (
        "build stdout before cleanup release\nbuild stdout after cleanup release\n"
    )
    assert receipt["stderr"] == (
        "build stderr before cleanup release\nbuild stderr after cleanup release\n"
    )
    with pytest.raises(ProcessLookupError):
        os.killpg(observed[0], 0)


def _binary_parser_input(tmp_path: Path, lab: str) -> Path:
    """Bind a never-executed sentinel to real source hashes for launcher tests."""
    binary = tmp_path / "never-executed-gpu-binary"
    binary.write_bytes(b"CPU launcher control fixture, not a GPU executable\n")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    if lab == "jacobi":
        receipt = {
            "schema": "multi_gpu_communication.build.v1",
            "binary_sha256": digest,
            "source_sha256": jacobi.source_hashes(),
            "with_nccl": False,
        }
    else:
        receipt = {
            "schema": nvshmem.BUILD_SCHEMA,
            "binary_sha256": digest,
            "source_sha256": nvshmem.source_hashes(),
            "compile_command": ["never-executed-parser-fixture"],
            "compiler_versions": {"nvcc": "parser fixture", "mpicxx": "parser fixture"},
        }
    binary.with_name(binary.name + ".build.json").write_text(json.dumps(receipt))
    return binary


def _invoke(lab: str, binary: Path, launcher: Path, prefix: Path, *args: str) -> dict:
    kwargs = {
        "binary": binary,
        "launcher": str(launcher),
        "launcher_args": tuple(args),
        "json_out": prefix,
        "dump_output": prefix.with_suffix(".f32"),
    }
    if lab == "jacobi":
        return jacobi.invoke_native(jacobi.NativeWorkload(backend="mpi"), **kwargs)
    return nvshmem.invoke_native(nvshmem.NativeWorkload(), **kwargs)


@pytest.mark.parametrize("lab", ["jacobi", "nvshmem"])
def test_native_launcher_waits_for_interrupt_cleanup_and_retains_streams(
    tmp_path: Path, lab: str
) -> None:
    binary = _binary_parser_input(tmp_path, lab)
    launcher = tmp_path / "cpu-launcher"
    launcher.write_text(
        f"#!{sys.executable}\n"
        + """
import os, signal, socket, sys
control = socket.create_connection(("127.0.0.1", int(sys.argv[1])))
interrupts = 0
def interrupted(signum, frame):
    global interrupts
    interrupts += 1
    print("launcher stdout before cleanup release", flush=True)
    print("launcher stderr before cleanup release", file=sys.stderr, flush=True)
    control.sendall(b"cleanup ready\\n")
    control.recv(1024)
    print("launcher stdout after cleanup release", flush=True)
    print("launcher stderr after cleanup release", file=sys.stderr, flush=True)
    control.sendall((str(interrupts) + "\\n").encode())
    control.close()
    raise SystemExit(0)
signal.signal(signal.SIGINT, interrupted)
control.sendall((str(os.getpid()) + " " + str(os.getpgrp()) + "\\n").encode())
while True:
    signal.pause()
"""
    )
    launcher.chmod(0o700)
    failures = []
    observed = []
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        address = listener.getsockname()

        def cancel_and_release() -> None:
            connection = None
            try:
                connection, _ = listener.accept()
                with connection.makefile("rb") as messages:
                    ready = messages.readline()
                    if ready == b"aborted\n":
                        return
                    pid, group = map(int, ready.split())
                    observed.append(pid)
                    assert group == pid == os.getpgid(pid)
                    os.kill(os.getpid(), signal.SIGINT)
                    assert messages.readline() == b"cleanup ready\n"
                    assert os.getpgid(pid) == group
                    connection.sendall(b"release\n")
                    assert messages.readline() == b"1\n"
            except BaseException as error:
                failures.append(error)
            finally:
                if connection is not None:
                    try:
                        connection.sendall(b"release\n")
                    except OSError:
                        pass
                    connection.close()

        controller = threading.Thread(target=cancel_and_release)
        controller.start()
        prefix = tmp_path / "result.json"
        try:
            with pytest.raises(KeyboardInterrupt) as interrupted:
                _invoke(lab, binary, launcher, prefix, str(address[1]))
        finally:
            if controller.is_alive():
                try:
                    with socket.create_connection(address) as wake:
                        wake.sendall(b"aborted\n")
                except OSError:
                    pass
            controller.join()
    assert not failures
    assert isinstance(interrupted.value, CooperativeProcessInterrupted)
    receipt = interrupted.value.result
    assert receipt["returncode"] == 0
    assert receipt["cancellation"]["status"] == "cleanup_completed"
    assert receipt["execution"]["live_process_group_members"] == []
    assert (
        receipt["stdout"]
        == "launcher stdout before cleanup release\nlauncher stdout after cleanup release\n"
    )
    assert (
        receipt["stderr"]
        == "launcher stderr before cleanup release\nlauncher stderr after cleanup release\n"
    )
    for stream in ("stdout", "stderr"):
        assert Path(str(prefix) + f".{stream}.log").read_text() == receipt[stream]
    assert json.loads(Path(str(prefix) + ".execution.json").read_text())["cancelled"] is True
    assert not prefix.exists()
    assert not prefix.with_suffix(".f32").exists()
    with pytest.raises(ProcessLookupError):
        os.killpg(observed[0], 0)


@pytest.mark.parametrize("lab", ["jacobi", "nvshmem"])
def test_native_launcher_failure_retains_both_streams_and_receipt(tmp_path: Path, lab: str) -> None:
    binary = _binary_parser_input(tmp_path, lab)
    launcher = tmp_path / "failing-cpu-launcher"
    launcher.write_text(
        f"#!{sys.executable}\nimport sys\nprint('actual CPU launcher failure')\n"
        "print('actual CPU launcher stderr', file=sys.stderr)\nraise SystemExit(7)\n"
    )
    launcher.chmod(0o700)
    prefix = tmp_path / "result.json"
    with pytest.raises(RuntimeError, match="exit 7"):
        _invoke(lab, binary, launcher, prefix)
    receipt = json.loads(Path(str(prefix) + ".execution.json").read_text())
    assert receipt["returncode"] == 7
    assert receipt["stdout"] == "actual CPU launcher failure\n"
    assert receipt["stderr"] == "actual CPU launcher stderr\n"
    assert receipt["cancelled"] is False
    assert not prefix.exists()


def test_failed_process_creation_keeps_a_diagnostic_without_claiming_cleanup(
    tmp_path: Path,
) -> None:
    prefix = tmp_path / "launch"
    with pytest.raises(FileNotFoundError):
        run_recorded_cooperative_process(
            [str(tmp_path / "absent-executable")], output_prefix=prefix
        )
    diagnostic = json.loads(Path(str(prefix) + ".execution.json").read_text())
    assert diagnostic["receipt_available"] is False
    assert diagnostic["error_type"] == "FileNotFoundError"
    assert "execution" not in diagnostic


@pytest.mark.parametrize("existing_suffix", [".stderr.log", ".execution.json"])
def test_failed_artifact_reservation_removes_partial_files(
    tmp_path: Path,
    existing_suffix: str,
) -> None:
    prefix = tmp_path / "reservation"
    existing = Path(str(prefix) + existing_suffix)
    existing.write_text("existing owner diagnostics\n")
    with pytest.raises(FileExistsError, match="Refusing to reuse"):
        run_recorded_cooperative_process(
            [sys.executable, "-c", "raise SystemExit('must not launch')"],
            output_prefix=prefix,
        )
    assert existing.read_text() == "existing owner diagnostics\n"
    for suffix in (".stdout.log", ".stderr.log", ".execution.json"):
        path = Path(str(prefix) + suffix)
        if path != existing:
            assert not path.exists()


@pytest.mark.parametrize("lab", ["jacobi", "nvshmem"])
def test_native_launcher_preserves_existing_diagnostic(tmp_path: Path, lab: str) -> None:
    binary = _binary_parser_input(tmp_path, lab)
    launcher = tmp_path / "must-not-launch"
    marker = tmp_path / "launched"
    launcher.write_text(
        f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(marker)!r}).touch()\n"
    )
    launcher.chmod(0o700)
    prefix = tmp_path / "result.json"
    receipt = Path(str(prefix) + ".execution.json")
    receipt.write_text("owner diagnostics\n")
    with pytest.raises(FileExistsError):
        _invoke(lab, binary, launcher, prefix)
    assert receipt.read_text() == "owner diagnostics\n"
    assert not marker.exists()
