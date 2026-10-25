from __future__ import annotations

import inspect
import json
import os
import signal
import subprocess
import sys
import time
from importlib import import_module
from pathlib import Path

import pytest

from core.scripts import b200_torch2141_smoke as smoke


def _write_module(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _write_manifest(root: Path, *entries: str) -> None:
    path = root / smoke.COMPATIBILITY_IMPORT_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries) + "\n", encoding="utf-8")


def _write_entrypoint_manifest(root: Path, entrypoints: list[dict[str, object]]) -> None:
    path = root / smoke.ENTRYPOINT_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": smoke.ENTRYPOINT_MANIFEST_SCHEMA_VERSION,
                "entrypoints": entrypoints,
            }
        ),
        encoding="utf-8",
    )


def test_discovery_and_real_imports_aggregate_failures(tmp_path: Path) -> None:
    _write_module(tmp_path / "ch97/__init__.py", "")
    _write_module(tmp_path / "ch97/baseline_ok.py", "VALUE = 42\n")
    _write_module(tmp_path / "ch97/optimized_exit.py", "raise SystemExit(7)\n")
    _write_module(tmp_path / "labs/__init__.py", "")
    _write_module(tmp_path / "labs/smoke_orchestration/__init__.py", "")
    _write_module(tmp_path / "labs/smoke_orchestration/kernels/__init__.py", "")
    _write_module(
        tmp_path / "labs/smoke_orchestration/baseline_broken.py",
        "raise RuntimeError('import defect')\n",
    )
    _write_module(
        tmp_path / "labs/smoke_orchestration/kernels/optimized_nested.py",
        "NESTED = True\n",
    )
    _write_module(tmp_path / "labs/smoke_orchestration/helper.py", "raise RuntimeError\n")
    _write_module(tmp_path / "ch97/demo.py", "raise RuntimeError\n")
    _write_module(tmp_path / "ch97/runnable_demo.py", "VALUE = 'manifest'\n")
    _write_manifest(tmp_path, "ch97/runnable_demo.py")

    result = smoke.import_repository_examples(
        tmp_path,
        deadline=time.monotonic() + 10.0,
        per_import_timeout_seconds=2.0,
    )

    assert result.discovered == 5
    assert result.passed == 3
    assert result.failed == 2
    assert result.skipped == 0
    assert result.not_run == 0
    assert result.complete
    assert not result.succeeded
    assert result.groups["ch97"].to_dict() == {
        "discovered": 3,
        "passed": 2,
        "failed": 1,
        "skipped": 0,
        "not_run": 0,
    }
    assert result.groups["labs/smoke_orchestration"].to_dict() == {
        "discovered": 2,
        "passed": 1,
        "failed": 1,
        "skipped": 0,
        "not_run": 0,
    }
    diagnostics = {failure.module: failure.diagnostic for failure in result.failures}
    assert diagnostics["ch97.optimized_exit"] == "SystemExit: 7"
    assert diagnostics["labs.smoke_orchestration.baseline_broken"] == (
        "RuntimeError: import defect"
    )


def test_import_timeout_kills_the_worker_group_and_continues(tmp_path: Path) -> None:
    orphan_marker = tmp_path / "orphan-finished.txt"
    child_source = (
        "import pathlib, time\n"
        "time.sleep(1.5)\n"
        f"pathlib.Path({str(orphan_marker)!r}).write_text('orphan', encoding='utf-8')\n"
    )
    _write_module(tmp_path / "ch97/__init__.py", "")
    _write_module(
        tmp_path / "ch97/baseline_hang.py",
        "import subprocess, sys, time\n"
        f"subprocess.Popen([sys.executable, '-c', {child_source!r}])\n"
        "time.sleep(30)\n",
    )
    _write_module(tmp_path / "ch97/optimized_after_hang.py", "VALUE = 1\n")
    _write_manifest(tmp_path)

    started = time.monotonic()
    result = smoke.import_repository_examples(
        tmp_path,
        deadline=time.monotonic() + 5.0,
        per_import_timeout_seconds=0.5,
    )

    assert time.monotonic() - started < 3.0
    assert result.passed == 1
    assert result.failed == 1
    assert result.not_run == 0
    assert result.complete
    failure = result.failures[0]
    assert failure.module == "ch97.baseline_hang"
    assert failure.outcome == smoke.IMPORT_FAILURE_TIMEOUT
    assert failure.returncode == -signal.SIGKILL
    assert failure.signal_number == signal.SIGKILL
    assert failure.timeout_seconds == pytest.approx(0.5)
    assert failure.diagnostic == "timed out after 0.500s"
    time.sleep(1.7)
    assert not orphan_marker.exists()


def test_import_timeout_uses_the_remaining_global_budget(tmp_path: Path) -> None:
    later_marker = tmp_path / "later-imported.txt"
    _write_module(tmp_path / "ch97/__init__.py", "")
    _write_module(tmp_path / "ch97/baseline_hang.py", "import time\ntime.sleep(30)\n")
    _write_module(
        tmp_path / "ch97/optimized_not_run.py",
        f"from pathlib import Path\nPath({str(later_marker)!r}).write_text('ran')\n",
    )
    _write_manifest(tmp_path)

    started = time.monotonic()
    result = smoke.import_repository_examples(
        tmp_path,
        deadline=time.monotonic() + 0.2,
        per_import_timeout_seconds=30.0,
    )

    assert time.monotonic() - started < 1.0
    assert result.failed == 1
    assert result.not_run == 1
    assert result.deadline_exceeded
    assert not result.complete
    assert result.failures[0].outcome == smoke.IMPORT_FAILURE_TIMEOUT
    assert 0 < (result.failures[0].timeout_seconds or 0) <= 0.2
    assert not later_marker.exists()


def test_import_worker_records_signal_and_exit_then_continues(tmp_path: Path) -> None:
    _write_module(tmp_path / "ch97/__init__.py", "")
    _write_module(
        tmp_path / "ch97/baseline_crash.py",
        "import os, signal\nos.kill(os.getpid(), signal.SIGKILL)\n",
    )
    _write_module(tmp_path / "ch97/optimized_exit.py", "import os\nos._exit(23)\n")
    _write_module(tmp_path / "ch97/runnable_after_failures.py", "VALUE = 1\n")
    _write_manifest(tmp_path, "ch97/runnable_after_failures.py")

    result = smoke.import_repository_examples(
        tmp_path,
        deadline=time.monotonic() + 5.0,
        per_import_timeout_seconds=1.0,
    )

    assert result.passed == 1
    assert result.failed == 2
    assert result.not_run == 0
    assert result.complete
    failures = {failure.module: failure for failure in result.failures}
    crash = failures["ch97.baseline_crash"]
    assert crash.outcome == smoke.IMPORT_FAILURE_SIGNAL
    assert crash.returncode == -signal.SIGKILL
    assert crash.signal_number == signal.SIGKILL
    assert crash.diagnostic == "import worker terminated by signal SIGKILL (9)"
    exited = failures["ch97.optimized_exit"]
    assert exited.outcome == smoke.IMPORT_FAILURE_EXIT
    assert exited.returncode == 23
    assert exited.signal_number is None
    assert exited.diagnostic == "import worker exited with status 23"


def test_temporary_import_sweep_restores_real_labs_package_state(tmp_path: Path) -> None:
    real_labs = import_module("labs")
    real_occupancy_tuning = import_module("labs.occupancy_tuning")
    assert real_labs.occupancy_tuning is real_occupancy_tuning

    _write_module(tmp_path / "labs/__init__.py", "SOURCE = 'fixture'\n")
    _write_module(tmp_path / "labs/smoke_orchestration/__init__.py", "")
    _write_module(
        tmp_path / "labs/smoke_orchestration/baseline_ok.py",
        "VALUE = 'fixture'\n",
    )
    _write_manifest(tmp_path)

    result = smoke.import_repository_examples(
        tmp_path,
        deadline=time.monotonic() + 10.0,
        per_import_timeout_seconds=2.0,
    )

    assert result.succeeded
    assert sys.modules["labs"] is real_labs
    assert sys.modules["labs.occupancy_tuning"] is real_occupancy_tuning
    assert real_labs.occupancy_tuning is real_occupancy_tuning


def test_expired_deadline_reports_exact_remaining_counts(tmp_path: Path) -> None:
    _write_module(tmp_path / "ch97/__init__.py", "")
    _write_module(tmp_path / "ch97/baseline_one.py", "VALUE = 1\n")
    _write_module(tmp_path / "ch97/optimized_two.py", "VALUE = 2\n")
    _write_manifest(tmp_path)

    result = smoke.import_repository_examples(
        tmp_path,
        deadline=time.monotonic() - 1.0,
        per_import_timeout_seconds=1.0,
    )

    assert result.discovered == 2
    assert result.passed == 0
    assert result.failed == 0
    assert result.not_run == 2
    assert result.deadline_exceeded
    assert not result.complete
    assert result.groups["ch97"].not_run == 2


def test_compatibility_import_manifest_expands_the_owned_example_roots() -> None:
    code_root = Path(__file__).resolve().parents[1]
    manifest_lines = [
        line.strip()
        for line in (code_root / smoke.COMPATIBILITY_IMPORT_MANIFEST)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    discovered = smoke.discover_repository_examples(code_root)
    tracked = set(
        subprocess.run(
            ["git", "ls-files", "-z", "--"],
            cwd=code_root,
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .split("\0")
    )
    baseline_optimized = {
        str(path.relative_to(code_root))
        for pattern in (
            "ch*/**/baseline_*.py",
            "ch*/**/optimized_*.py",
            "labs/**/baseline_*.py",
            "labs/**/optimized_*.py",
        )
        for path in code_root.glob(pattern)
        if path.is_file()
    }

    assert len(manifest_lines) == 233
    assert len(manifest_lines) == len(set(manifest_lines))
    assert len(discovered) == 1152
    assert all((code_root / relative).is_file() for relative in manifest_lines)
    assert set(manifest_lines) <= tracked
    assert set(manifest_lines).isdisjoint(baseline_optimized)


def test_real_entrypoint_manifest_uses_registered_public_cuda_demos() -> None:
    from core.demos.demos_commands import DEMOS

    code_root = Path(__file__).resolve().parents[1]
    specs = smoke.load_repository_entrypoints(code_root)

    assert [spec.name for spec in specs] == [
        "cuda_event_timing",
        "cuda_graph_capture",
        "cuda_inductor_recompilation",
        "online_softmax",
        "transformer_engine_fp8",
        "nvfp4_add_rmsnorm",
    ]
    assert all(spec.demo in DEMOS for spec in specs)
    assert all(spec.required_stdout for spec in specs)
    assert all(spec.timeout_seconds <= 60.0 for spec in specs)
    assert all(spec.command[1:5] == ["-m", "cli.aisp", "demos", spec.demo] for spec in specs)

    by_name = {spec.name: spec for spec in specs}
    assert by_name["online_softmax"].args == ("--rows", "4", "--cols", "257")
    assert by_name["online_softmax"].timeout_seconds == 30.0
    assert by_name["transformer_engine_fp8"].args == (
        "--batch",
        "16",
        "--hidden",
        "256",
    )
    assert by_name["transformer_engine_fp8"].timeout_seconds == 50.0
    assert by_name["nvfp4_add_rmsnorm"].args == ("--workload", "rmsnorm_2048")
    assert by_name["nvfp4_add_rmsnorm"].timeout_seconds == 40.0


def test_repository_entrypoints_execute_real_bounded_subprocesses(tmp_path: Path) -> None:
    _write_module(tmp_path / "cli/__init__.py", "")
    _write_module(
        tmp_path / "cli/aisp.py",
        "import sys\nprint('fixture argv=' + ','.join(sys.argv[1:]))\n",
    )
    _write_entrypoint_manifest(
        tmp_path,
        [
            {
                "name": "fixture_pass",
                "demo": "fixture-demo",
                "args": ["--size", "4"],
                "timeout_seconds": 2,
                "required_stdout": ["fixture argv=demos,fixture-demo,--,--size,4"],
                "coverage": "Real subprocess fixture",
            },
            {
                "name": "fixture_missing_evidence",
                "demo": "fixture-demo",
                "args": ["--size", "8"],
                "timeout_seconds": 2,
                "required_stdout": ["required CUDA evidence"],
                "coverage": "Missing evidence fixture",
            },
        ],
    )

    result = smoke.run_repository_entrypoints(
        tmp_path,
        deadline=time.monotonic() + 10.0,
    )

    assert result.discovered == 2
    assert result.passed == 1
    assert result.failed == 1
    assert result.not_run == 0
    assert result.complete
    assert not result.succeeded
    assert result.results[0].returncode == 0
    assert result.results[0].status == smoke.STATUS_PASS
    assert result.results[1].returncode == 0
    assert result.results[1].status == smoke.STATUS_FAIL
    assert result.results[1].diagnostic == (
        "stdout lacked required evidence: ['required CUDA evidence']"
    )


def test_repository_entrypoint_timeout_terminates_its_process_group(tmp_path: Path) -> None:
    _write_module(tmp_path / "cli/__init__.py", "")
    _write_module(
        tmp_path / "cli/aisp.py",
        "import subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])\n"
        "print(f'child_pid={child.pid}', flush=True)\n"
        "time.sleep(30)\n",
    )
    _write_entrypoint_manifest(
        tmp_path,
        [
            {
                "name": "fixture_timeout",
                "demo": "fixture-demo",
                "args": [],
                "timeout_seconds": 0.5,
                "required_stdout": ["unreachable"],
                "coverage": "Process group timeout fixture",
            }
        ],
    )

    started = time.monotonic()
    result = smoke.run_repository_entrypoints(
        tmp_path,
        deadline=time.monotonic() + 5.0,
    )

    assert time.monotonic() - started < 3.0
    assert result.failed == 1
    assert result.complete
    assert result.results[0].status == smoke.STATUS_FAIL
    assert result.results[0].returncode is not None
    assert result.results[0].diagnostic == "timed out after 0.500s"
    assert "child_pid=" in result.results[0].stdout


def test_parent_exit_cleanup_kills_and_reaps_isolated_child_group() -> None:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        start_new_session=True,
    )
    smoke._register_child_process(process)
    try:
        smoke._kill_active_child_process_groups()
        assert process.returncode == -signal.SIGKILL
        assert process.pid not in smoke._ACTIVE_CHILD_PROCESSES
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=2.0)
        smoke._unregister_child_process(process)


def test_required_gpu_checks_cover_real_flashinfer_nccl_and_dtensor_paths() -> None:
    flashinfer_source = inspect.getsource(smoke._check_flashinfer_block_sparse)
    nccl_dtensor_source = inspect.getsource(smoke._check_nccl_dtensor)
    nccl_command = smoke._gpu_check_command(
        "nccl_dtensor",
        Path(smoke.__file__).resolve().parents[2],
    )

    assert "flashinfer.BlockSparseAttentionWrapper" in flashinfer_source
    assert "wrapper.plan(" in flashinfer_source
    assert "wrapper.run(" in flashinfer_source
    assert "torch.isfinite(output)" in flashinfer_source
    assert "torch.testing.assert_close(output, reference" in flashinfer_source
    assert "dtensor_module.DTensorMeshBenchmark()" in nccl_dtensor_source
    assert "distributed.all_reduce(" in nccl_dtensor_source
    assert "async_op=True" in nccl_dtensor_source
    assert "wait()" in nccl_dtensor_source
    assert "benchmark.benchmark_fn()" in nccl_dtensor_source
    assert "benchmark.capture_verification_payload()" in nccl_dtensor_source
    assert "torch.distributed.run" in nccl_command
    assert "--nproc-per-node=1" in nccl_command
    assert "--rdzv-backend=static" in nccl_command
    assert "--rdzv-endpoint=127.0.0.1:0" in nccl_command


def test_nccl_dtensor_check_waits_for_public_async_collective(monkeypatch) -> None:
    events: list[str] = []

    class FakeTensor:
        def __init__(self, values: list[float]):
            self.values = values
            self.shape = (len(values),)

        def __mul__(self, multiplier: int) -> FakeTensor:
            return FakeTensor([value * multiplier for value in self.values])

    class FakeTruth:
        def all(self) -> FakeTruth:
            return self

        def item(self) -> bool:
            return True

    class FakeWork:
        def wait(self) -> None:
            events.append("work.wait")

    class FakeDistributed:
        class ReduceOp:
            SUM = object()

        @staticmethod
        def is_nccl_available() -> bool:
            return True

        @staticmethod
        def get_world_size() -> int:
            return 1

        @staticmethod
        def get_rank() -> int:
            return 0

        @staticmethod
        def get_backend() -> str:
            return "nccl"

        @staticmethod
        def all_reduce(tensor, *, op, async_op):
            assert isinstance(tensor, FakeTensor)
            assert op is FakeDistributed.ReduceOp.SUM
            assert async_op is True
            events.append("all_reduce.async")
            return FakeWork()

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize(device) -> None:
            events.append(f"synchronize:{device}")

    class FakeTesting:
        @staticmethod
        def assert_close(actual, expected, **_kwargs) -> None:
            events.append("assert_close")
            assert actual.values == expected.values

    class FakeTorch:
        cuda = FakeCuda()
        testing = FakeTesting()
        float32 = object()

        @staticmethod
        def tensor(values, *, dtype, device):
            assert dtype is FakeTorch.float32
            assert device == "cuda:0"
            return FakeTensor(list(values))

        @staticmethod
        def isfinite(_tensor):
            return FakeTruth()

    class FakeBenchmark:
        device = "cuda:0"

        def __init__(self) -> None:
            self.input = FakeTensor([3.0, 5.0])
            self.output = self.input * 2

        def setup(self) -> None:
            events.append("benchmark.setup")

        def benchmark_fn(self) -> None:
            events.append("benchmark.run")

        def capture_verification_payload(self) -> None:
            events.append("benchmark.capture")

        def get_verify_inputs(self):
            return {"input": self.input}

        def get_verify_output(self):
            return self.output

        def teardown(self) -> None:
            events.append("benchmark.teardown")

    class FakeDTensorModule:
        DTensorMeshBenchmark = FakeBenchmark

    modules = {
        "torch": FakeTorch(),
        "torch.distributed": FakeDistributed(),
        "ch13.dtensor_mesh_tool": FakeDTensorModule(),
    }
    monkeypatch.setattr(smoke, "import_module", modules.__getitem__)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "1")

    details = smoke._check_nccl_dtensor()

    assert events.index("all_reduce.async") < events.index("work.wait")
    assert events.index("work.wait") < events.index("synchronize:cuda:0")
    assert details["collective"] == (
        "one-element asynchronous all_reduce SUM followed by Work.wait"
    )
    assert events[-1] == "benchmark.teardown"


def test_isolated_gpu_check_returns_success_report(monkeypatch, tmp_path: Path) -> None:
    report = {
        "status": smoke.STATUS_PASS,
        "details": {"surface": "fixture"},
    }
    command = [
        sys.executable,
        "-c",
        f"print({(smoke._GPU_CHECK_REPORT_PREFIX + json.dumps(report))!r})",
    ]
    monkeypatch.setattr(smoke, "_gpu_check_command", lambda *args, **kwargs: command)

    details = smoke._run_isolated_gpu_check(
        "flashinfer_block_sparse",
        code_root=tmp_path,
        deadline=time.monotonic() + 2.0,
        timeout_seconds=1.0,
    )

    assert details["surface"] == "fixture"
    assert 0 < details["worker_timeout_seconds"] <= 1.0


def test_isolated_gpu_check_surfaces_worker_failure(monkeypatch, tmp_path: Path) -> None:
    report = {
        "status": smoke.STATUS_FAIL,
        "diagnostic": "RuntimeError: deliberate GPU check failure",
    }
    command = [
        sys.executable,
        "-c",
        f"print({(smoke._GPU_CHECK_REPORT_PREFIX + json.dumps(report))!r})",
    ]
    monkeypatch.setattr(smoke, "_gpu_check_command", lambda *args, **kwargs: command)

    with pytest.raises(RuntimeError, match="deliberate GPU check failure"):
        smoke._run_isolated_gpu_check(
            "flashinfer_block_sparse",
            code_root=tmp_path,
            deadline=time.monotonic() + 2.0,
            timeout_seconds=1.0,
        )


def test_isolated_gpu_check_deadline_kills_worker_process_group(
    monkeypatch,
    tmp_path: Path,
) -> None:
    orphan_marker = tmp_path / "gpu-worker-orphan.txt"
    child_source = (
        "import pathlib, time\n"
        "time.sleep(0.6)\n"
        f"pathlib.Path({str(orphan_marker)!r}).write_text('orphan', encoding='utf-8')\n"
    )
    worker_source = (
        "import signal, subprocess, sys, time\n"
        "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
        f"subprocess.Popen([sys.executable, '-c', {child_source!r}])\n"
        "time.sleep(30)\n"
    )
    command = [sys.executable, "-c", worker_source]
    monkeypatch.setattr(smoke, "_gpu_check_command", lambda *args, **kwargs: command)

    started = time.monotonic()
    work_deadline = started + 0.05
    overall_deadline = work_deadline + smoke.PROCESS_GROUP_CLEANUP_RESERVE_SECONDS
    with pytest.raises(smoke.SmokeDeadlineExceededError, match="timed out after"):
        smoke._run_isolated_gpu_check(
            "flashinfer_block_sparse",
            code_root=tmp_path,
            deadline=work_deadline,
            timeout_seconds=30.0,
        )

    assert time.monotonic() <= overall_deadline
    time.sleep(0.8)
    assert not orphan_marker.exists()
    assert not smoke._ACTIVE_CHILD_PROCESSES


def test_signal_during_spawn_is_deferred_until_child_registration() -> None:
    source = """
import os
import signal
import subprocess
import sys
from core.utils import process_groups

registry = process_groups.ProcessGroupRegistry()
original_popen = process_groups.subprocess.Popen
def interrupted_popen(*args, **kwargs):
    child = original_popen(*args, **kwargs)
    print(child.pid, flush=True)
    os.kill(os.getpid(), signal.SIGTERM)
    return child
process_groups.subprocess.Popen = interrupted_popen
with registry.cleanup_on_parent_exit():
    registry.spawn(
        [sys.executable, '-c', 'import time; time.sleep(30)'],
        start_new_session=True,
    )
"""
    completed = subprocess.run(
        [sys.executable, "-c", source],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=5.0,
        check=False,
    )

    assert completed.returncode == 128 + signal.SIGTERM
    child_pid = int(completed.stdout.strip())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


def test_expired_deadline_does_not_start_repository_entrypoints(tmp_path: Path) -> None:
    _write_entrypoint_manifest(
        tmp_path,
        [
            {
                "name": "fixture_not_run",
                "demo": "fixture-demo",
                "args": [],
                "timeout_seconds": 2,
                "required_stdout": ["unreachable"],
                "coverage": "Deadline fixture",
            }
        ],
    )

    result = smoke.run_repository_entrypoints(
        tmp_path,
        deadline=time.monotonic() - 1.0,
    )

    assert result.discovered == 1
    assert result.passed == 0
    assert result.failed == 0
    assert result.not_run == 1
    assert result.deadline_exceeded
    assert not result.complete
    assert not result.results


def test_smoke_success_requires_complete_entrypoints_and_repository_imports() -> None:
    passing_check = smoke.CheckResult(
        name="required",
        policy=smoke.POLICY_REQUIRED,
        status=smoke.STATUS_PASS,
        duration_ms=1.0,
    )
    failing_check = smoke.CheckResult(
        name="failed",
        policy=smoke.POLICY_REQUIRED,
        status=smoke.STATUS_FAIL,
        duration_ms=1.0,
        diagnostic="RuntimeError: failed",
    )
    imports = smoke.ImportSweepResult(
        discovered=1,
        passed=1,
        failed=0,
        skipped=0,
        not_run=0,
        duration_ms=1.0,
        deadline_exceeded=False,
        groups={"ch01": smoke.ImportGroupCounts(discovered=1, passed=1)},
        failures=[],
    )
    entrypoints = smoke.EntrypointSweepResult(
        discovered=1,
        passed=1,
        failed=0,
        not_run=0,
        duration_ms=1.0,
        deadline_exceeded=False,
        results=[],
    )

    assert smoke.smoke_succeeded([passing_check], entrypoints, imports)
    assert not smoke.smoke_succeeded([passing_check, failing_check], entrypoints, imports)

    entrypoints.not_run = 1
    entrypoints.passed = 0
    assert not smoke.smoke_succeeded([passing_check], entrypoints, imports)
    entrypoints.not_run = 0
    entrypoints.passed = 1

    imports.not_run = 1
    imports.passed = 0
    imports.groups["ch01"].not_run = 1
    imports.groups["ch01"].passed = 0
    assert not smoke.smoke_succeeded([passing_check], entrypoints, imports)

    omitted_imports = smoke._not_requested_repository_imports(
        reason="short-gpu mode excludes the repository import sweep"
    )
    assert not omitted_imports.complete
    assert not omitted_imports.succeeded
    assert not smoke.smoke_succeeded([passing_check], entrypoints, omitted_imports)
    assert smoke.smoke_succeeded(
        [passing_check],
        entrypoints,
        omitted_imports,
        require_repository_imports=False,
    )


def test_cli_rejects_nonpositive_time_bounds() -> None:
    parser = smoke.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--deadline-seconds", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--per-import-timeout-seconds", "-1"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--deadline-seconds", "241"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--deadline-seconds", "nan"])


def test_execute_smoke_rejects_deadlines_above_the_four_minute_cap() -> None:
    with pytest.raises(ValueError, match="at most 240"):
        smoke.execute_smoke(
            Path(smoke.__file__).resolve().parents[2],
            deadline_seconds=241.0,
            per_import_timeout_seconds=smoke.DEFAULT_IMPORT_TIMEOUT_SECONDS,
        )


def test_cli_defaults_to_full_and_accepts_short_gpu_mode() -> None:
    parser = smoke.build_parser()

    assert parser.parse_args([]).mode == smoke.SMOKE_MODE_FULL
    assert (
        parser.parse_args(["--mode", smoke.SMOKE_MODE_SHORT_GPU]).mode == smoke.SMOKE_MODE_SHORT_GPU
    )


def test_full_smoke_runs_bounded_entrypoints_before_broad_import_sweep(monkeypatch) -> None:
    observed: list[str] = []

    def record_check(name, policy, check, deadline):
        observed.append(f"check:{name}")
        return smoke.CheckResult(
            name=name,
            policy=policy,
            status=smoke.STATUS_PASS,
            duration_ms=0.0,
        )

    monkeypatch.setattr(smoke, "_run_check", record_check)
    monkeypatch.setattr(
        smoke,
        "import_repository_examples",
        lambda *args, **kwargs: observed.append("imports")
        or smoke.ImportSweepResult(
            discovered=1,
            passed=1,
            failed=0,
            skipped=0,
            not_run=0,
            duration_ms=0.0,
            deadline_exceeded=False,
            groups={"ch01": smoke.ImportGroupCounts(discovered=1, passed=1)},
            failures=[],
        ),
    )
    monkeypatch.setattr(
        smoke,
        "run_repository_entrypoints",
        lambda *args, **kwargs: observed.append("entrypoints")
        or smoke.EntrypointSweepResult(
            discovered=1,
            passed=1,
            failed=0,
            not_run=0,
            duration_ms=0.0,
            deadline_exceeded=False,
            results=[],
        ),
    )

    report = smoke.execute_smoke(
        Path(smoke.__file__).resolve().parents[2],
        deadline_seconds=smoke.DEFAULT_DEADLINE_SECONDS,
        per_import_timeout_seconds=smoke.DEFAULT_IMPORT_TIMEOUT_SECONDS,
    )

    assert "check:flashinfer_block_sparse" in observed
    assert "check:nccl_dtensor" in observed
    assert observed[-2:] == ["entrypoints", "imports"]
    assert report["status"] == smoke.STATUS_PASS
    assert report["mode"] == smoke.SMOKE_MODE_FULL
    assert report["repository_imports"]["requested"] is True
    assert report["repository_imports"]["coverage_status"] == smoke.IMPORT_COVERAGE_COMPLETE


def test_short_gpu_mode_skips_imports_and_preserves_the_shared_deadline(monkeypatch) -> None:
    observed: list[str] = []
    observed_deadlines: list[float] = []

    def record_check(name, policy, check, deadline):
        observed.append(f"check:{name}")
        observed_deadlines.append(deadline)
        return smoke.CheckResult(
            name=name,
            policy=policy,
            status=smoke.STATUS_PASS,
            duration_ms=0.0,
        )

    def reject_import_sweep(*args, **kwargs):
        raise AssertionError("short-gpu mode must not start the repository import sweep")

    def record_entrypoints(*args, **kwargs):
        observed.append("entrypoints")
        observed_deadlines.append(kwargs["deadline"])
        return smoke.EntrypointSweepResult(
            discovered=6,
            passed=6,
            failed=0,
            not_run=0,
            duration_ms=0.0,
            deadline_exceeded=False,
            results=[],
        )

    monkeypatch.setattr(smoke, "_run_check", record_check)
    monkeypatch.setattr(smoke, "import_repository_examples", reject_import_sweep)
    monkeypatch.setattr(smoke, "run_repository_entrypoints", record_entrypoints)
    monkeypatch.setattr(smoke.time, "monotonic", lambda: 100.0)

    report = smoke.execute_smoke(
        Path(smoke.__file__).resolve().parents[2],
        deadline_seconds=smoke.DEFAULT_DEADLINE_SECONDS,
        per_import_timeout_seconds=smoke.DEFAULT_IMPORT_TIMEOUT_SECONDS,
        mode=smoke.SMOKE_MODE_SHORT_GPU,
    )

    assert observed[-1] == "entrypoints"
    assert all(item.startswith("check:") for item in observed[:-1])
    assert "check:vllm_abi_imports" not in observed
    assert "check:flashinfer_block_sparse" in observed
    assert "check:nccl_dtensor" in observed
    assert set(observed_deadlines) == {
        100.0 + smoke.DEFAULT_DEADLINE_SECONDS - smoke.PROCESS_GROUP_CLEANUP_RESERVE_SECONDS
    }
    assert report["status"] == smoke.STATUS_PASS
    assert report["mode"] == smoke.SMOKE_MODE_SHORT_GPU
    assert report["repository_entrypoints"]["discovered"] == 6
    assert report["repository_entrypoints"]["passed"] == 6
    assert report["cleanup_reserve_seconds"] == smoke.PROCESS_GROUP_CLEANUP_RESERVE_SECONDS

    vllm_check = next(check for check in report["checks"] if check["name"] == "vllm_abi_imports")
    assert vllm_check["policy"] == smoke.POLICY_NOT_REQUESTED
    assert vllm_check["status"] == smoke.STATUS_NOT_RUN
    assert vllm_check["diagnostic"] == ("short-gpu mode excludes the optional vLLM ABI import")
    assert report["summary"]["checks_not_requested"] == ["vllm_abi_imports"]

    imports = report["repository_imports"]
    assert imports["requested"] is False
    assert imports["counts_available"] is False
    assert imports["discovered"] is None
    assert imports["not_run"] is None
    assert imports["coverage_status"] == smoke.IMPORT_COVERAGE_NOT_REQUESTED
    assert imports["complete"] is False
    assert imports["omission_reason"] == ("short-gpu mode excludes the repository import sweep")
    assert report["summary"]["repository_import_coverage"] == (smoke.IMPORT_COVERAGE_NOT_REQUESTED)
    assert report["summary"]["repository_import_failures"] is None
    assert report["summary"]["repository_imports_not_run"] is None

    rendered = smoke.render_text(report)
    assert "mode=short-gpu" in rendered
    assert rendered.count("environment python=") == 1
    assert "CHECK NOT_RUN policy=not_requested name=vllm_abi_imports" in rendered
    assert "IMPORT_COVERAGE NOT_REQUESTED" in rendered
    assert "IMPORT_SUMMARY requested=false coverage_status=not_requested" in rendered


def test_repository_identity_records_exact_clean_git_revision(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Compatibility Test",
            "-c",
            "user.email=compatibility@example.com",
            "commit",
            "-q",
            "-m",
            "test fixture",
        ],
        cwd=tmp_path,
        check=True,
    )

    identity = smoke._check_repository_identity(tmp_path)

    assert len(identity["git_revision"]) == 40
    assert set(identity["git_revision"]) <= set("0123456789abcdef")
    assert len(identity["git_tree"]) == 40
    assert set(identity["git_tree"]) <= set("0123456789abcdef")

    untracked = tmp_path / "ch97" / "baseline_untracked.py"
    _write_module(untracked, "VALUE = 'untracked'\n")
    with pytest.raises(RuntimeError, match="differ from the recorded Git revision"):
        smoke._check_repository_identity(tmp_path)
    untracked.unlink()
    untracked.parent.rmdir()

    tracked.write_text("dirty\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="differ from the recorded Git revision"):
        smoke._check_repository_identity(tmp_path)


def test_runtime_platform_gate_reports_the_real_host() -> None:
    is_target = (
        smoke.platform.system() == "Linux"
        and smoke.platform.machine().lower() in {"x86_64", "amd64"}
        and smoke.platform.python_implementation() == "CPython"
        and tuple(sys.version_info[:2]) == (3, 12)
    )
    if is_target:
        details = smoke._check_runtime_platform()
        assert details["system"] == "Linux"
        assert details["machine"] in {"x86_64", "amd64"}
        assert details["python_implementation"] == "CPython"
        assert details["python"].startswith("3.12.")
    else:
        with pytest.raises(RuntimeError, match="requires Linux x86-64 CPython 3.12"):
            smoke._check_runtime_platform()


def test_json_cli_keeps_stdout_machine_readable_under_failure() -> None:
    code_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.scripts.b200_torch2141_smoke",
            "--deadline-seconds",
            "0.001",
            "--per-import-timeout-seconds",
            "0.001",
            "--format",
            "json",
        ],
        cwd=code_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert completed.returncode == 1
    report = json.loads(completed.stdout)
    assert report["status"] == smoke.STATUS_FAIL
    assert (
        report["repository_entrypoints"]["not_run"]
        == report["repository_entrypoints"]["discovered"]
    )
    assert report["repository_imports"]["not_run"] == report["repository_imports"]["discovered"]
