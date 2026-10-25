from __future__ import annotations

import json
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
        "time.sleep(0.8)\n"
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
        per_import_timeout_seconds=0.2,
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
    assert failure.timeout_seconds == pytest.approx(0.2)
    assert failure.diagnostic == "timed out after 0.200s"
    time.sleep(1.0)
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
    ]
    assert all(spec.demo in DEMOS for spec in specs)
    assert all(spec.required_stdout for spec in specs)
    assert all(spec.timeout_seconds <= 60.0 for spec in specs)
    assert all(spec.command[1:5] == ["-m", "cli.aisp", "demos", spec.demo] for spec in specs)


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


def test_cli_rejects_nonpositive_time_bounds() -> None:
    parser = smoke.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--deadline-seconds", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--per-import-timeout-seconds", "-1"])


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
