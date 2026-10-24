from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from core.scripts import b200_torch2141_smoke as smoke


def _write_module(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def _write_manifest(root: Path, *entries: str) -> None:
    path = root / smoke.RUNNABLE_EXAMPLE_MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(entries) + "\n", encoding="utf-8")


def _remove_imported_test_modules() -> None:
    prefixes = ("ch97", "labs.smoke_orchestration")
    for module_name in list(sys.modules):
        if module_name == "labs" or module_name.startswith(prefixes):
            sys.modules.pop(module_name, None)


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

    _remove_imported_test_modules()
    try:
        result = smoke.import_repository_examples(
            tmp_path,
            deadline=time.monotonic() + 10.0,
            per_import_timeout_seconds=2.0,
        )
    finally:
        _remove_imported_test_modules()

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


def test_real_runnable_manifest_expands_the_owned_example_roots() -> None:
    code_root = Path(__file__).resolve().parents[1]
    manifest_lines = [
        line.strip()
        for line in (code_root / smoke.RUNNABLE_EXAMPLE_MANIFEST)
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


def test_smoke_success_requires_complete_repository_imports() -> None:
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

    assert smoke.smoke_succeeded([passing_check], imports)
    assert not smoke.smoke_succeeded([passing_check, failing_check], imports)

    imports.not_run = 1
    imports.passed = 0
    imports.groups["ch01"].not_run = 1
    imports.groups["ch01"].passed = 0
    assert not smoke.smoke_succeeded([passing_check], imports)


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
    assert report["repository_imports"]["not_run"] == report["repository_imports"]["discovered"]
