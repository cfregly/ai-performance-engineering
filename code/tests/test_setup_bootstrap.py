"""Tests for the privileged setup repository bootstrap."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CODE_ROOT.parent.resolve()
SETUP_SCRIPT = CODE_ROOT / "setup.sh"


def _fake_git(tmp_path: Path, *, fail_on: str = "") -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_path = tmp_path / "git.log"
    git_path = bin_dir / "git"
    git_path.write_text(
        """#!/bin/sh
printf '%s\\n' "$*" >> "$AISP_TEST_GIT_LOG"
if [ -n "$AISP_TEST_GIT_FAIL_ON" ]; then
    case "$*" in
        *"$AISP_TEST_GIT_FAIL_ON"*) exit 42 ;;
    esac
fi
exit 0
""",
        encoding="utf-8",
    )
    git_path.chmod(0o755)
    return bin_dir, log_path


def _run_bootstrap(tmp_path: Path, *, fail_on: str = "") -> subprocess.CompletedProcess[str]:
    bin_dir, log_path = _fake_git(tmp_path, fail_on=fail_on)
    setup_link = tmp_path / "setup-link.sh"
    setup_link.symlink_to(SETUP_SCRIPT)
    env = os.environ.copy()
    env.update(
        {
            "AISP_SETUP_BOOTSTRAP_ONLY": "1",
            "AISP_TEST_GIT_FAIL_ON": fail_on,
            "AISP_TEST_GIT_LOG": str(log_path),
            "PATH": f"{bin_dir}:/usr/bin:/bin",
        }
    )
    return subprocess.run(
        ["/bin/bash", str(setup_link)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_bootstrap_resolves_physical_root_and_marks_it_safe_before_git_reads(
    tmp_path: Path,
) -> None:
    result = _run_bootstrap(tmp_path)
    calls = (tmp_path / "git.log").read_text(encoding="utf-8").splitlines()

    assert result.returncode == 0, result.stderr
    assert result.stdout.endswith(f"Repository bootstrap complete: {REPOSITORY_ROOT}\n")
    assert calls[0] == f"config --global --add safe.directory {REPOSITORY_ROOT}"
    assert calls[1:] == [
        f"-C {REPOSITORY_ROOT} submodule sync --recursive",
        f"-C {REPOSITORY_ROOT} submodule update --init --recursive",
    ]
    assert all("rev-parse" not in call for call in calls)


def test_bootstrap_propagates_injected_submodule_git_failure(tmp_path: Path) -> None:
    result = _run_bootstrap(tmp_path, fail_on="submodule update --init --recursive")
    calls = (tmp_path / "git.log").read_text(encoding="utf-8").splitlines()

    assert result.returncode == 42
    assert calls[-1] == f"-C {REPOSITORY_ROOT} submodule update --init --recursive"
    assert "Repository bootstrap complete" not in result.stdout
