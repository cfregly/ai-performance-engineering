from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _worker_env(module_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    code_root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = os.pathsep.join((str(module_dir), str(code_root)))
    return env


def test_benchmark_worker_calls_only_the_named_entrypoint(tmp_path: Path) -> None:
    output = tmp_path / "called.txt"
    (tmp_path / "worker_fixture.py").write_text(
        """from pathlib import Path
import sys

def selected_entrypoint():
    Path(sys.argv[1]).write_text('|'.join(sys.argv[2:]), encoding='utf-8')

def main():
    raise RuntimeError('main must not be inferred')
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.harness.benchmark_worker",
            "--module",
            "worker_fixture",
            "--callable",
            "selected_entrypoint",
            "--",
            str(output),
            "alpha",
            "beta",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_worker_env(tmp_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_text(encoding="utf-8") == "alpha|beta"


def test_benchmark_worker_rejects_a_missing_explicit_entrypoint(tmp_path: Path) -> None:
    (tmp_path / "worker_fixture.py").write_text(
        "def main():\n    return None\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.harness.benchmark_worker",
            "--module",
            "worker_fixture",
            "--callable",
            "missing_entrypoint",
            "--",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_worker_env(tmp_path),
    )

    assert completed.returncode != 0
    assert "worker_fixture.missing_entrypoint is not callable" in completed.stderr


def test_benchmark_worker_propagates_a_nonzero_integer_status(tmp_path: Path) -> None:
    (tmp_path / "worker_fixture.py").write_text(
        "def selected_entrypoint():\n    return 7\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.harness.benchmark_worker",
            "--module",
            "worker_fixture",
            "--callable",
            "selected_entrypoint",
            "--",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_worker_env(tmp_path),
    )

    assert completed.returncode == 7


def test_benchmark_worker_forwards_target_help_after_delimiter(tmp_path: Path) -> None:
    output = tmp_path / "called.txt"
    (tmp_path / "worker_fixture.py").write_text(
        """from pathlib import Path
import sys

def selected_entrypoint():
    Path(sys.argv[1]).write_text('|'.join(sys.argv[2:]), encoding='utf-8')
""",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.harness.benchmark_worker",
            "--module",
            "worker_fixture",
            "--callable",
            "selected_entrypoint",
            "--",
            str(output),
            "--help",
            "--config=value",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_worker_env(tmp_path),
    )

    assert completed.returncode == 0, completed.stderr
    assert output.read_text(encoding="utf-8") == "--help|--config=value"


def test_benchmark_worker_requires_argument_delimiter(tmp_path: Path) -> None:
    (tmp_path / "worker_fixture.py").write_text(
        "def selected_entrypoint():\n    return None\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "core.harness.benchmark_worker",
            "--module",
            "worker_fixture",
            "--callable",
            "selected_entrypoint",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=_worker_env(tmp_path),
    )

    assert completed.returncode != 0
    assert "an explicit '--' delimiter is required" in completed.stderr
