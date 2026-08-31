"""Behavioral checks for the repository's static import-boundary gate."""

from pathlib import Path
import subprocess
import sys

import pytest

from core.scripts import check_import_edges as checker


def write_module(root, relative, source):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)
    return path


@pytest.mark.parametrize("statement", [
    "import labs.example", "from labs import example", "import ch01.example",
    "import ch1.example", "from ch10.example import run",
])
def test_absolute_core_paths_reject_chapter_and_lab_back_edges(tmp_path, statement):
    path = write_module(tmp_path, "core/example.py", statement)
    errors = checker.check_file(path, tmp_path)
    assert len(errors) == 1
    assert "back-edge" in errors[0][1]


def test_cli_scans_padded_chapters_and_operational_scripts(tmp_path):
    write_module(tmp_path, "ch01/baseline_example.py", "from common import example")
    write_module(tmp_path, "scripts/example.py", "import profiling.example")
    result = subprocess.run(
        [sys.executable, str(Path(checker.__file__).resolve()), "--root", str(tmp_path)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 1
    assert "ch01/baseline_example.py:1" in result.stdout
    assert "scripts/example.py:1" in result.stdout
    assert "core.common" in result.stdout
    assert "core.profiling" in result.stdout


@pytest.mark.parametrize("statement", [
    "import scripts.queue_restore", "from scripts import queue_restore",
    "from scripts.queue_restore import restore",
])
def test_existing_operational_namespace_is_allowed_without_executing_it(tmp_path, statement):
    write_module(tmp_path, "scripts/queue_restore.py", "raise RuntimeError('must not execute')")
    source = write_module(tmp_path, "tests/test_example.py", statement)
    assert checker.check_file(source, tmp_path) == []


@pytest.mark.parametrize("statement", [
    "import scripts.validate_imports", "from scripts import validate_imports",
    "import scripts", "from scripts import *",
])
def test_operational_namespace_does_not_exempt_migrated_or_unspecified_helpers(tmp_path, statement):
    write_module(tmp_path, "core/scripts/validate_imports.py", "def main(): pass")
    (tmp_path / "scripts").mkdir()
    source = write_module(tmp_path, "tests/test_example.py", statement)
    errors = checker.check_file(source, tmp_path)
    assert len(errors) == 1
    assert "core.scripts" in errors[0][1]


def test_relative_core_import_stays_in_its_package(tmp_path):
    source = write_module(tmp_path, "core/helpers/example.py", "from ..common import example")
    assert checker.check_file(source, tmp_path) == []


def test_script_symlink_cannot_authorize_a_module_outside_the_scripts_tree(tmp_path):
    outside = write_module(tmp_path, "unrelated/example.py", "pass")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "example.py").symlink_to(outside)
    source = write_module(tmp_path, "tests/test_example.py", "import scripts.example")
    assert "core.scripts" in checker.check_file(source, tmp_path)[0][1]


def test_scan_root_is_applied_to_core_back_edges_in_cli(tmp_path):
    write_module(tmp_path, "core/example.py", "from labs.example import run")
    result = subprocess.run(
        [sys.executable, str(Path(checker.__file__).resolve()), "--root", str(tmp_path)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 1
    assert "core/example.py:1" in result.stdout
    assert "back-edge" in result.stdout
