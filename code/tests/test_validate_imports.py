"""Run the actual import-validation CLI against real temporary CPU modules."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "core/scripts/validate_imports.py"


def fixture_repository(tmp_path, modules):
    root = tmp_path / "fixture-code"
    script = root / "core/scripts/validate_imports.py"
    script.parent.mkdir(parents=True)
    script.write_bytes(SOURCE.read_bytes())
    (root / "core/__init__.py").touch()
    (root / "core/scripts/__init__.py").touch()
    for name, source in modules.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent / "__init__.py").touch()
        path.write_text(source)
    return root, script


def run_validator(tmp_path, modules, *args):
    root, _ = fixture_repository(tmp_path, modules)
    # Use the documented module entrypoint from the code directory, without an
    # installed repository or PYTHONPATH concealing broken CLI root discovery.
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-m", "core.scripts.validate_imports", *args], cwd=root,
                            env=env, capture_output=True, text=True, timeout=10)
    return root, result


MARK_IMPORTED = "from pathlib import Path\nPath(__file__).with_suffix('.imported').write_text('executed')\n"


def test_default_root_cli_actually_imports_both_chapter_variants(tmp_path):
    root, result = run_validator(tmp_path, {
        "ch01/baseline_fixture.py": MARK_IMPORTED,
        "ch01/optimized_fixture.py": MARK_IMPORTED,
    }, "--verbose")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Results: 2/2 passed (100.0%)" in result.stdout
    assert sorted(path.name for path in root.rglob("*.imported")) == [
        "baseline_fixture.imported", "optimized_fixture.imported"]


def test_chapter_cli_supports_padded_and_legacy_paths_without_other_imports(tmp_path):
    root, result = run_validator(tmp_path, {
        "ch07/baseline_fixture.py": MARK_IMPORTED,
        "ch7/optimized_fixture.py": MARK_IMPORTED,
        "ch01/baseline_not_selected.py": "raise RuntimeError('must not import another chapter')\n",
    }, "--chapter", "7")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Results: 2/2 passed (100.0%)" in result.stdout
    assert len(list(root.rglob("*.imported"))) == 2


@pytest.mark.parametrize("arguments", [(), ("--chapter", "19")])
def test_empty_inventory_is_explicit_failure_without_zero_division(tmp_path, arguments):
    _, result = run_validator(tmp_path, {}, *arguments)
    assert result.returncode == 1
    assert "No benchmark files found" in result.stdout
    assert "ZeroDivisionError" not in result.stderr
    assert "All imports successful" not in result.stdout


@pytest.mark.parametrize("broken_source,diagnostic", [
    ("import deliberately_absent_validation_dependency\n", "ModuleNotFoundError"),
    ("raise RuntimeError('SKIPPED: actual CUDA capability unavailable')\n", "SKIPPED: actual CUDA capability unavailable"),
    ("raise SystemExit(0)\n", "SystemExit: 0"),
])
def test_real_import_failures_are_reported_and_never_count_as_pass(tmp_path, broken_source, diagnostic):
    root, result = run_validator(tmp_path, {
        "ch01/baseline_broken.py": broken_source,
        "ch01/optimized_actual_import.py": MARK_IMPORTED,
    })
    assert result.returncode == 1, result.stdout + result.stderr
    assert "Results: 1/2 passed (50.0%)" in result.stdout
    assert diagnostic in result.stdout
    assert (root / "ch01/optimized_actual_import.imported").read_text() == "executed"
    assert "All imports successful" not in result.stdout


def test_multiline_import_error_preserves_class_and_flattens_full_message(tmp_path):
    broken_source = (
        "raise ImportError('cuda-python is required for this benchmark. Install with:\\n"
        "  pip install cuda-python\\n"
        "Or run: bash setup.sh')\n"
    )
    _, result = run_validator(tmp_path, {"ch14/optimized_cuda_python.py": broken_source})

    assert result.returncode == 1, result.stdout + result.stderr
    assert (
        "ImportError: cuda-python is required for this benchmark. "
        "Install with: pip install cuda-python Or run: bash setup.sh"
    ) in result.stdout
    assert "\n  ImportError (1 files):\n" in result.stdout
    assert "\nOr run: bash setup.sh" not in result.stdout
    assert "\n  Or run (" not in result.stdout


def test_zero_chapter_cannot_silently_select_all_chapters(tmp_path):
    root, result = run_validator(tmp_path, {"ch01/baseline_fixture.py": MARK_IMPORTED}, "--chapter", "0")
    assert result.returncode == 2
    assert "chapter must be positive" in result.stderr
    assert not list(root.rglob("*.imported"))


def test_actual_make_chapter_recipe_imports_without_pythonpath(tmp_path):
    root, _ = fixture_repository(tmp_path, {"ch07/baseline_fixture.py": MARK_IMPORTED})
    (root / "Makefile").write_bytes((SOURCE.parents[2] / "Makefile").read_bytes())
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    make = shutil.which("make")
    assert make is not None, "make is required to validate the repository Makefile entrypoint"
    result = subprocess.run([make, "validate-ch7", f"PYTHON={sys.executable}"], cwd=root,
                            env=env, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"{sys.executable} -m core.scripts.validate_imports --chapter 7" in result.stdout
    assert "Results: 1/1 passed (100.0%)" in result.stdout
    assert (root / "ch07/baseline_fixture.imported").read_text() == "executed"
