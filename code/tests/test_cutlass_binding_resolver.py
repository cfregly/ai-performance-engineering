from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from core.benchmark import cutlass_binding


def _write_main_header(root):
    header = root / "include/cutlass/cutlass.h"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.touch()


def _write_util_header(root):
    header = root / "tools/util/include/cutlass/util/packed_stride.hpp"
    header.parent.mkdir(parents=True, exist_ok=True)
    header.touch()


def _complete_root(root):
    _write_main_header(root)
    _write_util_header(root)
    return root


def _install_package_fixture(monkeypatch, root):
    module_file = root / "cutlass_library/__init__.py"
    module_file.parent.mkdir(parents=True, exist_ok=True)
    module_file.touch()
    package_root = _complete_root(module_file.parent / "source")
    monkeypatch.setitem(
        sys.modules, "cutlass_library", SimpleNamespace(__file__=str(module_file))
    )
    return package_root


def test_cutlass_root_precedence_is_explicit_then_repository_then_package(
    tmp_path, monkeypatch
):
    explicit = _complete_root(tmp_path / "explicit")
    repository = _complete_root(tmp_path / "repo/third_party/cutlass")
    package = _install_package_fixture(monkeypatch, tmp_path / "package")
    monkeypatch.setenv("CUTLASS_PATH", str(explicit))

    assert cutlass_binding._detect_cutlass_include_dirs(
        repo_root=tmp_path / "repo"
    ) == [explicit / "include", explicit / "tools/util/include"]

    monkeypatch.delenv("CUTLASS_PATH")
    assert cutlass_binding._detect_cutlass_include_dirs(
        repo_root=tmp_path / "repo"
    ) == [repository / "include", repository / "tools/util/include"]

    (repository / "include/cutlass/cutlass.h").unlink()
    assert cutlass_binding._detect_cutlass_include_dirs(
        repo_root=tmp_path / "repo"
    ) == [package / "include", package / "tools/util/include"]


def test_invalid_explicit_cutlass_path_fails_without_fallback(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    _write_main_header(explicit)
    _complete_root(tmp_path / "repo/third_party/cutlass")
    _install_package_fixture(monkeypatch, tmp_path / "package")
    monkeypatch.setenv("CUTLASS_PATH", str(explicit))

    with pytest.raises(RuntimeError, match="CUTLASS_PATH must name one complete"):
        cutlass_binding._detect_cutlass_include_dirs(repo_root=tmp_path / "repo")


def test_cutlass_resolver_does_not_mix_headers_from_different_roots(
    tmp_path, monkeypatch
):
    repository = tmp_path / "repo/third_party/cutlass"
    _write_main_header(repository)
    module_file = tmp_path / "package/cutlass_library/__init__.py"
    module_file.parent.mkdir(parents=True, exist_ok=True)
    module_file.touch()
    package = module_file.parent / "source"
    _write_util_header(package)
    monkeypatch.delenv("CUTLASS_PATH", raising=False)
    monkeypatch.setitem(
        sys.modules, "cutlass_library", SimpleNamespace(__file__=str(module_file))
    )

    with pytest.raises(RuntimeError, match="No complete CUTLASS source root"):
        cutlass_binding._detect_cutlass_include_dirs(repo_root=tmp_path / "repo")
