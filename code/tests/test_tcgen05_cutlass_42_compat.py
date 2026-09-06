from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

import core.common.tcgen05 as tcgen05

_PACKED_STRIDE_FIXTURE = """\
#pragma once
#define CUTLASS_HOST_DEVICE
namespace cutlass {
template <class Stride, class Shape>
int make_cute_packed_stride(Stride, Shape const&) { return 7; }
}  // namespace cutlass
"""


def _create_cutlass_candidate(
    tmp_path: Path,
    *,
    version: tuple[int, int, int] = (4, 2, 0),
) -> tuple[Path, Path]:
    cutlass_root = tmp_path / "cutlass"
    include_dir = cutlass_root / "include"
    for header in tcgen05._REQUIRED_CUTLASS_HEADERS:
        if header == tcgen05._CUTLASS_MOE_STRIDE_HEADER:
            continue
        header_path = include_dir / header
        header_path.parent.mkdir(parents=True, exist_ok=True)
        header_path.write_text(f"// {header}\n", encoding="utf-8")

    major, minor, patch = version
    version_header = include_dir / "cutlass" / "version.h"
    version_header.write_text(
        f"#define CUTLASS_MAJOR {major}\n"
        f"#define CUTLASS_MINOR {minor}\n"
        f"#define CUTLASS_PATCH {patch}\n",
        encoding="utf-8",
    )

    utility_include = cutlass_root / "tools" / "util" / "include"
    utility_header = utility_include / tcgen05._CUTLASS_42_PACKED_STRIDE_HEADER
    utility_header.parent.mkdir(parents=True, exist_ok=True)
    utility_header.write_text(_PACKED_STRIDE_FIXTURE, encoding="utf-8")
    return include_dir, utility_include


def _configure_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    include_dir: Path,
) -> Path:
    build_root = tmp_path / "extensions"
    monkeypatch.setattr(tcgen05, "_SM100_CUTLASS_CANDIDATES", (include_dir,))
    monkeypatch.setattr(tcgen05, "_FALLBACK_CUTLASS_CANDIDATES", ())
    monkeypatch.setattr(
        tcgen05,
        "_get_extension_build_dir",
        lambda name: build_root / name,
    )
    return build_root


def test_pinned_cutlass_42_uses_verified_packed_stride_compatibility(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    include_dir, utility_include = _create_cutlass_candidate(tmp_path)
    build_root = _configure_candidate(monkeypatch, tmp_path, include_dir)
    fixture_digest = hashlib.sha256(_PACKED_STRIDE_FIXTURE.encode("utf-8")).hexdigest()
    monkeypatch.setattr(tcgen05, "_CUTLASS_42_PACKED_STRIDE_SHA256", fixture_digest)

    include_roots = tcgen05._cutlass_includes_for_capability((10, 0))

    overlay_include = include_roots[0]
    assert include_roots == (overlay_include, include_dir, utility_include)
    compat_header = overlay_include / tcgen05._CUTLASS_MOE_STRIDE_HEADER
    assert compat_header.read_text(encoding="utf-8") == (
        tcgen05._CUTLASS_42_MOE_STRIDE_COMPAT_SOURCE
    )
    assert all(
        any((root / header).is_file() for root in include_roots)
        for header in tcgen05._REQUIRED_CUTLASS_HEADERS
    )
    assert compat_header.is_relative_to(build_root)

    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler unavailable for compatibility-header syntax check")
    probe = tmp_path / "probe.cpp"
    executable = tmp_path / "probe"
    probe.write_text(
        "#include <cutlass/detail/collective/moe_stride_utils.hpp>\n"
        "int main() {\n"
        "  return cutlass::make_internal_packed_stride(1, 2) == 7 ? 0 : 1;\n"
        "}\n",
        encoding="utf-8",
    )
    compile_result = subprocess.run(
        [
            compiler,
            "-std=c++17",
            *(f"-I{root}" for root in include_roots),
            str(probe),
            "-o",
            str(executable),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr
    run_result = subprocess.run(
        [str(executable)],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr


def test_cutlass_42_compatibility_rejects_unverified_utility_header(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    include_dir, _ = _create_cutlass_candidate(tmp_path)
    build_root = _configure_candidate(monkeypatch, tmp_path, include_dir)

    with pytest.raises(RuntimeError, match="unverified cutlass/util/packed_stride.hpp digest"):
        tcgen05._cutlass_includes_for_capability((10, 0))

    assert not build_root.exists()


def test_cutlass_compatibility_is_not_applied_to_other_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    include_dir, _ = _create_cutlass_candidate(tmp_path, version=(4, 3, 0))
    build_root = _configure_candidate(monkeypatch, tmp_path, include_dir)
    fixture_digest = hashlib.sha256(_PACKED_STRIDE_FIXTURE.encode("utf-8")).hexdigest()
    monkeypatch.setattr(tcgen05, "_CUTLASS_42_PACKED_STRIDE_SHA256", fixture_digest)

    with pytest.raises(RuntimeError, match=r"CUTLASS version is \(4, 3, 0\)"):
        tcgen05._cutlass_includes_for_capability((10, 0))

    assert not build_root.exists()
