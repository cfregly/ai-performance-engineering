from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import core.benchmark.tcgen05_requirements as requirements
import core.common.tcgen05 as tcgen05

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ONLY_TESTS = (
    "tests/test_benchmark_story_metadata.py",
    "tests/test_ch10_tcgen05_cluster_pipeline_sources.py",
    "tests/test_custom_metrics_cleanup.py",
)
TCGEN05_SOURCES = (
    REPO_ROOT / "ch08" / "tiling_kernels_tcgen05.cu",
    REPO_ROOT / "ch09" / "tcgen05_basic.cu",
    REPO_ROOT / "ch09" / "tcgen05_pipelined.cu",
    REPO_ROOT / "ch10" / "matmul_tcgen05.cu",
    REPO_ROOT / "ch10" / "tcgen05_cluster.cu",
    REPO_ROOT / "ch10" / "tcgen05_warp_specialized.cu",
    REPO_ROOT / "ch10" / "tcgen05_warp_specialized_cutlass.cu",
    REPO_ROOT / "ch10" / "tcgen05_warpgroup_specialized.cu",
)


def _create_cutlass_headers(include_dir: Path) -> Path:
    for header in tcgen05._REQUIRED_CUTLASS_HEADERS:
        header_path = include_dir / header
        header_path.parent.mkdir(parents=True, exist_ok=True)
        header_path.write_text(f"// {header}\n")
    return include_dir


def _configure_supported_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capability: tuple[int, int] = (10, 0),
) -> tuple[Path, Callable[[str], Path]]:
    include_dir = _create_cutlass_headers(tmp_path / "cutlass" / "include")
    build_root = tmp_path / "extensions"

    def build_dir(name: str) -> Path:
        return build_root / name

    monkeypatch.setattr(tcgen05, "_detect_compute_capability", lambda: capability)
    monkeypatch.setattr(tcgen05, "_SM100_CUTLASS_CANDIDATES", (include_dir,))
    monkeypatch.setattr(tcgen05, "_FALLBACK_CUTLASS_CANDIDATES", ())
    monkeypatch.setattr(tcgen05, "_get_extension_build_dir", build_dir)
    return include_dir, build_dir


def test_tcgen05_source_only_tests_collect_with_cuda_hidden() -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            *SOURCE_ONLY_TESTS,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )

    assert completed.returncode == pytest.ExitCode.OK, completed.stdout + completed.stderr


def test_tcgen05_required_cutlass_headers_cover_direct_source_includes() -> None:
    direct_cutlass_headers: set[Path] = set()
    for source in TCGEN05_SOURCES:
        for line in source.read_text().splitlines():
            stripped = line.strip()
            if not stripped.startswith("#include <"):
                continue
            header = stripped.removeprefix("#include <").split(">", maxsplit=1)[0]
            if header.startswith(("cute/", "cutlass/")):
                direct_cutlass_headers.add(Path(header))

    assert direct_cutlass_headers == set(tcgen05._REQUIRED_CUTLASS_HEADERS)


@pytest.mark.parametrize(
    ("capability", "message"),
    (
        (None, "requires a visible CUDA device"),
        ((9, 0), "requires SM100-class Tensor Cores"),
        ((12, 0), "has no natively validated SM120 implementation"),
        ((12, 1), r"is not supported on sm_121 \(GB10\)"),
        ((13, 0), "does not support sm_130"),
    ),
)
def test_tcgen05_loader_fails_closed_without_supported_capability(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capability: tuple[int, int] | None,
    message: str,
) -> None:
    build_root = tmp_path / "extensions"
    monkeypatch.setattr(tcgen05, "_detect_compute_capability", lambda: capability)
    monkeypatch.setattr(
        tcgen05,
        "_get_extension_build_dir",
        lambda name: build_root / name,
    )
    monkeypatch.setattr(
        tcgen05,
        "load",
        lambda **_kwargs: pytest.fail("extension build must not start without CUDA capability"),
    )
    tcgen05.load_tiling_tcgen05_module.cache_clear()

    try:
        with pytest.raises(RuntimeError, match=f"SKIPPED: tcgen05 extension loading {message}"):
            tcgen05.load_tiling_tcgen05_module()
    finally:
        tcgen05.load_tiling_tcgen05_module.cache_clear()
    assert not build_root.exists()


@pytest.mark.parametrize(
    ("capability", "target"),
    (
        ((10, 0), "100a"),
        ((10, 3), "103a"),
    ),
)
def test_tcgen05_flags_use_the_exact_validated_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capability: tuple[int, int],
    target: str,
) -> None:
    include_dir, _ = _configure_supported_loader(monkeypatch, tmp_path, capability)

    flags = tcgen05._tcgen05_cuda_flags()

    gencode_flags = [flag for flag in flags if flag.startswith("-gencode=")]
    assert gencode_flags == [f"-gencode=arch=compute_{target},code=sm_{target}"]
    assert f"-I{include_dir}" in flags


@pytest.mark.parametrize(
    ("capability", "message"),
    (
        ((9, 0), "requires SM100-class Tensor Cores"),
        ((12, 0), "has no natively validated SM120 implementation"),
        ((12, 1), r"is not supported on sm_121 \(GB10\)"),
        ((13, 0), "does not support sm_130"),
    ),
)
def test_tcgen05_public_gate_matches_loader_rejections(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
    message: str,
) -> None:
    monkeypatch.setattr(requirements, "ensure_blackwell_tma_supported", lambda _name: None)
    monkeypatch.setattr(
        requirements.torch.cuda,
        "get_device_capability",
        lambda: capability,
    )

    with pytest.raises(RuntimeError, match=message):
        requirements.ensure_tcgen05_supported(
            loader=lambda: pytest.fail("unsupported capability must not invoke the loader")
        )


@pytest.mark.parametrize("capability", ((10, 0), (10, 3)))
def test_tcgen05_public_gate_accepts_exact_validated_capabilities(
    monkeypatch: pytest.MonkeyPatch,
    capability: tuple[int, int],
) -> None:
    loader_calls = 0

    def loader() -> object:
        nonlocal loader_calls
        loader_calls += 1
        return object()

    monkeypatch.setattr(requirements, "ensure_blackwell_tma_supported", lambda _name: None)
    monkeypatch.setattr(
        requirements.torch.cuda,
        "get_device_capability",
        lambda: capability,
    )

    requirements.ensure_tcgen05_supported(loader=loader)

    assert loader_calls == 1


def test_tcgen05_include_selection_skips_incomplete_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    complete = _create_cutlass_headers(tmp_path / "complete")
    monkeypatch.setattr(tcgen05, "_SM100_CUTLASS_CANDIDATES", (incomplete, complete))
    monkeypatch.setattr(tcgen05, "_FALLBACK_CUTLASS_CANDIDATES", ())

    assert tcgen05._cutlass_includes_for_capability((10, 0)) == (complete,)


@pytest.mark.parametrize("invalid_kind", ("missing_headers", "regular_file"))
def test_tcgen05_invalid_cutlass_install_fails_before_cache_or_load(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    invalid_kind: str,
) -> None:
    invalid_include = tmp_path / "invalid-include"
    if invalid_kind == "missing_headers":
        invalid_include.mkdir()
        (invalid_include / "cute").mkdir()
    else:
        invalid_include.write_text("not a directory\n")
    build_root = tmp_path / "extensions"
    source = tmp_path / "kernel.cu"
    source.write_text("// source\n")
    monkeypatch.setattr(tcgen05, "_detect_compute_capability", lambda: (10, 0))
    monkeypatch.setattr(tcgen05, "_SM100_CUTLASS_CANDIDATES", (invalid_include,))
    monkeypatch.setattr(tcgen05, "_FALLBACK_CUTLASS_CANDIDATES", ())
    monkeypatch.setattr(
        tcgen05,
        "_get_extension_build_dir",
        lambda name: build_root / name,
    )
    monkeypatch.setattr(
        tcgen05,
        "load",
        lambda **_kwargs: pytest.fail("invalid includes must not invoke PyTorch load"),
    )

    with pytest.raises(RuntimeError, match="SKIPPED:.*valid CUTLASS include directory"):
        tcgen05._load_extension("test_ext", [source])

    assert not build_root.exists()


def test_tcgen05_matching_fingerprint_delegates_header_check_to_exact_pytorch_build_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    include_dir, build_dir_for = _configure_supported_loader(monkeypatch, tmp_path)
    source = tmp_path / "kernel.cu"
    source.write_text("// source\n")
    name = "test_ext"
    flags = tcgen05._tcgen05_cuda_flags()
    fingerprint = tcgen05._compute_build_fingerprint([source], flags)
    (include_dir / "cute" / "tensor.hpp").write_text("// changed header\n")
    assert tcgen05._compute_build_fingerprint([source], flags) == fingerprint
    build_dir = build_dir_for(name)
    build_dir.mkdir(parents=True)
    (build_dir / f"{name}.so").write_bytes(b"stale shared library")
    (build_dir / ".build_fingerprint").write_text('{"fingerprint": "' + fingerprint + '"}\n')
    module = object()
    calls: list[dict[str, Any]] = []

    def fake_load(**kwargs: Any) -> object:
        calls.append(kwargs)
        return module

    monkeypatch.setattr(tcgen05, "load", fake_load)

    assert tcgen05._load_extension(name, [source]) is module
    assert len(calls) == 1
    assert calls[0]["build_directory"] == str(build_dir)
    assert (build_dir / f"{name}.so").read_bytes() == b"stale shared library"
    assert not list(build_dir.glob(".build_fingerprint.*"))


def test_tcgen05_invalidation_failure_is_fatal_and_preserves_old_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    build_dir = tmp_path / "test_ext"
    build_dir.mkdir()
    fingerprint_file = build_dir / ".build_fingerprint"
    fingerprint_file.write_text('{"fingerprint": "old"}\n')
    stale_output = build_dir / "test_ext.so"
    stale_output.write_bytes(b"old")
    monkeypatch.setattr(tcgen05, "_get_extension_build_dir", lambda _name: build_dir)
    monkeypatch.setattr(tcgen05, "_compute_build_fingerprint", lambda *_args: "new")

    def fail_remove(_path: Path) -> None:
        raise PermissionError("cache is not removable")

    monkeypatch.setattr(tcgen05.shutil, "rmtree", fail_remove)

    with pytest.raises(PermissionError, match="cache is not removable"):
        tcgen05._check_and_invalidate_cache("test_ext", [], [])

    assert fingerprint_file.read_text() == '{"fingerprint": "old"}\n'
    assert stale_output.read_bytes() == b"old"


def test_tcgen05_failed_build_does_not_record_fingerprint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, build_dir_for = _configure_supported_loader(monkeypatch, tmp_path)
    source = tmp_path / "kernel.cu"
    source.write_text("// source\n")
    monkeypatch.setattr(
        tcgen05,
        "load",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("compiler failed")),
    )

    with pytest.raises(RuntimeError, match="compiler failed"):
        tcgen05._load_extension("test_ext", [source])

    build_dir = build_dir_for("test_ext")
    assert not (build_dir / ".build_fingerprint").exists()
    assert not list(build_dir.glob(".build_fingerprint.*"))


def test_tcgen05_retry_uses_same_build_dir_and_records_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, build_dir_for = _configure_supported_loader(monkeypatch, tmp_path)
    source = tmp_path / "kernel.cu"
    source.write_text("// source\n")
    module = object()
    calls: list[dict[str, Any]] = []

    def fake_load(**kwargs: Any) -> object:
        calls.append(kwargs)
        if len(calls) == 1:
            raise RuntimeError("No such file")
        return module

    monkeypatch.setattr(tcgen05, "load", fake_load)

    assert tcgen05._load_extension("test_ext", [source]) is module
    expected_build_dir = str(build_dir_for("test_ext"))
    assert [call["build_directory"] for call in calls] == [
        expected_build_dir,
        expected_build_dir,
    ]
    assert calls[0]["verbose"] is False
    assert calls[1]["verbose"] is True
    assert (build_dir_for("test_ext") / ".build_fingerprint").is_file()
    assert not list(build_dir_for("test_ext").glob(".build_fingerprint.*"))


def test_tcgen05_successful_public_loader_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = object()
    calls = 0

    def fake_load_extension(_name: str, _sources: list[Path]) -> object:
        nonlocal calls
        calls += 1
        return module

    monkeypatch.setattr(tcgen05, "_load_extension", fake_load_extension)
    tcgen05.load_tiling_tcgen05_module.cache_clear()
    try:
        assert tcgen05.load_tiling_tcgen05_module() is module
        assert tcgen05.load_tiling_tcgen05_module() is module
    finally:
        tcgen05.load_tiling_tcgen05_module.cache_clear()

    assert calls == 1
