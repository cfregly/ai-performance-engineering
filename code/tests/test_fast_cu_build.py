"""Build provenance checks that execute without a CUDA toolchain."""

from __future__ import annotations

import pytest

from labs.fast_cu.build import extension_name, parse_nvcc_release


def test_nvcc_release_requires_explicit_toolkit_version():
    assert parse_nvcc_release("Cuda compilation tools, release 13.1, V13.1.80") == (13, 1)
    assert parse_nvcc_release("Cuda compilation tools, release 12.8, V12.8.61") == (12, 8)
    with pytest.raises(RuntimeError, match="Cannot determine"):
        parse_nvcc_release("driver version 590.0")


def test_cache_key_tracks_source_headers_and_flags(tmp_path):
    source = tmp_path / "extension.cu"
    source.write_text("// original\n")
    manifest = {"revision": "pinned", "files": {"gemm.cuh": "a" * 64}}
    original = extension_name("fast_cu", [source], ["-O3"], ["-lcuda"], manifest)
    assert original == extension_name("fast_cu", [source], ["-O3"], ["-lcuda"], manifest)
    source.write_text("// changed\n")
    assert original != extension_name("fast_cu", [source], ["-O3"], ["-lcuda"], manifest)
    source.write_text("// original\n")
    assert original != extension_name("fast_cu", [source], ["-O2"], ["-lcuda"], manifest)
    assert original != extension_name("fast_cu", [source], ["-O3"], ["-lcublas"], manifest)
    manifest["files"]["gemm.cuh"] = "b" * 64
    assert original != extension_name("fast_cu", [source], ["-O3"], ["-lcuda"], manifest)

    header = tmp_path / "port.cuh"
    header.write_text("// original port\n")
    with_header = extension_name("fast_cu", [source, header], [], [], manifest)
    header.write_text("// changed port\n")
    assert with_header != extension_name("fast_cu", [source, header], [], [], manifest)


def test_extension_name_is_safe_and_sources_required(tmp_path):
    with pytest.raises(ValueError, match="identifier"):
        extension_name("bad-name", [], [], [], {})
    with pytest.raises(ValueError, match="source"):
        extension_name("fast_cu", [], [], [], {})


def test_cache_key_tracks_in_place_toolkit_upgrade(tmp_path):
    source = tmp_path / "extension.cu"
    source.write_text("// unchanged source\n")
    manifest = {"revision": "pinned", "files": {"gemm.cuh": "a" * 64}}
    old = extension_name(
        "fast_cu",
        [source],
        [],
        [],
        manifest,
        toolchain_identity="/cuda/bin/nvcc\nrelease 13.0 V13.0.88",
    )
    new = extension_name(
        "fast_cu",
        [source],
        [],
        [],
        manifest,
        toolchain_identity="/cuda/bin/nvcc\nrelease 13.1 V13.1.80",
    )
    assert old != new
