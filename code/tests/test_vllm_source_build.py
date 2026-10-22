"""Focused contracts for the pinned vLLM native source build."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from scripts import build_vllm_for_torch as builder

CODE_ROOT = Path(__file__).resolve().parents[1]


def _contract() -> dict[str, Any]:
    return json.loads((CODE_ROOT / "vllm_source_build.json").read_text(encoding="utf-8"))


def _write_compiler(path: Path, body: str) -> None:
    path.write_text("#!/bin/sh\nset -eu\n" + body, encoding="utf-8")
    path.chmod(0o755)


def test_contract_requires_the_reviewed_elfutils_header_probe() -> None:
    contract = _contract()

    builder._validate_contract(contract)

    assert contract["build"]["required_cxx_header_probe"] == {
        "compiler": "g++",
        "language": "c++",
        "standard": "c++20",
        "headers": ["elfutils/libdwfl.h"],
        "debian_packages": ["libdw-dev", "libelf-dev"],
    }


def test_header_probe_preprocesses_the_exact_source_with_the_supplied_environment(
    tmp_path: Path,
) -> None:
    compiler = tmp_path / "g++"
    arguments = tmp_path / "arguments.txt"
    source = tmp_path / "source.cc"
    include_path = tmp_path / "include"
    include_path.mkdir()
    _write_compiler(
        compiler,
        'printf "%s\\n" "$@" > "$AISP_TEST_ARGUMENTS"\n'
        'cat > "$AISP_TEST_SOURCE"\n'
        'test "$CPATH" = "$AISP_TEST_EXPECTED_CPATH"\n',
    )
    environment = {
        **os.environ,
        "AISP_TEST_ARGUMENTS": str(arguments),
        "AISP_TEST_SOURCE": str(source),
        "AISP_TEST_EXPECTED_CPATH": str(include_path),
        "CPATH": str(include_path),
    }

    receipt = builder._validate_required_cxx_headers(
        _contract(), {"g++": str(compiler)}, env=environment
    )

    expected_source = "#include <elfutils/libdwfl.h>\n"
    assert arguments.read_text(encoding="utf-8").splitlines() == [
        "-std=c++20",
        "-x",
        "c++",
        "-E",
        "-",
    ]
    assert source.read_text(encoding="utf-8") == expected_source
    assert receipt == {
        "command": [str(compiler), "-std=c++20", "-x", "c++", "-E", "-"],
        "headers": ["elfutils/libdwfl.h"],
        "source_sha256": hashlib.sha256(expected_source.encode()).hexdigest(),
        "debian_packages": ["libdw-dev", "libelf-dev"],
    }


def test_header_probe_failure_names_both_required_debian_packages(
    tmp_path: Path,
) -> None:
    compiler = tmp_path / "g++"
    _write_compiler(
        compiler,
        "cat >/dev/null\n"
        'printf "%s\\n" "fatal error: elfutils/libdwfl.h: No such file" >&2\n'
        "exit 42\n",
    )

    with pytest.raises(
        builder.BuildContractError,
        match=r"elfutils/libdwfl\.h\. Install libdw-dev, libelf-dev",
    ):
        builder._validate_required_cxx_headers(_contract(), {"g++": str(compiler)}, env=os.environ)


def test_continuation_build_environment_keeps_only_explicit_sdk_paths() -> None:
    contract = _contract()
    environment = builder._create_build_environment(
        contract,
        {
            "VLLM_CUTLASS_SRC_DIR": "/sources/cutlass",
            "TRITON_KERNELS_SRC_DIR": "/sources/triton-kernels/python/triton_kernels",
        },
        cuda_arch_list="10.0",
        max_jobs=8,
        nvcc_threads=2,
        base_environment={
            "PATH": "/venv/bin:/usr/bin",
            "CPATH": "/sdk/usr/include",
            "CPLUS_INCLUDE_PATH": "/sdk/usr/include",
            "LIBRARY_PATH": "/sdk/usr/lib/x86_64-linux-gnu",
        },
    )

    assert environment["CPATH"] == "/sdk/usr/include"
    assert environment["CPLUS_INCLUDE_PATH"] == "/sdk/usr/include"
    assert environment["LIBRARY_PATH"] == "/sdk/usr/lib/x86_64-linux-gnu"
    assert environment["MAX_JOBS"] == "8"
    assert environment["CARGO_BUILD_JOBS"] == "8"
    assert environment["PYTORCH_NVML_BASED_CUDA_CHECK"] == "1"
    assert environment["NVCC_THREADS"] == "2"
    assert environment["VLLM_USE_PRECOMPILED"] == "0"
    assert environment["VLLM_USE_PRECOMPILED_RUST"] == "0"
    assert environment["VLLM_VERSION_OVERRIDE"] == contract["package"]["version"]
    assert environment["VLLM_CUTLASS_SRC_DIR"] == "/sources/cutlass"
    assert environment["TRITON_KERNELS_SRC_DIR"].endswith("/python/triton_kernels")


def test_build_command_uses_the_pinned_build_frontend_without_isolation(
    tmp_path: Path,
) -> None:
    assert builder._create_build_command(tmp_path / "wheel") == [
        builder.sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(tmp_path / "wheel"),
        ".",
    ]
