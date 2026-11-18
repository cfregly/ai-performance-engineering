"""NCCL build receipt, loader preflight, and per-rank provenance checks."""

import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from ch04 import nccl_symmetric_memory_build_receipt as receipt
from ch04 import run_nccl_symmetric_memory as runner


@pytest.fixture
def private_library(tmp_path: Path) -> Path:
    directory = tmp_path / "private NCCL" / "lib"
    directory.mkdir(parents=True)
    library = directory / "libnccl.so.2.27.3"
    library.write_bytes(b"private library bytes")
    (directory / "libnccl.so.2").symlink_to(library.name)
    return library


def build_for(library: Path) -> dict:
    return {
        "build_options": {"nccl_libs": shlex.join(["-L", str(library.parent), "-lnccl"])},
        "nccl_header_version_code": 22703,
        "nccl_runtime_library_path": str(library),
        "nccl_runtime_library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }


def test_build_receipt_records_canonical_library_and_digest(private_library, tmp_path):
    binary = tmp_path / "native"
    binary.write_bytes(b"fixture")
    compiler = tmp_path / "compiler"
    compiler.write_text(
        "#!/bin/sh\n"
        'if [ "$1" = --version ]; then echo fixture; else\n'
        "printf '%s\\n' '#define NCCL_MAJOR 2' '#define NCCL_MINOR 27' '#define NCCL_PATCH 3'\n"
        "fi\n"
    )
    compiler.chmod(0o755)
    build = build_for(private_library)
    command = [
        sys.executable,
        receipt.__file__,
        "--binary",
        str(binary),
        "--arch",
        "sm100",
        "--nvcc",
        str(compiler),
        "--mpicxx",
        str(compiler),
    ]
    for option in ("nvcc_flags", "mpi_compile_flags", "nccl_flags", "nvtx_libs", "nccl_libs"):
        command += ["--build-option", f"{option}={build['build_options'].get(option, '')}"]
    subprocess.run(command, check=True)
    recorded = runner.validate_build_receipt(binary)
    assert recorded["nccl_runtime_library_path"] == str(private_library)
    assert recorded["nccl_runtime_library_sha256"] == build["nccl_runtime_library_sha256"]


def test_atomic_receipt_replacement_preserves_the_previous_file(tmp_path, monkeypatch):
    target = tmp_path / "native.build.json"
    target.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(_source, _target):
        raise OSError("injected replacement failure")

    monkeypatch.setattr(receipt.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replacement failure"):
        receipt.write_json_atomically(target, {"new": True})

    assert target.read_text(encoding="utf-8") == '{"old": true}\n'
    assert list(tmp_path.glob("*.tmp")) == []


def test_linker_dirs_resolve_relative_flags_and_aliases(private_library, tmp_path):
    alias = tmp_path / "lib64"
    alias.mkdir()
    (alias / "libnccl.so.2").symlink_to(private_library)
    linker_flags = shlex.join(["-L", "private NCCL/lib", "-Llib64", "-Llib64", "-lnccl"])
    directories = receipt.nccl_library_directories(linker_flags, tmp_path)
    assert directories == [str(private_library.parent), str(alias)]
    assert receipt.expected_nccl_library(directories) == private_library


def test_linker_dirs_discover_ubuntu_multiarch_library(private_library, tmp_path):
    library_root = tmp_path / "usr" / "lib"
    multiarch = library_root / "x86_64-linux-gnu"
    multiarch.mkdir(parents=True)
    (multiarch / "libnccl.so.2").symlink_to(private_library)

    directories = receipt.nccl_library_directories("-Lusr/lib -lnccl", tmp_path)

    assert directories == [str(library_root), str(multiarch)]
    assert receipt.expected_nccl_library(directories) == private_library


def test_missing_ambiguous_and_dangling_libraries_fail_closed(private_library, tmp_path):
    other = tmp_path / "lib64"
    other.mkdir()
    with pytest.raises(receipt.RuntimeBindingError, match="exactly one libnccl.so.2"):
        receipt.expected_nccl_library([str(other)])
    (other / "libnccl.so.2").write_bytes(private_library.read_bytes())
    with pytest.raises(receipt.RuntimeBindingError, match="exactly one libnccl.so.2"):
        receipt.expected_nccl_library([str(private_library.parent), str(other)])
    broken = tmp_path / "broken"
    broken.mkdir()
    (broken / "libnccl.so.2").symlink_to("absent")
    with pytest.raises(receipt.RuntimeBindingError, match="regular file"):
        receipt.expected_nccl_library([str(private_library.parent), str(broken)])


@pytest.mark.parametrize("options", ["-lnccl", "-L", "-L -lnccl", "-L'unclosed"])
def test_unusable_search_flags_fail_closed(options, tmp_path):
    with pytest.raises(receipt.RuntimeBindingError):
        receipt.nccl_library_directories(options, tmp_path)


def test_preload_rejected_without_parent_mutation(private_library, monkeypatch):
    monkeypatch.setenv("LD_PRELOAD", "/preloaded/library.so")
    with pytest.raises(receipt.RuntimeBindingError, match="LD_PRELOAD"):
        runner.child_runtime_environment(build_for(private_library))
    assert os.environ["LD_PRELOAD"] == "/preloaded/library.so"


def fake_ldd_environment(tmp_path, monkeypatch, library, output, status=0):
    monkeypatch.delenv("LD_PRELOAD", raising=False)
    monkeypatch.setenv("LD_LIBRARY_PATH", "/inherited")
    ldd = tmp_path / "ldd"
    ldd.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$1" "$LD_LIBRARY_PATH" > "$LDD_ARGS_FILE"\n'
        "printf '%s\\n' \"$LDD_OUTPUT\"\n"
        'exit "$LDD_STATUS"\n'
    )
    ldd.chmod(0o755)
    build = build_for(library)
    env, directories = runner.child_runtime_environment(build)
    env.update(
        PATH=str(tmp_path),
        LDD_ARGS_FILE=str(tmp_path / "ldd_args"),
        LDD_OUTPUT=output,
        LDD_STATUS=str(status),
    )
    return build, directories, env


def test_ldd_runs_under_child_environment(private_library, tmp_path, monkeypatch):
    binary = tmp_path / "native"
    soname = private_library.parent / "libnccl.so.2"
    output = f"\tlibnccl.so.2 => {soname} (0x1)\n\tlibc.so.6 => /lib/libc.so.6 (0x2)"
    args = fake_ldd_environment(tmp_path, monkeypatch, private_library, output)
    assert runner.verify_loader_binding(binary, *args) == private_library
    assert (tmp_path / "ldd_args").read_text().splitlines() == [
        str(binary),
        os.pathsep.join([str(private_library.parent), "/inherited"]),
    ]
    assert os.environ["LD_LIBRARY_PATH"] == "/inherited"


@pytest.mark.parametrize(
    ("output", "status", "message"),
    [
        ("libnccl.so.2 => /other/libnccl.so.2 (0x1)", 0, "expected"),
        ("libnccl.so.2 => not found", 0, "did not resolve"),
        ("libc.so.6 => /lib/libc.so.6 (0x2)", 0, "one libnccl.so.2 dependency"),
        ("libnccl.so.2 => {library} (0x1)\nlibnccl.so.2 => {library} (0x2)", 0, "one libnccl.so.2"),
        ("libnccl.so.2 => {library} (0x1)", 1, "ldd exited with 1"),
    ],
)
def test_ldd_mismatch_or_ambiguity_fails(
    private_library, tmp_path, monkeypatch, output, status, message
):
    args = fake_ldd_environment(
        tmp_path, monkeypatch, private_library, output.format(library=private_library), status
    )
    with pytest.raises(receipt.RuntimeBindingError, match=message):
        runner.verify_loader_binding(tmp_path / "native", *args)


def test_library_replaced_after_build_fails_before_ldd(private_library, tmp_path):
    build = build_for(private_library)
    private_library.write_bytes(b"replaced")
    with pytest.raises(receipt.RuntimeBindingError, match="SHA256"):
        runner.verify_loader_binding(tmp_path / "native", build, [str(private_library.parent)], {})


def test_rank_report_must_match_library_version_path_count_and_digest(private_library):
    build = build_for(private_library)
    observation = {
        "nccl_runtime_library_verified_ranks": 2,
        "nccl_runtime_library_path": str(private_library),
        "nccl_runtime_version_code": 22703,
    }
    assert (
        runner.verify_rank_binding(observation, build, private_library, 2)
        == build["nccl_runtime_library_sha256"]
    )
    for field, other in [
        ("nccl_runtime_library_verified_ranks", 1),
        ("nccl_runtime_library_verified_ranks", 2.0),
        ("nccl_runtime_library_path", "/another/libnccl.so.2"),
        ("nccl_runtime_version_code", 22803),
    ]:
        with pytest.raises(receipt.RuntimeBindingError):
            runner.verify_rank_binding({**observation, field: other}, build, private_library, 2)
    private_library.write_bytes(b"changed during execution")
    with pytest.raises(receipt.RuntimeBindingError, match="changed during execution"):
        runner.verify_rank_binding(observation, build, private_library, 2)


def test_cli_records_binding_failure_before_mpi(tmp_path, private_library):
    binary = tmp_path / "native"
    binary.write_bytes(b"fixture")
    build = build_for(private_library)
    build.update(
        schema=runner.BUILD_SCHEMA,
        source_sha256=runner.build_source_hashes(),
        binary_sha256=runner.file_sha256(binary),
        minimum_nccl_version_code=22703,
        compiler_versions={"nvcc": "fixture", "mpicxx": "fixture"},
        compile_command=["fixture"],
    )
    binary.with_name("native.build.json").write_text(json.dumps(build))
    result = tmp_path / "result.json"
    private_library.write_bytes(b"changed after build")
    environment = os.environ.copy()
    environment.pop("LD_PRELOAD", None)
    process = subprocess.run(
        [
            sys.executable,
            runner.__file__,
            "--binary",
            str(binary),
            "--ranks",
            "2",
            "--all-arms",
            "--json-out",
            str(result),
        ],
        env=environment,
        text=True,
        capture_output=True,
    )
    assert process.returncode == 1, process.stdout + process.stderr
    failure = json.loads(result.read_text())
    assert failure["status"] == "FAILED"
    assert failure["failed_arm"] == "cuda_malloc"
    assert failure["completed_arms"] == {}
    assert "NCCL runtime binding: SHA256" in failure["reason"]


@pytest.mark.parametrize(
    ("index", "rank_values", "expected"),
    [
        (0, (176, 245), 421.0),
        (1, (194, 104), 298.0),
        (1024, (205, 139), 344.0),
        (2049, (38, 172), 210.0),
        (1 << 32, (57, 62), 119.0),
        ((1 << 45) + 7, (202, 125), 327.0),
    ],
)
def test_native_full_width_reference_values(index, rank_values, expected):
    assert tuple(runner._input_value(rank, index) for rank in range(2)) == rank_values
    assert runner._expected_value(2, index) == expected
