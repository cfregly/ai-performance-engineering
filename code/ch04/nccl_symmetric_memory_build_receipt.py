"""Record the exact native NCCL symmetric-memory build and its input hashes."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import tempfile
from collections.abc import Iterator
from pathlib import Path

BUILD_SCHEMA = "nccl_symmetric_memory.build.v1"
MINIMUM_NCCL_VERSION_CODE = 22703
NCCL_VERSION_PROBE_MARKER = "AISP_NCCL_HEADER_VERSION_PROBE"
SOURCE_FILES = (
    "nccl_symmetric_memory_native.cu",
    "Makefile",
    "nccl_symmetric_memory_build_receipt.py",
    "../core/common/cuda_arch.mk",
    "../core/common/nvtx_utils.cuh",
    "../core/common/nvtx_label_utils.cuh",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomically(path: Path, content: dict) -> None:
    """Replace a generated receipt only after its complete contents are durable."""
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(content, stream, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


@contextlib.contextmanager
def serialized_build(directory: Path) -> Iterator[None]:
    """Keep the native binary and its receipt under one cross-process lock."""
    identity = hashlib.sha256(str(directory.resolve()).encode("utf-8")).hexdigest()[:16]
    lock_path = Path(tempfile.gettempdir()) / f"aisp-nccl-symmetric-memory-{identity}.lock"
    with lock_path.open("a", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class RuntimeBindingError(RuntimeError):
    """The study cannot establish the identity of the loaded NCCL library."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"NCCL runtime binding: {reason}")


def nccl_library_directories(linker_flags: str, build_directory: Path) -> list[str]:
    """Resolve -L options and their installed GNU multiarch library directories."""
    try:
        tokens = iter(shlex.split(linker_flags))
    except ValueError as error:
        raise RuntimeBindingError(f"invalid NCCL linker options ({error})") from error
    directories = []
    for token in tokens:
        if token == "-L":
            directory = next(tokens, "")
            if not directory or directory.startswith("-"):
                raise RuntimeBindingError("missing directory after -L")
        elif token.startswith("-L"):
            directory = token[2:]
        else:
            continue
        explicit = (build_directory / directory).resolve()
        candidates = [explicit]
        try:
            candidates.extend(
                sorted(
                    (
                        candidate.resolve()
                        for candidate in explicit.glob("*-linux-gnu*")
                        if candidate.is_dir()
                    ),
                    key=str,
                )
            )
        except OSError as error:
            raise RuntimeBindingError(
                f"could not inspect NCCL library directory {explicit} ({error})"
            ) from error
        for candidate in candidates:
            path = str(candidate)
            if path not in directories:
                directories.append(path)
    if not directories:
        raise RuntimeBindingError("the build receipt has no NCCL -L directory")
    return directories


def expected_nccl_library(directories: list[str]) -> Path:
    libraries = set()
    for directory in directories:
        soname = Path(directory) / "libnccl.so.2"
        if soname.is_symlink() or soname.exists():
            if not soname.is_file():
                raise RuntimeBindingError(f"{soname} does not resolve to a regular file")
            libraries.add(soname.resolve(strict=True))
    if len(libraries) != 1:
        raise RuntimeBindingError(
            f"expected exactly one libnccl.so.2 in {directories}, found {sorted(map(str, libraries))}"
        )
    return libraries.pop()


def parse_nccl_version_macros(preprocessor_output: str) -> tuple[int, str]:
    """Return NCCL's numeric and dotted header versions from compiler macros."""
    macros: dict[str, str] = {}
    for line in preprocessor_output.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) == 3 and fields[0] == "#define":
            macros[fields[1]] = fields[2]
    required = ("NCCL_MAJOR", "NCCL_MINOR", "NCCL_PATCH")
    missing = [name for name in required if name not in macros]
    if missing:
        raise RuntimeError(f"NCCL header preprocessing did not expose {missing}")
    try:
        major, minor, patch = (int(macros[name]) for name in required)
    except ValueError as error:
        raise RuntimeError("NCCL header version macros must be decimal integers") from error
    version_code = major * 10000 + minor * 100 + patch
    return version_code, f"{major}.{minor}.{patch}"


def compile_command_prefix(nvcc: str, mpicxx: str, options: dict[str, str]) -> list[str]:
    """Return the ordered compiler and flag prefix shared by build and probe."""
    command = [nvcc]
    for name in ("nvcc_flags", "mpi_compile_flags", "nccl_flags"):
        try:
            command.extend(shlex.split(options[name]))
        except ValueError as error:
            raise ValueError(f"Invalid {name} build option ({error})") from error
    command.append(f"-ccbin={mpicxx}")
    return command


def parse_nccl_version_probe(preprocessor_output: str) -> tuple[int, str]:
    """Parse the one marker line emitted after NCCL macros are expanded."""
    matches = []
    for line in preprocessor_output.splitlines():
        fields = line.split()
        if fields and fields[0] == NCCL_VERSION_PROBE_MARKER:
            matches.append(fields)
    if len(matches) != 1 or len(matches[0]) != 4:
        raise RuntimeError("NCCL header preprocessing did not emit one version probe")
    try:
        major, minor, patch = (int(value) for value in matches[0][1:])
    except ValueError as error:
        raise RuntimeError("NCCL header version macros must be decimal integers") from error
    version_code = major * 10000 + minor * 100 + patch
    return version_code, f"{major}.{minor}.{patch}"


def _preprocessor_error(completed: subprocess.CompletedProcess[str], reason: str) -> RuntimeError:
    stderr = completed.stderr.strip() or "<empty stderr>"
    return RuntimeError(
        f"{reason}. Compiler exited with {completed.returncode}.\n"
        f"NCCL header preprocessor stderr:\n{stderr}"
    )


def query_nccl_header_version(
    nvcc: str,
    mpicxx: str,
    options: dict[str, str],
    *,
    directory: Path,
) -> tuple[int, str, list[str], str]:
    command = [*compile_command_prefix(nvcc, mpicxx, options), "-E", "-x", "cu", "-"]
    probe_source = (
        f"#include <nccl.h>\n{NCCL_VERSION_PROBE_MARKER} NCCL_MAJOR NCCL_MINOR NCCL_PATCH\n"
    )
    completed = subprocess.run(
        command,
        cwd=directory,
        input=probe_source,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise _preprocessor_error(completed, "NCCL header preprocessing failed")
    try:
        version_code, version = parse_nccl_version_probe(completed.stdout)
    except RuntimeError as error:
        raise _preprocessor_error(completed, str(error)) from error
    return version_code, version, command, completed.stderr


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--mpicxx", required=True)
    parser.add_argument("--nccl-home", type=Path, required=True)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the native binary while holding the build-output lock",
    )
    parser.add_argument(
        "--build-option",
        action="append",
        default=[],
        help="Effective Makefile option as a unique name=value",
    )
    args = parser.parse_args()

    directory = Path(__file__).resolve().parent
    options: dict[str, str] = {}
    for option in args.build_option:
        name, separator, value = option.partition("=")
        if not separator or not name or name in options:
            raise ValueError("Each build option must have a unique name=value")
        options[name] = value
    required_options = {
        "nvcc_flags",
        "mpi_compile_flags",
        "nccl_home",
        "nccl_flags",
        "nccl_libs",
        "nvtx_libs",
    }
    missing_options = sorted(required_options - options.keys())
    if missing_options:
        raise ValueError(f"Missing effective build options: {missing_options}")

    nccl_home_input = args.nccl_home.expanduser()
    nccl_home = (
        nccl_home_input if nccl_home_input.is_absolute() else directory / nccl_home_input
    ).resolve(strict=True)
    if not nccl_home.is_dir():
        raise ValueError(f"NCCL_HOME is not a directory: {nccl_home}")
    option_home = Path(options["nccl_home"]).expanduser()
    option_home = (option_home if option_home.is_absolute() else directory / option_home).resolve(
        strict=True
    )
    if option_home != nccl_home:
        raise ValueError("--nccl-home and the recorded nccl_home build option differ")

    compile_command = compile_command_prefix(args.nvcc, args.mpicxx, options)
    compile_command.extend(["nccl_symmetric_memory_native.cu", "-o", str(args.binary)])
    for name in ("nccl_libs", "nvtx_libs"):
        compile_command.extend(shlex.split(options[name]))

    build_context = serialized_build(directory) if args.compile else contextlib.nullcontext()
    with build_context:
        if args.compile:
            subprocess.run(compile_command, cwd=directory, check=True)

        header_code, header_version, preprocessor_command, preprocessor_stderr = (
            query_nccl_header_version(
                args.nvcc,
                args.mpicxx,
                options,
                directory=directory,
            )
        )
        if header_code < MINIMUM_NCCL_VERSION_CODE:
            raise RuntimeError(
                f"NCCL headers {header_version} are too old. Version 2.27.3 or newer is required"
            )

        compiler_versions: dict[str, str] = {}
        for name, executable in (("nvcc", args.nvcc), ("mpicxx", args.mpicxx)):
            completed = subprocess.run(
                [executable, "--version"], capture_output=True, text=True, check=True
            )
            version = completed.stdout.strip() or completed.stderr.strip()
            if not version:
                raise RuntimeError(f"{name} did not report a compiler version")
            compiler_versions[name] = version

        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=directory,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        library = expected_nccl_library(nccl_library_directories(options["nccl_libs"], directory))
        receipt = {
            "schema": BUILD_SCHEMA,
            "source_commit": source_commit,
            "source_sha256": {name: file_sha256(directory / name) for name in SOURCE_FILES},
            "binary_sha256": file_sha256(args.binary),
            "architecture": args.arch,
            "minimum_nccl_version": "2.27.3",
            "minimum_nccl_version_code": MINIMUM_NCCL_VERSION_CODE,
            "nccl_home": str(nccl_home),
            "nccl_header_version": header_version,
            "nccl_header_version_code": header_code,
            "nccl_runtime_library_path": str(library),
            "nccl_runtime_library_sha256": file_sha256(library),
            "compiler_versions": compiler_versions,
            "build_options": options,
            "nccl_header_preprocessor_command": preprocessor_command,
            "nccl_header_preprocessor_stderr": preprocessor_stderr,
            "compile_command": compile_command,
        }
        target = args.binary.with_name(args.binary.name + ".build.json")
        write_json_atomically(target, receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
