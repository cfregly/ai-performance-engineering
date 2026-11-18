"""Record the exact native NCCL symmetric-memory build and its input hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

BUILD_SCHEMA = "nccl_symmetric_memory.build.v1"
MINIMUM_NCCL_VERSION_CODE = 22703
SOURCE_FILES = (
    "nccl_symmetric_memory_native.cu",
    "Makefile",
    "nccl_symmetric_memory_build_receipt.py",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def query_nccl_header_version(mpicxx: str, nccl_flags: str) -> tuple[int, str, list[str]]:
    command = [mpicxx, *shlex.split(nccl_flags), "-dM", "-E", "-x", "c++", "-"]
    completed = subprocess.run(
        command,
        input="#include <nccl.h>\n",
        capture_output=True,
        text=True,
        check=True,
    )
    version_code, version = parse_nccl_version_macros(completed.stdout)
    return version_code, version, command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--mpicxx", required=True)
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
        "nccl_flags",
        "nccl_libs",
        "nvtx_libs",
    }
    missing_options = sorted(required_options - options.keys())
    if missing_options:
        raise ValueError(f"Missing effective build options: {missing_options}")

    header_code, header_version, preprocessor_command = query_nccl_header_version(
        args.mpicxx, options["nccl_flags"]
    )
    if header_code < MINIMUM_NCCL_VERSION_CODE:
        raise RuntimeError(
            f"NCCL headers {header_version} are too old. Version 2.27.3 or newer is required"
        )

    compile_command = [args.nvcc]
    for name in ("nvcc_flags", "mpi_compile_flags", "nccl_flags"):
        compile_command.extend(shlex.split(options[name]))
    compile_command.extend(
        [f"-ccbin={args.mpicxx}", "nccl_symmetric_memory_native.cu", "-o", str(args.binary)]
    )
    for name in ("nccl_libs", "nvtx_libs"):
        compile_command.extend(shlex.split(options[name]))

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
    receipt = {
        "schema": BUILD_SCHEMA,
        "source_commit": source_commit,
        "source_sha256": {
            name: file_sha256(directory / name) for name in SOURCE_FILES
        },
        "binary_sha256": file_sha256(args.binary),
        "architecture": args.arch,
        "minimum_nccl_version": "2.27.3",
        "minimum_nccl_version_code": MINIMUM_NCCL_VERSION_CODE,
        "nccl_header_version": header_version,
        "nccl_header_version_code": header_code,
        "compiler_versions": compiler_versions,
        "build_options": options,
        "nccl_header_preprocessor_command": preprocessor_command,
        "compile_command": compile_command,
    }
    target = args.binary.with_name(args.binary.name + ".build.json")
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
