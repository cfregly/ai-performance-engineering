"""Record the actual native build and its shared numerical dependency."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


BUILD_SCHEMA = "nvshmem_device.build.v1"
SOURCE_FILES = (
    "nvshmem_device_native.cu",
    "Makefile",
    "build_receipt.py",
    "../multi_gpu_communication/jacobi_contract.hpp",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--mpicxx", required=True)
    parser.add_argument("--build-option", action="append", default=[], help="Effective Makefile option as name=value")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    options = {}
    for option in args.build_option:
        name, separator, value = option.partition("=")
        if not separator or not name or name in options:
            raise ValueError("Each build option must have a unique name=value")
        options[name] = value
    required = {"nvcc_flags", "mpi_compile_flags", "nvshmem_flags", "nvshmem_libs", "nvtx_libs"}
    if not required <= options.keys():
        raise ValueError(f"Missing effective build options: {sorted(required - options.keys())}")
    command = [args.nvcc]
    for name in ("nvcc_flags", "mpi_compile_flags", "nvshmem_flags"):
        command.extend(shlex.split(options[name]))
    command.extend([f"-ccbin={args.mpicxx}", "nvshmem_device_native.cu", "-o", str(args.binary)])
    for name in ("nvshmem_libs", "nvtx_libs"):
        command.extend(shlex.split(options[name]))
    versions = {}
    for name, executable in (("nvcc", args.nvcc), ("mpicxx", args.mpicxx)):
        result = subprocess.run([executable, "--version"], capture_output=True, text=True, check=True)
        versions[name] = result.stdout.strip() or result.stderr.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=directory, capture_output=True, text=True, check=True,
    ).stdout.strip()
    receipt = {
        "schema": BUILD_SCHEMA,
        "source_commit": commit,
        "source_sha256": {name: file_sha256(directory / name) for name in SOURCE_FILES},
        "binary_sha256": file_sha256(args.binary),
        "architecture": args.arch,
        "compiler_versions": versions,
        "build_options": options,
        "compile_command": command,
    }
    target = args.binary.with_name(args.binary.name + ".build.json")
    target.write_text(json.dumps(receipt, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
