"""Record the actual native binary, source, compiler, and build configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--arch", required=True)
    parser.add_argument("--with-nccl", choices=("0", "1"), required=True)
    parser.add_argument("--with-nvshmem", choices=("0", "1"), required=True)
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--mpicxx", required=True)
    parser.add_argument("--build-option", action="append", default=[], help="Effective Makefile option as name=value")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    paths = ("jacobi_contract.hpp", "jacobi_native.cu", "Makefile")
    build_options = {}
    for option in args.build_option:
        name, separator, value = option.partition("=")
        if not name or not separator:
            raise ValueError("--build-option requires name=value")
        build_options[name] = value
    compile_command = [args.nvcc]
    for name in ("nvcc_flags", "mpi_compile_flags", "nccl_flags", "nvshmem_flags"):
        compile_command.extend(shlex.split(build_options.get(name, "")))
    compile_command.extend([f"-ccbin={args.mpicxx}", "jacobi_native.cu", "-o", str(args.binary)])
    for name in ("nccl_libs", "nvshmem_libs", "nvtx_libs"):
        compile_command.extend(shlex.split(build_options.get(name, "")))
    versions = {}
    for name, executable in (("nvcc", args.nvcc), ("mpicxx", args.mpicxx)):
        result = subprocess.run([executable, "--version"], capture_output=True, text=True, check=True)
        versions[name] = result.stdout.strip()
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=directory,
                            capture_output=True, text=True, check=True).stdout.strip()
    receipt = {
        "schema": "multi_gpu_communication.build.v1",
        "source_commit": commit,
        "source_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in paths},
        "binary_sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
        "architecture": args.arch,
        "with_nccl": args.with_nccl == "1",
        "with_nvshmem": args.with_nvshmem == "1",
        "compiler_versions": versions,
        "build_options": build_options,
        "compile_command": compile_command,
    }
    args.binary.with_name(args.binary.name + ".build.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
