"""Build fast.cu CUDA extensions during setup and cache them by source hash."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

from labs.fast_cu.source import verify_upstream


def parse_nvcc_release(output: str) -> tuple[int, int]:
    match = re.search(r"\brelease\s+(\d+)\.(\d+)\b", output)
    if match is None:
        raise RuntimeError("Cannot determine CUDA toolkit release from nvcc --version")
    return int(match[1]), int(match[2])


def extension_name(
    name: str,
    sources: list[Path],
    cuda_flags: list[str],
    link_flags: list[str],
    manifest: dict,
    *,
    toolchain_identity: str = "",
) -> str:
    """Include transitive upstream headers in the JIT cache identity."""
    if not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]*", name):
        raise ValueError("Extension name must be a C/Python identifier")
    if not sources:
        raise ValueError("At least one CUDA extension source is required")
    payload = {
        "sources": [(path.name, hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources],
        "upstream_revision": manifest["revision"],
        "upstream_files": manifest["files"],
        "cuda_flags": cuda_flags,
        "link_flags": link_flags,
        "toolchain_identity": toolchain_identity,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return f"{name}_{digest[:20]}"


def load_cuda_extension(
    name: str,
    sources: list[Path],
    *,
    extra_cuda_cflags: list[str],
    extra_ldflags: list[str],
    minimum_cuda: tuple[int, int],
    dependencies: list[Path] | None = None,
):
    """Build before timing without installing dependencies or changing global flags."""
    manifest = verify_upstream()
    from torch.utils.cpp_extension import CUDA_HOME, load

    if CUDA_HOME is None:
        raise RuntimeError("SKIPPED: fast.cu requires a local CUDA toolkit with nvcc")
    nvcc = Path(CUDA_HOME) / "bin" / "nvcc"
    if not nvcc.is_file():
        raise RuntimeError(f"SKIPPED: fast.cu nvcc is missing: {nvcc}")
    compiler_version = subprocess.run(
        [str(nvcc), "--version"], check=True, capture_output=True, text=True
    ).stdout
    version = parse_nvcc_release(compiler_version)
    if version < minimum_cuda:
        required = ".".join(map(str, minimum_cuda))
        raise RuntimeError(
            f"SKIPPED: fast.cu requires CUDA toolkit {required} or newer. Found {version}"
        )
    module_name = extension_name(
        name,
        sources + (dependencies or []),
        extra_cuda_cflags,
        extra_ldflags,
        manifest,
        toolchain_identity=f"{nvcc.resolve()}\n{compiler_version}",
    )
    return load(
        name=module_name,
        sources=[str(path) for path in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        with_cuda=True,
        verbose=False,
    )
