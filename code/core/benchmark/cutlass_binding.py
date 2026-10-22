"""CUTLASS PyTorch binding for Chapter 14."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch

from core.utils.extension_loader_template import load_cuda_extension


def _detect_cutlass_include() -> Path:
    """Resolve a usable CUTLASS include directory for extension builds.

    Preferred order:
    1) nvidia-cutlass-dsl Python package include path
    2) repository local third_party/cutlass/include
    3) CUTLASS_PATH/include (if exported)
    """
    candidates: list[Path] = []

    # Option 1: packaged CUTLASS headers from nvidia-cutlass-dsl.
    try:
        import cutlass_library
    except ImportError:
        cutlass_library = None  # type: ignore[assignment]

    if cutlass_library is not None:
        candidates.append(Path(cutlass_library.__file__).resolve().parent / "source" / "include")

    repo_root = Path(__file__).resolve().parents[2]
    # Option 2: vendored CUTLASS headers in this repository.
    candidates.append(repo_root / "third_party" / "cutlass" / "include")

    # Option 3: explicit CUTLASS_PATH from the environment.
    import os

    cutlass_path = os.environ.get("CUTLASS_PATH")
    if cutlass_path:
        candidates.append(Path(cutlass_path).expanduser().resolve() / "include")

    for include_dir in candidates:
        if include_dir.exists():
            return include_dir

    searched = ", ".join(str(path) for path in candidates) if candidates else "<none>"
    raise RuntimeError(
        "CUTLASS include directory not found. Install nvidia-cutlass-dsl>=4.2 "
        "or ensure third_party/cutlass/include exists. "
        f"Searched: {searched}"
    )


@lru_cache()
def _load_cutlass_module(verbose: bool = False):
    """Compile and cache the CUTLASS GEMM extension."""
    repo_root = Path(__file__).resolve().parents[2]
    cuda_source = repo_root / "core" / "benchmark" / "cuda" / "cutlass_gemm_extension.cu"
    include_dir = _detect_cutlass_include()
    # make_cute_packed_stride lives in cutlass/tools/util/include (not the main
    # include dir), so add it alongside, preferring the sibling of the detected
    # include then the repo-vendored copy.
    include_dirs = [include_dir]
    for util in (
        include_dir.parent / "tools" / "util" / "include",
        repo_root / "third_party" / "cutlass" / "tools" / "util" / "include",
    ):
        if util.exists() and util not in include_dirs:
            include_dirs.append(util)
            break
    # The CUTLASS 3.x Sm100 collective (tcgen05 / TMA) must be built for the arch
    # 'a' variant (sm_103a on GB300, sm_100a on B200); torch auto-detect emits plain
    # sm_103, which cannot compile the tcgen05/TMA path. An explicit -gencode makes
    # torch respect it (and skip its default arch). cute also needs the relaxed
    # constexpr / extended lambda flags.
    arch = "100a"
    try:
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}a"
    except Exception:
        pass
    extra_flags = [
        "-O3", "--use_fast_math", "-std=c++20",
        "--expt-relaxed-constexpr", "--expt-extended-lambda",
        f"-gencode=arch=compute_{arch},code=sm_{arch}",
    ]
    return load_cuda_extension(
        extension_name="cutlass_gemm_ext",
        cuda_source_file=str(cuda_source),
        include_dirs=include_dirs,
        extra_cuda_cflags=extra_flags,
        extra_ldflags=["-lcublas"],
        verbose=verbose,
    )


def cublas_gemm_fp16(a: torch.Tensor, b: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    """Invoke the explicit cuBLAS FP16 GEMM helper from Python."""
    module = _load_cutlass_module(verbose=verbose)
    return module.cublas_gemm_fp16(a, b)


def cutlass_gemm_fp16(a: torch.Tensor, b: torch.Tensor, verbose: bool = False) -> torch.Tensor:
    """Invoke the CUTLASS GEMM kernel from Python."""
    module = _load_cutlass_module(verbose=verbose)
    return module.cutlass_gemm_fp16(a, b)
