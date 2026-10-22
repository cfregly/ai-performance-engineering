"""CUTLASS PyTorch binding for Chapter 14."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Optional

import torch

from core.utils.extension_loader_template import load_cuda_extension


_CUTLASS_REQUIRED_HEADERS = (
    Path("include/cutlass/cutlass.h"),
    Path("tools/util/include/cutlass/util/packed_stride.hpp"),
)


def _cutlass_include_dirs(root: Path) -> tuple[Path, Path] | None:
    resolved_root = root.expanduser().resolve()
    if not all(
        (resolved_root / relative).is_file() for relative in _CUTLASS_REQUIRED_HEADERS
    ):
        return None
    return (
        resolved_root / "include",
        resolved_root / "tools" / "util" / "include",
    )


def _detect_cutlass_include_dirs(*, repo_root: Optional[Path] = None) -> list[Path]:
    """Resolve main and utility headers from one complete CUTLASS source root."""
    explicit_value = os.environ.get("CUTLASS_PATH")
    if explicit_value is not None:
        if not explicit_value.strip():
            raise RuntimeError("CUTLASS_PATH is set but empty")
        explicit_root = Path(explicit_value)
        include_dirs = _cutlass_include_dirs(explicit_root)
        if include_dirs is None:
            resolved_root = explicit_root.expanduser().resolve()
            missing = [
                str(resolved_root / relative)
                for relative in _CUTLASS_REQUIRED_HEADERS
                if not (resolved_root / relative).is_file()
            ]
            raise RuntimeError(
                "CUTLASS_PATH must name one complete CUTLASS source root. "
                f"Missing required headers: {', '.join(missing)}"
            )
        return list(include_dirs)

    repository_root = repo_root or Path(__file__).resolve().parents[2]
    repository_cutlass = repository_root / "third_party" / "cutlass"
    include_dirs = _cutlass_include_dirs(repository_cutlass)
    if include_dirs is not None:
        return list(include_dirs)

    package_root: Optional[Path] = None
    try:
        import cutlass_library
    except ImportError:
        pass
    else:
        package_file = getattr(cutlass_library, "__file__", None)
        if package_file:
            package_root = Path(package_file).resolve().parent / "source"
            include_dirs = _cutlass_include_dirs(package_root)
            if include_dirs is not None:
                return list(include_dirs)

    required = ", ".join(str(path) for path in _CUTLASS_REQUIRED_HEADERS)
    searched = [str(repository_cutlass.resolve())]
    if package_root is not None:
        searched.append(str(package_root.resolve()))
    raise RuntimeError(
        "No complete CUTLASS source root was found. Each root must contain "
        f"{required}. Searched: {', '.join(searched)}"
    )


@lru_cache()
def _load_cutlass_module(verbose: bool = False):
    """Compile and cache the CUTLASS GEMM extension."""
    repo_root = Path(__file__).resolve().parents[2]
    cuda_source = repo_root / "core" / "benchmark" / "cuda" / "cutlass_gemm_extension.cu"
    include_dirs = _detect_cutlass_include_dirs(repo_root=repo_root)
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
