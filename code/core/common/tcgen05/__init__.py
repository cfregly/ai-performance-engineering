"""Shared tcgen05 kernel loaders and Python wrappers."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from core.benchmark.tcgen05_requirements import (
    SUPPORTED_TCGEN05_CAPABILITIES,
    ensure_tcgen05_capability_supported,
    ensure_tcgen05_supported,
)
from core.harness.hardware_capabilities import detect_capabilities

try:
    import fcntl
except ImportError:  # pragma: no cover - tcgen05 extension builds target Linux
    fcntl = None  # type: ignore[assignment]

try:  # Ensure TORCH_CUDA_ARCH_LIST stays clamped for GB-series hosts.
    import arch_config  # noqa: F401
except ImportError:  # pragma: no cover - optional bootstrap
    arch_config = None  # type: ignore[assignment]

_REPO_ROOT = Path(__file__).resolve().parents[3]
# For SM100 (Blackwell) tcgen05/TMEM kernels, prefer the standalone CUTLASS which has
# the required SM100-specific headers (mma_sm100_umma.hpp, tmem_allocator_sm100.hpp).
# TransformerEngine's bundled CUTLASS may not include these newer headers.
_SM100_CUTLASS_CANDIDATES = (
    _REPO_ROOT / "third_party" / "cutlass" / "include",
    _REPO_ROOT / "third_party" / "cutlass_latest" / "cutlass-main" / "include",
)
_FALLBACK_CUTLASS_CANDIDATES = (
    _REPO_ROOT / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include",
)
_SUPPORTED_TCGEN05_TARGETS = {
    (10, 0): "100a",
    (10, 3): "103a",
}
if frozenset(_SUPPORTED_TCGEN05_TARGETS) != SUPPORTED_TCGEN05_CAPABILITIES:
    raise RuntimeError("tcgen05 capability and CUDA target tables are inconsistent")

# These are the CUTLASS headers included directly by the tcgen05 sources loaded
# from this module. Checking them here gives a clear skip before PyTorch mutates
# an extension cache or launches Ninja.
_REQUIRED_CUTLASS_HEADERS = (
    Path("cutlass/arch/barrier.h"),
    Path("cutlass/cutlass.h"),
    Path("cutlass/detail/collective/moe_stride_utils.hpp"),
    Path("cutlass/epilogue/collective/collective_builder.hpp"),
    Path("cutlass/gemm/collective/collective_builder.hpp"),
    Path("cutlass/gemm/device/gemm_universal_adapter.h"),
    Path("cutlass/gemm/dispatch_policy.hpp"),
    Path("cutlass/gemm/kernel/gemm_universal.hpp"),
    Path("cutlass/half.h"),
    Path("cute/arch/cluster_sm90.hpp"),
    Path("cute/arch/copy_sm90_tma.hpp"),
    Path("cute/arch/mma_sm100_umma.hpp"),
    Path("cute/arch/tmem_allocator_sm100.hpp"),
    Path("cute/atom/copy_traits_sm90_tma.hpp"),
    Path("cute/atom/mma_traits_sm100.hpp"),
    Path("cute/numeric/integral_constant.hpp"),
    Path("cute/tensor.hpp"),
)

# CUTLASS 4.2.0 at the repository-pinned 57e3cfb revision predates
# cutlass/detail/collective/moe_stride_utils.hpp. The same four packed-stride
# overloads exist in its tools utility header under make_cute_packed_stride.
# This exact-content allowlist keeps the compatibility path scoped to that
# known implementation. Retire it when the pinned CUTLASS provides the native
# moe_stride_utils.hpp header.
_CUTLASS_42_VERSION = (4, 2, 0)
_CUTLASS_MOE_STRIDE_HEADER = Path("cutlass/detail/collective/moe_stride_utils.hpp")
_CUTLASS_42_PACKED_STRIDE_HEADER = Path("cutlass/util/packed_stride.hpp")
_CUTLASS_42_PACKED_STRIDE_SHA256 = (
    "ca7f6b722a87848a53730cf8049991dcd781a5d79b37d7cda683ab6367710650"
)
_CUTLASS_42_MOE_STRIDE_COMPAT_SOURCE = """\
// Generated compatibility header for repository-pinned CUTLASS 4.2.0.
#pragma once
#include <cutlass/util/packed_stride.hpp>

namespace cutlass {
template <class Stride, class Shape>
CUTLASS_HOST_DEVICE
auto make_internal_packed_stride(Stride stride, Shape const& shape)
    -> decltype(make_cute_packed_stride(stride, shape)) {
  return make_cute_packed_stride(stride, shape);
}
}  // namespace cutlass
"""


def _detect_compute_capability() -> tuple[int, int] | None:
    """Return the visible CUDA compute capability without requiring a GPU at import time."""
    if not torch.cuda.is_available():
        return None
    cap = detect_capabilities()
    if cap is not None:
        parts = cap.compute_capability.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return major, minor
    return torch.cuda.get_device_capability()


def _current_compute_capability() -> tuple[int, int]:
    """Return a supported tcgen05 capability or fail closed when a loader is invoked."""
    capability = _detect_compute_capability()
    if capability is None:
        raise RuntimeError(
            "SKIPPED: tcgen05 extension loading requires a visible CUDA device "
            "with a detectable compute capability."
        )
    ensure_tcgen05_capability_supported(
        capability,
        module_name="tcgen05 extension loading",
    )
    return capability


def _cutlass_version(include_dir: Path) -> tuple[int, int, int] | None:
    """Read the numeric CUTLASS version macros from an include directory."""
    version_path = include_dir / "cutlass" / "version.h"
    try:
        lines = version_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None

    values: dict[str, int] = {}
    wanted = {"CUTLASS_MAJOR", "CUTLASS_MINOR", "CUTLASS_PATCH"}
    for line in lines:
        parts = line.split()
        if len(parts) < 3 or parts[0] != "#define" or parts[1] not in wanted:
            continue
        try:
            values[parts[1]] = int(parts[2])
        except ValueError:
            return None
    if set(values) != wanted:
        return None
    return values["CUTLASS_MAJOR"], values["CUTLASS_MINOR"], values["CUTLASS_PATCH"]


def _write_cutlass_42_compat_header(destination: Path) -> None:
    """Atomically materialize the narrow CUTLASS 4.2 stride-name adapter."""
    if destination.is_file():
        try:
            if destination.read_text(encoding="utf-8") == _CUTLASS_42_MOE_STRIDE_COMPAT_SOURCE:
                return
        except (OSError, UnicodeError):
            pass

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            delete=False,
        ) as temporary_file:
            temporary_file.write(_CUTLASS_42_MOE_STRIDE_COMPAT_SOURCE)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            with suppress(FileNotFoundError):
                temporary_path.unlink()


def _cutlass_42_compat_includes(candidate: Path) -> tuple[tuple[Path, ...] | None, str]:
    """Return verified include roots for the repository-pinned CUTLASS 4.2 shim."""
    version = _cutlass_version(candidate)
    if version != _CUTLASS_42_VERSION:
        return None, f"CUTLASS version is {version!r}, expected {_CUTLASS_42_VERSION!r}"

    utility_include = candidate.parent / "tools" / "util" / "include"
    utility_header = utility_include / _CUTLASS_42_PACKED_STRIDE_HEADER
    if not utility_header.is_file():
        return None, f"missing {_CUTLASS_42_PACKED_STRIDE_HEADER}"

    try:
        utility_digest = hashlib.sha256(utility_header.read_bytes()).hexdigest()
    except OSError as error:
        return None, f"cannot read {_CUTLASS_42_PACKED_STRIDE_HEADER}: {error}"
    if utility_digest != _CUTLASS_42_PACKED_STRIDE_SHA256:
        return None, (
            f"unverified {_CUTLASS_42_PACKED_STRIDE_HEADER} digest {utility_digest}"
        )

    overlay_digest = hashlib.sha256(
        (_CUTLASS_42_MOE_STRIDE_COMPAT_SOURCE + utility_digest).encode("utf-8")
    ).hexdigest()[:16]
    overlay_include = (
        _get_extension_build_dir("tcgen05_cutlass_42_compat")
        / overlay_digest
        / "include"
    )
    _write_cutlass_42_compat_header(overlay_include / _CUTLASS_MOE_STRIDE_HEADER)
    return (overlay_include, candidate, utility_include), ""


def _cutlass_includes_for_capability(
    capability: tuple[int, int],
) -> tuple[Path, ...]:
    ensure_tcgen05_capability_supported(
        capability,
        module_name="tcgen05 extension loading",
    )
    candidates = _SM100_CUTLASS_CANDIDATES + _FALLBACK_CUTLASS_CANDIDATES
    checked: list[str] = []
    for candidate in candidates:
        if not candidate.is_dir():
            checked.append(f"{candidate} (not a directory)")
            continue
        missing = [
            header for header in _REQUIRED_CUTLASS_HEADERS if not (candidate / header).is_file()
        ]
        if not missing:
            return (candidate,)
        if missing == [_CUTLASS_MOE_STRIDE_HEADER]:
            compat_includes, incompatibility = _cutlass_42_compat_includes(candidate)
            if compat_includes is not None:
                return compat_includes
            checked.append(
                f"{candidate} (missing {_CUTLASS_MOE_STRIDE_HEADER}; "
                f"CUTLASS 4.2 compatibility unavailable: {incompatibility})"
            )
            continue
        missing_names = ", ".join(str(header) for header in missing)
        checked.append(f"{candidate} (missing {missing_names})")

    details = "; ".join(checked)
    raise RuntimeError(
        "SKIPPED: tcgen05 extension loading requires a valid CUTLASS include "
        f"directory with its required SM100 headers. Checked: {details}."
    )


_CLANG_HOST = _REPO_ROOT / "third_party" / "llvm" / "bin" / "clang++"

# Build fingerprint version. Bump this when changing build logic.
_BUILD_FINGERPRINT_VERSION = "v4"


def _tcgen05_cuda_flags() -> list[str]:
    capability = _current_compute_capability()
    cutlass_includes = _cutlass_includes_for_capability(capability)
    flags = [
        "-std=c++20",
    ]
    for inc in cutlass_includes:
        flags.append(f"-I{inc}")
    target = _SUPPORTED_TCGEN05_TARGETS[capability]
    flags.append(f"-gencode=arch=compute_{target},code=sm_{target}")
    if _CLANG_HOST.exists():
        flags.append(f"-ccbin={_CLANG_HOST}")
    return flags


def _get_cuda_version() -> str:
    """Get CUDA toolkit version string for fingerprinting."""
    try:
        if hasattr(torch.version, "cuda") and torch.version.cuda:
            return torch.version.cuda
        cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", ""))
        if cuda_home:
            return cuda_home
    except Exception:
        pass
    return "unknown"


def _get_env_fingerprint() -> str:
    """Get relevant environment variables for fingerprinting."""
    env_vars = ["TORCH_CUDA_ARCH_LIST", "CUDA_HOME", "CUDA_PATH", "MAX_JOBS", "CC", "CXX"]
    parts = []
    for var in sorted(env_vars):
        val = os.environ.get(var, "")
        if val:
            parts.append(f"{var}={val}")
    return "|".join(parts) if parts else "default"


def _get_include_dir_fingerprint(include_dirs: Sequence[Path]) -> str:
    """Fingerprint broad CUTLASS metadata outside Ninja's header dependency graph."""
    hasher = hashlib.sha256()
    for inc_dir in include_dirs:
        if inc_dir.exists():
            try:
                mtime = inc_dir.stat().st_mtime
                hasher.update(f"{inc_dir}:{mtime}\n".encode())
                # Check key version files
                version_file = inc_dir / "cutlass" / "version.h"
                if version_file.exists():
                    hasher.update(f"{version_file}:{version_file.stat().st_mtime}\n".encode())
            except OSError:
                pass
    return hasher.hexdigest()[:8]


def _compute_build_fingerprint(sources: Sequence[Path], cuda_flags: list[str]) -> str:
    """Compute a hash fingerprint of all build inputs.

    This includes:
    - Source file contents
    - All compiler flags (including include paths)
    - Build fingerprint version (for manual invalidation)
    - Python, torch, and CUDA versions
    - GPU architecture
    - Environment variables
    - Broad include directory metadata

    PyTorch and Ninja still inspect their generated header dependency graph on
    every load. The manual fingerprint does not replace that dependency check.
    """
    hasher = hashlib.sha256()

    # Include fingerprint version for manual cache invalidation
    hasher.update(f"version:{_BUILD_FINGERPRINT_VERSION}\n".encode())

    # Include torch version
    hasher.update(f"torch:{torch.__version__}\n".encode())

    # The workspace build directory is shared across Python environments.
    hasher.update(f"python:{sys.implementation.cache_tag}\n".encode())

    # Include CUDA version - important for toolkit upgrades
    hasher.update(f"cuda:{_get_cuda_version()}\n".encode())

    # Include environment variables
    hasher.update(f"env:{_get_env_fingerprint()}\n".encode())

    # Include GPU architecture
    capability = _current_compute_capability()
    major, minor = capability
    cutlass_includes = _cutlass_includes_for_capability(capability)
    hasher.update(f"gpu_arch:sm_{major}{minor}\n".encode())

    # Include all compiler flags (sorted for consistency)
    for flag in sorted(cuda_flags):
        hasher.update(f"flag:{flag}\n".encode())

    # Include source file contents
    for src in sorted(sources):
        if src.exists():
            hasher.update(f"source:{src}:\n".encode())
            hasher.update(src.read_bytes())
            hasher.update(b"\n")

    # Include CUTLASS path and broad metadata. Ninja tracks individual headers.
    for inc in cutlass_includes:
        hasher.update(f"include:{inc}\n".encode())
    hasher.update(f"inc_fp:{_get_include_dir_fingerprint(cutlass_includes)}\n".encode())

    return hasher.hexdigest()[:16]  # Short hash is sufficient


def _check_and_invalidate_cache(
    name: str,
    sources: Sequence[Path],
    cuda_flags: list[str],
) -> str:
    """Check if cached build matches current inputs; invalidate if not.

    This prevents stale cache issues when include paths, compiler flags,
    or source files change. Return the fingerprint to record after a successful
    PyTorch/Ninja build.
    """
    build_dir = _get_extension_build_dir(name)
    fingerprint_file = build_dir / ".build_fingerprint"
    current_fingerprint = _compute_build_fingerprint(sources, cuda_flags)

    # Check if we have a cached build with a matching fingerprint
    if fingerprint_file.exists():
        try:
            stored = json.loads(fingerprint_file.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            stored = None
        if isinstance(stored, dict) and stored.get("fingerprint") == current_fingerprint:
            return current_fingerprint

    # Cache miss or fingerprint mismatch - invalidate cache
    if build_dir.exists():
        shutil.rmtree(build_dir)

    # PyTorch expects an existing build_directory. The fingerprint is written
    # only after load() completes successfully.
    build_dir.mkdir(parents=True, exist_ok=True)
    return current_fingerprint


def _write_build_fingerprint(
    name: str,
    sources: Sequence[Path],
    cuda_flags: list[str],
    fingerprint: str,
) -> None:
    """Atomically record build inputs after PyTorch successfully loads the extension."""
    build_dir = _get_extension_build_dir(name)
    build_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_file = build_dir / ".build_fingerprint"
    payload = {
        "fingerprint": fingerprint,
        "sources": [str(source) for source in sources],
        "cuda_flags": cuda_flags,
    }
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=build_dir,
            prefix=".build_fingerprint.",
            delete=False,
        ) as temporary_file:
            json.dump(payload, temporary_file, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
            temporary_path = Path(temporary_file.name)
        temporary_path.replace(fingerprint_file)
    finally:
        if temporary_path is not None:
            with suppress(FileNotFoundError):
                temporary_path.unlink()


def _get_extension_build_dir(name: str) -> Path:
    """Get the torch extension build directory for a given extension name."""
    # torch extensions default to ~/.cache/torch_extensions or TORCH_EXTENSIONS_DIR
    base = os.environ.get("TORCH_EXTENSIONS_DIR")
    if base:
        return Path(base) / name
    # Fall back to workspace .torch_extensions
    return _REPO_ROOT / ".torch_extensions" / name


@contextmanager
def _extension_build_lock(name: str) -> Iterator[None]:
    """Serialize mutation of one extension cache across benchmark processes.

    PyTorch's JIT lock lives inside ``build_directory``. Our fingerprint and
    stale-build handling may remove that directory, so the outer lock must live
    beside it and remain valid while the cache is invalidated or rebuilt.
    """
    if fcntl is None:
        raise RuntimeError(
            "tcgen05 extension builds require POSIX advisory file locking so "
            "concurrent benchmark processes cannot corrupt the shared build cache"
        )

    build_dir = _get_extension_build_dir(name)
    lock_path = build_dir.parent / f".{name}.build.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _clean_stale_build(name: str) -> None:
    """Remove stale build artifacts if .so is missing but build.ninja exists."""
    build_dir = _get_extension_build_dir(name)
    ninja_file = build_dir / "build.ninja"
    so_file = build_dir / f"{name}.so"

    if ninja_file.exists() and not so_file.exists():
        # Stale build directory - ninja exists but .so missing means build failed
        shutil.rmtree(build_dir)


def _load_extension(name: str, sources: Sequence[Path]):
    # Reject unsupported hardware/toolchains before creating a cache or lock file.
    cuda_flags = _tcgen05_cuda_flags()
    with _extension_build_lock(name):
        # Check if cached build matches current inputs; invalidate if not.
        # This prevents stale cache issues when include paths or flags change.
        current_fingerprint = _check_and_invalidate_cache(name, sources, cuda_flags)

        # Clean up stale build artifacts (incomplete builds).
        _clean_stale_build(name)

        build_dir = _get_extension_build_dir(name)
        build_dir.mkdir(parents=True, exist_ok=True)

        try:
            module = load(
                name=name,
                sources=[str(src) for src in sources],
                extra_cuda_cflags=cuda_flags,
                extra_cflags=["-std=c++20"],
                extra_ldflags=["-lcuda"],
                verbose=False,
                build_directory=str(build_dir),
            )
        except Exception as e:
            # On failure, retry with verbose=True to capture build errors.
            error_msg = str(e)
            if "cannot open shared object file" in error_msg or "No such file" in error_msg:
                # Clean up and retry with verbose output.
                _clean_stale_build(name)
                build_dir.mkdir(parents=True, exist_ok=True)
                try:
                    module = load(
                        name=name,
                        sources=[str(src) for src in sources],
                        extra_cuda_cflags=cuda_flags,
                        extra_cflags=["-std=c++20"],
                        extra_ldflags=["-lcuda"],
                        verbose=True,  # Show build errors on retry
                        build_directory=str(build_dir),
                    )
                except Exception as retry_e:
                    raise RuntimeError(
                        f"Failed to build tcgen05 extension '{name}'. "
                        f"Build errors (see above). Original error: {retry_e}"
                    ) from retry_e
            else:
                raise

        _write_build_fingerprint(
            name,
            sources,
            cuda_flags,
            current_fingerprint,
        )
        return module


@lru_cache(None)
def load_matmul_tcgen05_module():
    """Compile (if needed) and return the Chapter 10 tcgen05 matmul extension."""
    return _load_extension("ch10_matmul_tcgen05_ext", [_REPO_ROOT / "ch10" / "matmul_tcgen05.cu"])


@lru_cache(None)
def load_tiling_tcgen05_module():
    """Compile (if needed) and return the Chapter 8 tcgen05 tiling extension."""
    return _load_extension(
        "ch08_tiling_tcgen05_ext", [_REPO_ROOT / "ch08" / "tiling_kernels_tcgen05.cu"]
    )


@lru_cache(None)
def load_tcgen05_basic_module():
    """Compile (if needed) and return the Chapter 9 basic tcgen05 matmul extension."""
    return _load_extension("ch09_tcgen05_basic_ext", [_REPO_ROOT / "ch09" / "tcgen05_basic.cu"])


@lru_cache(None)
def load_tcgen05_pipelined_module():
    """Compile (if needed) and return the Chapter 9 pipelined tcgen05 matmul extension."""
    return _load_extension(
        "ch09_tcgen05_pipelined_ext", [_REPO_ROOT / "ch09" / "tcgen05_pipelined.cu"]
    )


@lru_cache(None)
def load_tcgen05_cluster_module():
    """Compile (if needed) and return the Chapter 10 cluster tcgen05 matmul extension."""
    return _load_extension("ch10_tcgen05_cluster_ext", [_REPO_ROOT / "ch10" / "tcgen05_cluster.cu"])


@lru_cache(None)
def load_tcgen05_warp_specialized_module():
    """Compile (if needed) and return the Chapter 10 warp-specialized tcgen05 matmul extension."""
    return _load_extension(
        "ch10_tcgen05_warp_specialized_ext", [_REPO_ROOT / "ch10" / "tcgen05_warp_specialized.cu"]
    )


@lru_cache(None)
def load_tcgen05_warp_specialized_cutlass_module():
    """Compile (if needed) and return the Chapter 10 CUTLASS-style warp-specialized tcgen05 matmul extension."""
    return _load_extension(
        "ch10_tcgen05_warp_specialized_cutlass_ext",
        [_REPO_ROOT / "ch10" / "tcgen05_warp_specialized_cutlass.cu"],
    )


@lru_cache(None)
def load_tcgen05_warpgroup_specialized_module():
    """Compile (if needed) and return the Chapter 10 warpgroup-specialized tcgen05 matmul extension."""
    return _load_extension(
        "ch10_tcgen05_warpgroup_specialized_ext",
        [_REPO_ROOT / "ch10" / "tcgen05_warpgroup_specialized.cu"],
    )


def matmul_tcgen05(
    a: torch.Tensor, b: torch.Tensor, *, module_name: str = "tcgen05 matmul"
) -> torch.Tensor:
    """Execute the CUTLASS tcgen05 GEMM after ensuring hardware/toolchain support."""
    ensure_tcgen05_supported(loader=load_matmul_tcgen05_module, module_name=module_name)
    module = load_matmul_tcgen05_module()
    return module.matmul_tcgen05(a, b)


def matmul_tcgen05_bias_silu(
    a: torch.Tensor,
    b: torch.Tensor,
    bias: torch.Tensor,
    *,
    module_name: str = "tcgen05 matmul bias+SiLU",
) -> torch.Tensor:
    """Execute the tcgen05 GEMM with TMEM-resident bias+SiLU epilogue."""
    ensure_tcgen05_supported(loader=load_matmul_tcgen05_module, module_name=module_name)
    module = load_matmul_tcgen05_module()
    return module.matmul_tcgen05_bias_silu(a, b, bias)


def matmul_tiling_tcgen05(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    module_name: str = "tcgen05 tiling matmul",
) -> torch.Tensor:
    """Execute the CUTLASS tcgen05 tiling GEMM."""
    ensure_tcgen05_supported(loader=load_tiling_tcgen05_module, module_name=module_name)
    module = load_tiling_tcgen05_module()
    return module.matmul_tiling_tcgen05(a, b)


__all__ = [
    "load_matmul_tcgen05_module",
    "load_tiling_tcgen05_module",
    "load_tcgen05_basic_module",
    "load_tcgen05_pipelined_module",
    "load_tcgen05_cluster_module",
    "load_tcgen05_warp_specialized_module",
    "load_tcgen05_warp_specialized_cutlass_module",
    "load_tcgen05_warpgroup_specialized_module",
    "matmul_tcgen05",
    "matmul_tcgen05_bias_silu",
    "matmul_tiling_tcgen05",
]
