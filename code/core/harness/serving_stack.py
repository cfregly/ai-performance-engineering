"""Shared serving stack pin resolution for benchmark hosts.

Single source of truth:
- Require the base exact pins from requirements_latest.txt.
- Require the ABI-bound vLLM CUDA 13 pin from vllm_no_deps.pin.
- Fail fast if pins are missing or unreadable.
"""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import re
import site
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REQUIREMENTS = _REPO_ROOT / "requirements_latest.txt"
_DEFAULT_VLLM_PIN = _REPO_ROOT / "vllm_no_deps.pin"

_DEFAULT_TORCH = "2.9.1+cu130"
_DEFAULT_VLLM = "0.16.0+cu130"
_DEFAULT_FLASHINFER = "0.6.3"

_NVIDIA_WHEEL_LIB_SUBDIRS: Tuple[Tuple[str, ...], ...] = (
    ("nvidia", "cublas", "lib"),
    ("nvidia", "cuda_runtime", "lib"),
    ("nvidia", "cuda_nvrtc", "lib"),
    ("nvidia", "cudnn", "lib"),
    ("nvidia", "cufft", "lib"),
    ("nvidia", "curand", "lib"),
    ("nvidia", "cusolver", "lib"),
    ("nvidia", "cusparse", "lib"),
    ("nvidia", "cusparselt", "lib"),
    ("nvidia", "nccl", "lib"),
    ("nvidia", "nvjitlink", "lib"),
    ("nvidia", "nvshmem", "lib"),
    ("nvidia", "cufile", "lib"),
)

_DEFAULT_PRELOAD_LIBS: Tuple[str, ...] = (
    "libcublas.so.12",
    "libcudnn.so.9",
    "libcudnn_graph.so.9",
)


@dataclass(frozen=True)
class ServingStackPins:
    torch_version: str
    vllm_version: str
    flashinfer_version: str

    @property
    def vllm_pip_spec(self) -> str:
        return f"vllm=={self.vllm_version}"

    @property
    def pinned_stack_str(self) -> str:
        return (
            f"torch=={self.torch_version}, "
            f"vllm=={self.vllm_version}, "
            f"flashinfer-python=={self.flashinfer_version}"
        )


def _read_pinned_version(package_name: str, requirements_path: Path) -> Optional[str]:
    if not requirements_path.exists():
        return None
    try:
        lines = requirements_path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        raise RuntimeError(
            "FAIL FAST: Unable to read serving stack requirements file. "
            f"Expected readable file at {requirements_path}: {exc}"
        ) from exc
    pattern = re.compile(rf"^\s*{re.escape(package_name)}==([^\s#]+)\s*(?:#.*)?$")
    for raw_line in lines:
        match = pattern.match(raw_line)
        if match:
            return match.group(1).strip()
    return None


def _site_package_roots() -> List[Path]:
    roots: List[Path] = []
    try:
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(Path(user_site))
    except Exception as exc:
        warnings.warn(
            f"Failed to query user site-packages for serving-stack CUDA wheels: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    try:
        for site_path in site.getsitepackages():
            if site_path:
                roots.append(Path(site_path))
    except Exception as exc:
        warnings.warn(
            f"Failed to query system site-packages for serving-stack CUDA wheels: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
    deduped: List[Path] = []
    seen = set()
    for root in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(root)
    return deduped


def _discover_cuda_wheel_lib_dirs(extra_roots: Optional[Sequence[Path]] = None) -> List[str]:
    roots = list(_site_package_roots())
    if extra_roots:
        roots.extend(Path(p) for p in extra_roots)

    lib_dirs: List[str] = []
    seen = set()
    for root in roots:
        for parts in _NVIDIA_WHEEL_LIB_SUBDIRS:
            candidate = root.joinpath(*parts)
            if not candidate.is_dir():
                continue
            path_str = str(candidate)
            if path_str in seen:
                continue
            seen.add(path_str)
            lib_dirs.append(path_str)
    return lib_dirs


def configure_serving_stack_runtime_env(extra_roots: Optional[Sequence[Path]] = None) -> List[str]:
    """Prepend CUDA wheel library directories to LD_LIBRARY_PATH for vLLM imports.

    Returns the directories that were prepended in this call.
    """

    discovered = _discover_cuda_wheel_lib_dirs(extra_roots=extra_roots)
    if not discovered:
        return []

    existing_entries = [
        entry for entry in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if entry
    ]
    merged: List[str] = []
    seen = set()
    for entry in [*discovered, *existing_entries]:
        if entry in seen:
            continue
        seen.add(entry)
        merged.append(entry)
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(merged)
    return discovered


def preload_serving_stack_shared_libs(
    required_libs: Optional[Sequence[str]] = None,
    extra_roots: Optional[Sequence[Path]] = None,
) -> Dict[str, str]:
    """Preload serving-stack CUDA shared libs into the current process.

    This handles the case where mutating LD_LIBRARY_PATH inside Python is too late
    for some dynamic-link resolution paths used by extension modules.
    """

    libs = tuple(required_libs or _DEFAULT_PRELOAD_LIBS)
    search_dirs = _discover_cuda_wheel_lib_dirs(extra_roots=extra_roots)
    loaded: Dict[str, str] = {}
    rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0)

    for lib_name in libs:
        candidate_path: Optional[Path] = None
        for lib_dir in search_dirs:
            path = Path(lib_dir) / lib_name
            if path.is_file():
                candidate_path = path
                break
        if candidate_path is None:
            continue
        try:
            ctypes.CDLL(str(candidate_path), mode=rtld_global)
            loaded[lib_name] = str(candidate_path)
        except OSError as exc:
            raise RuntimeError(
                "FAIL FAST: Unable to preload required serving-stack shared library "
                f"{candidate_path}: {exc}"
            ) from exc
    return loaded


def configure_serving_stack_cache_env(cache_root: Optional[Path] = None) -> Dict[str, str]:
    """Configure writable cache directories for vLLM/HF runtime artifacts."""

    configured: Dict[str, str] = {}
    root = Path(
        cache_root
        or os.environ.get("AISP_SERVING_CACHE_ROOT", "")
        or (Path("/tmp") / "aisp_serving_cache")
    )
    root.mkdir(parents=True, exist_ok=True)

    cache_map = {
        "HF_HOME": root / "huggingface",
        "HF_HUB_CACHE": root / "huggingface" / "hub",
        "HUGGINGFACE_HUB_CACHE": root / "huggingface" / "hub",
        "TRANSFORMERS_CACHE": root / "huggingface" / "transformers",
        "VLLM_CACHE_ROOT": root / "vllm",
        "VLLM_TORCH_COMPILE_CACHE_DIR": root / "vllm" / "torch_compile_cache",
        "TORCHINDUCTOR_CACHE_DIR": root / "torchinductor",
    }
    for key, path in cache_map.items():
        current = os.environ.get(key)
        if current:
            resolved = Path(current)
            resolved.mkdir(parents=True, exist_ok=True)
            configured[key] = str(resolved)
            continue
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
        configured[key] = str(path)
    return configured


def get_serving_stack_pins(
    requirements_path: Optional[Path] = None,
    vllm_pin_path: Optional[Path] = None,
) -> ServingStackPins:
    req_path = (requirements_path or _DEFAULT_REQUIREMENTS).resolve()
    vllm_path = (vllm_pin_path or _DEFAULT_VLLM_PIN).resolve()
    if not req_path.exists():
        raise RuntimeError(
            "FAIL FAST: Serving stack requirements file is missing. "
            f"Expected: {req_path}"
        )

    torch_version = _read_pinned_version("torch", req_path)
    if not vllm_path.exists():
        raise RuntimeError(
            "FAIL FAST: ABI-bound vLLM pin manifest is missing. "
            f"Expected: {vllm_path}"
        )

    vllm_version = _read_pinned_version("vllm", vllm_path)
    flashinfer_version = _read_pinned_version("flashinfer-python", req_path)
    if not torch_version or not vllm_version or not flashinfer_version:
        raise RuntimeError(
            "FAIL FAST: Serving stack pin(s) missing in requirements file. "
            f"Expected torch=={_DEFAULT_TORCH}, vllm=={_DEFAULT_VLLM}, "
            f"flashinfer-python=={_DEFAULT_FLASHINFER} in {req_path}, and "
            f"vllm=={_DEFAULT_VLLM} in {vllm_path}"
        )

    return ServingStackPins(
        torch_version=torch_version,
        vllm_version=vllm_version,
        flashinfer_version=flashinfer_version,
    )
