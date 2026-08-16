"""Helpers for applying environment defaults and reporting capabilities."""

from __future__ import annotations

from core.utils import compile_utils as _compile_utils_patch  # noqa: F401
from core.harness.cuda_capabilities import (
    pipeline_support_status,
    pipeline_runtime_allowed,
    tma_support_status,
)
from core.harness.hardware_capabilities import (
    detect_capabilities,
    format_capability_report,
)
import ctypes
import glob
import os
import site
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

try:
    from core.utils.build_utils import ensure_clean_build_directory
except ImportError:  # pragma: no cover - psutil optional in some environments
    ensure_clean_build_directory = None  # type: ignore[assignment]

ENV_DEFAULTS: Dict[str, str] = {
    "PYTHONFAULTHANDLER": "1",
    "TORCH_SHOW_CPP_STACKTRACES": "1",
    "TORCH_CUDNN_V8_API_ENABLED": "1",
    "CUDA_LAUNCH_BLOCKING": "0",
    "CUDA_CACHE_DISABLE": "0",
    "NCCL_IB_DISABLE": "0",
    "NCCL_P2P_DISABLE": "0",
    "NCCL_SHM_DISABLE": "0",
    "CUDA_DEVICE_MAX_CONNECTIONS": "32",
    # Hugging Face network calls can be slow/transient on some clusters.
    # Raise defaults to reduce flaky dataset/model fetch failures.
    "HF_HUB_DOWNLOAD_TIMEOUT": "120",
    "HF_HUB_ETAG_TIMEOUT": "60",
    "TORCH_COMPILE_DEBUG": "0",
    # "TORCH_LOGS": "",  # Disabled - remove verbose dynamo logging to reduce noise
    "CUDA_HOME": "/usr/local/cuda",
    # Runtime policy for cuDNN/cuda user-mode library selection:
    # - auto: prefer torch/lib when available, otherwise fall back to system runtime
    # - torch: force torch/lib precedence
    # - system: force system runtime precedence
    "AISP_CUDNN_RUNTIME_POLICY": "auto",
}

CUDA_PATH_SUFFIXES: Tuple[str, ...] = ("bin",)
CUDA_LIBRARY_SUFFIXES: Tuple[str, ...] = ("lib64",)
CUDA_WHEEL_PRELOAD_LIBS: Tuple[str, ...] = ("libcublasLt.so.12", "libcublas.so.12")
_PRELOADED_CUDA_WHEEL_LIBS: Set[str] = set()

# Try to find NCCL library for current architecture
def _find_nccl_library() -> str:
    """Find NCCL library for the current architecture."""
    import platform
    machine = platform.machine()
    
    # Try architecture-specific paths
    candidates = []
    if machine == "x86_64":
        candidates = [
            "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
            "/usr/lib/x86_64-linux-gnu/libnccl.so",
        ]
    elif machine in ("aarch64", "arm64"):
        candidates = [
            "/usr/lib/aarch64-linux-gnu/libnccl.so.2",
            "/usr/lib/aarch64-linux-gnu/libnccl.so",
        ]
    
    # Also try generic paths
    candidates.extend([
        "/usr/local/lib/libnccl.so.2",
        "/usr/local/lib/libnccl.so",
        "/usr/lib/libnccl.so.2",
        "/usr/lib/libnccl.so",
    ])
    
    # Return first existing file, or default to x86_64 path (will be ignored if not found)
    for path in candidates:
        if os.path.exists(path):
            return path
    
    # Return empty string if not found (will be skipped when adding to LD_PRELOAD)
    return ""

NCCL_LIBRARY_PATH = _find_nccl_library()

REPORTED_ENV_KEYS: Tuple[str, ...] = (
    "PYTHONFAULTHANDLER",
    "TORCH_SHOW_CPP_STACKTRACES",
    "TORCH_CUDNN_V8_API_ENABLED",
    "CUDA_LAUNCH_BLOCKING",
    "CUDA_CACHE_DISABLE",
    "NCCL_IB_DISABLE",
    "NCCL_P2P_DISABLE",
    "NCCL_SHM_DISABLE",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "HF_HUB_DOWNLOAD_TIMEOUT",
    "HF_HUB_ETAG_TIMEOUT",
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "TRANSFORMERS_CACHE",
    "PYTORCH_ALLOC_CONF",
    "TORCH_COMPILE_DEBUG",
    "TORCH_LOGS",
    "AISP_CUDNN_RUNTIME_POLICY",
    "CUDA_HOME",
    "PATH",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
)

_ENV_AND_CAPABILITIES_LOGGED = False


def _runtime_policy() -> str:
    policy = os.environ.get("AISP_CUDNN_RUNTIME_POLICY", "auto").strip().lower()
    if policy not in {"auto", "torch", "system"}:
        policy = "auto"
    os.environ["AISP_CUDNN_RUNTIME_POLICY"] = policy
    return policy


def _split_paths(value: str) -> List[str]:
    return [entry for entry in value.split(":") if entry]


def _append_if_valid(path: str, ordered: List[str], seen: Set[str]) -> None:
    if not path or path in seen or not os.path.isdir(path):
        return
    ordered.append(path)
    seen.add(path)


def _looks_like_torch_lib(path: str) -> bool:
    normalized = path.rstrip("/")
    return normalized.endswith("/torch/lib") or "/site-packages/torch/lib" in normalized


def _candidate_torch_site_roots() -> Iterable[Path]:
    roots: List[Path] = []
    py_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    roots.append(Path(sys.prefix) / "lib" / py_version / "site-packages")
    roots.append(Path("/usr/local/lib") / py_version / "dist-packages")

    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        roots.append(Path(venv) / "lib" / py_version / "site-packages")

    try:
        roots.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass

    try:
        roots.append(Path(site.getusersitepackages()))
    except Exception:
        pass

    seen: Set[Path] = set()
    for root in roots:
        if root in seen:
            continue
        seen.add(root)
        yield root


def _discover_nvidia_wheel_lib_dirs() -> List[str]:
    """Discover CUDA/NVIDIA wheel shared-library directories.

    These libraries (for example libcublas.so.12) are required by some Python
    packages such as transformer_engine when imported transitively via
    transformers/accelerate.
    """
    discovered: List[str] = []
    seen: Set[str] = set()
    for root in _candidate_torch_site_roots():
        nvidia_root = root / "nvidia"
        if not nvidia_root.is_dir():
            continue
        for lib_dir in sorted(nvidia_root.glob("*/lib")):
            if not lib_dir.is_dir():
                continue
            path_str = str(lib_dir)
            if path_str in seen:
                continue
            seen.add(path_str)
            discovered.append(path_str)
    return discovered


def _ensure_cuda_wheel_runtime_libs() -> None:
    """Ensure NVIDIA wheel runtime libs are available on LD_LIBRARY_PATH."""
    lib_dirs = _discover_nvidia_wheel_lib_dirs()
    # Preserve discovered order when prepending.
    for lib_dir in reversed(lib_dirs):
        _prepend_if_missing("LD_LIBRARY_PATH", lib_dir)


def _preload_cuda_wheel_runtime_libs() -> None:
    """Preload critical CUDA wheel libs for in-process extension loading.

    Updating LD_LIBRARY_PATH after process start does not always affect dynamic
    linker resolution for ctypes/extension modules. Preloading with RTLD_GLOBAL
    ensures symbols are available for transitive imports (e.g. transformers ->
    accelerate -> transformer_engine).
    """
    search_dirs = _discover_nvidia_wheel_lib_dirs()
    if not search_dirs:
        return

    rtld_global = getattr(ctypes, "RTLD_GLOBAL", 0)
    for lib_name in CUDA_WHEEL_PRELOAD_LIBS:
        if lib_name in _PRELOADED_CUDA_WHEEL_LIBS:
            continue
        for lib_dir in search_dirs:
            candidate = Path(lib_dir) / lib_name
            if not candidate.is_file():
                continue
            try:
                ctypes.CDLL(str(candidate), mode=rtld_global)
                _PRELOADED_CUDA_WHEEL_LIBS.add(lib_name)
                break
            except OSError:
                continue


def _discover_torch_lib_dirs() -> List[str]:
    dirs: List[str] = []
    seen: Set[str] = set()
    for root in _candidate_torch_site_roots():
        candidate = root / "torch" / "lib"
        candidate_str = str(candidate)
        if candidate_str in seen or not candidate.is_dir():
            continue
        dirs.append(candidate_str)
        seen.add(candidate_str)
    return dirs


def _discover_system_cudnn_dirs() -> List[str]:
    candidates = (
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/aarch64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
    )
    found: List[str] = []
    seen: Set[str] = set()
    for directory in candidates:
        if directory in seen or not os.path.isdir(directory):
            continue
        if list(Path(directory).glob("libcudnn*.so*")):
            found.append(directory)
            seen.add(directory)
    return found


def _prioritize_runtime_libraries() -> None:
    policy = _runtime_policy()
    current_paths = _split_paths(os.environ.get("LD_LIBRARY_PATH", ""))
    ordered: List[str] = []
    seen: Set[str] = set()

    using_torch_runtime = False
    if policy in {"auto", "torch"}:
        for path in current_paths:
            if _looks_like_torch_lib(path):
                _append_if_valid(path, ordered, seen)
        if not any(_looks_like_torch_lib(path) for path in ordered):
            for path in _discover_torch_lib_dirs():
                _append_if_valid(path, ordered, seen)
        using_torch_runtime = any(_looks_like_torch_lib(path) for path in ordered)

    if policy == "system":
        for path in _discover_system_cudnn_dirs():
            _append_if_valid(path, ordered, seen)

    # Keep CUDA/system paths in existing order after runtime-preferred paths.
    for path in current_paths:
        if policy == "system" and _looks_like_torch_lib(path):
            continue
        _append_if_valid(path, ordered, seen)

    if policy != "system" and not using_torch_runtime:
        # If no torch runtime was found, preserve whatever CUDA/system ordering we already had.
        os.environ["LD_LIBRARY_PATH"] = ":".join(ordered)
        return

    os.environ["LD_LIBRARY_PATH"] = ":".join(ordered)


def _ensure_huggingface_cache_dirs() -> None:
    """Ensure writable Hugging Face cache directories for benchmarked entrypoints."""
    root = Path.cwd() / ".cache" / "huggingface"
    cache_map = {
        "HF_HOME": root,
        "HF_HUB_CACHE": root / "hub",
        "HUGGINGFACE_HUB_CACHE": root / "hub",
    }

    for key, default_path in cache_map.items():
        current = os.environ.get(key)
        resolved = Path(current).expanduser() if current else default_path
        resolved.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(resolved)

    transformers_cache = os.environ.get("TRANSFORMERS_CACHE")
    if transformers_cache:
        resolved = Path(transformers_cache).expanduser()
        resolved.mkdir(parents=True, exist_ok=True)
        os.environ["TRANSFORMERS_CACHE"] = str(resolved)


def apply_env_defaults() -> Dict[str, str]:
    """Apply default environment configuration and return the resulting values.
    
    Only sets variables that are not already set, to avoid overwriting user configurations.
    """
    applied: Dict[str, str] = {}

    for key, value in ENV_DEFAULTS.items():
        previous = os.environ.get(key)
        if previous is None:
            os.environ.setdefault(key, value)
        applied[key] = os.environ[key]

    if "PYTORCH_ALLOC_CONF" not in os.environ:
        legacy = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        os.environ["PYTORCH_ALLOC_CONF"] = legacy or "max_split_size_mb:128,expandable_segments:True"
    applied["PYTORCH_ALLOC_CONF"] = os.environ["PYTORCH_ALLOC_CONF"]
    
    # Ensure PyTorch inductor cache directory exists to prevent C++ compilation errors
    # PyTorch inductor needs this directory to exist for C++ code generation
    # Use absolute path to avoid working directory issues
    inductor_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if not inductor_cache_dir:
        # Default to .torch_inductor in current working directory (convert to absolute)
        inductor_cache_dir = str(Path.cwd() / ".torch_inductor")
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache_dir
    else:
        # Convert relative paths to absolute paths to avoid working directory issues
        if not os.path.isabs(inductor_cache_dir):
            inductor_cache_dir = str(Path.cwd() / inductor_cache_dir)
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = inductor_cache_dir
    
    if inductor_cache_dir:
        inductor_cache_path = Path(inductor_cache_dir)
        # Create directory and subdirectories (used by inductor for C++ compilation)
        # 'od' is for output directory, 'tk' is for temporary kernel files
        try:
            inductor_cache_path.mkdir(parents=True, exist_ok=True)
            (inductor_cache_path / "od").mkdir(parents=True, exist_ok=True)
            (inductor_cache_path / "tk").mkdir(parents=True, exist_ok=True)
            if ensure_clean_build_directory is not None:
                # Kill stale compiler processes/locks that keep torch.compile hanging
                ensure_clean_build_directory(inductor_cache_path)
        except (OSError, PermissionError):
            # If we can't create the directory, that's okay - PyTorch will handle it
            # or fail with a clearer error message
            pass

    # Ensure cpp-extension builds use a workspace-local cache to avoid /tmp lockups
    torch_extensions_dir = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not torch_extensions_dir:
        torch_extensions_dir = str(Path.cwd() / ".torch_extensions")
        os.environ["TORCH_EXTENSIONS_DIR"] = torch_extensions_dir
    elif not os.path.isabs(torch_extensions_dir):
        torch_extensions_dir = str(Path.cwd() / torch_extensions_dir)
        os.environ["TORCH_EXTENSIONS_DIR"] = torch_extensions_dir

    try:
        torch_extensions_path = Path(torch_extensions_dir)
        torch_extensions_path.mkdir(parents=True, exist_ok=True)
        if ensure_clean_build_directory is not None:
            ensure_clean_build_directory(torch_extensions_path)
    except (OSError, PermissionError):
        pass

    _ensure_huggingface_cache_dirs()

    # Only ensure CUDA paths if CUDA_HOME was not already set by user
    # This prevents overwriting user-configured CUDA installations
    if "CUDA_HOME" not in os.environ or os.environ["CUDA_HOME"] == ENV_DEFAULTS.get("CUDA_HOME"):
        _ensure_cuda_paths()
    else:
        # CUDA_HOME is set by user - only prepend paths if they're missing (don't force our defaults)
        _ensure_cuda_paths(use_existing_cuda_home=True)

    # Ensure CUDA wheel user-mode runtime libs are discoverable (e.g. libcublas.so.12).
    _ensure_cuda_wheel_runtime_libs()
    _preload_cuda_wheel_runtime_libs()

    # Ensure runtime library precedence is explicit and deterministic.
    _prioritize_runtime_libraries()

    _ensure_nsight_paths()
    applied["PATH"] = os.environ.get("PATH", "")
    applied["LD_LIBRARY_PATH"] = os.environ.get("LD_LIBRARY_PATH", "")

    _ensure_ld_preload()
    applied["LD_PRELOAD"] = os.environ.get("LD_PRELOAD", "")

    return {key: os.environ.get(key, "") for key in REPORTED_ENV_KEYS}


def _ensure_cuda_paths(use_existing_cuda_home: bool = False) -> None:
    """Ensure CUDA paths are present in PATH and LD_LIBRARY_PATH.
    
    Args:
        use_existing_cuda_home: If True, use existing CUDA_HOME value even if it matches defaults.
                                If False, only add paths if CUDA_HOME was set from defaults.
    """
    # Only use default CUDA_HOME if it's not already set
    if use_existing_cuda_home:
        cuda_home = os.environ.get("CUDA_HOME")
        if not cuda_home:
            # No CUDA_HOME set and we're not using defaults - skip
            return
    else:
        # Use default only if not set
        cuda_home = os.environ.get("CUDA_HOME", ENV_DEFAULTS.get("CUDA_HOME", ""))
        if not cuda_home:
            return

    # Only add paths if the CUDA_HOME directory actually exists
    if not os.path.exists(cuda_home):
        return

    path_prefixes = _build_paths(cuda_home, CUDA_PATH_SUFFIXES)
    lib_prefixes = _build_paths(cuda_home, CUDA_LIBRARY_SUFFIXES)

    # Only prepend if missing (this function already checks)
    for prefix in path_prefixes:
        if os.path.exists(prefix):  # Only add if path exists
            _prepend_if_missing("PATH", prefix)
    for prefix in lib_prefixes:
        if os.path.exists(prefix):  # Only add if path exists
            _prepend_if_missing("LD_LIBRARY_PATH", prefix)


def _maybe_add_binary_to_path(binary_name: str, patterns: Iterable[str]) -> None:
    """Search common install locations for a binary and prepend its directory to PATH."""
    for pattern in patterns:
        # Prefer newer installs by reversing sorted order
        for candidate in sorted(glob.glob(pattern), reverse=True):
            candidate_path = Path(candidate)
            if candidate_path.is_dir():
                candidate_path = candidate_path / binary_name
            if not candidate_path.exists() or not os.access(candidate_path, os.X_OK):
                continue
            _prepend_if_missing("PATH", str(candidate_path.parent))
            return


def _ensure_nsight_paths() -> None:
    """Ensure Nsight Systems/Compute binaries are discoverable even when not on PATH."""
    if shutil.which("nsys") is None:
        _maybe_add_binary_to_path(
            "nsys",
            (
                "/opt/nvidia/nsight-systems/*/bin",
                "/opt/nvidia/nsight-systems/*/target-linux*",
                "/opt/nvidia/nsight-systems/*",
                "/usr/local/cuda-*/bin",
                "/usr/local/cuda/bin",
            ),
        )
    if shutil.which("ncu") is None:
        _maybe_add_binary_to_path(
            "ncu",
            (
                "/opt/nvidia/nsight-compute/*/bin",
                "/opt/nvidia/nsight-compute/*",
                "/usr/local/cuda-*/bin",
                "/usr/local/cuda/bin",
            ),
        )


def _build_paths(root: str, suffixes: Iterable[str]) -> List[str]:
    return [os.path.join(root, suffix) for suffix in suffixes]


def _prepend_if_missing(key: str, prefix: str) -> None:
    os.environ.setdefault(key, "")
    existing = os.environ.get(key, "")
    components = [segment for segment in existing.split(os.pathsep) if segment]
    if prefix not in components:
        components.insert(0, prefix)
        os.environ[key] = os.pathsep.join(components)


def _ensure_ld_preload() -> None:
    # Preloading distro NCCL can conflict with the NCCL bundled in modern CUDA
    # wheels. Prefer the wheel runtime unless the caller explicitly selected
    # the system runtime policy.
    import platform
    if platform.machine() in ("aarch64", "arm64"):
        return

    if _runtime_policy() != "system":
        for lib_dir in _discover_nvidia_wheel_lib_dirs():
            if any(Path(lib_dir).glob("libnccl.so*")):
                return

    os.environ.setdefault("LD_PRELOAD", "")
    preload_entries = [segment for segment in os.environ["LD_PRELOAD"].split(os.pathsep) if segment]
    
    # Only add NCCL library if it exists and isn't already in LD_PRELOAD
    if NCCL_LIBRARY_PATH and os.path.exists(NCCL_LIBRARY_PATH):
        if NCCL_LIBRARY_PATH not in preload_entries:
            preload_entries.insert(0, NCCL_LIBRARY_PATH)
            os.environ["LD_PRELOAD"] = os.pathsep.join(preload_entries)
    elif NCCL_LIBRARY_PATH:
        # Library path was set but doesn't exist - this is okay, just skip it
        pass


def snapshot_environment() -> Dict[str, str]:
    """Return a snapshot of the relevant environment variables."""
    return {key: os.environ.get(key, "") for key in REPORTED_ENV_KEYS}


def dump_environment_and_capabilities(stream=None, *, force: bool = False) -> None:
    """Emit environment configuration and hardware capabilities."""
    global _ENV_AND_CAPABILITIES_LOGGED
    if _ENV_AND_CAPABILITIES_LOGGED and not force:
        return

    if stream is None:
        stream = sys.stdout

    env_snapshot = snapshot_environment()
    print("=" * 80, file=stream)
    print("ENVIRONMENT CONFIGURATION", file=stream)
    print("=" * 80, file=stream)
    for key in REPORTED_ENV_KEYS:
        print(f"{key}={env_snapshot.get(key, '')}", file=stream)

    print("\n" + "=" * 80, file=stream)
    print("HARDWARE CAPABILITIES", file=stream)
    print("=" * 80, file=stream)

    cap_report = format_capability_report()
    print(cap_report, file=stream)
    
    pipeline_supported, pipeline_reason = pipeline_support_status()
    tma_supported, tma_reason = tma_support_status()
    print(f"CUDA Pipeline API Support: {'yes' if pipeline_supported else 'no'} ({pipeline_reason})", file=stream)
    runtime_allowed, runtime_reason = pipeline_runtime_allowed()
    print(f"Pipeline Runtime Enabled: {'yes' if runtime_allowed else 'no'} ({runtime_reason})", file=stream)
    print(f"TMA Support: {'yes' if tma_supported else 'no'} ({tma_reason})", file=stream)
    
    # Check profiling tool availability
    print("\n" + "=" * 80, file=stream)
    print("PROFILING TOOLS", file=stream)
    print("=" * 80, file=stream)
    
    # Check nsys availability
    nsys_available = False
    try:
        import subprocess
        import shutil
        if shutil.which("nsys"):
            result = subprocess.run(
                ["nsys", "--version"],
                capture_output=True,
                timeout=5,
                check=False
            )
            nsys_available = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass  # nsys not available or timed out
    print(f"nsys available: {nsys_available}", file=stream)
    
    # Check ncu availability
    ncu_available = False
    try:
        if shutil.which("ncu"):
            result = subprocess.run(
                ["ncu", "--version"],
                capture_output=True,
                timeout=5,
                check=False
            )
            ncu_available = result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass  # ncu not available or timed out
    print(f"ncu available: {ncu_available}", file=stream)
    
    _ENV_AND_CAPABILITIES_LOGGED = True
