#!/usr/bin/env python3
"""Architecture helpers for Blackwell and Grace-Blackwell GPUs."""

from typing import Any, Dict, List, Optional
import os
import subprocess
import shutil
import warnings
from pathlib import Path
from importlib import metadata as importlib_metadata
from contextlib import nullcontext

from core.utils.compile_utils import enable_tf32
from core.utils.warning_filters import (
    suppress_known_cuda_capability_warnings,
    warn_optional_component_unavailable,
)
from core.benchmark.triton_compat import (
    ENABLE_TRITON_PATCH as _TRITON_PATCH_ENABLED,
    ensure_triton_compat,
)

with suppress_known_cuda_capability_warnings(context="core.harness.arch_config torch import"):
    import torch

try:
    from torch.nn.attention import sdpa_kernel as _sdpa_kernel
    from torch.nn.attention import SDPBackend as _SDPBackend
    _NEW_SDPA_API_AVAILABLE = True
except ImportError:
    _sdpa_kernel = None  # type: ignore[assignment]
    _SDPBackend = None  # type: ignore[assignment]
    _NEW_SDPA_API_AVAILABLE = False

def _default_sdpa_backends() -> List[Any]:
    if _SDPBackend is None:
        return []
    order: List[Any] = []
    # Prefer TE fused attention on Blackwell/GB200 where available, then Flash, then other fused paths.
    for name in ("TRANSFORMER_ENGINE", "FLASH_ATTENTION", "EFFICIENT_ATTENTION", "CUDNN"):
        if hasattr(_SDPBackend, name):
            order.append(getattr(_SDPBackend, name))
    return order


_PREFERRED_SDPA_BACKENDS: List[Any] = _default_sdpa_backends()


def prefer_sdpa_backends(order: Optional[List[Any]] = None):
    """
    Return a context manager that routes scaled_dot_product_attention to preferred backends.
    
    Uses the new torch.nn.attention.sdpa_kernel() API. Never falls back to the
    deprecated torch.backends.cuda.sdp_kernel() API.

    Example:
        with prefer_sdpa_backends():
            F.scaled_dot_product_attention(...)
    """
    if not _NEW_SDPA_API_AVAILABLE:
        # Return no-op context manager - do NOT use deprecated API
        return nullcontext()
    
    if order is None:
        order = _PREFERRED_SDPA_BACKENDS
    if _sdpa_kernel is None or not order:
        return nullcontext()
    return _sdpa_kernel(order)


def prefer_flash_sdpa():
    """Alias retained for backwards compatibility with earlier chapter drafts."""
    return prefer_sdpa_backends()

BLACKWELL_CC = "10.0"
BLACKWELL_ULTRA_CC = "10.3"
GRACE_BLACKWELL_MAJOR = 12

def _parse_version_tuple(version: str) -> tuple:
    parts = []
    for token in version.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            parts.append(0)
    return tuple(parts)

class ArchitectureConfig:
    """Provide configuration details for NVIDIA Blackwell GPUs."""

    def __init__(self) -> None:
        self.arch = self._detect_architecture()
        self.config = self._get_architecture_config()
        self.cutlass_version = None

    @staticmethod
    def metadata_for_capability(capability: Optional[tuple[int, int]], device_name: str = "") -> Dict[str, Any]:
        """Describe an observed capability without inferring an unobserved SKU/topology.

        The mapping is NVIDIA's CUDA GPU table, not a compile/runtime qualification.
        https://developer.nvidia.com/cuda/gpus
        """
        families = {
            (10, 0): ("blackwell", "Blackwell B200/GB200 family"),
            (10, 3): ("blackwell_ultra", "Blackwell Ultra B300/GB300 family"),
            (12, 0): ("blackwell_consumer", "Blackwell GeForce RTX 50 / RTX PRO family"),
            (12, 1): ("grace_blackwell", "GB10 / DGX Spark family"),
        }
        arch, family_name = families.get(capability, ("cpu" if capability is None else "other", "CPU" if capability is None else "Unclassified CUDA GPU"))
        known_blackwell = capability in families
        cc = f"{capability[0]}.{capability[1]}" if capability else None
        return {
            "architecture": arch,
            "name": device_name or family_name,
            "family": family_name,
            "compute_capability": cc or "N/A",
            "sm_version": f"sm_{capability[0]}{capability[1]}" if capability else "cpu",
            "memory_bandwidth": "SKU-dependent; not inferred from compute capability",
            "tensor_cores": "5th Gen family; SKU throughput not inferred" if known_blackwell else "Unknown",
            "features": ["Stream-ordered Memory", "TMA"] if known_blackwell else [],
            "cuda_features": ["Stream-ordered Memory", "TMA"] if known_blackwell else [],
            "pytorch_optimizations": ["torch.compile with actual device capability"] if known_blackwell else [],
            "triton_features": ["Actual-device code generation; toolchain support required"] if known_blackwell else [],
            "profiling_tools": ["Nsight Systems", "Nsight Compute", "PyTorch Profiler"] if capability else [],
            "tcgen05_supported": capability in {(10, 0), (10, 3)},
            "runtime_qualified": False,
        }

    def _detect_architecture(self) -> str:
        self.compute_capability = None
        self.device_name = ""
        with suppress_known_cuda_capability_warnings(context="ArchitectureConfig._detect_architecture"):
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                self.compute_capability = (props.major, props.minor)
                self.device_name = props.name
        return self.metadata_for_capability(self.compute_capability)["architecture"]

    def _get_architecture_config(self) -> Dict[str, Any]:
        return self.metadata_for_capability(self.compute_capability, self.device_name)

    def require_tcgen05(self) -> None:
        if not self.config["tcgen05_supported"]:
            raise RuntimeError(
                f"tcgen05 is unsupported for {self.config['sm_version']}; "
                "a different GPU target cannot be substituted"
            )

    def get_sm_version(self) -> str:
        return self.config["sm_version"]

    def get_architecture_name(self) -> str:
        return self.config["name"]

    def get_features(self) -> list:
        return self.config["features"]

    def get_cuda_features(self) -> list:
        return self.config["cuda_features"]

    def get_pytorch_optimizations(self) -> list:
        return self.config["pytorch_optimizations"]

    def get_triton_features(self) -> list:
        return self.config["triton_features"]

    def get_profiling_tools(self) -> list:
        return self.config["profiling_tools"]

    def _sanitize_arch_value(self, value: Optional[str]) -> Optional[str]:
        """Compatibility helper: preserve the caller's exact requested target."""
        return value

    def _set_arch_env(self, key: str, fallback: str) -> None:
        os.environ.setdefault(key, fallback)

    def _configure_arch_environment(self) -> None:
        if self.compute_capability is None or self.arch == "other":
            return
        major, minor = self.compute_capability
        self._set_arch_env("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")
        self._set_arch_env("CMAKE_CUDA_ARCHITECTURES", f"{major}{minor}")
        self._set_arch_env("CUDAARCHS", f"{major}{minor}")

    def configure_pytorch_optimizations(self) -> None:
        with suppress_known_cuda_capability_warnings(context="ArchitectureConfig.configure_pytorch_optimizations"):
            if not torch.cuda.is_available():
                return

        self._configure_arch_environment()
        # Unsupported targets must fail in their own toolchain; never pretend
        # that a 12.1 device is 12.0 or that future capabilities are SM100.

        # PyTorch Inductor configuration
        inductor = getattr(torch, "_inductor", None)
        cfg = None
        if inductor and hasattr(inductor, "config"):
            cfg = inductor.config
            # Enable PyTorch 2.10 features
            if hasattr(cfg, "triton"):
                triton_cfg = cfg.triton
                if hasattr(triton_cfg, "unique_kernel_names"):
                    triton_cfg.unique_kernel_names = True
                # Avoid automatic cudagraph wrapping to prevent RNG capture issues in setup code.
                if hasattr(triton_cfg, "cudagraph_trees"):
                    triton_cfg.cudagraph_trees = False
                if hasattr(triton_cfg, "cudagraphs"):
                    triton_cfg.cudagraphs = False
            
            # Enable max-autotune GEMM backends (PyTorch 2.10)
            # CUTLASS provides optimized GEMM kernels for NVIDIA GPUs
            if hasattr(cfg, "max_autotune_gemm_backends"):
                cfg.max_autotune_gemm_backends = "CUTLASS,TRITON,ATEN"
            
            # Enable CUTLASS for all operations
            if hasattr(cfg, "cuda") and hasattr(cfg.cuda, "cutlass_enabled_ops"):
                cfg.cuda.cutlass_enabled_ops = "all"
            
            # Enable aggressive Triton optimization for Blackwell
            if hasattr(cfg, "aggressive_fusion"):
                cfg.aggressive_fusion = True
        
        # Leave Triton's actual runtime capability untouched.
        if self.arch in ("blackwell", "blackwell_ultra", "blackwell_consumer", "grace_blackwell"):
            try:
                import triton
            except ImportError:
                triton = None

            # Configure CUTLASS for torch.compile backend
            # Fix the cutlass_dir path to point to nvidia-cutlass-dsl installation
            if hasattr(cfg, "cuda") and hasattr(cfg.cuda, "cutlass_dir"):
                try:
                    import cutlass
                    # Get the nvidia_cutlass_dsl root directory
                    cutlass_module_path = os.path.dirname(cutlass.__file__)
                    nvidia_cutlass_root = os.path.dirname(os.path.dirname(cutlass_module_path))
                    cfg.cuda.cutlass_dir = nvidia_cutlass_root
                    try:
                        cutlass_pkg_version = importlib_metadata.version("nvidia-cutlass-dsl")
                        self.cutlass_version = cutlass_pkg_version
                        if _parse_version_tuple(cutlass_pkg_version) < (4, 2, 0):
                            warnings.warn(
                                "nvidia-cutlass-dsl < 4.2 detected; upgrade recommended for full Blackwell support.",
                                RuntimeWarning,
                            )
                    except importlib_metadata.PackageNotFoundError:
                        warnings.warn(
                            "nvidia-cutlass-dsl package not found; CUTLASS kernels may be skipped.",
                            RuntimeWarning,
                        )
                except ImportError:
                    # If cutlass not installed, unset cutlass_dir
                    # PyTorch will skip CUTLASS backend
                    pass

            if "TRITON_PTXAS_PATH" not in os.environ:
                try:
                    triton_root = Path(triton.__file__).resolve().parent
                    bundled_ptxas = triton_root / "backends" / "nvidia" / "bin" / "ptxas"
                    system_ptxas = shutil.which("ptxas")
                    version_ok = False
                    if bundled_ptxas.exists():
                        try:
                            result = subprocess.run(
                                [str(bundled_ptxas), "--version"],
                                capture_output=True,
                                text=True,
                                timeout=2,
                                check=False,
                            )
                        except (subprocess.SubprocessError, OSError):
                            result = None
                        if result and "release 13." in result.stdout:
                            version_ok = True
                    if not version_ok and system_ptxas:
                        os.environ["TRITON_PTXAS_PATH"] = system_ptxas
                        if VERBOSE_EXPERIMENTAL_FEATURES:
                            print(f"Triton: selected system ptxas at {system_ptxas}; target support still requires compilation")
                except Exception as ex:
                    if VERBOSE_EXPERIMENTAL_FEATURES:
                        print(f"WARNING: Triton ptxas selection failed: {ex}")
        
        # Standard CUDA configurations
        os.environ.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")
        # Use new PYTORCH_ALLOC_CONF (preferred), fallback to legacy PYTORCH_CUDA_ALLOC_CONF
        alloc_conf = os.environ.get("PYTORCH_ALLOC_CONF")
        legacy_alloc = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        if alloc_conf is None:
            alloc_conf = legacy_alloc or "max_split_size_mb:256,expandable_segments:True"
            if VERBOSE_EXPERIMENTAL_FEATURES and legacy_alloc:
                print("Migrated PYTORCH_CUDA_ALLOC_CONF to PYTORCH_ALLOC_CONF (new API)")
        os.environ["PYTORCH_ALLOC_CONF"] = alloc_conf
        
        # Configure TF32 using the shared helper to avoid legacy/new API mixing.
        enable_tf32()

    def print_info(self) -> None:
        cfg = self.config
        print(f"Architecture: {cfg['name']}")
        print(f"Compute Capability: {cfg['compute_capability']}")
        print(f"SM Version: {cfg['sm_version']}")
        print(f"Memory Bandwidth: {cfg['memory_bandwidth']}")
        print(f"Tensor Cores: {cfg['tensor_cores']}")
        if cfg['features']:
            print(f"Features: {', '.join(cfg['features'])}")
        if cfg['cuda_features']:
            print(f"CUDA Features: {', '.join(cfg['cuda_features'])}")
        if cfg['pytorch_optimizations']:
            print(f"PyTorch Optimisations: {', '.join(cfg['pytorch_optimizations'])}")
        if cfg['triton_features']:
            print(f"Triton Features: {', '.join(cfg['triton_features'])}")
        if cfg['profiling_tools']:
            print(f"Profiling Tools: {', '.join(cfg['profiling_tools'])}")

_OPTIMIZATIONS_APPLIED = False

# Feature flags (can be disabled via environment variables)
VERBOSE_EXPERIMENTAL_FEATURES = os.environ.get("VERBOSE_EXPERIMENTAL_FEATURES", "0") == "1"
ENABLE_TRITON_PATCH = _TRITON_PATCH_ENABLED


def configure_optimizations() -> None:
    global _OPTIMIZATIONS_APPLIED
    if _OPTIMIZATIONS_APPLIED:
        return
    ArchitectureConfig().configure_pytorch_optimizations()
    ensure_triton_compat()
    _OPTIMIZATIONS_APPLIED = True
    
    # Optionally pre-warm CUDA extensions in background
    # Enable via: export PREWARM_CUDA_EXTENSIONS=1
    if os.environ.get("PREWARM_CUDA_EXTENSIONS", "0") == "1":
        try:
            from core.utils.extension_prewarm import prewarm_extensions
            prewarm_extensions(background=True)
        except ImportError as exc:
            warn_optional_component_unavailable(
                "core.utils.extension_prewarm",
                exc,
                impact="PREWARM_CUDA_EXTENSIONS=1 was requested but extension prewarm did not run",
                context="core.harness.arch_config.configure_optimizations",
            )


arch_config = ArchitectureConfig()
configure_optimizations()
