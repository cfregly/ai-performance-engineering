"""
Self-contained tcgen05 kernel loader for the Matching cuBLAS lab.

This module JIT-compiles the tcgen05 GEMM kernels without depending on
any other chapter or common code.

ONLY includes working kernels that exist in this directory.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
from functools import lru_cache
from pathlib import Path
import sys

import torch
from torch.utils.cpp_extension import _get_build_directory, load

_LAB_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _LAB_DIR.parents[1]

# CUTLASS include paths
_CUTLASS_CANDIDATES = [
    _REPO_ROOT / "third_party" / "cutlass" / "include",
    _REPO_ROOT / "third_party" / "TransformerEngine" / "3rdparty" / "cutlass" / "include",
]


def _find_cutlass_include() -> Path | None:
    """Find CUTLASS include directory."""
    for cand in _CUTLASS_CANDIDATES:
        if cand.exists():
            return cand
    return None


def _get_cuda_flags() -> list[str]:
    """Get CUDA compiler flags for tcgen05."""
    flags = ["-std=c++20"]
    
    cutlass_inc = _find_cutlass_include()
    if cutlass_inc:
        flags.append(f"-I{cutlass_inc}")
    else:
        raise RuntimeError("CUTLASS include directory not found.")
    
    major, minor = torch.cuda.get_device_capability()
    if major == 10 and minor >= 3:
        # Blackwell Ultra (GB300, sm_103). sm_100a cubins are arch-locked and give
        # "no kernel image is available" on sm_103, so target sm_103a explicitly.
        flags.append("-gencode=arch=compute_103a,code=sm_103a")
    elif major >= 10:
        flags.append("-gencode=arch=compute_100a,code=sm_100a")
    else:
        raise RuntimeError(f"tcgen05 requires SM 10.0+ (Blackwell). Got SM {major}.{minor}")
    
    return flags


def _load_kernel(source_file: Path, name_prefix: str, extra_cuda_flags: tuple[str, ...] = ()):
    """Generic kernel loader with caching."""
    if not source_file.exists():
        raise FileNotFoundError(f"{source_file.name} not found in {_LAB_DIR}")

    cuda_flags = _get_cuda_flags() + list(extra_cuda_flags)
    src_hash = hashlib.md5(
        source_file.read_bytes() + "|".join(extra_cuda_flags).encode()
    ).hexdigest()[:8]
    build_name = f"{name_prefix}_{src_hash}"
    build_dir = Path(_get_build_directory(build_name, verbose=False))
    shared_object = build_dir / f"{build_name}.so"

    if shared_object.exists():
        loaded = sys.modules.get(build_name)
        if loaded is not None and getattr(loaded, "__file__", None) == str(shared_object):
            return loaded

        spec = importlib.util.spec_from_file_location(build_name, shared_object)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to import cached extension {shared_object}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[build_name] = module
        spec.loader.exec_module(module)
        return module
    
    print(f"  [Compiling {source_file.name} (first time only)...]")
    module = load(
        name=build_name,
        sources=[str(source_file)],
        extra_cuda_cflags=cuda_flags,
        extra_cflags=["-std=c++20"],
        extra_ldflags=["-lcuda"],
        verbose=False,
    )
    return module


# =============================================================================
# Stage 2: Basic tcgen05
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_module():
    """JIT-compile the basic tcgen05 GEMM kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_gemm.cu", "lab_tcgen05")


def matmul_tcgen05(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM: C = A @ B^T"""
    return load_tcgen05_module().matmul_tcgen05(a, b)


# =============================================================================
# Stage 3: 2-Stage Pipeline
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_pipelined_module():
    """JIT-compile the 2-stage pipelined kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_pipelined.cu", "lab_tcgen05_pipelined")


def matmul_tcgen05_pipelined(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute 2-stage pipelined tcgen05 GEMM: C = A @ B^T"""
    return load_tcgen05_pipelined_module().matmul_tcgen05_pipelined(a, b)


# =============================================================================
# Stage 4: 3-Stage Pipeline
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_3stage_module():
    """JIT-compile the 3-stage pipelined kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_3stage.cu", "lab_tcgen05_3stage")


def matmul_tcgen05_3stage(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute 3-stage pipelined tcgen05 GEMM: C = A @ B^T"""
    return load_tcgen05_3stage_module().matmul_tcgen05_3stage(a, b)


# =============================================================================
# Stage 5: Swizzled Tiles
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_swizzled_module():
    """JIT-compile the swizzled tile scheduling kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_swizzled.cu", "lab_tcgen05_swizzled")


def matmul_tcgen05_swizzled(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute swizzled tcgen05 GEMM: C = A @ B^T"""
    return load_tcgen05_swizzled_module().matmul_tcgen05_swizzled(a, b)


# =============================================================================
# Stage 6: Cluster (2x1) 
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_cluster_module():
    """JIT-compile the cluster launch kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_cluster.cu", "lab_tcgen05_cluster")


def matmul_tcgen05_cluster(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute tcgen05 GEMM with 2x1 cluster: C = A @ B^T"""
    return load_tcgen05_cluster_module().matmul_tcgen05_cluster(a, b)


# =============================================================================
# Stage 7: 4-Stage Deep Pipeline
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_warp_spec_module():
    """JIT-compile the 4-stage warp-specialized kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_warp_spec.cu", "lab_tcgen05_warp_spec")


def matmul_tcgen05_warp_spec(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute 4-stage deep pipelined tcgen05 GEMM: C = A @ B^T"""
    return load_tcgen05_warp_spec_module().matmul_tcgen05_warp_spec(a, b)


# =============================================================================
# Stage 8: No-Wait Pattern (KEY BREAKTHROUGH!)
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_no_wait_module():
    """JIT-compile the no-wait pattern kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_no_wait.cu", "lab_tcgen05_no_wait")


def matmul_tcgen05_no_wait(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute no-wait tcgen05 GEMM: C = A @ B^T
    
    KEY OPTIMIZATION: Don't wait for MMA barrier after each k-tile!
    +43% performance improvement.
    """
    return load_tcgen05_no_wait_module().matmul_tcgen05_no_wait(a, b)


# =============================================================================
# Stage 9: No-Wait + Swizzle
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_no_wait_swizzle_module():
    """JIT-compile the no-wait swizzled kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_no_wait_swizzle.cu", "lab_tcgen05_no_wait_swizzle")


def matmul_tcgen05_no_wait_swizzle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute no-wait + swizzled tcgen05 GEMM: C = A @ B^T"""
    return load_tcgen05_no_wait_swizzle_module().matmul_tcgen05_no_wait_swizzle(a, b)


# =============================================================================
# Stage 10: TMA Before Wait (Warp Parallel)
# =============================================================================

@lru_cache(maxsize=1)
def load_tcgen05_warp_parallel_module():
    """JIT-compile the warp-parallel kernel."""
    return _load_kernel(_LAB_DIR / "tcgen05_warp_parallel.cu", "lab_tcgen05_warp_parallel")


def matmul_tcgen05_warp_parallel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute warp-parallel tcgen05 GEMM: C = A @ B^T

    Issues next TMA before waiting for current one.
    """
    return load_tcgen05_warp_parallel_module().matmul_tcgen05_warp_parallel(a, b)


# =============================================================================
# Stage 13: Dual-CTA Occupancy (2 CTAs/SM)
# =============================================================================

@lru_cache(maxsize=None)
def _load_tcgen05_dual_cta_module(tile_n: int, stages: int, cluster_m: int = 1):
    extra = (
        f"-DDUAL_TILE_N={tile_n}",
        f"-DDUAL_STAGES={stages}",
        f"-DDUAL_CLUSTER_M={cluster_m}",
    )
    return _load_kernel(
        _LAB_DIR / "tcgen05_dual_cta.cu",
        f"lab_tcgen05_dual_cta_n{tile_n}s{stages}c{cluster_m}",
        extra_cuda_flags=extra,
    )


def load_tcgen05_dual_cta_module():
    """JIT-compile the dual-CTA (2 CTAs/SM) occupancy kernel.

    Tunables (env, read at first load):
      AISP_DUAL_TILE_N: MMA tile N (default 256; 256-col fp32 acc in TMEM)
      AISP_DUAL_STAGES: smem pipeline stages (default 2; ~96KB/CTA)
      AISP_DUAL_CLUSTER_M: 1 (default, plain launch) or 2/4 = (M,1,1)
        cluster + TMA multicast of B across the cluster (E3 lever vs the
        long_scoreboard TMA-latency stall; B L2->SM traffic / cluster_m)
    Defaults are the measured-best config from the GB300 sweep (2026-06-10,
    GPU 2): (256,2) = 838-915us vs (128,3) = 1050-1109us; see
    docs/gb300-gemm-occupancy-rewrite.md. Both CTAs/SM fit because TMEM
    (2x256 of 512 cols) and smem (2x~96KB of 227KB) leave room for a
    co-resident CTA, unlike the cluster kernel's full-TMEM 192KB footprint.
    """
    tile_n = int(os.environ.get("AISP_DUAL_TILE_N", "256"))
    stages = int(os.environ.get("AISP_DUAL_STAGES", "2"))
    cluster_m = int(os.environ.get("AISP_DUAL_CLUSTER_M", "1"))
    return _load_tcgen05_dual_cta_module(tile_n, stages, cluster_m)


def matmul_tcgen05_dual_cta(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute dual-CTA (2 CTAs/SM) tcgen05 GEMM: C = A @ B^T

    KEY OPTIMIZATION: half-size TMEM allocation (128 cols) + ~96KB smem +
    early tcgen05 alloc-permit release, so TWO CTAs co-reside per SM and
    cover each other's TMA latency.
    """
    return load_tcgen05_dual_cta_module().matmul_tcgen05_dual_cta(a, b)


# =============================================================================
# Stage 14: 2-SM UMMA pair (tcgen05 cta_group::2) on the dual-CTA footprint
# =============================================================================

@lru_cache(maxsize=None)
def _load_tcgen05_dual_cta_2sm_module(tile_n: int, stages: int, warp_split: int = 0, amcast: int = 0,
                                      min_blocks: int = 2, tile_k: int = 64,
                                      epi_overlap: int = 0, epi_atom: int = 32):
    extra = (
        f"-DDUAL2SM_TILE_N={tile_n}",
        f"-DDUAL2SM_STAGES={stages}",
        f"-DDUAL2SM_WARP_SPLIT={warp_split}",
        f"-DDUAL2SM_AMCAST={amcast}",
        f"-DDUAL2SM_MIN_BLOCKS={min_blocks}",
        f"-DDUAL2SM_TILE_K={tile_k}",
        f"-DDUAL2SM_EPI_OVERLAP={epi_overlap}",
        f"-DDUAL2SM_EPI_ATOM={epi_atom}",
    )
    return _load_kernel(
        _LAB_DIR / "tcgen05_dual_cta_2sm.cu",
        f"lab_tcgen05_dual_cta_2sm_n{tile_n}s{stages}w{warp_split}a{amcast}"
        f"mb{min_blocks}k{tile_k}eo{epi_overlap}ea{epi_atom}",
        extra_cuda_flags=extra,
    )


def load_tcgen05_dual_cta_2sm_module():
    """JIT-compile the 2-SM UMMA pair (cta_group::2) kernel.

    Fuses the dual-CTA pair into ONE 256-wide SM100_MMA_F16BF16_2x1SM_SS
    per (pair, k-block): the even cluster rank issues the MMA for both SMs
    (halved instruction/barrier issue), the odd rank free-runs as a pure
    TMA producer. Per-CTA smem/TMEM footprint matches the plain dual-CTA
    (256,2) config, so 2 CTAs/SM remains reachable.

    Tunables (env, read at first load):
      AISP_DUAL2SM_TILE_N: MMA tile N (default 128; pair tile is 256xN)
      AISP_DUAL2SM_STAGES: smem stages (default 3; 24KB/CTA/stage at N=128)
      AISP_DUAL2SM_WARP_SPLIT: 1 = whole-warp producer/consumer split of
        the leader's mainloop (V-front lever a; DEFAULT after the V2-front
        A/B: 875.3us vs 892.3us round-robin, 12/12 order-alternated paired
        wins, median 1.0725x hot; set 0 for the U-front single-warp base)
      AISP_DUAL2SM_AMCAST: 1 = (2,2,1) cluster + TMA-multicast of A across
        the cluster N mode (V-front lever b); 0 = (2,1,1) cluster
      AISP_DUAL2SM_MIN_BLOCKS: __launch_bounds__ min-CTAs/SM hint (default
        2 = 255-reg budget; 4 caps regs at 128 for a 4th co-resident CTA on
        tile_n=64 footprints -- V3-front lever c)
      AISP_DUAL2SM_TILE_K: K-extent per pipeline stage (default 64; 128
        doubles the TMA box and halves barrier round-trips per fed byte --
        V3-front lever d)
      AISP_DUAL2SM_EPI_OVERLAP: 0 (default, off) or 2|4|8 = cross-tile TMEM
        double-buffered epilogue overlap (V4-front lever e): each cluster
        walks that many consecutive n-tiles with 2 TMEM acc buffers and a
        SECOND warpgroup (256 threads/CTA) draining buffer (t%2) while the
        MMA stream fills buffer ((t+1)%2). TMEM 2x128=256 cols/CTA -> TWO
        CTAs/SM (vs the incumbent's three); WARP_SPLIT is ignored (the
        overlap mainloop is always producer/consumer split).
      AISP_DUAL2SM_EPI_ATOM: t2r column-repeat of the overlap epilogue's
        chunked drain (default 32; B55 atom-width trap knob)
    Defaults are the measured-best config from the GB300 U-front session
    (2026-06-11, GPU 2, 8192^3): (128,3) = 867-895us / 33.2-33.8% SoL,
    16/16 interleaved-rep wins vs plain dual (256,2). ncu: 152 regs/thread
    and 72KB smem/CTA -> Block Limits 3/3 -> THREE CTAs/SM (TMEM 3x128 of
    512 cols); per-CTA B traffic halves because the 2x1SM atom splits B
    N/2-per-CTA across the pair. (256,2) 2SM ties plain dual: 255 regs cap
    it at 2 CTAs/SM and TMEM 2x256 is an exact fit.
    """
    tile_n = int(os.environ.get("AISP_DUAL2SM_TILE_N", "128"))
    stages = int(os.environ.get("AISP_DUAL2SM_STAGES", "3"))
    warp_split = int(os.environ.get("AISP_DUAL2SM_WARP_SPLIT", "1"))
    amcast = int(os.environ.get("AISP_DUAL2SM_AMCAST", "0"))
    min_blocks = int(os.environ.get("AISP_DUAL2SM_MIN_BLOCKS", "2"))
    tile_k = int(os.environ.get("AISP_DUAL2SM_TILE_K", "64"))
    epi_overlap = int(os.environ.get("AISP_DUAL2SM_EPI_OVERLAP", "0"))
    epi_atom = int(os.environ.get("AISP_DUAL2SM_EPI_ATOM", "32"))
    return _load_tcgen05_dual_cta_2sm_module(tile_n, stages, warp_split, amcast, min_blocks, tile_k,
                                             epi_overlap, epi_atom)


def matmul_tcgen05_dual_cta_2sm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute 2-SM UMMA pair tcgen05 GEMM: C = A @ B^T."""
    return load_tcgen05_dual_cta_2sm_module().matmul_tcgen05_dual_cta_2sm(a, b)
