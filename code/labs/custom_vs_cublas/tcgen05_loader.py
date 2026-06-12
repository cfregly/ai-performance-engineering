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
                                      epi_overlap: int = 0, epi_atom: int = 32,
                                      persist: int = 0, raster_gm: int = 0,
                                      prefetch: int = 0, pf_issuer: int = 0,
                                      tma_epi: int = 0):
    extra = (
        f"-DDUAL2SM_TILE_N={tile_n}",
        f"-DDUAL2SM_STAGES={stages}",
        f"-DDUAL2SM_WARP_SPLIT={warp_split}",
        f"-DDUAL2SM_AMCAST={amcast}",
        f"-DDUAL2SM_MIN_BLOCKS={min_blocks}",
        f"-DDUAL2SM_TILE_K={tile_k}",
        f"-DDUAL2SM_EPI_OVERLAP={epi_overlap}",
        f"-DDUAL2SM_EPI_ATOM={epi_atom}",
        f"-DDUAL2SM_PERSIST={persist}",
        f"-DDUAL2SM_RASTER_GM={raster_gm}",
        f"-DDUAL2SM_PREFETCH={prefetch}",
        f"-DDUAL2SM_PF_ISSUER={pf_issuer}",
        f"-DDUAL2SM_TMA_EPI={tma_epi}",
    )
    return _load_kernel(
        _LAB_DIR / "tcgen05_dual_cta_2sm.cu",
        f"lab_tcgen05_dual_cta_2sm_n{tile_n}s{stages}w{warp_split}a{amcast}"
        f"mb{min_blocks}k{tile_k}eo{epi_overlap}ea{epi_atom}p{persist}rg{raster_gm}"
        f"pf{prefetch}pi{pf_issuer}te{tma_epi}",
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
        3 = 170-reg budget for the persistent 3-CTAs/SM default; 2 = the
        255-reg budget of the non-persistent champion; 4 caps regs at 128
        for a 4th co-resident CTA on tile_n=64 footprints -- V3 lever c)
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
      AISP_DUAL2SM_PERSIST: 1 = persistent clusters (V5-front lever f):
        the grid is exactly the co-residable cluster count and cluster r
        walks raster indices r, r+C, r+2C...; composes with EPI_OVERLAP=2
        (V4 double-TMEM inner loop, 2 CTAs/SM, use STAGES=4 per the B63
        sizing law) or EPI_OVERLAP=0 (champion 3 CTAs/SM, per-round
        epilogue, use MIN_BLOCKS=3 to hold the 170-reg budget). Default 1
        (flipped by the V6 ratification; see below).
      AISP_DUAL2SM_RASTER_GM: GROUP_M tile-raster group size in pair-rows
        (V5-front lever f; 0 = off). Groups of gm pair-rows sweep n
        together so the in-flight window keeps gm A row-panels L2-resident
        while B col-panels stream (8 -> 32MiB A-window at 8192^3). Without
        PERSIST this flattens the launch grid to 1D and relies on
        ascending-blockIdx rasterization. Default 8 (flipped by the V6
        ratification; see below).
      AISP_DUAL2SM_PREFETCH: raster-aware L2 prefetch depth in leading
        k-stages of the NEXT round's A/B panels (V6-front lever g; scoped
        to PERSIST + EPI_OVERLAP=0, whose smem ring refills cold at every
        round boundary). 0 = off (default). Issued via the TMA atoms'
        cp.async.bulk.prefetch.tensor on the same descriptors.
      AISP_DUAL2SM_PF_ISSUER: prefetch issue point (default 0): 0 =
        producer warp after the round's last TMA issue; 1 = epilogue
        warp 2 after the round's MMAs complete; 2 = parked warp 3 at
        round start (max lead, eviction risk).
      AISP_DUAL2SM_TMA_EPI: 1 = TMA-store epilogue through the drained
        ring stage (V7-front lever h; scoped to PERSIST + EPI_OVERLAP=0,
        EPI_ATOM=32): each round's D chunks are staged t2r -> swizzled
        st.shared into smem_A[c % stages] (free at the round boundary) ->
        ONE cp.async.bulk.tensor.2d store per 128x32 chunk, replacing the
        per-thread scoreboarded st.global path (50% sector efficiency,
        structural to the 32dp32b t2r layout). Default 1 (flipped by the
        V7 ratification: 1.0905x and 1.0863x paired medians, 12/12 +
        12/12 order-alternated wins vs te=0 at 8192^3; L2 D-write sectors
        16.78M -> 8.39M = the exact floor; tensor-pipe +9.5pts; see
        docs/gb300-gemm-occupancy-rewrite.md V7). Set 0 for the B65/V6
        st.global epilogue.
    Defaults are the B65/V5 persistent+raster winner, RATIFIED by the V6
    front (2026-06-11, GPU 2, 8192^3, fresh session): (128,3,ws=1,mb=3,
    p=1,rg=8) = 882.9us median vs the prior (128,3,ws=1,mb=2) champion's
    928.5us -- 1.0562x paired median, 12/12 order-alternated wins
    (replicating B65's 1.0598x, 12/12; see
    docs/gb300-gemm-occupancy-rewrite.md V5/V6). MIN_BLOCKS=3 holds the
    170-reg budget for 3 CTAs/SM under the persistent per-round epilogue;
    for the NON-persistent champion shape set PERSIST=0 RASTER_GM=0
    MIN_BLOCKS=2 (the pre-V6 default, 152 regs / 3 CTAs/SM). PREFETCH
    defaults 0: the V6 sweep measured it a tie at best (1.0021, 7/12 at
    depth 2/issuer 0) -- the round-boundary refill is already
    latency-covered by co-resident CTAs. V7 (2026-06-11) adds TMA_EPI=1
    on top (the final default: 128,3,ws=1,mb=3,p=1,rg=8,te=1): 826.9-
    830.3us median / 69.7-70.0% real SoL vs the te=0 incumbent's 898.5-
    906.5us / ~64%, paired 1.0905x + 1.0863x, 24/24 order-alternated wins
    across two sessions, rel_err 0.0 at 4 sizes.
    """
    tile_n = int(os.environ.get("AISP_DUAL2SM_TILE_N", "128"))
    stages = int(os.environ.get("AISP_DUAL2SM_STAGES", "3"))
    warp_split = int(os.environ.get("AISP_DUAL2SM_WARP_SPLIT", "1"))
    amcast = int(os.environ.get("AISP_DUAL2SM_AMCAST", "0"))
    min_blocks = int(os.environ.get("AISP_DUAL2SM_MIN_BLOCKS", "3"))
    tile_k = int(os.environ.get("AISP_DUAL2SM_TILE_K", "64"))
    epi_overlap = int(os.environ.get("AISP_DUAL2SM_EPI_OVERLAP", "0"))
    epi_atom = int(os.environ.get("AISP_DUAL2SM_EPI_ATOM", "32"))
    persist = int(os.environ.get("AISP_DUAL2SM_PERSIST", "1"))
    raster_gm = int(os.environ.get("AISP_DUAL2SM_RASTER_GM", "8"))
    prefetch = int(os.environ.get("AISP_DUAL2SM_PREFETCH", "0"))
    pf_issuer = int(os.environ.get("AISP_DUAL2SM_PF_ISSUER", "0"))
    tma_epi = int(os.environ.get("AISP_DUAL2SM_TMA_EPI", "1"))
    return _load_tcgen05_dual_cta_2sm_module(tile_n, stages, warp_split, amcast, min_blocks, tile_k,
                                             epi_overlap, epi_atom, persist, raster_gm,
                                             prefetch, pf_issuer, tma_epi)


def matmul_tcgen05_dual_cta_2sm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute 2-SM UMMA pair tcgen05 GEMM: C = A @ B^T."""
    return load_tcgen05_dual_cta_2sm_module().matmul_tcgen05_dual_cta_2sm(a, b)


# =============================================================================
# Stage 15: FP8 (e4m3) port of the 2-SM UMMA pair kernel (Front F8)
# =============================================================================

@lru_cache(maxsize=None)
def _load_tcgen05_dual_2sm_fp8_module(tile_n: int, stages: int, warp_split: int = 1,
                                      min_blocks: int = 3, tile_k: int = 128,
                                      epi_atom: int = 32,
                                      persist: int = 1, raster_gm: int = 8,
                                      tma_epi: int = 1, d_half: int = 0,
                                      epi_overlap: int = 0):
    extra = (
        f"-DDUAL2SM_TILE_N={tile_n}",
        f"-DDUAL2SM_STAGES={stages}",
        f"-DDUAL2SM_WARP_SPLIT={warp_split}",
        "-DDUAL2SM_AMCAST=0",
        f"-DDUAL2SM_MIN_BLOCKS={min_blocks}",
        f"-DDUAL2SM_TILE_K={tile_k}",
        f"-DDUAL2SM_EPI_OVERLAP={epi_overlap}",
        f"-DDUAL2SM_EPI_ATOM={epi_atom}",
        f"-DDUAL2SM_PERSIST={persist}",
        f"-DDUAL2SM_RASTER_GM={raster_gm}",
        "-DDUAL2SM_PREFETCH=0",
        "-DDUAL2SM_PF_ISSUER=0",
        f"-DDUAL2SM_TMA_EPI={tma_epi}",
        f"-DDUAL2SM_D_HALF={d_half}",
    )
    return _load_kernel(
        _LAB_DIR / "tcgen05_dual_2sm_fp8.cu",
        f"lab_tcgen05_dual_2sm_fp8_n{tile_n}s{stages}w{warp_split}"
        f"mb{min_blocks}k{tile_k}ea{epi_atom}p{persist}rg{raster_gm}te{tma_epi}dh{d_half}"
        f"eo{epi_overlap}",
        extra_cuda_flags=extra,
    )


def load_tcgen05_dual_2sm_fp8_module():
    """JIT-compile the FP8 (e4m3 x e4m3 -> fp32 accum) 2-SM UMMA pair kernel.

    Type-level port of the FP16 champion (tcgen05_dual_cta_2sm.cu) to dense
    FP8 via SM100_MMA_F8F6F4_2x1SM_SS: atom K-extent 32, kTileK default 128
    so the per-stage byte footprint (24KB at TILE_N=128) and the 3-CTAs/SM
    occupancy math are IDENTICAL to the FP16 winner while barrier
    round-trips per fed byte halve. No block scales (dense GEMM lab).

    Tunables (env, read at first load; defaults = the MEASURED FP8 champion
    after the F8b in-kernel-fp16-D ratification (2026-06-11 GPU2 12-rep
    interleaves x2: 313.5-315.0us / ~3500 TF at 8192^3 = 0.907-0.908x
    cuBLASLt FP8; the B72 dh0 incumbent was 377.6us / 0.7228x; see
    docs/gb300-fp8-dual2sm.md) -- the n256 big tile BEATS the FP16
    champion geometry n128 at FP8 rates):
      AISP_DUAL2SM_FP8_TILE_N (256), AISP_DUAL2SM_FP8_STAGES (3),
      AISP_DUAL2SM_FP8_WARP_SPLIT (1), AISP_DUAL2SM_FP8_MIN_BLOCKS (2),
      AISP_DUAL2SM_FP8_TILE_K (128; 64 selects the SW64 smem atom and
      12KB/stage -> stages up to 6 at the same 72KB ring),
      AISP_DUAL2SM_FP8_EPI_ATOM (32), AISP_DUAL2SM_FP8_PERSIST (1),
      AISP_DUAL2SM_FP8_RASTER_GM (8), AISP_DUAL2SM_FP8_TMA_EPI (1; requires
      TILE_K=128 -- the 16KB fp32 staging chunk must fit a drained A stage),
      AISP_DUAL2SM_FP8_D_HALF (1 = in-kernel fp16 D through the TMA_EPI
      staging path: D-store bytes halve and the host-side fp32->fp16
      conversion kernel is retired entirely; requires TMA_EPI=1. DEFAULT 1
      by the F8b ratification (2026-06-11, GPU 2, 8192^3): paired 1.2527x
      and 1.2525x vs the dh0 champion, 24/24 interleaved wins across two
      sessions, 313.5-315.0 us = 0.907-0.908x of same-run cuBLASLt FP8,
      rel_err 0.0 exactly at 2048/4096/8192 -- in-kernel RN rounding is
      bit-identical to torch .to(fp16) on the exact dataset),
      AISP_DUAL2SM_FP8_EPI_OVERLAP (0; 2 = the F8c in-CTA epilogue/mainloop
      overlap: persistent eo=2 structure -- 2x TILE_N-col TMEM accumulator
      buffers + a dedicated epilogue WARPGROUP (256 threads/CTA) draining
      buffer t%2 through the staged fp16 TMA store while the consumer
      fills buffer (t+1)%2. Requires TILE_N=128 with TMA_EPI=1 + D_HALF=1
      (2x128 TMEM cols/CTA = the n256 champion's 256-col budget, so
      2 CTAs/SM is preserved; n256 double-buffering needs the whole TMEM
      and is inadmissible). Dedicated 32KB staging pair in smem -- the
      eo=2 ring never drains, so the eo=0 drained-stage trick is out).
    Note: MIN_BLOCKS=1 selects the F8b deep-ring 1-CTA/SM build variant
    (lifts the 110KB smem cap to the 227KB per-block opt-in; the grid
    static-sizes to 152). Measured a LOSS at every depth s4/s5/s6 vs the
    s3/mb2 2-CTAs/SM champion (0.86x/0.93x/0.95x paired, 0/36 wins), so
    mb stays 2; see docs/gb300-fp8-dual2sm.md F8b for the mechanism.
    """
    tile_n = int(os.environ.get("AISP_DUAL2SM_FP8_TILE_N", "256"))
    stages = int(os.environ.get("AISP_DUAL2SM_FP8_STAGES", "3"))
    warp_split = int(os.environ.get("AISP_DUAL2SM_FP8_WARP_SPLIT", "1"))
    min_blocks = int(os.environ.get("AISP_DUAL2SM_FP8_MIN_BLOCKS", "2"))
    tile_k = int(os.environ.get("AISP_DUAL2SM_FP8_TILE_K", "128"))
    epi_atom = int(os.environ.get("AISP_DUAL2SM_FP8_EPI_ATOM", "32"))
    persist = int(os.environ.get("AISP_DUAL2SM_FP8_PERSIST", "1"))
    raster_gm = int(os.environ.get("AISP_DUAL2SM_FP8_RASTER_GM", "8"))
    tma_epi = int(os.environ.get("AISP_DUAL2SM_FP8_TMA_EPI", "1"))
    d_half = int(os.environ.get("AISP_DUAL2SM_FP8_D_HALF", "1"))
    epi_overlap = int(os.environ.get("AISP_DUAL2SM_FP8_EPI_OVERLAP", "0"))
    return _load_tcgen05_dual_2sm_fp8_module(tile_n, stages, warp_split, min_blocks, tile_k,
                                             epi_atom, persist, raster_gm, tma_epi, d_half,
                                             epi_overlap)


def matmul_tcgen05_dual_2sm_fp8(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Execute FP8 2-SM UMMA pair GEMM: C = A @ B^T (e4m3 in, fp16 out)."""
    return load_tcgen05_dual_2sm_fp8_module().matmul_tcgen05_dual_2sm_fp8(a, b)


def _load_tcgen05_dual_2sm_nvfp4_module(tile_n: int = 128, stages: int = 3,
                                        min_blocks: int = 2, tile_k: int = 256,
                                        raster_gm: int = 4, tma_epi: int = 1,
                                        d_half: int = 1, epi_atom: int = 32):
    """JIT-compile the NVFP4 (block-scaled e2m1, SF_VEC=16 ue4m3) 2-SM port.

    Front N4b: SM100_MMA_MXF4_2x1SM_SS (tcgen05.mma.cta_group::2.kind::
    mxf4nvf4.block_scale.block16, atom K=64) on the FP8 champion's
    persistent eo=0 structure. SFA/SFB ride the same stage ring
    (TMA -> smem -> UTCCP 2cta -> TMEM); TMEM alloc is 256 cols/CTA
    (acc 128 + SF; pow2 alloc law) -> 2 CTAs/SM. kTileN=128 only (B75:
    n256 would need a 512-col alloc = 1 CTA/SM, inadmissible).
    kTileK=256 -> SW128 e2m1 smem atom (A 16KB + B 8KB + SFA 2KB + SFB 2KB
    per stage); 128 -> SW64. te1+dh1 carried verbatim from the FP8 champion.

    Defaults = the N4b-ratified champion (2026-06-12, GPU 1, 12-rep
    interleaves at 8192^3, exact dataset rel_err == 0.0 at 2048/4096/8192):
    n128 s3 mb2 k256 rg4 te1 dh1 = 219.8us median = 5003 TFLOPS = 58.0% of
    the warmed cuBLASLt NVFP4 ceiling (127.5us / 8622 TF; B61 method) =
    0.594x cuBLASLt. Beats the FP8 champion's 313.5us ABSOLUTE time at the
    same FLOP count (1.43x). Sweep: rg4 > rg16 > rg8 (1.02x, 12/12); s2
    0.77x; k128 (SW64, te0) ~0.54x; see docs/gb300-nvfp4-dual2sm.md.
    """
    extra = (
        f"-DDUAL2SM_TILE_N={tile_n}",
        f"-DDUAL2SM_STAGES={stages}",
        f"-DDUAL2SM_MIN_BLOCKS={min_blocks}",
        f"-DDUAL2SM_TILE_K={tile_k}",
        "-DDUAL2SM_PERSIST=1",
        f"-DDUAL2SM_RASTER_GM={raster_gm}",
        f"-DDUAL2SM_EPI_ATOM={epi_atom}",
        f"-DDUAL2SM_TMA_EPI={tma_epi}",
        f"-DDUAL2SM_D_HALF={d_half}",
    )
    return _load_kernel(
        _LAB_DIR / "tcgen05_dual_2sm_nvfp4.cu",
        f"lab_tcgen05_dual_2sm_nvfp4_n{tile_n}s{stages}mb{min_blocks}k{tile_k}"
        f"rg{raster_gm}ea{epi_atom}te{tma_epi}dh{d_half}",
        extra_cuda_flags=extra,
    )


def matmul_tcgen05_dual_2sm_nvfp4(a: torch.Tensor, b: torch.Tensor,
                                  sfa: torch.Tensor, sfb: torch.Tensor) -> torch.Tensor:
    """NVFP4 2-SM pair GEMM: D = (A*SFA) @ (B*SFB)^T.

    a/b: packed uint8 [rows, k/2] (2 e2m1 per byte, low nibble first);
    sfa/sfb: float8_e4m3fn flat in the cuBLASLt 128x4-blocked layout
    (same convention as torch._scaled_mm NVFP4). fp16 out.
    """
    return _load_tcgen05_dual_2sm_nvfp4_module().matmul_tcgen05_dual_2sm_nvfp4(a, b, sfa, sfb)
