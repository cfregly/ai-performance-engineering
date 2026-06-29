"""Custom CUDA runtime integration for the NVFP4 grouped GEMM lab.

This module is the single canonical implementation path used by baseline and optimized wrappers:
- `prepare_custom_cuda`: setup-time metadata and descriptor preparation
- `custom_kernel_custom_cuda`: timed kernel launch path
"""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional, Sequence

import torch

from core.utils.extension_loader_template import load_cuda_extension
from labs.nvfp4_group_gemm.task import input_t, output_t

REPO_ROOT = Path(__file__).resolve().parents[2]

_EXT_BASE_NAME = "nvfp4_group_gemm_custom_cuda"
_SOURCE_FILE = "custom_cuda_group_gemm_kernel.cu"
_EXT: Optional[object] = None


def _get_process_extension_cache() -> dict[str, object]:
    cache_name = "_AISP_EXT_PROCESS_CACHE"
    cache = getattr(builtins, cache_name, None)
    if not isinstance(cache, dict):
        cache = {}
        setattr(builtins, cache_name, cache)
    return cache


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return int(default)
    return int(raw)


def _env_flag(name: str, default: int = 0) -> bool:
    return _env_int(name, default) != 0


def _env_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None:
        return str(default)
    value = raw.strip()
    if value == "":
        return str(default)
    return value


def _compile_config_hash(*, source_file: str, extra_cuda_cflags: Sequence[str]) -> str:
    """Stable short hash for compile-time config to avoid stale extension reuse."""
    payload = {
        "source_file": str(source_file),
        "extra_cuda_cflags": [str(flag) for flag in extra_cuda_cflags],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(encoded).hexdigest()[:12]


def _build_ab_tma_descs(
    ext: object,
    a_ptrs_cpu: list[int],
    b_ptrs_cpu: list[int],
    m_sizes_tma: list[int],
    n_sizes_tma_ab: list[int],
    k_halves: list[int],
    a_box_height: int,
    b_box_height: int,
):
    return ext.nvfp4_group_gemm_build_ab_tma_descs_cuda(
        torch.tensor(a_ptrs_cpu, dtype=torch.int64, device="cpu").contiguous(),
        torch.tensor(b_ptrs_cpu, dtype=torch.int64, device="cpu").contiguous(),
        torch.tensor(m_sizes_tma, dtype=torch.int32, device="cpu").contiguous(),
        torch.tensor(n_sizes_tma_ab, dtype=torch.int32, device="cpu").contiguous(),
        torch.tensor(k_halves, dtype=torch.int32, device="cpu").contiguous(),
        int(a_box_height),
        int(b_box_height),
    )


def _build_scale_tma_descs(
    ext: object,
    sfa_ptrs_cpu: list[int],
    sfb_ptrs_cpu: list[int],
    m_sizes_tma: list[int],
    n_sizes_tma_sf: list[int],
    k_halves: list[int],
    sfa_box_height: int,
    sfb_box_height: int,
):
    return ext.nvfp4_group_gemm_build_scale_tma_descs_cuda(
        torch.tensor(sfa_ptrs_cpu, dtype=torch.int64, device="cpu").contiguous(),
        torch.tensor(sfb_ptrs_cpu, dtype=torch.int64, device="cpu").contiguous(),
        torch.tensor(m_sizes_tma, dtype=torch.int32, device="cpu").contiguous(),
        torch.tensor(n_sizes_tma_sf, dtype=torch.int32, device="cpu").contiguous(),
        torch.tensor(k_halves, dtype=torch.int32, device="cpu").contiguous(),
        int(sfa_box_height),
        int(sfb_box_height),
    )


def load_custom_cuda_nvfp4_group_gemm(*, verbose: bool = False) -> object:
    """Load (and JIT-build if needed) the custom CUDA extension.

    Optional build namespace override:
      - ``AISP_NVFP4_GROUP_GEMM_EXT_NAME=<name>``
        Useful when running concurrent benchmark/profiling jobs.
        The effective extension name appends a deterministic compile-config hash.
    """
    global _EXT

    lab_dir = Path(__file__).resolve().parent
    source = lab_dir / _SOURCE_FILE

    extra_cuda_cflags = [
        "--std=c++17",
        "-O3",
        "--use_fast_math",
        "-lineinfo",
        "-gencode=arch=compute_100a,code=sm_100a",
        "-gencode=arch=compute_100a,code=compute_100a",
        "-gencode=arch=compute_103a,code=sm_103a",
        "-gencode=arch=compute_103a,code=compute_103a",
    ]
    # Compile-time tuning knobs (kept explicit to avoid global default drift).
    # These control constexprs in `custom_cuda_group_gemm_kernel.cu`; compile-config hashing in
    # `ext_name` below ensures knob changes trigger a distinct extension build automatically.
    pipeline_stages = os.getenv("AISP_NVFP4_GROUP_GEMM_PIPELINE_STAGES")
    if pipeline_stages is not None and pipeline_stages.strip() != "":
        extra_cuda_cflags.append(f"-DNVFP4_GROUP_GEMM_PIPELINE_STAGES={int(pipeline_stages)}")
    tmem_columns = os.getenv("AISP_NVFP4_GROUP_GEMM_TMEM_COLUMNS")
    if tmem_columns is not None and tmem_columns.strip() != "":
        extra_cuda_cflags.append(f"-DNVFP4_GROUP_GEMM_TMEM_COLUMNS={int(tmem_columns)}")
    use_utccp_64x128b = os.getenv("AISP_NVFP4_GROUP_GEMM_USE_UTCCP_64X128B")
    if use_utccp_64x128b is not None and use_utccp_64x128b.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_USE_UTCCP_64X128B={int(use_utccp_64x128b)}"
        )
    use_utccp_64x128b_sfa = os.getenv("AISP_NVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFA")
    if use_utccp_64x128b_sfa is not None and use_utccp_64x128b_sfa.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFA={int(use_utccp_64x128b_sfa)}"
        )
    use_utccp_64x128b_sfb = os.getenv("AISP_NVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFB")
    if use_utccp_64x128b_sfb is not None and use_utccp_64x128b_sfb.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_USE_UTCCP_64X128B_SFB={int(use_utccp_64x128b_sfb)}"
        )
    utccp_64x128b_schedule = os.getenv("AISP_NVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE")
    if utccp_64x128b_schedule is not None and utccp_64x128b_schedule.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_UTCCP_64X128B_SCHEDULE={int(utccp_64x128b_schedule)}"
        )
    use_utccp_128x128b_sf = os.getenv("AISP_NVFP4_GROUP_GEMM_USE_UTCCP_128X128B_SF")
    if use_utccp_128x128b_sf is not None and use_utccp_128x128b_sf.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_USE_UTCCP_128X128B_SF={int(use_utccp_128x128b_sf)}"
        )
    multicast_a = os.getenv("AISP_NVFP4_GROUP_GEMM_MULTICAST_A")
    if multicast_a is not None and multicast_a.strip() != "":
        extra_cuda_cflags.append(f"-DNVFP4_GROUP_GEMM_MULTICAST_A={int(multicast_a)}")
    multicast_sfa = os.getenv("AISP_NVFP4_GROUP_GEMM_MULTICAST_SFA")
    if multicast_sfa is not None and multicast_sfa.strip() != "":
        extra_cuda_cflags.append(f"-DNVFP4_GROUP_GEMM_MULTICAST_SFA={int(multicast_sfa)}")
    ws_unroll2_mma = os.getenv("AISP_NVFP4_GROUP_GEMM_WS_UNROLL2_MMA")
    if ws_unroll2_mma is not None and ws_unroll2_mma.strip() != "":
        extra_cuda_cflags.append(f"-DNVFP4_GROUP_GEMM_WS_UNROLL2_MMA={int(ws_unroll2_mma)}")
    ws_sfb1_segment_helpers = os.getenv("AISP_NVFP4_GROUP_GEMM_WS_SFB1_SEGMENT_HELPERS")
    if ws_sfb1_segment_helpers is not None and ws_sfb1_segment_helpers.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_WS_SFB1_SEGMENT_HELPERS={int(ws_sfb1_segment_helpers)}"
        )
    ws_nanosleep_cycles = os.getenv("AISP_NVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES")
    if ws_nanosleep_cycles is not None and ws_nanosleep_cycles.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_WS_NANOSLEEP_CYCLES={int(ws_nanosleep_cycles)}"
        )
    ws_tma_producer = os.getenv("AISP_NVFP4_GROUP_GEMM_WS_TMA_PRODUCER")
    if ws_tma_producer is not None and ws_tma_producer.strip() != "":
        extra_cuda_cflags.append(f"-DNVFP4_GROUP_GEMM_WS_TMA_PRODUCER={int(ws_tma_producer)}")
    ws_split_u0_segs = os.getenv("AISP_NVFP4_GROUP_GEMM_WS_SPLIT_U0_SEGS")
    if ws_split_u0_segs is not None and ws_split_u0_segs.strip() != "":
        extra_cuda_cflags.append(f"-DNVFP4_GROUP_GEMM_WS_SPLIT_U0_SEGS={int(ws_split_u0_segs)}")
    ws_segment_parallel = os.getenv("AISP_NVFP4_GROUP_GEMM_WS_SEGMENT_PARALLEL")
    if ws_segment_parallel is not None and ws_segment_parallel.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_WS_SEGMENT_PARALLEL={int(ws_segment_parallel)}"
        )
    cta1_commit_barrier = os.getenv("AISP_NVFP4_GROUP_GEMM_CTA1_COMMIT_BARRIER")
    if cta1_commit_barrier is not None and cta1_commit_barrier.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_CTA1_COMMIT_BARRIER={int(cta1_commit_barrier)}"
        )
    stage1_prefetch = os.getenv("AISP_NVFP4_GROUP_GEMM_STAGE1_PREFETCH")
    if stage1_prefetch is not None and stage1_prefetch.strip() != "":
        extra_cuda_cflags.append(f"-DNVFP4_GROUP_GEMM_STAGE1_PREFETCH={int(stage1_prefetch)}")
    warp0_only_mainloop = os.getenv("AISP_NVFP4_GROUP_GEMM_WARP0_ONLY_MAINLOOP")
    if warp0_only_mainloop is not None and warp0_only_mainloop.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_WARP0_ONLY_MAINLOOP={int(warp0_only_mainloop)}"
        )
    use_cutlass_tmem_sf_frg = os.getenv("AISP_NVFP4_GROUP_GEMM_USE_CUTLASS_TMEM_SF_FRG")
    if use_cutlass_tmem_sf_frg is not None and use_cutlass_tmem_sf_frg.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_USE_CUTLASS_TMEM_SF_FRG={int(use_cutlass_tmem_sf_frg)}"
        )
    cta2_sf_dp_bank = os.getenv("AISP_NVFP4_GROUP_GEMM_CTA2_SF_DP_BANK")
    if cta2_sf_dp_bank is not None and cta2_sf_dp_bank.strip() != "":
        extra_cuda_cflags.append(f"-DNVFP4_GROUP_GEMM_CTA2_SF_DP_BANK={int(cta2_sf_dp_bank)}")
    cta2_sfa_pair_segs = os.getenv("AISP_NVFP4_GROUP_GEMM_CTA2_SFA_PAIR_SEGS")
    if cta2_sfa_pair_segs is not None and cta2_sfa_pair_segs.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_CTA2_SFA_PAIR_SEGS={int(cta2_sfa_pair_segs)}"
        )
    cta2_sfa_sf_id = os.getenv("AISP_NVFP4_GROUP_GEMM_CTA2_SFA_SF_ID")
    if cta2_sfa_sf_id is not None and cta2_sfa_sf_id.strip() != "":
        extra_cuda_cflags.append(f"-DNVFP4_GROUP_GEMM_CTA2_SFA_SF_ID={int(cta2_sfa_sf_id)}")
    cta2_sfb_sf_id = os.getenv("AISP_NVFP4_GROUP_GEMM_CTA2_SFB_SF_ID")
    if cta2_sfb_sf_id is not None and cta2_sfb_sf_id.strip() != "":
        extra_cuda_cflags.append(f"-DNVFP4_GROUP_GEMM_CTA2_SFB_SF_ID={int(cta2_sfb_sf_id)}")
    cta2_accum_dp_bank = os.getenv("AISP_NVFP4_GROUP_GEMM_CTA2_ACCUM_DP_BANK")
    if cta2_accum_dp_bank is not None and cta2_accum_dp_bank.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_CTA2_ACCUM_DP_BANK={int(cta2_accum_dp_bank)}"
        )
    cta2_sfb_alloc_mode = os.getenv("AISP_NVFP4_GROUP_GEMM_CTA2_SFB_ALLOC_MODE")
    if cta2_sfb_alloc_mode is not None and cta2_sfb_alloc_mode.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_CTA2_SFB_ALLOC_MODE={int(cta2_sfb_alloc_mode)}"
        )
    cta2_sfb_pair_segs = os.getenv("AISP_NVFP4_GROUP_GEMM_CTA2_SFB_PAIR_SEGS")
    if cta2_sfb_pair_segs is not None and cta2_sfb_pair_segs.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_CTA2_SFB_PAIR_SEGS={int(cta2_sfb_pair_segs)}"
        )
    cta2_sfb_populate_u1 = os.getenv("AISP_NVFP4_GROUP_GEMM_CTA2_SFB_POPULATE_U1")
    if cta2_sfb_populate_u1 is not None and cta2_sfb_populate_u1.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_CTA2_SFB_POPULATE_U1={int(cta2_sfb_populate_u1)}"
        )
    cta2_skip_utccp_sfa = os.getenv("AISP_NVFP4_GROUP_GEMM_CTA2_SKIP_UTCCP_SFA")
    if cta2_skip_utccp_sfa is not None and cta2_skip_utccp_sfa.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_CTA2_SKIP_UTCCP_SFA={int(cta2_skip_utccp_sfa)}"
        )
    cta2_skip_utccp_sfb = os.getenv("AISP_NVFP4_GROUP_GEMM_CTA2_SKIP_UTCCP_SFB")
    if cta2_skip_utccp_sfb is not None and cta2_skip_utccp_sfb.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_CTA2_SKIP_UTCCP_SFB={int(cta2_skip_utccp_sfb)}"
        )
    cta2_utccp_rank0_only = os.getenv("AISP_NVFP4_GROUP_GEMM_CTA2_UTCCP_RANK0_ONLY")
    if cta2_utccp_rank0_only is not None and cta2_utccp_rank0_only.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_CTA2_UTCCP_RANK0_ONLY={int(cta2_utccp_rank0_only)}"
        )
    mma_lane0_all_warps = os.getenv("AISP_NVFP4_GROUP_GEMM_MMA_LANE0_ALL_WARPS")
    if mma_lane0_all_warps is not None and mma_lane0_all_warps.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_MMA_LANE0_ALL_WARPS={int(mma_lane0_all_warps)}"
        )
    cta2_mma_segment_parallel = os.getenv("AISP_NVFP4_GROUP_GEMM_CTA2_MMA_SEGMENT_PARALLEL")
    if cta2_mma_segment_parallel is not None and cta2_mma_segment_parallel.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_CTA2_MMA_SEGMENT_PARALLEL={int(cta2_mma_segment_parallel)}"
        )
    cta2_mma_all_threads = os.getenv("AISP_NVFP4_GROUP_GEMM_CTA2_MMA_ALL_THREADS")
    if cta2_mma_all_threads is not None and cta2_mma_all_threads.strip() != "":
        extra_cuda_cflags.append(
            f"-DNVFP4_GROUP_GEMM_CTA2_MMA_ALL_THREADS={int(cta2_mma_all_threads)}"
        )
    debug_stage = os.getenv("AISP_NVFP4_GROUP_GEMM_DEBUG_STAGE")
    if debug_stage is not None and debug_stage.strip() != "":
        extra_cuda_cflags.append(f"-DNVFP4_GROUP_GEMM_DEBUG_STAGE={int(debug_stage)}")
    epilogue_ld_x16 = os.getenv("AISP_NVFP4_GROUP_GEMM_EPILOGUE_LD_X16")
    if epilogue_ld_x16 is not None and epilogue_ld_x16.strip() != "":
        extra_cuda_cflags.append(f"-DNVFP4_GROUP_GEMM_EPILOGUE_LD_X16={int(epilogue_ld_x16)}")
    epilogue_ld_x32 = os.getenv("AISP_NVFP4_GROUP_GEMM_EPILOGUE_LD_X32")
    if epilogue_ld_x32 is not None and epilogue_ld_x32.strip() != "":
        extra_cuda_cflags.append(f"-DNVFP4_GROUP_GEMM_EPILOGUE_LD_X32={int(epilogue_ld_x32)}")
    maxrregcount = os.getenv("AISP_NVFP4_GROUP_GEMM_MAXRREGCOUNT")
    if maxrregcount is not None and maxrregcount.strip() != "":
        extra_cuda_cflags.append(f"--maxrregcount={int(maxrregcount)}")

    ext_base = os.getenv("AISP_NVFP4_GROUP_GEMM_EXT_NAME", _EXT_BASE_NAME).strip() or _EXT_BASE_NAME
    cfg_hash = _compile_config_hash(source_file=_SOURCE_FILE, extra_cuda_cflags=extra_cuda_cflags)
    ext_name = f"{ext_base}__cfg_{cfg_hash}"
    build_dir = REPO_ROOT / ".torch_extensions" / ext_name

    if _EXT is not None and getattr(_EXT, "__name__", None) == ext_name:
        return _EXT

    process_cache = _get_process_extension_cache()
    cached = process_cache.get(ext_name)
    if cached is not None:
        _EXT = cached
        return _EXT
    if ext_name in sys.modules:
        _EXT = sys.modules[ext_name]
        process_cache[ext_name] = _EXT
        return _EXT

    _EXT = load_cuda_extension(
        extension_name=ext_name,
        cuda_source_file=str(source),
        build_dir=build_dir,
        extra_cuda_cflags=extra_cuda_cflags,
        # We use the CUDA Driver API (cuTensorMapEncodeTiled/cuGetErrorString) for TMA descriptor
        # encoding, so we must link libcuda explicitly (torch extensions only link cudart by default).
        extra_ldflags=["-lcuda"],
        verbose=verbose,
    )
    process_cache[ext_name] = _EXT
    sys.modules[ext_name] = _EXT
    return _EXT


def _pack_scale_tiles_for_tcgen05(
    sfa_inv_u8: torch.Tensor,
    sfb_inv_u8: torch.Tensor,
    *,
    m: int,
    n: int,
    k_scales: int,
    sfb_k_major: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack reordered scale tensors into per-tile byte layouts (CUTLASS layout transliteration).

    This packs GPU MODE's reordered scale layout:
      `s*_inv_u8[tile_mn, kk, mm32, mm4, kk4]`
    into a shared-memory-friendly 2D tile that matches CUTLASS's SM100 block-scaled
    scale-factor layout (see `cutlass/detail/sm100_blockscaled_layout.hpp`,
    `Sm1xxBlockScaledConfig<SFVecSize=16>::SfKMajorAtom`).

    Concretely, for each K tile (256 FP4 elems => 16 scale factors), we form 4 blocks
    of 4 scale factors. Each block is stored as a 32x16 matrix:
      rows = mm32 (0..31)
      cols = mm4*4 + kk4 (0..15)

    We then stack the 4 blocks along rows, yielding 128 rows:
      row = seg*32 + mm32, where seg = 0..3 within the K tile.

    Output layouts match the kernel's shared-memory staging format:
      - SFA: [m_tiles, k_tiles, 128, 16]  (4 blocks * 32 rows)
      - SFB: [n_tiles, k_tiles, 128, 16]  (4 blocks * 32 rows) by default.

    For `AISP_NVFP4_GROUP_GEMM_UNROLL_N=2`, the kernel can reduce TMA transaction count by
    loading 2 adjacent N tiles per K tile. That requires SFB to be K-major across N tiles so
    (tile_n, tile_n+1) are contiguous for a fixed k_tile:
      - SFB (K-major): [k_tiles, n_tiles, 128, 16]
    """
    if sfa_inv_u8.dtype != torch.uint8 or sfb_inv_u8.dtype != torch.uint8:
        raise ValueError("Expected uint8 scale tensors for tcgen05 packing")

    device = sfa_inv_u8.device
    # Tile counts come from the actual tensor shapes. We mask out rows/cols beyond the true
    # M/N sizes so callers may provide extra padded tiles (needed for cta_group::2 bring-up).
    m_tiles_required = (m + 127) // 128
    n_tiles_required = (n + 127) // 128
    m_tiles = int(sfa_inv_u8.size(0))
    n_tiles = int(sfb_inv_u8.size(0))
    k_tiles = (k_scales + 15) // 16

    sfa_tiles = torch.empty((m_tiles, k_tiles, 128, 16), dtype=torch.uint8, device=device)
    if sfb_k_major:
        sfb_tiles = torch.empty((k_tiles, n_tiles, 128, 16), dtype=torch.uint8, device=device)
    else:
        sfb_tiles = torch.empty((n_tiles, k_tiles, 128, 16), dtype=torch.uint8, device=device)

    if sfa_inv_u8.device != sfb_inv_u8.device:
        raise ValueError("Expected SFA/SFB scales to be on the same device")
    if sfa_inv_u8.ndim != 5 or sfb_inv_u8.ndim != 5:
        raise ValueError("Expected packed scales with shape [tiles, kk, 32, 4, 4]")
    if m_tiles < m_tiles_required or n_tiles < n_tiles_required:
        raise ValueError("Scale tile dims mismatch with m/n")
    if int(sfa_inv_u8.size(2)) != 32 or int(sfa_inv_u8.size(3)) != 4 or int(sfa_inv_u8.size(4)) != 4:
        raise ValueError("Unexpected SFA layout; expected [..., 32, 4, 4]")
    if int(sfb_inv_u8.size(2)) != 32 or int(sfb_inv_u8.size(3)) != 4 or int(sfb_inv_u8.size(4)) != 4:
        raise ValueError("Unexpected SFB layout; expected [..., 32, 4, 4]")

    kk_blocks = int(sfa_inv_u8.size(1))
    if int(sfb_inv_u8.size(1)) != kk_blocks:
        raise ValueError("Expected SFA/SFB to have the same kk dimension")

    mm32 = torch.arange(32, device=device, dtype=torch.int32).view(32, 1)
    mm4 = torch.arange(4, device=device, dtype=torch.int32).view(1, 4)
    local = mm4 * 32 + mm32  # [32,4] -> 0..127

    # Scale validity mask for the last partial K tile (k_scales may not be multiple of 16).
    # Shape [k_tiles, 4, 4] where dims are (seg, kk4).
    scale_idx = (torch.arange(k_tiles, device=device, dtype=torch.int32).view(-1, 1) * 16) + torch.arange(
        16, device=device, dtype=torch.int32
    ).view(1, -1)
    scale_valid = (scale_idx < int(k_scales)).view(k_tiles, 4, 4).to(torch.uint8)  # 1 if scale exists else 0
    src = torch.empty((4, 32, 4, 4), dtype=torch.uint8, device=device)

    for mt in range(m_tiles):
        valid_m = max(0, min(128, m - mt * 128))
        m_mask = (local < valid_m).to(torch.uint8)  # [32,4]
        for kt in range(k_tiles):
            kk_base = kt * 4
            seg_avail = max(0, min(4, kk_blocks - kk_base))
            if seg_avail:
                src[:seg_avail].copy_(sfa_inv_u8[mt, kk_base : kk_base + seg_avail])
            if seg_avail < 4:
                src[seg_avail:].zero_()

            # Zero out invalid rows/cols in the MN tile.
            src *= m_mask.view(1, 32, 4, 1)
            # Zero out scale factors beyond k_scales (padding values in reordered tensor are random).
            src *= scale_valid[kt].view(4, 1, 1, 4)

            # [seg, mm32, mm4, kk4] -> [seg*32 + mm32, mm4*4 + kk4]
            sfa_tiles[mt, kt].copy_(src.contiguous().reshape(128, 16))

    for nt in range(n_tiles):
        valid_n = max(0, min(128, n - nt * 128))
        n_mask = (local < valid_n).to(torch.uint8)  # [32,4]
        for kt in range(k_tiles):
            kk_base = kt * 4
            seg_avail = max(0, min(4, kk_blocks - kk_base))
            if seg_avail:
                src[:seg_avail].copy_(sfb_inv_u8[nt, kk_base : kk_base + seg_avail])
            if seg_avail < 4:
                src[seg_avail:].zero_()

            src *= n_mask.view(1, 32, 4, 1)
            src *= scale_valid[kt].view(4, 1, 1, 4)

            # [seg, mm32, mm4, kk4] -> [seg*32 + mm32, mm4*4 + kk4]
            if sfb_k_major:
                sfb_tiles[kt, nt].copy_(src.contiguous().reshape(128, 16))
            else:
                sfb_tiles[nt, kt].copy_(src.contiguous().reshape(128, 16))

    return sfa_tiles.contiguous(), sfb_tiles.contiguous()


def _fuse_grouped_ctx_tensor(grouped_ctxs: Sequence[dict[str, Any]], key: str) -> torch.Tensor:
    first = grouped_ctxs[0][key]
    total_rows = sum(int(ctx[key].shape[0]) for ctx in grouped_ctxs)
    fused = first.new_empty((total_rows, *first.shape[1:]))
    offset = 0
    for grouped_ctx in grouped_ctxs:
        tensor = grouped_ctx[key]
        next_offset = offset + int(tensor.shape[0])
        fused[offset:next_offset].copy_(tensor)
        offset = next_offset
    return fused


def _fuse_cta_group_idx_map(grouped_ctxs: Sequence[dict[str, Any]]) -> torch.Tensor:
    first = grouped_ctxs[0]["cta_group_idx_map"]
    total_rows = sum(int(ctx["cta_group_idx_map"].shape[0]) for ctx in grouped_ctxs)
    fused = first.new_empty((total_rows, *first.shape[1:]))
    row_offset = 0
    group_offset = 0
    for grouped_ctx in grouped_ctxs:
        cta_group_idx_map = grouped_ctx["cta_group_idx_map"]
        next_row_offset = row_offset + int(cta_group_idx_map.shape[0])
        fused_slice = fused[row_offset:next_row_offset]
        fused_slice.copy_(cta_group_idx_map)
        if group_offset:
            fused_slice.add_(int(group_offset))
        row_offset = next_row_offset
        group_offset += int(grouped_ctx["a_ptrs"].numel())
    return fused


def prepare_custom_cuda(data_list: Sequence[input_t]) -> Optional[Sequence[tuple[Any, ...]]]:
    """Build pointer metadata + tensormap descriptors for the tcgen05 path (outside timed path)."""
    if not data_list:
        return None

    prepared: list[tuple[Any, ...]] = []
    for data in data_list:
        abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data

        output_refs: list[torch.Tensor] = []
        a_ptrs_cpu: list[int] = []
        b_ptrs_cpu: list[int] = []
        c_ptrs_cpu: list[int] = []
        sfa_ptrs_cpu: list[int] = []
        sfb_ptrs_cpu: list[int] = []

        m_sizes: list[int] = []
        n_sizes: list[int] = []
        # TMA descriptors are encoded against padded M/N dimensions. Keep the padded sizes explicit
        # so the descriptor heights always match the backing allocations.
        m_sizes_tma: list[int] = []
        n_sizes_tma_ab: list[int] = []
        n_sizes_tma_sf: list[int] = []
        k_halves: list[int] = []
        k_scales: list[int] = []

        # Keep tensors alive so pointers remain valid through benchmark_fn().
        a_padded_tensors: list[torch.Tensor] = []
        b_padded_tensors: list[torch.Tensor] = []
        sfa_packed_tensors: list[torch.Tensor] = []
        sfb_packed_tensors: list[torch.Tensor] = []

        # cta_group::2 loads B starting at `n_offset + 64` but still uses a 128-row tensormap box.
        # Without extra padding, rank1 will read beyond the last 128-row tile on the tail N tile.
        # Pad B's N dimension by +64 rows when experimental cta2 is enabled to keep all TMA reads in-bounds.
        cluster_dim_x = _env_int("AISP_NVFP4_GROUP_GEMM_CLUSTER_DIM_X", 1)
        enable_cta2 = _env_int("AISP_NVFP4_GROUP_GEMM_ENABLE_EXPERIMENTAL_CTA2", 0)
        use_cta2 = (cluster_dim_x == 2) and (enable_cta2 != 0)
        unroll_n = _env_int("AISP_NVFP4_GROUP_GEMM_UNROLL_N", 1)
        if unroll_n not in (1, 2):
            raise ValueError(f"AISP_NVFP4_GROUP_GEMM_UNROLL_N must be 1 or 2, got {unroll_n}")
        cta2_partition_b = _env_int("AISP_NVFP4_GROUP_GEMM_CTA2_PARTITION_B", 1)
        # UnrollN=2 relies on a different (K-major) packed SFB layout and 256-row TMA loads.
        # This is now supported for both cta_group::1 and experimental cta_group::2 launches.
        # Extra B padding is not required for our current cta_group::2 bring-up modes.
        extra_b_rows = 0

        for (a, b, c), (sfa_cpu, _sfb_cpu), (sfa_reordered, sfb_reordered) in zip(
            abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors
        ):
            if a.dim() != 3 or b.dim() != 3 or c.dim() != 3:
                raise ValueError("Expected A/B/C tensors with shape [M|N, K/2|N, 1]")
            if a.size(2) != 1 or b.size(2) != 1 or c.size(2) != 1:
                raise ValueError("Only l=1 inputs are supported in custom CUDA path")

            # A/B are float4_e2m1fn_x2 views over packed bytes.
            a_u8 = a[:, :, 0].view(torch.uint8)
            b_u8 = b[:, :, 0].view(torch.uint8)
            c_out = c[:, :, 0]
            if not a_u8.is_contiguous() or not b_u8.is_contiguous() or not c_out.is_contiguous():
                raise ValueError("Expected contiguous A/B/C views in custom CUDA path")

            m = int(a_u8.size(0))
            n = int(b_u8.size(0))
            k_bytes = int(a_u8.size(1))  # packed bytes == K/2
            if k_bytes <= 0:
                raise ValueError("Invalid K/2 bytes")

            # TMA descriptor encoding uses padded M/N dims.
            # Keep the historical 256-row M padding contract for descriptor/input allocation.
            m_padded = ((m + 255) // 256) * 256
            # UnrollN=2 optimization: pad N to a multiple of 256 so (tile_n, tile_n+1) always stays
            # in-bounds for 256-row TMA scale loads (SFB). The second N128 tile on the tail is
            # zero-padded and never computed (tile_n+1 >= n_tiles_group), but keeping it in-bounds
            # avoids OOB reads when the tensormap box height is 256.
            #
            # For UnrollN=1 keep the canonical N128 padding.
            if unroll_n == 2:
                n_padded_sf = ((n + 255) // 256) * 256
            else:
                n_padded_sf = ((n + 127) // 128) * 128
            n_padded_ab = n_padded_sf + extra_b_rows

            a_pad = torch.empty((m_padded, k_bytes), dtype=torch.uint8, device="cuda")
            a_pad[:m, :].copy_(a_u8)
            if m_padded > m:
                a_pad[m:, :].zero_()
            b_pad = torch.empty((n_padded_ab, k_bytes), dtype=torch.uint8, device="cuda")
            b_pad[:n, :].copy_(b_u8)
            if n_padded_ab > n:
                b_pad[n:, :].zero_()

            # Scale factors: use the GPU MODE-reordered float8 tensors, but invert the permute
            # to get a contiguous rank-5 layout [mm, kk, 32, 4, 4] (l=1 dropped).
            # Original reorder layout: [mm32, mm4, mm, kk4, kk, l]
            sfa_inv = sfa_reordered.permute(5, 2, 4, 0, 1, 3).contiguous()[0]
            sfb_inv = sfb_reordered.permute(5, 2, 4, 0, 1, 3).contiguous()[0]

            # tcgen05 block-scaled UMMA consumes raw FP8 bytes directly from TMEM.
            # Canonicalize to float8_e4m3fn encoding in setup() so the timed path stays pure kernel time.
            if sfa_inv.dtype != torch.float8_e4m3fn:
                sfa_inv = sfa_inv.to(dtype=torch.float16).to(dtype=torch.float8_e4m3fn)
            if sfb_inv.dtype != torch.float8_e4m3fn:
                sfb_inv = sfb_inv.to(dtype=torch.float16).to(dtype=torch.float8_e4m3fn)

            sfa_inv_u8 = sfa_inv.view(torch.uint8)
            sfb_inv_u8 = sfb_inv.view(torch.uint8)
            k_scale = int(sfa_cpu.size(1))

            # Pad the tile dimension of the reordered SFA tensor so scale packing (and the scale
            # tensormap height) matches the padded M used by A's tensormap.
            m_tiles_actual = int(sfa_inv_u8.size(0))
            m_tiles_tma = m_padded // 128
            if m_tiles_tma < m_tiles_actual:
                raise ValueError("Internal error: padded m_tiles is smaller than reordered tensor tiles")
            if m_tiles_tma != m_tiles_actual:
                padded = torch.empty((m_tiles_tma,) + tuple(sfa_inv_u8.shape[1:]), dtype=torch.uint8, device="cuda")
                padded[:m_tiles_actual].copy_(sfa_inv_u8)
                padded[m_tiles_actual:].zero_()
                sfa_inv_u8 = padded

            # Likewise, pad SFB's tile dimension so its tensormap height (based on padded N) matches
            # the backing allocation. This is required for UnrollN=2 when N tiles are rounded up to
            # an even count (multiple of 256 rows).
            n_tiles_actual = int(sfb_inv_u8.size(0))
            n_tiles_tma = n_padded_sf // 128
            if n_tiles_tma < n_tiles_actual:
                raise ValueError("Internal error: padded n_tiles is smaller than reordered tensor tiles")
            if n_tiles_tma != n_tiles_actual:
                padded = torch.empty((n_tiles_tma,) + tuple(sfb_inv_u8.shape[1:]), dtype=torch.uint8, device="cuda")
                padded[:n_tiles_actual].copy_(sfb_inv_u8)
                padded[n_tiles_actual:].zero_()
                sfb_inv_u8 = padded

            # For UnrollN=2 we keep K-major SFB packing so adjacent N tiles are contiguous
            # for each K tile in both cta_group::1 and cta_group::2 paths.
            use_sfb_k_major = unroll_n == 2

            sfa_packed, sfb_packed = _pack_scale_tiles_for_tcgen05(
                sfa_inv_u8,
                sfb_inv_u8,
                m=m,
                n=n,
                k_scales=k_scale,
                sfb_k_major=use_sfb_k_major,
            )

            a_padded_tensors.append(a_pad)
            b_padded_tensors.append(b_pad)
            sfa_packed_tensors.append(sfa_packed)
            sfb_packed_tensors.append(sfb_packed)

            output_refs.append(c)
            a_ptrs_cpu.append(int(a_pad.data_ptr()))
            b_ptrs_cpu.append(int(b_pad.data_ptr()))
            c_ptrs_cpu.append(int(c_out.data_ptr()))
            sfa_ptrs_cpu.append(int(sfa_packed.data_ptr()))
            sfb_ptrs_cpu.append(int(sfb_packed.data_ptr()))

            m_sizes.append(m)
            n_sizes.append(n)
            m_sizes_tma.append(m_padded)
            n_sizes_tma_ab.append(n_padded_ab)
            n_sizes_tma_sf.append(n_padded_sf)
            k_halves.append(k_bytes)
            # Scale factors are [M|N, K/16, 1] in reference layout; use that as K_scales.
            k_scales.append(k_scale)

        # GPU MODE-style packed CTA mapping: avoid launching max(M_tiles)*max(N_tiles) for groups with small M/N.
        # Setup-time only (not timed). We build explicit CTA->(group,tile_m,tile_n) maps and launch a
        # single linear grid of exactly `total_ctas` blocks with grid=(1,1,total_ctas).
        cta_order = _env_str("AISP_NVFP4_GROUP_GEMM_CTA_ORDER", "tm_major").lower()
        if cta_order not in ("tm_major", "tn_major"):
            raise ValueError(
                "AISP_NVFP4_GROUP_GEMM_CTA_ORDER must be one of: 'tm_major' (default), 'tn_major'"
            )
        cta_group_idx_map: list[int] = []
        cta_tile_m_map: list[int] = []
        cta_tile_n_map: list[int] = []
        for gi, (m, n) in enumerate(zip(m_sizes, n_sizes)):
            m_tiles = (int(m) + 127) // 128
            n_tiles = (int(n) + 127) // 128
            if cta_order == "tm_major":
                # Order CTAs by (group, tile_m, tile_n). This keeps adjacent N-CTAs for
                # the same M tile contiguous and aligns with cluster/multicast assumptions.
                for tm in range(m_tiles):
                    if int(m) - tm * 128 <= 0:
                        continue
                    for tn in range(0, n_tiles, unroll_n):
                        cta_group_idx_map.append(int(gi))
                        cta_tile_m_map.append(int(tm))
                        cta_tile_n_map.append(int(tn))
            else:
                # Order CTAs by (group, tile_n, tile_m) to improve temporal reuse of B/SFB
                # across M tiles when N is large.
                for tn in range(0, n_tiles, unroll_n):
                    for tm in range(m_tiles):
                        if int(m) - tm * 128 <= 0:
                            continue
                        cta_group_idx_map.append(int(gi))
                        cta_tile_m_map.append(int(tm))
                        cta_tile_n_map.append(int(tn))

        cta_group_idx_map_t = torch.tensor(cta_group_idx_map, dtype=torch.int32, device="cuda").contiguous()
        cta_tile_m_map_t = torch.tensor(cta_tile_m_map, dtype=torch.int32, device="cuda").contiguous()
        cta_tile_n_map_t = torch.tensor(cta_tile_n_map, dtype=torch.int32, device="cuda").contiguous()

        # Build tensormap descriptors on the GPU (outside timed path).
        ext = load_custom_cuda_nvfp4_group_gemm()
        # Keep descriptor contracts aligned with the runtime-selected CTA2 partition mode.
        cta2_partition_b_eff = cta2_partition_b

        if not use_cta2:
            # UnrollN=2 optimization: load both adjacent N128 tiles in one TMA transaction.
            b_box_height = 256 if unroll_n == 2 else 128
        elif cta2_partition_b_eff == 1:
            # UnrollN=1 partitioned-B mode: each CTA rank loads only N/2 rows.
            b_box_height = 64
        elif unroll_n == 2:
            # cta_group::2 + UnrollN=2 unpartitioned path issues one 256-row B transaction.
            b_box_height = 256
        else:
            b_box_height = 128
        # cta_group::2 uses a 64-row A tile per CTA (cluster tile M=128). cta_group::1 uses 128.
        a_box_height = 64 if use_cta2 else 128
        a_descs, b_descs = _build_ab_tma_descs(
            ext=ext,
            a_ptrs_cpu=a_ptrs_cpu,
            b_ptrs_cpu=b_ptrs_cpu,
            m_sizes_tma=m_sizes_tma,
            n_sizes_tma_ab=n_sizes_tma_ab,
            k_halves=k_halves,
            a_box_height=int(a_box_height),
            b_box_height=int(b_box_height),
        )
        # Scale-factor tensor maps:
        # SFA remains 128-row in all modes.
        # For UnrollN=2 mode0 (unpartitioned-B), keep SFB on a 256-row tensormap so u=0/u=1
        # share one contiguous packed tile in shared memory.
        # For CTA2 mode1 (partitioned-N), keep the established 128-row SFB tensormap contract.
        sfa_box_height = 128
        if use_cta2 and cta2_partition_b_eff == 1:
            sfb_box_height = 128
        else:
            sfb_box_height = 256 if unroll_n == 2 else 128
        sfb_box_height_override = _env_int("AISP_NVFP4_GROUP_GEMM_CTA2_SFB_BOX_HEIGHT_OVERRIDE", 0)
        if sfb_box_height_override > 0:
            sfb_box_height = int(sfb_box_height_override)
        sfa_descs, sfb_descs = _build_scale_tma_descs(
            ext=ext,
            sfa_ptrs_cpu=sfa_ptrs_cpu,
            sfb_ptrs_cpu=sfb_ptrs_cpu,
            m_sizes_tma=m_sizes_tma,
            n_sizes_tma_sf=n_sizes_tma_sf,
            k_halves=k_halves,
            sfa_box_height=int(sfa_box_height),
            sfb_box_height=int(sfb_box_height),
        )

        grouped_ctx = {
            "a_ptrs": torch.tensor(a_ptrs_cpu, dtype=torch.int64, device="cuda").contiguous(),
            "b_ptrs": torch.tensor(b_ptrs_cpu, dtype=torch.int64, device="cuda").contiguous(),
            "c_ptrs": torch.tensor(c_ptrs_cpu, dtype=torch.int64, device="cuda").contiguous(),
            "sfa_ptrs": torch.tensor(sfa_ptrs_cpu, dtype=torch.int64, device="cuda").contiguous(),
            "sfb_ptrs": torch.tensor(sfb_ptrs_cpu, dtype=torch.int64, device="cuda").contiguous(),
            "m_sizes": torch.tensor(m_sizes, dtype=torch.int32, device="cuda").contiguous(),
            "n_sizes": torch.tensor(n_sizes, dtype=torch.int32, device="cuda").contiguous(),
            "k_halves": torch.tensor(k_halves, dtype=torch.int32, device="cuda").contiguous(),
            "k_scales": torch.tensor(k_scales, dtype=torch.int32, device="cuda").contiguous(),
            "a_descs": a_descs,
            "b_descs": b_descs,
            "sfa_descs": sfa_descs,
            "sfb_descs": sfb_descs,
            "cta_group_idx_map": cta_group_idx_map_t,
            "cta_tile_m_map": cta_tile_m_map_t,
            "cta_tile_n_map": cta_tile_n_map_t,
            "max_m": int(max(m_sizes)) if m_sizes else 0,
            "max_n": int(max(n_sizes)) if n_sizes else 0,
        }

        ctx = {
            "output_refs": output_refs,
            "grouped_ctx": grouped_ctx,
            "a_padded_tensors": a_padded_tensors,
            "b_padded_tensors": b_padded_tensors,
            "sfa_packed_tensors": sfa_packed_tensors,
            "sfb_packed_tensors": sfb_packed_tensors,
        }
        prepared.append((abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes, ctx))

    # Optional launch optimization: fuse the full inputs-per-iteration request set into one
    # grouped tcgen05 launch. Workload is unchanged (same requests and same tensors), but
    # per-iteration launch overhead is reduced by issuing one large grouped kernel.
    if _env_flag("AISP_NVFP4_GROUP_GEMM_FUSE_INPUTS", 1) and len(prepared) > 1:
        compress_fused_inputs = _env_flag("AISP_NVFP4_GROUP_GEMM_FUSE_INPUTS_COMPRESS_LIST", 1)
        grouped_ctxs = [entry[4]["grouped_ctx"] for entry in prepared]
        cat_keys = (
            "a_ptrs",
            "b_ptrs",
            "c_ptrs",
            "sfa_ptrs",
            "sfb_ptrs",
            "m_sizes",
            "n_sizes",
            "k_halves",
            "k_scales",
            "a_descs",
            "b_descs",
            "sfa_descs",
            "sfb_descs",
            "cta_tile_m_map",
            "cta_tile_n_map",
        )
        fused_grouped_ctx: dict[str, torch.Tensor | int] = {
            key: _fuse_grouped_ctx_tensor(grouped_ctxs, key) for key in cat_keys
        }

        # CTA->group map must be offset per input so each CTA points at the correct flattened group.
        fused_grouped_ctx["cta_group_idx_map"] = _fuse_cta_group_idx_map(grouped_ctxs)
        fused_grouped_ctx["max_m"] = int(max(int(ctx["max_m"]) for ctx in grouped_ctxs))
        fused_grouped_ctx["max_n"] = int(max(int(ctx["max_n"]) for ctx in grouped_ctxs))

        for fused_input_idx, entry in enumerate(prepared):
            ctx = entry[4]
            ctx["fused_grouped_ctx"] = fused_grouped_ctx
            ctx["fused_input_idx"] = int(fused_input_idx)

        if compress_fused_inputs:
            # Optional host-overhead reduction: keep only one logical callsite when fused mode is enabled.
            # The single fused launch still executes all original requests in the same iteration.
            # Keep strong references for all fused slots so pointer-backed tensors remain alive.
            primary_ctx = prepared[0][4]
            primary_ctx["fused_keepalive_ctxs"] = [entry[4] for entry in prepared]
            return [prepared[0]]

    return prepared


def _launch_tcgen05_grouped(ext: object, grouped_ctx: dict[str, Any]) -> None:
    ext.nvfp4_group_gemm_forward_grouped_tcgen05_cuda(
        grouped_ctx["a_ptrs"],
        grouped_ctx["b_ptrs"],
        grouped_ctx["sfa_ptrs"],
        grouped_ctx["sfb_ptrs"],
        grouped_ctx["c_ptrs"],
        grouped_ctx["m_sizes"],
        grouped_ctx["n_sizes"],
        grouped_ctx["k_halves"],
        grouped_ctx["k_scales"],
        grouped_ctx["a_descs"],
        grouped_ctx["b_descs"],
        grouped_ctx["sfa_descs"],
        grouped_ctx["sfb_descs"],
        grouped_ctx["cta_group_idx_map"],
        grouped_ctx["cta_tile_m_map"],
        grouped_ctx["cta_tile_n_map"],
        int(grouped_ctx["max_m"]),
        int(grouped_ctx["max_n"]),
    )


def custom_kernel_custom_cuda(data: tuple[Any, ...]) -> output_t:
    """Run the grouped tcgen05 kernel (timed path).

    Optional launch-mode knob:
      - `AISP_NVFP4_GROUP_GEMM_CLUSTER_DIM_X=2` enables cluster launch via cudaLaunchKernelEx.
      - `AISP_NVFP4_GROUP_GEMM_ENABLE_EXPERIMENTAL_CTA2=1` (with cluster_dim_x=2) enables the
        in-progress cta_group::2 path; default cluster mode remains correctness-safe cta_group::1.
      - `AISP_NVFP4_GROUP_GEMM_TMA_L2_PROMOTION` controls TMA L2 promotion policy for all A/B/SF
        tensor maps (setup-time descriptor encoding):
          0 = NONE (default),
          1 = L2_64B,
          2 = L2_128B,
          3 = L2_256B.
      - Experimental cta2 mapping overrides (rows): `AISP_NVFP4_GROUP_GEMM_CTA2_A_ROW_OFFSET`,
        `AISP_NVFP4_GROUP_GEMM_CTA2_B_ROW_OFFSET`, `AISP_NVFP4_GROUP_GEMM_CTA2_SFA_ROW_OFFSET`,
        `AISP_NVFP4_GROUP_GEMM_CTA2_EPILOGUE_ROW_BASE`.
      - Experimental cta2 TMEM epilogue address mode:
        `AISP_NVFP4_GROUP_GEMM_CTA2_EPILOGUE_ADDR_MODE`:
          0 = CUTLASS `tmem_frg_2sm<float>` mapping (default),
          1 = naive dp=m_local, col=n,
          2 = alternate rank-folded candidate.
      - Experimental cta2 SFB-slot mapping mode:
        `AISP_NVFP4_GROUP_GEMM_CTA2_SFB_SLOT_MODE`:
          0 = even-slot mode,
          1 = rank-based even/odd candidate,
          2 = odd-slot candidate.
    """
    if len(data) < 5:
        raise ValueError("custom_kernel_custom_cuda requires prepare_custom_cuda() output")
    ctx = data[4]
    grouped_ctx = ctx["grouped_ctx"]
    fused_grouped_ctx = ctx.get("fused_grouped_ctx")
    fused_input_idx = int(ctx.get("fused_input_idx", 0))

    ext = load_custom_cuda_nvfp4_group_gemm()
    if fused_grouped_ctx is not None:
        # Execute the fused launch only once per timed loop (on the first input slot); all C
        # tensors for every slot are updated by that single grouped call.
        if fused_input_idx == 0:
            _launch_tcgen05_grouped(ext, fused_grouped_ctx)
    else:
        _launch_tcgen05_grouped(ext, grouped_ctx)
    return ctx["output_refs"]


__all__ = [
    "custom_kernel_custom_cuda",
    "load_custom_cuda_nvfp4_group_gemm",
    "prepare_custom_cuda",
]
