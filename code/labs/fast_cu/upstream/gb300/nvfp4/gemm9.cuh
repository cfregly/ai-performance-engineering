#pragma once
// r9: assign A reuse by L2 side and select the visit order per shape.
// This is a complete, directly editable ladder snapshot.
// It intentionally duplicates the implementation in the other gemm*.cuh files.

// Conditional branches are already materialized; edit this implementation directly.
#define NVFP4_BUILD_NAME "r9"
// Read-only metadata for main.cu; these do not select code in this header.
#define NVFP4_HAS_K64_TAIL 1
#define NVFP4_HAS_SCHEDULE 1

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(expr) do {                                                  \
    cudaError_t _e = (expr);                                                   \
    if (_e != cudaSuccess) {                                                   \
        std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                   \
                     cudaGetErrorName(_e), __FILE__, __LINE__,                 \
                     cudaGetErrorString(_e));                                  \
        std::abort();                                                          \
    }                                                                          \
} while (0)

#define CU_CHECK(expr) do {                                                    \
    CUresult _e = (expr);                                                      \
    if (_e != CUDA_SUCCESS) {                                                  \
        const char* _s = nullptr;                                              \
        cuGetErrorString(_e, &_s);                                             \
        std::fprintf(stderr, "CU error at %s:%d: %s\n",                        \
                     __FILE__, __LINE__, _s ? _s : "?");                       \
        std::abort();                                                          \
    }                                                                          \
} while (0)


// ============================================================================
// PTX helpers. Thin zero-cost wrappers over the exact instruction forms the
// kernel needs — nothing generic, nothing speculative.
// ============================================================================
namespace ptx {

// Generic pointer -> 32-bit `.shared` window byte address.
template <typename T>
static __device__ __forceinline__ uint32_t to_shared(T* ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// Local `.shared` address -> the same offset in CTA `cta_rank`'s smem
// (`.shared::cluster` instructions take this mapped form).
static __device__ __forceinline__ uint32_t mapa_shared_cluster(
        uint32_t local_addr, uint32_t cta_rank) {
    uint32_t mapped;
    asm("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(mapped) : "r"(local_addr), "r"(cta_rank));
    return mapped;
}

static __device__ __forceinline__ void bar_sync(uint32_t id, uint32_t count) {
    asm volatile("barrier.cta.sync %0, %1;" :: "r"(id), "r"(count));
}

// Cluster barrier with release/acquire — the publish boundary between the
// mbarrier inits and any cross-CTA use of those mbarriers.
static __device__ __forceinline__ void cluster_sync_rel_acq() {
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
}

// %cluster_ctarank: 0 or 1 in this kernel's (2,1,1) clusters.
static __device__ __forceinline__ uint32_t cluster_rank() {
    uint32_t r;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
    return r;
}

static __device__ __forceinline__ uint32_t lane_id() {
    uint32_t r;
    asm("mov.u32 %0, %%laneid;" : "=r"(r));
    return r;
}

// ---- mbarriers -------------------------------------------------------------

static __device__ __forceinline__ void mbar_init(uint64_t* bar, uint32_t count) {
    asm volatile("mbarrier.init.shared.b64 [%0], %1;"
                 :: "r"(to_shared(bar)), "r"(count));
}

// Publish the inits to the async proxy + the peer CTA before any TMA
// completes on them.
static __device__ __forceinline__ void fence_mbarrier_init_release_cluster() {
    asm volatile("fence.mbarrier_init.release.cluster;");
}

// Cross-CTA arrive with release ordering (teardown handshake).
static __device__ __forceinline__ void mbar_arrive_cluster_release(
        uint64_t* bar, uint32_t cta_rank) {
    const uint32_t mapped = mapa_shared_cluster(to_shared(bar), cta_rank);
    asm volatile("mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0], 1;"
                 :: "r"(mapped));
}

// Fused `tcgen05.wait::ld` + "buffer free" arrive, one asm block. The arrive
// must not become visible while TMEM loads are still in flight (it frees the
// accumulator buffer to the next tile's MMA — the bring-up chapter's
// drain overlap); the fused block
// keeps the arrive pinned behind the wait. (An earlier form also manufactured
// a register dependency on a drained value, `dep * 0` folded into the
// address; verified dead on nvcc 13.1 — ptxas const-folds it, SASS
// byte-identical without it — and retired.)
static __device__ __forceinline__ void tcgen05_wait_ld_then_arrive_release(
        uint64_t* bar, uint32_t cta_rank, bool do_arrive) {
    const uint32_t mapped = mapa_shared_cluster(to_shared(bar), cta_rank);
    asm volatile(
        "{\n\t"
        ".reg .pred %%pdo;\n\t"
        "setp.ne.s32 %%pdo, %1, 0;\n\t"
        "tcgen05.wait::ld.sync.aligned;\n\t"
        "@%%pdo mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0], 1;\n\t"
        "}\n\t"
        :: "r"(mapped), "r"((int)do_arrive) : "memory");
}


// Blocking parity wait at the consuming site (producer ring/tail waits +
// teardown). The 10 ms suspend hint is the polite form: a full-rate spin
// competes for issue slots and measurably loses; a nanosleep-backoff spin
// measured the same as this hint across five windows, so the simpler form
// ships. The carried one-stage-ahead try_wait peeks that used to gate these
// waits measured as pure overhead and were removed.
static __device__ __forceinline__ void mbar_wait_parity(
        uint64_t* bar, uint32_t parity) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 p, [%0], %1, 10000000;\n\t"
        "@!p bra WAIT_%=;\n\t}\n"
        :: "r"(to_shared(bar)), "r"(parity));
}

// ---- TMA -------------------------------------------------------------------

// Producer expect_tx arrive on the cluster-shared ring barrier: both
// peers' transfer bytes account against the even CTA's barrier, so one
// barrier per ring slot covers the whole cluster-wide gather. (Mechanism:
// `& 0xFEFFFFFF` clears bit 24 of the shared address — the hardware's
// peer-CTA bit — re-pointing the arrive at the even CTA's copy.)
static __device__ __forceinline__ void mbar_arrive_expect_tx_cluster(
        uint64_t* bar, uint32_t bytes) {
    const uint32_t mbar_addr = to_shared(bar) & 0xFEFFFFFFu;
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes));
}

// 3D tiled load, cta_group::2 + multicast (the A feed; the mask is the
// issuing CTA itself in a (2,1) cluster). Completion routes to the even
// CTA's mbarrier via the bit-24 clear.
static __device__ __forceinline__ void cp_async_bulk_tensor_3d_load_multicast(
        uint32_t dst_smem, const CUtensorMap* tmap,
        int32_t x, int32_t y, int32_t z, uint64_t* bar,
        uint16_t multicast_mask) {
    const uint32_t mbar_addr = to_shared(bar) & 0xFEFFFFFFu;
    asm volatile(
        "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%4, %5, %6}], [%2], %3;"
        :: "r"(dst_smem), "l"(tmap), "r"(mbar_addr), "h"(multicast_mask),
           "r"(x), "r"(y), "r"(z)
        : "memory");
}

// 3D tiled load, cta_group::2, no multicast (the B feed — each peer loads
// its own 128-wide N-block).
static __device__ __forceinline__ void cp_async_bulk_tensor_3d_load_2sm_bit24(
        uint32_t dst_smem, const CUtensorMap* tmap,
        int32_t x, int32_t y, int32_t z, uint64_t* bar) {
    const uint32_t mbar_addr = to_shared(bar) & 0xFEFFFFFFu;
    asm volatile(
        "cp.async.bulk.tensor.3d.cta_group::2.shared::cluster.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst_smem), "l"(tmap), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr)
        : "memory");
}

// 4D tiled load, cta_group::2 + multicast (the scale-factor feed). The
// 512-byte SF tiles of one ring slot form a 4D box; `w` selects the slot.
static __device__ __forceinline__ void cp_async_bulk_tensor_4d_load_multicast(
        uint32_t dst_smem, const CUtensorMap* tmap,
        int32_t x, int32_t y, int32_t z2, int32_t w, uint64_t* bar,
        uint16_t multicast_mask) {
    const uint32_t mbar_addr = to_shared(bar) & 0xFEFFFFFFu;
    asm volatile(
        "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%4, %5, %6, %7}], [%2], %3;"
        :: "r"(dst_smem), "l"(tmap), "r"(mbar_addr), "h"(multicast_mask),
           "r"(x), "r"(y), "r"(z2), "r"(w)
        : "memory");
}


// ---- tcgen05 (tensor memory + MMA plumbing) ---------------------------------

static __device__ __forceinline__ void tcgen05_alloc_2sm(
        uint32_t smem_addr_for_taddr, uint32_t n_cols) {
    asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;"
                 :: "r"(smem_addr_for_taddr), "r"(n_cols));
}

static __device__ __forceinline__ void tcgen05_dealloc_2sm(
        uint32_t taddr, uint32_t n_cols) {
    asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;"
                 :: "r"(taddr), "r"(n_cols));
}

static __device__ __forceinline__ void tcgen05_relinquish_2sm() {
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
}

// The widest single-instruction TMEM read: 32 b32 registers per lane
// (blog: the epilogue — a quarter the load/wait count of the narrow forms).
static __device__ __forceinline__ void tcgen05_ld_32x32b_x32(
        uint32_t taddr, uint32_t* r) {
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x32.b32 "
                 " {%0, %1, %2, %3, %4, %5, %6, %7,"
                 "  %8, %9, %10, %11, %12, %13, %14, %15,"
                 "  %16, %17, %18, %19, %20, %21, %22, %23,"
                 "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]),
                   "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]),
                   "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15]),
                   "=r"(r[16]), "=r"(r[17]), "=r"(r[18]), "=r"(r[19]),
                   "=r"(r[20]), "=r"(r[21]), "=r"(r[22]), "=r"(r[23]),
                   "=r"(r[24]), "=r"(r[25]), "=r"(r[26]), "=r"(r[27]),
                   "=r"(r[28]), "=r"(r[29]), "=r"(r[30]), "=r"(r[31])
                 : "r"(taddr));
}

// ---- stores + conversion -----------------------------------------------------

static __device__ __forceinline__ uint32_t cvt_pack_f16x2(float a, float b) {
    uint32_t d;
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(d) : "f"(a), "f"(b));
    return d;
}

// L2::evict_first requires a 256-bit store, but the 128-bit tail can still
// adopt the no-L1-allocation part of the policy at this rung.
static __device__ __forceinline__ void st_global_na_b128(
        uint4* ptr, uint4 v) {
    asm volatile("st.global.L1::no_allocate.v4.b32 "
                 "[%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
}

// Add the output cache policy to the 256-bit store: bypass L1 and evict the
// write-once line first from L2 so it does not displace reusable operands.
static __device__ __forceinline__ void st_global_na_ef_b256(
        void* ptr, uint4 lo, uint4 hi) {
    asm volatile("st.global.L1::no_allocate.L2::evict_first.v8.b32 [%0], "
                 "{%1, %2, %3, %4, %5, %6, %7, %8};"
                 :: "l"(ptr), "r"(lo.x), "r"(lo.y), "r"(lo.z), "r"(lo.w),
                    "r"(hi.x), "r"(hi.y), "r"(hi.z), "r"(hi.w));
}


// ---- MMA descriptors ---------------------------------------------------------

enum class Major : uint8_t { K = 0, MN = 1 };
enum class ScaleFormat : uint8_t { E4M3 = 0, E8M0 = 1 };

// Source descriptor for the tcgen05.cp that stages scale factors from smem
// into TMEM: unswizzled, SBO=128, LBO=16. The low word later carries the
// per-slot byte offset (in 16-byte units).
__host__ __device__ static __forceinline__ constexpr uint64_t
mma_sf_cp_src_desc_cd(uint32_t sf_smem_addr) {
    uint64_t addr_field = uint64_t((sf_smem_addr >> 4) & 0x3FC0u);
    return addr_field | 0x400800010000ull;
}

// Operand descriptor for a K=96 (48-byte) slice inside a 128-byte-swizzled
// smem K-window. OR-constant 0x4010404000000000: version=1, absolute-address
// LBO mode (bit 52), 128B swizzle, SBO=1024. Both the address field and the
// absolute-LBO field encode the window's own 1024-aligned base; the
// intra-window offset advances the address field only.
__host__ __device__ static __forceinline__ constexpr uint64_t
mma_smem_desc_k96_sw128(uint32_t window_base_1024aligned, int intra_window_off = 0) {
    uint64_t addr_field = uint64_t(((window_base_1024aligned >> 4) & 0x7FC0u)
                                   + uint32_t(intra_window_off >> 4));
    uint64_t lbo_field  = uint64_t((window_base_1024aligned << 12) & 0x7FC00000u);
    return addr_field | lbo_field | 0x4010404000000000ull;
}

// Instruction descriptor for tcgen05.mma.kind::mxf4nvf4.block_scale.
// The K extent is selected by BIT 31, not by the asm mnemonic: k_dim=0 runs
// K=64 (0x10400480), k_dim=1 runs K=96 (0x90400480, sm_103a-only) — the
// cheapest quarter of the gap if your kernel is built on the K=64 form
// (blog: the MMA warp). Bit 13 flips which TMEM scale-factor column the MMA reads.
__host__ __device__ static __forceinline__ constexpr uint32_t
mma_inst_desc_mxf4nvf4_block16(
        uint32_t M, uint32_t N,
        uint32_t tmem_sfa_addr, uint32_t tmem_sfb_addr,
        Major a_major, Major b_major,
        bool negate_a, bool negate_b,
        bool k_dim, ScaleFormat sf) {
    constexpr uint32_t MXF4_E2M1 = 1u;
    uint32_t d = 0;
    d |= ((tmem_sfb_addr & 0xC0000000u) >> 30) << 4;   // b_sf_id
    d |= MXF4_E2M1 << 7;                               // a_format = E2M1
    d |= MXF4_E2M1 << 10;                              // b_format = E2M1
    if (negate_a) d |= 1u << 13;
    if (negate_b) d |= 1u << 14;
    d |= (static_cast<uint32_t>(a_major) & 0x1u) << 15;
    d |= (static_cast<uint32_t>(b_major) & 0x1u) << 16;
    d |= ((N >> 3) & 0x3Fu) << 17;                     // N in units of 8
    d |= (static_cast<uint32_t>(sf) & 0x1u) << 23;     // 0 = UE4M3
    d |= ((M >> 4) & 0x1Fu) << 24;                     // m_dim at bits 27-28
    d |= ((tmem_sfa_addr & 0xC0000000u) >> 30) << 29;  // a_sf_id
    if (k_dim) d |= 1u << 31;                          // 1 = K=96
    return d;
}

}  // namespace ptx

// ============================================================================
// Kernel geometry from the blog's opening kernel section. One tile shape,
// one cluster shape:
// 256x256 output per 2-CTA cluster (128 rows per CTA), K advanced in
// 96-element MMA steps staged as 768-element K-groups of 8 steps, persistent
// grid of 76 clusters = 152 CTAs = one per SM.
// ============================================================================
namespace nvfp4 {

constexpr int CTA_GROUP    = 2;      // cta_group::2 — the MMA spans both CTAs
constexpr int BLOCK_M      = 128;    // A rows per CTA
constexpr int BLOCK_N      = 256;    // output tile N extent (per cluster)
constexpr int BLOCK_N_PEER = BLOCK_N / CTA_GROUP;   // 128 — B cols per CTA
constexpr int CLUSTER_M    = CTA_GROUP * BLOCK_M;   // 256 — tile M extent
[[maybe_unused]] constexpr int MMA_K = 96;  // K per tcgen05.mma (idesc bit 31)
constexpr int WIN_BYTES    = 128;    // one 128B-swizzled smem K-window
constexpr int AB_LOAD_BYTES = WIN_BYTES;
constexpr int AB_TMA_SUBS   = 3;
constexpr int TB_SIZE      = 224;    // 7 warps (blog: the warp table)
constexpr int N_EPI_WARPS  = 4;      // warps 0-3

constexpr int SF_TMEM_COLS = 9 * 4;  // 3 SFA + 6 SFB cp tiles, 4 cols each
// Two 256-wide accumulators use a 220-column stride and overlap by the 36
// columns needed for scale factors.
constexpr int D_STRIDE     = 220;
constexpr int D_DB         = 2;
constexpr int N_D_COLS     = D_STRIDE + BLOCK_N;
constexpr int SF_TMEM_BASE = 476;
constexpr int N_ALLOC_COLS = 512;
static_assert(N_D_COLS <= SF_TMEM_BASE &&
              SF_TMEM_BASE + SF_TMEM_COLS <= N_ALLOC_COLS,
              "accumulators + scale factors exceed the 512 TMEM columns");
constexpr int EPI_CHUNK = 32;  // cols per regular drain

// ---- the shared-memory map (blog: the feed) -----------------------------------
// Every offset is a compile-time expression.
constexpr int CD_HDR        = 1024;                 // mbarrier header pad
constexpr int CD_AB_SLOTS   = 6;
constexpr int CD_SF_SLOTS   = 7;
constexpr int CD_AB_STRIDE  = BLOCK_M * WIN_BYTES;  // 16384 per A or B slot
constexpr int CD_SF_TILE    = 512;                  // one (32 lanes x 16B) x4 tile
constexpr int CD_SFA_TILES  = 3;                    // per SFA slot (M = 1 block)
constexpr int CD_SFB_TILES  = 6;                    // per SFB slot (N = 2 blocks)
constexpr int CD_SFA_STRIDE = CD_SFA_TILES * CD_SF_TILE;   // 1536
constexpr int CD_SFB_STRIDE = CD_SFB_TILES * CD_SF_TILE;   // 3072

// Expected transfer bytes are derived, never tuned: a wrong-by-one
// expectation doesn't crash, it deadlocks (blog: the feed).
constexpr uint32_t AB_TX = uint32_t(CTA_GROUP) *
        uint32_t(BLOCK_M + BLOCK_N_PEER) * uint32_t(AB_LOAD_BYTES);
constexpr uint32_t SF_TX = uint32_t(CD_SFA_STRIDE + CD_SFB_STRIDE) *
        uint32_t(CTA_GROUP);                                           // 9216
static_assert(AB_TX == 65536u && SF_TX == 9216u,
              "expect_tx derivation drifted");

// The map IS the struct: member sizes and alignment produce every byte
// offset; the raw-address constants the mainloop uses read off it with
// offsetof, and the static_asserts pin the published values.
struct SmemCD {
    // mbarrier header: four ring regions in handshake order, then the
    // singleton accumulator + teardown barriers (alignas pads it to 1024).
    struct alignas(1024) Header {
        uint64_t ab_ready[CD_AB_SLOTS];   // TMA -> MMA
        uint64_t ab_free [CD_AB_SLOTS];   // MMA -> TMA
        uint64_t sf_ready[CD_SF_SLOTS];   // TMA -> MMA
        uint64_t sf_free [CD_SF_SLOTS];   // MMA -> TMA
        uint64_t acc_ready;               // MMA -> epi
        uint64_t acc_free;                // epi -> MMA
        uint64_t dealloc;                 // 2-CTA teardown
        uint32_t tmem_addr;               // TMEM base addr storage
    } hdr;
    uint8_t a  [CD_AB_SLOTS][CD_AB_STRIDE];
    uint8_t b  [CD_AB_SLOTS][CD_AB_STRIDE];
    uint8_t sfa[CD_SF_SLOTS][CD_SFA_STRIDE];
    // SFB starts 1024-aligned (the cp source descriptor drops address bits 4-9).
    alignas(1024) uint8_t sfb[CD_SF_SLOTS][CD_SFB_STRIDE];
};
static_assert(sizeof(SmemCD::Header) == CD_HDR, "mbarrier header overflows the pad");

constexpr int CD_A_OFF   = offsetof(SmemCD, a);
constexpr int CD_B_OFF   = offsetof(SmemCD, b);
constexpr int CD_SFA_OFF = offsetof(SmemCD, sfa);
constexpr int CD_SFB_OFF = offsetof(SmemCD, sfb);
constexpr int CD_TOTAL   = sizeof(SmemCD);
static_assert(CD_TOTAL <= 232448, "smem map exceeds the opt-in budget");
static_assert(CD_AB_SLOTS != 6 ||
              (CD_B_OFF == 99328 && CD_SFA_OFF == 197632 &&
               CD_SFB_OFF == 208896 && CD_TOTAL == 230400),
              "the 6-slot byte map drifted from its published values");

// The mainloop computes raw `.shared` byte addresses from these.
constexpr int MB_AB_READY  = offsetof(SmemCD, hdr.ab_ready);
constexpr int MB_AB_FREE   = offsetof(SmemCD, hdr.ab_free);
constexpr int MB_SF_READY  = offsetof(SmemCD, hdr.sf_ready);
constexpr int MB_SF_FREE   = offsetof(SmemCD, hdr.sf_free);
constexpr int MB_ACC_READY = offsetof(SmemCD, hdr.acc_ready);
constexpr int MB_ACC_FREE  = offsetof(SmemCD, hdr.acc_free);


// ---- schedule table + placement audit (blog: the L2 trick) --------------------
constexpr int L2A_SM_CAP        = 256;
constexpr int L2A_CLUSTER_CAP   = 128;
constexpr int L2A_ROUTE_MODES   = 2;      // slot 0: gate table, slot 1: schedule
constexpr int L2A_ROUTE_WORK_CAP = 64 * 64;
__device__ __constant__ int l2a_sm_side[L2A_SM_CAP];
__device__ __constant__ int l2a_cluster_side[L2A_CLUSTER_CAP];
__device__ __constant__ int l2a_route_tables[L2A_ROUTE_MODES * L2A_ROUTE_WORK_CAP];
__device__ unsigned l2a_placement_errors;

}  // namespace nvfp4

// ============================================================================
// The consumer mainloop from the blog's pipeline section. One warp — warp 4
// of the even CTA —
// issues everything: the tcgen05.cp copies that stage scale factors from the
// smem ring into TMEM, immediately followed by the tcgen05.mma stream, in the
// SAME warp. A same-warp tcgen05 stream is one ordered async pipeline, so the
// write-after-read hazard on the single reused scale-factor TMEM region needs
// no fence and no cross-warp barrier.
//
// Per 768-element K-group, logical K=96 steps 0..7 issue in K order. Their
// starting offsets, expressed as 16-byte descriptor atoms modulo a 128-byte
// swizzle window, are 0,3,6,1,4,7,2,5. Atoms 6 and 7 straddle windows.
// ============================================================================
namespace nvfp4 {

// Integral tag for the K64-tail subclass dispatch (per-(a,b) folded variants).
template <uint32_t V> struct kdi { static constexpr uint32_t value = V; };

// Ring/parity state carried across tiles by the MMA warp.
struct MainloopState {
    uint32_t d_parity;   // which TMEM accumulator buffer the next tile writes
    uint32_t ab_phase, ab_slot;   // A/B ring cursor
    uint32_t sf_phase, sf_slot;   // scale-factor ring cursor
};


// Relaxed blocking wait (no .acquire): the tcgen05 pipeline's own ordering
// covers these ring waits, and the .acquire form perturbs the scoreboard
// cadence on the issuing warp. Waits sit AT the consuming site — the carried
// one-stage-ahead try_wait peeks that used to gate them measured as pure
// overhead and were removed.
static __device__ __forceinline__ void mbar_wait_relaxed(
        uint32_t bar_smem, uint32_t parity) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1, 10000000;\n\t"
        "@!p bra WAIT_%=;\n\t}\n"
        :: "r"(bar_smem), "r"(parity) : "memory");
}

// Gated form for the @216 buffer-free wait, which fires once per tile: the
// gate branches over the spin for every group past the first.
static __device__ __forceinline__ void mbar_wait_gated_relaxed(
        uint32_t bar_smem, uint32_t parity, bool peek) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.u32 p, %2, 0;\n\t"
        "@p bra DONE_%=;\n\t"
        "WAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1, 10000000;\n\t"
        "@!p bra WAIT_%=;\n\t"
        "DONE_%=:\n\t}\n"
        :: "r"(bar_smem), "r"(parity), "r"((uint32_t)peek) : "memory");
}

// redux.sync.min over 32 equal lanes is an identity, but ptxas trusts its
// result as warp-uniform. Schedule bit-math then stays on the uniform register
// datapath instead of paying a broadcast per tcgen05 operation.
static __device__ __forceinline__ uint32_t to_uniform(uint32_t v) {
    uint32_t r;
    asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;\n\t" : "=r"(r) : "r"(v));
    return r;
}

// One scale-factor staging copy, smem -> TMEM. Single-lane semantics behind
// the converged elect predicate: tcgen05.cp is a single-thread instruction —
// issued bare from a converged warp you get 32 initiations and garbage.
static __device__ __forceinline__ void cp_sf(
        uint32_t el, uint32_t dst, uint32_t lo, uint32_t hi) {
    const uint64_t desc = (uint64_t(hi) << 32) | uint64_t(lo);
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.u32 p, %2, 0;\n\t"
        "@p tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t}\n"
        :: "r"(dst), "l"(desc), "r"(el) : "memory");
}

// Ring-slot "consumed" arrive, multicast to both CTAs of the pair.
// arrive::one must fire exactly once — hence the elect predicate.
static __device__ __forceinline__ void commit_consumed(
        uint32_t el, uint32_t smbase, uint32_t mb_off, uint16_t mask) {
    asm volatile(
        "{\n\t.reg .pred p;\n\t"
        "setp.ne.u32 p, %2, 0;\n\t"
        "@p tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t}\n"
        :: "r"(smbase + mb_off), "h"(mask), "r"(el) : "memory");
}

// One K=96 NVFP4 MMA. The asm token says scale_vec::4X regardless — K=96 is
// selected by idesc bit 31 (blog: the MMA warp). acc_pred = 0 overwrites the
// accumulator (the tile's first step), nonzero accumulates.
static __device__ __forceinline__ void mma_cell(
        uint32_t el, uint32_t d,
        uint32_t desc_a_lo, uint32_t desc_a_hi,
        uint32_t desc_b_lo, uint32_t desc_b_hi,
        uint32_t idesc, uint32_t sf_a, uint32_t sf_b, uint32_t acc_pred) {
    const uint64_t desc_a = (uint64_t(desc_a_hi) << 32) | uint64_t(desc_a_lo);
    const uint64_t desc_b = (uint64_t(desc_b_hi) << 32) | uint64_t(desc_b_lo);
    asm volatile(
        "{\n\t.reg .pred p, acc;\n\t"
        "setp.ne.u32 p, %7, 0;\n\t"
        "setp.ne.b32 acc, %4, 0;\n\t"
        "@p tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X"
        "  [%0], %1, %2, %3, [%5], [%6], acc;\n\t}\n"
        :: "r"(d), "l"(desc_a), "l"(desc_b), "r"(idesc),
           "r"(acc_pred), "r"(sf_a), "r"(sf_b), "r"(el) : "memory");
}

// One SF ring slot -> TMEM: 9 tcgen05.cp tiles (3 SFA + 6 SFB). Source
// address advances are in 16-byte descriptor units. The SFB tile order
// 0,3,1,4,2,5 interleaves the two 128-wide N-blocks.
static __device__ __forceinline__ void cp_sf_buffer(
        uint32_t el, uint32_t slot,
        uint32_t sfa0, uint32_t sfa1, uint32_t sfa2,
        uint32_t sfb0, uint32_t sfb1, uint32_t sfb2,
        uint32_t sfb3, uint32_t sfb4, uint32_t sfb5,
        uint32_t sfa_hi, uint32_t sfa_lo, uint32_t sfb_hi, uint32_t sfb_lo) {
    constexpr uint32_t SFA_SLOT_U = CD_SFA_STRIDE / 16u;   // 96
    constexpr uint32_t SFB_SLOT_U = CD_SFB_STRIDE / 16u;   // 192
    constexpr uint32_t TILE_U     = CD_SF_TILE / 16u;      // 32
    const uint32_t a = slot * SFA_SLOT_U + sfa_lo;
    cp_sf(el, sfa0, a,              sfa_hi);
    cp_sf(el, sfa1, a + 1u*TILE_U,  sfa_hi);
    cp_sf(el, sfa2, a + 2u*TILE_U,  sfa_hi);
    const uint32_t b = slot * SFB_SLOT_U + sfb_lo;
    cp_sf(el, sfb0, b + 0u*TILE_U,  sfb_hi);
    cp_sf(el, sfb1, b + 3u*TILE_U,  sfb_hi);
    cp_sf(el, sfb2, b + 1u*TILE_U,  sfb_hi);
    cp_sf(el, sfb3, b + 4u*TILE_U,  sfb_hi);
    cp_sf(el, sfb4, b + 2u*TILE_U,  sfb_hi);
    cp_sf(el, sfb5, b + 5u*TILE_U,  sfb_hi);
}

// Operand descriptor LOW word for A/B ring slot `slot`, K-step `k_atom`
// (0..7 inside the 768-element group; each step is 48 bytes = +3 in the
// 16-byte address field).
static __device__ __forceinline__ uint32_t ab_desc_lo(
        uint32_t smbase, uint32_t ab_base_off, uint32_t slot, uint32_t k_atom) {
    const uint64_t d = ptx::mma_smem_desc_k96_sw128(
            smbase + ab_base_off + slot * CD_AB_STRIDE, int(k_atom) * 16);
    return uint32_t(d & 0xFFFFFFFFu);
}

// K-steps 6 and 7 cross a 128-byte window boundary. Their descriptor address
// comes from `addr_slot`, while the absolute-LBO field comes from the next
// slot whose swizzle window contains the second half.
static __device__ __forceinline__ uint32_t ab_desc_lo_straddle(
        uint32_t smbase, uint32_t ab_base_off,
        uint32_t addr_slot, uint32_t lbo_slot, uint32_t k_atom) {
    const uint64_t d_addr = ptx::mma_smem_desc_k96_sw128(
            smbase + ab_base_off + addr_slot * CD_AB_STRIDE, int(k_atom) * 16);
    const uint64_t d_lbo  = ptx::mma_smem_desc_k96_sw128(
            smbase + ab_base_off + lbo_slot  * CD_AB_STRIDE, 0);
    return uint32_t(d_addr & 0x0000FFFFu) | uint32_t(d_lbo & 0xFFFF0000u);
}

// One output tile: complete 768-element K-groups followed by one partial group
// containing `tail_cells` live K=96 steps. Dead cells fold their liveness into
// the issue predicate, so the warp stays converged.
static __device__ __forceinline__ MainloopState mainloop_tile(
        uint32_t taddr, uint32_t smbase, MainloopState in, uint16_t commit_mask,
        uint32_t full_groups,
        uint32_t tail_cells,
        uint32_t b64, uint32_t t_win
        ) {
    constexpr uint32_t AB_RING_SLOTS = uint32_t(CD_AB_SLOTS);
    constexpr uint32_t SF_RING_SLOTS = uint32_t(CD_SF_SLOTS);
    constexpr uint32_t AB_FREE_DELTA = uint32_t(MB_AB_FREE - MB_AB_READY);
    constexpr uint32_t SF_READY_OFF  = uint32_t(MB_SF_READY);
    constexpr uint32_t SF_FREE_DELTA = uint32_t(MB_SF_FREE - MB_SF_READY);
    constexpr uint32_t ACC_FREE_OFF  = uint32_t(MB_ACC_FREE);
    constexpr uint32_t A_OFF = uint32_t(CD_A_OFF);
    constexpr uint32_t B_OFF = uint32_t(CD_B_OFF);

    uint32_t d_parity = in.d_parity;
    uint32_t ab_phase = in.ab_phase, ab_slot = in.ab_slot;
    uint32_t sf_phase = in.sf_phase, sf_slot = in.sf_slot;

    // The caller has already selected exactly one lane for the whole MMA role.
    // Keep the helper predicate constant throughout the tile.
    const uint32_t el = 1u;

    // Fixed TMEM targets for the single reused scale-factor region: 9 cp
    // tiles of 4 columns each, starting past the accumulator region.
    constexpr uint32_t SF = uint32_t(SF_TMEM_BASE);
    const uint32_t sfa_t0 = taddr + SF +  0;      // SFA word 0 (MMA reads here)
    const uint32_t sfa_t1 = taddr + SF +  4;
    const uint32_t sfa_t2 = taddr + SF +  8;
    const uint32_t sfb_t0 = taddr + SF + 12;      // SFB word 0
    const uint32_t sfb_t1 = taddr + SF + 16;
    const uint32_t sfb_t2 = taddr + SF + 20;
    const uint32_t sfb_t3 = taddr + SF + 24;
    const uint32_t sfb_t4 = taddr + SF + 28;
    const uint32_t sfb_t5 = taddr + SF + 32;

    // Scale-factor cp source descriptors (low word carries the per-slot
    // offset; high word is constant).
    const uint64_t sfb_src = ptx::mma_sf_cp_src_desc_cd(smbase + uint32_t(CD_SFB_OFF));
    const uint32_t sfb_src_lo = uint32_t(sfb_src & 0xFFFFFFFFu);
    const uint32_t sfb_src_hi = uint32_t(sfb_src >> 32);
    const uint64_t sfa_src = ptx::mma_sf_cp_src_desc_cd(smbase + uint32_t(CD_SFA_OFF));
    const uint32_t sfa_src_lo = uint32_t(sfa_src & 0xFFFFFFFFu);
    const uint32_t sfa_src_hi = uint32_t(sfa_src >> 32);

    // Operand descriptor HIGH words (low words come from ab_desc_lo per step).
    const uint32_t a_desc_hi =
            uint32_t(ptx::mma_smem_desc_k96_sw128(smbase + A_OFF) >> 32);
    const uint32_t b_desc_hi =
            uint32_t(ptx::mma_smem_desc_k96_sw128(smbase + B_OFF) >> 32);

    // Each K=96 step needs 6 scale bytes per row, split across two staged
    // 32-bit words. Odd-issue steps read word 1: bit 31 of the SF address
    // selects the with-id encoding, and the id bits fold into the idesc.
    constexpr uint32_t SF_ADDR_ID = 0x80000000u;
    const uint32_t sfa_w1 = (taddr + SF +  4) | SF_ADDR_ID;
    const uint32_t sfb_w1 = (taddr + SF + 20) | SF_ADDR_ID;
    const uint32_t w1_id_bits =
            ((sfa_w1 >> 1) & 0x60000000u) | ((sfb_w1 >> 26) & 48u);

    // The tile's first MMA overwrites the accumulator buffer this tile owns;
    // the @216 "buffer free" wait runs once, before the first group.
    const uint32_t acc_wait_parity = d_parity;
    d_parity ^= 1u;
    const uint32_t d_tmem = d_parity * uint32_t(D_STRIDE) + taddr;

    // Instruction descriptor: kind::mxf4nvf4, M=256, N=256, E2M1 x E2M1,
    // UE4M3 block-16, K-major, k_dim=1 (K=96). Base value 0x90400480.
    constexpr uint32_t IDESC = ptx::mma_inst_desc_mxf4nvf4_block16(
            256u, 256u, 0u, 0u, ptx::Major::K, ptx::Major::K,
            false, false, /*k_dim=*/true, ptx::ScaleFormat::E4M3);
    static_assert(IDESC == 0x90400480u, "K=96 NVFP4 idesc drifted");
    const uint32_t w0_id_bits =
            ((sfa_t0 >> 1) & 0x60000000u) | ((sfb_t0 >> 26) & 48u);
    const uint32_t idesc_w0 = w0_id_bits | IDESC;
    const uint32_t idesc_w1 = w1_id_bits | IDESC;


    uint32_t acc_pred = 0;        // first MMA of the tile overwrites
    bool need_acc_wait = true;    // @216 wait fires once per tile

    if (full_groups >= 1) {
        uint32_t g = 0;
        do {
            // ---- SF batch 0 -> logical steps 0,1 (atoms 0,3) ----
            const uint32_t sf_bar0 = smbase + (sf_slot << 3) + SF_READY_OFF;
            mbar_wait_relaxed(sf_bar0, sf_phase);
            uint32_t sf_slot1, sf_phase1;
            {
                const uint32_t nx = sf_slot + 1;
                const bool wrap = (nx == SF_RING_SLOTS);
                sf_slot1  = wrap ? 0u : nx;
                sf_phase1 = sf_phase ^ (wrap ? 1u : 0u);
            }
            cp_sf_buffer(el, sf_slot, sfa_t0, sfa_t1, sfa_t2, sfb_t0, sfb_t1,
                         sfb_t2, sfb_t3, sfb_t4, sfb_t5,
                         sfa_src_hi, sfa_src_lo, sfb_src_hi, sfb_src_lo);
            commit_consumed(el, smbase, (sf_bar0 + SF_FREE_DELTA) - smbase,
                            commit_mask);
            const uint32_t sf_bar1 = smbase + (sf_slot1 << 3) + SF_READY_OFF;
            const uint32_t ab_bar0 = smbase + (ab_slot << 3);
            mbar_wait_relaxed(ab_bar0, ab_phase);
            uint32_t ab_slot1, ab_phase1;
            {
                const uint32_t nx = ab_slot + 1;
                const bool wrap = (nx == AB_RING_SLOTS);
                ab_slot1  = wrap ? 0u : nx;
                ab_phase1 = ab_phase ^ (wrap ? 1u : 0u);
            }
            const uint32_t ab_bar1 = smbase + (ab_slot1 << 3);
            {
                const uint32_t acc_bar = smbase + ACC_FREE_OFF;
                mbar_wait_gated_relaxed(acc_bar, acc_wait_parity, !need_acc_wait);
            }
            mma_cell(el, d_tmem, ab_desc_lo(smbase, A_OFF, ab_slot, 0), a_desc_hi,
                                 ab_desc_lo(smbase, B_OFF, ab_slot, 0), b_desc_hi,
                     idesc_w0, sfa_t0, sfb_t0, acc_pred);              // step 0, atom 0
            mma_cell(el, d_tmem, ab_desc_lo(smbase, A_OFF, ab_slot, 3), a_desc_hi,
                                 ab_desc_lo(smbase, B_OFF, ab_slot, 3), b_desc_hi,
                     idesc_w1, sfa_w1, sfb_w1, 0xFFFFFFFFu);           // step 1, atom 3

            // ---- SF batch 1 -> logical steps 2,3 (atoms 6,1) ----
            mbar_wait_relaxed(sf_bar1, sf_phase1);
            uint32_t sf_slot2, sf_phase2;
            {
                const uint32_t nx = sf_slot1 + 1;
                const bool wrap = (nx == SF_RING_SLOTS);
                sf_slot2  = wrap ? 0u : nx;
                sf_phase2 = sf_phase1 ^ (wrap ? 1u : 0u);
            }
            cp_sf_buffer(el, sf_slot1, sfa_t0, sfa_t1, sfa_t2, sfb_t0, sfb_t1,
                         sfb_t2, sfb_t3, sfb_t4, sfb_t5,
                         sfa_src_hi, sfa_src_lo, sfb_src_hi, sfb_src_lo);
            commit_consumed(el, smbase, (sf_bar1 + SF_FREE_DELTA) - smbase,
                            commit_mask);
            const uint32_t sf_bar2 = smbase + (sf_slot2 << 3) + SF_READY_OFF;
            mbar_wait_relaxed(ab_bar1, ab_phase1);
            uint32_t ab_slot2, ab_phase2;
            {
                const uint32_t nx = ab_slot1 + 1;
                const bool wrap = (nx == AB_RING_SLOTS);
                ab_slot2  = wrap ? 0u : nx;
                ab_phase2 = ab_phase1 ^ (wrap ? 1u : 0u);
            }
            const uint32_t ab_bar2 = smbase + (ab_slot2 << 3);
            mma_cell(el, d_tmem,
                     ab_desc_lo_straddle(smbase, A_OFF, ab_slot, ab_slot1, 6), a_desc_hi,
                     ab_desc_lo_straddle(smbase, B_OFF, ab_slot, ab_slot1, 6), b_desc_hi,
                     idesc_w0, sfa_t0, sfb_t0, 0xFFFFFFFFu);           // step 2, atom 6
            // AB0's last reader is MMA2.
            commit_consumed(el, smbase, (ab_bar0 + AB_FREE_DELTA) - smbase,
                            commit_mask);
            mma_cell(el, d_tmem, ab_desc_lo(smbase, A_OFF, ab_slot1, 1), a_desc_hi,
                                 ab_desc_lo(smbase, B_OFF, ab_slot1, 1), b_desc_hi,
                     idesc_w1, sfa_w1, sfb_w1, 0xFFFFFFFFu);           // step 3, atom 1

            // ---- SF batch 2 -> logical steps 4,5 (atoms 4,7) ----
            mbar_wait_relaxed(sf_bar2, sf_phase2);
            uint32_t sf_slot3, sf_phase3;
            {
                const uint32_t nx = sf_slot2 + 1;
                const bool wrap = (nx == SF_RING_SLOTS);
                sf_slot3  = wrap ? 0u : nx;
                sf_phase3 = sf_phase2 ^ (wrap ? 1u : 0u);
            }
            cp_sf_buffer(el, sf_slot2, sfa_t0, sfa_t1, sfa_t2, sfb_t0, sfb_t1,
                         sfb_t2, sfb_t3, sfb_t4, sfb_t5,
                         sfa_src_hi, sfa_src_lo, sfb_src_hi, sfb_src_lo);
            commit_consumed(el, smbase, (sf_bar2 + SF_FREE_DELTA) - smbase,
                            commit_mask);
            const uint32_t sf_bar3 = smbase + (sf_slot3 << 3) + SF_READY_OFF;
            mma_cell(el, d_tmem, ab_desc_lo(smbase, A_OFF, ab_slot1, 4), a_desc_hi,
                                 ab_desc_lo(smbase, B_OFF, ab_slot1, 4), b_desc_hi,
                     idesc_w0, sfa_t0, sfb_t0, 0xFFFFFFFFu);           // step 4, atom 4
            mbar_wait_relaxed(ab_bar2, ab_phase2);
            {
                const uint32_t nx = ab_slot2 + 1;
                const bool wrap = (nx == AB_RING_SLOTS);
                ab_slot  = wrap ? 0u : nx;             // ring cursor writeback
                ab_phase = ab_phase2 ^ (wrap ? 1u : 0u);
            }
            mma_cell(el, d_tmem,
                     ab_desc_lo_straddle(smbase, A_OFF, ab_slot1, ab_slot2, 7), a_desc_hi,
                     ab_desc_lo_straddle(smbase, B_OFF, ab_slot1, ab_slot2, 7), b_desc_hi,
                     idesc_w1, sfa_w1, sfb_w1, 0xFFFFFFFFu);           // step 5, atom 7

            // ---- SF batch 3 -> logical steps 6,7 (atoms 2,5) ----
            mbar_wait_relaxed(sf_bar3, sf_phase3);
            {
                const uint32_t nx = sf_slot3 + 1;
                const bool wrap = (nx == SF_RING_SLOTS);
                sf_slot  = wrap ? 0u : nx;             // ring cursor writeback
                sf_phase = sf_phase3 ^ (wrap ? 1u : 0u);
            }
            cp_sf_buffer(el, sf_slot3, sfa_t0, sfa_t1, sfa_t2, sfb_t0, sfb_t1,
                         sfb_t2, sfb_t3, sfb_t4, sfb_t5,
                         sfa_src_hi, sfa_src_lo, sfb_src_hi, sfb_src_lo);
            commit_consumed(el, smbase, (sf_bar3 + SF_FREE_DELTA) - smbase,
                            commit_mask);
            // AB1's last reader is MMA5; this measured placement leaves the
            // commit behind the following scale copy.
            commit_consumed(el, smbase, (ab_bar1 + AB_FREE_DELTA) - smbase,
                            commit_mask);
            mma_cell(el, d_tmem, ab_desc_lo(smbase, A_OFF, ab_slot2, 2), a_desc_hi,
                                 ab_desc_lo(smbase, B_OFF, ab_slot2, 2), b_desc_hi,
                     idesc_w0, sfa_t0, sfb_t0, 0xFFFFFFFFu);           // step 6, atom 2
            acc_pred = 0xFFFFFFFFu;   // accumulate from the second group on
            mma_cell(el, d_tmem, ab_desc_lo(smbase, A_OFF, ab_slot2, 5), a_desc_hi,
                                 ab_desc_lo(smbase, B_OFF, ab_slot2, 5), b_desc_hi,
                     idesc_w1, sfa_w1, sfb_w1, 0xFFFFFFFFu);           // step 7, atom 5
            need_acc_wait = false;
            // AB2's last reader is MMA7.
            commit_consumed(el, smbase, (ab_bar2 + AB_FREE_DELTA) - smbase,
                            commit_mask);
            ++g;
        } while (g != full_groups);
    }

    if (tail_cells == 0u)
        return MainloopState{d_parity, ab_phase, ab_slot, sf_phase, sf_slot};

    // ---- the K64-dispatch tail: r = a x K96 + b x K64, a even, b in {1,2}.
    // ONE fully-folded tail variant per active (a, b) subclass, selected by
    // a once-per-tile branch ladder: every per-position encoding below is a
    // compile-time constant inside its variant (the fixed-shape production
    // receipt: full folding costs zero select/branch artifact, while the
    // blended runtime-select form measured -0.75pp and a live uniform
    // preamble evicts the hot mainloop's descriptor math off the uniform
    // register file — +147 R2UR in the full-group loop, measured in SASS).
    // Encodings are the metal-proven set: k_dim=0 idesc, LEGACY operand
    // descriptors (bit 52 clear, even 16-byte units — abs-LBO pairs with
    // k_dim=1 only), plain id-0 SF words ("block16 + K=64 mandates sf_id
    // 0"). Cell count a + b and t_sf equal the K96 tail's, so the SF ring
    // walk is IDENTICAL; only T_WIN = ceil((r/2)/128) A/B windows are
    // consumed (the producer skips the same subs).
    const auto k64_tail = [&](auto a_tag, auto b_tag) -> MainloopState {
        constexpr uint32_t A_C  = decltype(a_tag)::value;   // even
        constexpr uint32_t B64V = decltype(b_tag)::value;   // 1 or 2
        constexpr uint32_t N_C  = A_C + B64V;               // == tail_cells
        constexpr uint32_t T_WIN =
            ((A_C * 96u + B64V * 64u) / 2u + 127u) / 128u;
        static_assert((IDESC & 0x7FFFFFFFu) == 0x10400480u,
                      "K=64 NVFP4 idesc drifted");
        // The two K64 idesc forms share the runtime id bits already staged
        // for the K96 skeleton; the K64#2 word-1-plain form derives its own.
        const uint32_t idesc_k64a = idesc_w0 & 0x7FFFFFFFu;
        const uint32_t idesc_k64b = ((sfa_t1 >> 1) & 0x60000000u)
                                  | ((sfb_t2 >> 26) & 48u)
                                  | (IDESC & 0x7FFFFFFFu);
        const uint32_t a_hi64 = a_desc_hi & ~(1u << 20);   // desc bit 52
        const uint32_t b_hi64 = b_desc_hi & ~(1u << 20);
        // Ring slots/phases for the tail windows: pure cursor math.
        const bool w0_wrap = (ab_slot + 1u == AB_RING_SLOTS);
        const uint32_t s1  = w0_wrap ? 0u : ab_slot + 1u;
        const uint32_t ph1 = ab_phase ^ (w0_wrap ? 1u : 0u);
        const bool w1_wrap = (s1 + 1u == AB_RING_SLOTS);
        const uint32_t s2  = w1_wrap ? 0u : s1 + 1u;
        const uint32_t ph2 = ph1 ^ (w1_wrap ? 1u : 0u);
        const uint32_t ab_bar0 = smbase + (ab_slot << 3);
        const uint32_t ab_bar1 = smbase + (s1 << 3);
        const uint32_t ab_bar2 = smbase + (s2 << 3);

        // SF batch 0 (cells 0,1) — always live.
        const uint32_t sf_bar0 = smbase + (sf_slot << 3) + SF_READY_OFF;
        mbar_wait_relaxed(sf_bar0, sf_phase);
        uint32_t sf_slot1, sf_phase1;
        {
            const uint32_t nx = sf_slot + 1;
            const bool wrap = (nx == SF_RING_SLOTS);
            sf_slot1  = wrap ? 0u : nx;
            sf_phase1 = sf_phase ^ (wrap ? 1u : 0u);
        }
        cp_sf_buffer(el, sf_slot, sfa_t0, sfa_t1, sfa_t2, sfb_t0, sfb_t1,
                     sfb_t2, sfb_t3, sfb_t4, sfb_t5,
                     sfa_src_hi, sfa_src_lo, sfb_src_hi, sfb_src_lo);
        commit_consumed(el, smbase, (sf_bar0 + SF_FREE_DELTA) - smbase,
                        commit_mask);
        const uint32_t sf_bar1 = smbase + (sf_slot1 << 3) + SF_READY_OFF;
        mbar_wait_relaxed(ab_bar0, ab_phase);
        {
            const uint32_t acc_bar = smbase + ACC_FREE_OFF;
            mbar_wait_gated_relaxed(acc_bar, acc_wait_parity, !need_acc_wait);
        }
        // Cell 0: K96 (A_C > 0) or the lone/leading K64 (A_C == 0).
        if constexpr (A_C == 0u) {
            mma_cell(el, d_tmem,
                     ab_desc_lo(smbase, A_OFF, ab_slot, 0) & 0xFFFFu, a_hi64,
                     ab_desc_lo(smbase, B_OFF, ab_slot, 0) & 0xFFFFu, b_hi64,
                     idesc_k64a, sfa_t0, sfb_t0, acc_pred);
        } else {
            mma_cell(el, d_tmem,
                     ab_desc_lo(smbase, A_OFF, ab_slot, 0), a_desc_hi,
                     ab_desc_lo(smbase, B_OFF, ab_slot, 0), b_desc_hi,
                     idesc_w0, sfa_t0, sfb_t0, acc_pred);
        }
        // Cell 1: K96 / K64#2 (A_C == 0, unit 2) / dead.
        if constexpr (N_C > 1u) {
            if constexpr (A_C == 0u) {
                mma_cell(el, d_tmem,
                         ab_desc_lo(smbase, A_OFF, ab_slot, 2) & 0xFFFFu, a_hi64,
                         ab_desc_lo(smbase, B_OFF, ab_slot, 2) & 0xFFFFu, b_hi64,
                         idesc_k64b, sfa_t1, sfb_t2, 0xFFFFFFFFu);
            } else {
                mma_cell(el, d_tmem,
                         ab_desc_lo(smbase, A_OFF, ab_slot, 3), a_desc_hi,
                         ab_desc_lo(smbase, B_OFF, ab_slot, 3), b_desc_hi,
                         idesc_w1, sfa_w1, sfb_w1, 0xFFFFFFFFu);
            }
        }

        // SF batch 1 (cells 2,3).
        uint32_t sf_slot2 = sf_slot1, sf_phase2 = sf_phase1;
        if constexpr (N_C > 2u) {
            mbar_wait_relaxed(sf_bar1, sf_phase1);
            {
                const uint32_t nx = sf_slot1 + 1;
                const bool wrap = (nx == SF_RING_SLOTS);
                sf_slot2  = wrap ? 0u : nx;
                sf_phase2 = sf_phase1 ^ (wrap ? 1u : 0u);
            }
            cp_sf_buffer(el, sf_slot1, sfa_t0, sfa_t1, sfa_t2, sfb_t0, sfb_t1,
                         sfb_t2, sfb_t3, sfb_t4, sfb_t5,
                         sfa_src_hi, sfa_src_lo, sfb_src_hi, sfb_src_lo);
            commit_consumed(el, smbase, (sf_bar1 + SF_FREE_DELTA) - smbase,
                            commit_mask);
        }
        const uint32_t sf_bar2 = smbase + (sf_slot2 << 3) + SF_READY_OFF;
        if constexpr (T_WIN >= 2u) mbar_wait_relaxed(ab_bar1, ph1);
        // Cell 2: K96 straddle (A_C > 2) / K64#1 at unit 6 of window 0
        // (A_C == 2; no crossing, legacy) / dead.
        if constexpr (A_C == 2u) {
            mma_cell(el, d_tmem,
                     ab_desc_lo(smbase, A_OFF, ab_slot, 6) & 0xFFFFu, a_hi64,
                     ab_desc_lo(smbase, B_OFF, ab_slot, 6) & 0xFFFFu, b_hi64,
                     idesc_k64a, sfa_t0, sfb_t0, 0xFFFFFFFFu);
        } else if constexpr (A_C > 2u) {
            mma_cell(el, d_tmem,
                     ab_desc_lo_straddle(smbase, A_OFF, ab_slot, s1, 6), a_desc_hi,
                     ab_desc_lo_straddle(smbase, B_OFF, ab_slot, s1, 6), b_desc_hi,
                     idesc_w0, sfa_t0, sfb_t0, 0xFFFFFFFFu);
        }
        commit_consumed(el, smbase, (ab_bar0 + AB_FREE_DELTA) - smbase,
                        commit_mask);
        // Cell 3: K96 / K64#2 (A_C == 2, unit 0 of window 1) / dead.
        if constexpr (A_C == 2u && B64V == 2u) {
            mma_cell(el, d_tmem,
                     ab_desc_lo(smbase, A_OFF, s1, 0) & 0xFFFFu, a_hi64,
                     ab_desc_lo(smbase, B_OFF, s1, 0) & 0xFFFFu, b_hi64,
                     idesc_k64b, sfa_t1, sfb_t2, 0xFFFFFFFFu);
        } else if constexpr (A_C > 3u) {
            mma_cell(el, d_tmem,
                     ab_desc_lo(smbase, A_OFF, s1, 1), a_desc_hi,
                     ab_desc_lo(smbase, B_OFF, s1, 1), b_desc_hi,
                     idesc_w1, sfa_w1, sfb_w1, 0xFFFFFFFFu);
        }

        // SF batch 2 (cells 4,5).
        uint32_t sf_slot3 = sf_slot2, sf_phase3 = sf_phase2;
        if constexpr (N_C > 4u) {
            mbar_wait_relaxed(sf_bar2, sf_phase2);
            {
                const uint32_t nx = sf_slot2 + 1;
                const bool wrap = (nx == SF_RING_SLOTS);
                sf_slot3  = wrap ? 0u : nx;
                sf_phase3 = sf_phase2 ^ (wrap ? 1u : 0u);
            }
            cp_sf_buffer(el, sf_slot2, sfa_t0, sfa_t1, sfa_t2, sfb_t0, sfb_t1,
                         sfb_t2, sfb_t3, sfb_t4, sfb_t5,
                         sfa_src_hi, sfa_src_lo, sfb_src_hi, sfb_src_lo);
            commit_consumed(el, smbase, (sf_bar2 + SF_FREE_DELTA) - smbase,
                            commit_mask);
        }
        const uint32_t sf_bar3 = smbase + (sf_slot3 << 3) + SF_READY_OFF;
        // Cell 4: K96 / K64#1 at unit 4 of window 1 (A_C == 4) / dead.
        if constexpr (A_C == 4u) {
            mma_cell(el, d_tmem,
                     ab_desc_lo(smbase, A_OFF, s1, 4) & 0xFFFFu, a_hi64,
                     ab_desc_lo(smbase, B_OFF, s1, 4) & 0xFFFFu, b_hi64,
                     idesc_k64a, sfa_t0, sfb_t0, 0xFFFFFFFFu);
        } else if constexpr (A_C > 4u) {
            mma_cell(el, d_tmem,
                     ab_desc_lo(smbase, A_OFF, s1, 4), a_desc_hi,
                     ab_desc_lo(smbase, B_OFF, s1, 4), b_desc_hi,
                     idesc_w0, sfa_t0, sfb_t0, 0xFFFFFFFFu);
        }
        // A/B cursor writeback: entry + T_WIN (the producer advances the same).
        if constexpr (T_WIN >= 3u) {
            mbar_wait_relaxed(ab_bar2, ph2);
            const bool w2_wrap = (s2 + 1u == AB_RING_SLOTS);
            ab_slot  = w2_wrap ? 0u : s2 + 1u;
            ab_phase = ph2 ^ (w2_wrap ? 1u : 0u);
        } else if constexpr (T_WIN == 2u) {
            ab_slot  = s2;
            ab_phase = ph2;
        } else {
            ab_slot  = s1;
            ab_phase = ph1;
        }
        // Cell 5: K96 straddle (A_C == 6) / K64#2 at unit 6 of window 1
        // (A_C == 4, no crossing) / dead.
        if constexpr (A_C == 4u && B64V == 2u) {
            mma_cell(el, d_tmem,
                     ab_desc_lo(smbase, A_OFF, s1, 6) & 0xFFFFu, a_hi64,
                     ab_desc_lo(smbase, B_OFF, s1, 6) & 0xFFFFu, b_hi64,
                     idesc_k64b, sfa_t1, sfb_t2, 0xFFFFFFFFu);
        } else if constexpr (A_C > 5u) {
            mma_cell(el, d_tmem,
                     ab_desc_lo_straddle(smbase, A_OFF, s1, s2, 7), a_desc_hi,
                     ab_desc_lo_straddle(smbase, B_OFF, s1, s2, 7), b_desc_hi,
                     idesc_w1, sfa_w1, sfb_w1, 0xFFFFFFFFu);
        }

        // SF batch 3 (cells 6,7).
        if constexpr (N_C > 6u) {
            mbar_wait_relaxed(sf_bar3, sf_phase3);
            {
                const uint32_t nx = sf_slot3 + 1;
                const bool wrap = (nx == SF_RING_SLOTS);
                sf_slot  = wrap ? 0u : nx;
                sf_phase = sf_phase3 ^ (wrap ? 1u : 0u);
            }
            cp_sf_buffer(el, sf_slot3, sfa_t0, sfa_t1, sfa_t2, sfb_t0, sfb_t1,
                         sfb_t2, sfb_t3, sfb_t4, sfb_t5,
                         sfa_src_hi, sfa_src_lo, sfb_src_hi, sfb_src_lo);
            commit_consumed(el, smbase, (sf_bar3 + SF_FREE_DELTA) - smbase,
                            commit_mask);
        } else {
            sf_slot  = sf_slot3;
            sf_phase = sf_phase3;
        }
        commit_consumed(T_WIN >= 2u ? el : 0u, smbase,
                        (ab_bar1 + AB_FREE_DELTA) - smbase, commit_mask);
        // Cell 6: K96 / K64#1 at unit 2 of window 2 (A_C == 6) / dead.
        if constexpr (A_C == 6u) {
            mma_cell(el, d_tmem,
                     ab_desc_lo(smbase, A_OFF, s2, 2) & 0xFFFFu, a_hi64,
                     ab_desc_lo(smbase, B_OFF, s2, 2) & 0xFFFFu, b_hi64,
                     idesc_k64a, sfa_t0, sfb_t0, 0xFFFFFFFFu);
        }
        // Cell 7: K64#2 at unit 4 of window 2 (A_C == 6) / dead (no K96
        // cell 7 exists in the active class: A_C <= 6).
        if constexpr (A_C == 6u && B64V == 2u) {
            mma_cell(el, d_tmem,
                     ab_desc_lo(smbase, A_OFF, s2, 4) & 0xFFFFu, a_hi64,
                     ab_desc_lo(smbase, B_OFF, s2, 4) & 0xFFFFu, b_hi64,
                     idesc_k64b, sfa_t1, sfb_t2, 0xFFFFFFFFu);
        }
        commit_consumed(T_WIN >= 3u ? el : 0u, smbase,
                        (ab_bar2 + AB_FREE_DELTA) - smbase, commit_mask);

        return MainloopState{d_parity, ab_phase, ab_slot, sf_phase, sf_slot};
    };
    if (b64 != 0u) {
        // Once-per-tile subclass dispatch (a = tail_cells - b, even).
        const uint32_t a_c = tail_cells - b64;
        if (b64 == 1u) {
            if (a_c == 0u) return k64_tail(kdi<0>{}, kdi<1>{});
            if (a_c == 2u) return k64_tail(kdi<2>{}, kdi<1>{});
            if (a_c == 4u) return k64_tail(kdi<4>{}, kdi<1>{});
            return k64_tail(kdi<6>{}, kdi<1>{});
        }
        if (a_c == 0u) return k64_tail(kdi<0>{}, kdi<2>{});
        if (a_c == 2u) return k64_tail(kdi<2>{}, kdi<2>{});
        if (a_c == 4u) return k64_tail(kdi<4>{}, kdi<2>{});
        return k64_tail(kdi<6>{}, kdi<2>{});
    }

    // ---- the exact-K tail group: tail_cells live K=96 steps out of 8 ----
    // Cell liveness rides the elect predicate (a 0-predicate op never enters
    // the tcgen05 pipe); SF batch b exists iff tail_cells > 2b, and skipped
    // batches advance NO ring cursor — the SF producer skips symmetrically.
    const uint32_t el_c1 = (tail_cells >= 2u) ? el : 0u;
    const uint32_t el_c2 = (tail_cells >= 3u) ? el : 0u;
    const uint32_t el_c3 = (tail_cells >= 4u) ? el : 0u;
    const uint32_t el_c4 = (tail_cells >= 5u) ? el : 0u;
    const uint32_t el_c5 = (tail_cells >= 6u) ? el : 0u;
    const uint32_t el_c6 = (tail_cells >= 7u) ? el : 0u;
    const uint32_t el_c7 = (tail_cells >= 8u) ? el : 0u;
    const bool b1_live = (tail_cells > 2u);
    const bool b2_live = (tail_cells > 4u);
    const bool b3_live = (tail_cells > 6u);

    // SF batch 0 (cells 0,1) — a live tail always has cell 0.
    const uint32_t sf_bar0 = smbase + (sf_slot << 3) + SF_READY_OFF;
    mbar_wait_relaxed(sf_bar0, sf_phase);
    uint32_t sf_slot1, sf_phase1;
    {
        const uint32_t nx = sf_slot + 1;
        const bool wrap = (nx == SF_RING_SLOTS);
        sf_slot1  = wrap ? 0u : nx;
        sf_phase1 = sf_phase ^ (wrap ? 1u : 0u);
    }
    cp_sf_buffer(el, sf_slot, sfa_t0, sfa_t1, sfa_t2, sfb_t0, sfb_t1,
                 sfb_t2, sfb_t3, sfb_t4, sfb_t5,
                 sfa_src_hi, sfa_src_lo, sfb_src_hi, sfb_src_lo);
    commit_consumed(el, smbase, (sf_bar0 + SF_FREE_DELTA) - smbase, commit_mask);
    const uint32_t sf_bar1 = smbase + (sf_slot1 << 3) + SF_READY_OFF;
    const uint32_t ab_bar0 = smbase + (ab_slot << 3);
    mbar_wait_relaxed(ab_bar0, ab_phase);
    uint32_t ab_slot1, ab_phase1;
    {
        const uint32_t nx = ab_slot + 1;
        const bool wrap = (nx == AB_RING_SLOTS);
        ab_slot1  = wrap ? 0u : nx;
        ab_phase1 = ab_phase ^ (wrap ? 1u : 0u);
    }
    const uint32_t ab_bar1 = smbase + (ab_slot1 << 3);
    {
        const uint32_t acc_bar = smbase + ACC_FREE_OFF;
        mbar_wait_gated_relaxed(acc_bar, acc_wait_parity, !need_acc_wait);
    }
    mma_cell(el, d_tmem, ab_desc_lo(smbase, A_OFF, ab_slot, 0), a_desc_hi,
                         ab_desc_lo(smbase, B_OFF, ab_slot, 0), b_desc_hi,
             idesc_w0, sfa_t0, sfb_t0, acc_pred);                      // step 0, atom 0
    mma_cell(el_c1, d_tmem, ab_desc_lo(smbase, A_OFF, ab_slot, 3), a_desc_hi,
                            ab_desc_lo(smbase, B_OFF, ab_slot, 3), b_desc_hi,
             idesc_w1, sfa_w1, sfb_w1, 0xFFFFFFFFu);                   // step 1, atom 3

    // SF batch 1 (cells 2,3).
    uint32_t sf_slot2 = sf_slot1, sf_phase2 = sf_phase1;
    if (b1_live) {
        mbar_wait_relaxed(sf_bar1, sf_phase1);
        {
            const uint32_t nx = sf_slot1 + 1;
            const bool wrap = (nx == SF_RING_SLOTS);
            sf_slot2  = wrap ? 0u : nx;
            sf_phase2 = sf_phase1 ^ (wrap ? 1u : 0u);
        }
        cp_sf_buffer(el, sf_slot1, sfa_t0, sfa_t1, sfa_t2, sfb_t0, sfb_t1,
                     sfb_t2, sfb_t3, sfb_t4, sfb_t5,
                     sfa_src_hi, sfa_src_lo, sfb_src_hi, sfb_src_lo);
        commit_consumed(el, smbase, (sf_bar1 + SF_FREE_DELTA) - smbase,
                        commit_mask);
    }
    const uint32_t sf_bar2 = smbase + (sf_slot2 << 3) + SF_READY_OFF;
    mbar_wait_relaxed(ab_bar1, ab_phase1);
    uint32_t ab_slot2, ab_phase2;
    {
        const uint32_t nx = ab_slot1 + 1;
        const bool wrap = (nx == AB_RING_SLOTS);
        ab_slot2  = wrap ? 0u : nx;
        ab_phase2 = ab_phase1 ^ (wrap ? 1u : 0u);
    }
    const uint32_t ab_bar2 = smbase + (ab_slot2 << 3);
    mma_cell(el_c2, d_tmem,
             ab_desc_lo_straddle(smbase, A_OFF, ab_slot, ab_slot1, 6), a_desc_hi,
             ab_desc_lo_straddle(smbase, B_OFF, ab_slot, ab_slot1, 6), b_desc_hi,
             idesc_w0, sfa_t0, sfb_t0, 0xFFFFFFFFu);                   // step 2, atom 6
    commit_consumed(el, smbase, (ab_bar0 + AB_FREE_DELTA) - smbase, commit_mask);
    mma_cell(el_c3, d_tmem, ab_desc_lo(smbase, A_OFF, ab_slot1, 1), a_desc_hi,
                            ab_desc_lo(smbase, B_OFF, ab_slot1, 1), b_desc_hi,
             idesc_w1, sfa_w1, sfb_w1, 0xFFFFFFFFu);                   // step 3, atom 1

    // SF batch 2 (cells 4,5).
    uint32_t sf_slot3 = sf_slot2, sf_phase3 = sf_phase2;
    if (b2_live) {
        mbar_wait_relaxed(sf_bar2, sf_phase2);
        {
            const uint32_t nx = sf_slot2 + 1;
            const bool wrap = (nx == SF_RING_SLOTS);
            sf_slot3  = wrap ? 0u : nx;
            sf_phase3 = sf_phase2 ^ (wrap ? 1u : 0u);
        }
        cp_sf_buffer(el, sf_slot2, sfa_t0, sfa_t1, sfa_t2, sfb_t0, sfb_t1,
                     sfb_t2, sfb_t3, sfb_t4, sfb_t5,
                     sfa_src_hi, sfa_src_lo, sfb_src_hi, sfb_src_lo);
        commit_consumed(el, smbase, (sf_bar2 + SF_FREE_DELTA) - smbase,
                        commit_mask);
    }
    const uint32_t sf_bar3 = smbase + (sf_slot3 << 3) + SF_READY_OFF;
    mma_cell(el_c4, d_tmem, ab_desc_lo(smbase, A_OFF, ab_slot1, 4), a_desc_hi,
                            ab_desc_lo(smbase, B_OFF, ab_slot1, 4), b_desc_hi,
             idesc_w0, sfa_t0, sfb_t0, 0xFFFFFFFFu);                   // step 4, atom 4
    mbar_wait_relaxed(ab_bar2, ab_phase2);
    {
        const uint32_t nx = ab_slot2 + 1;
        const bool wrap = (nx == AB_RING_SLOTS);
        ab_slot  = wrap ? 0u : nx;
        ab_phase = ab_phase2 ^ (wrap ? 1u : 0u);
    }
    mma_cell(el_c5, d_tmem,
             ab_desc_lo_straddle(smbase, A_OFF, ab_slot1, ab_slot2, 7), a_desc_hi,
             ab_desc_lo_straddle(smbase, B_OFF, ab_slot1, ab_slot2, 7), b_desc_hi,
             idesc_w1, sfa_w1, sfb_w1, 0xFFFFFFFFu);                   // step 5, atom 7

    // SF batch 3 (cells 6,7).
    if (b3_live) {
        mbar_wait_relaxed(sf_bar3, sf_phase3);
        {
            const uint32_t nx = sf_slot3 + 1;
            const bool wrap = (nx == SF_RING_SLOTS);
            sf_slot  = wrap ? 0u : nx;
            sf_phase = sf_phase3 ^ (wrap ? 1u : 0u);
        }
        cp_sf_buffer(el, sf_slot3, sfa_t0, sfa_t1, sfa_t2, sfb_t0, sfb_t1,
                     sfb_t2, sfb_t3, sfb_t4, sfb_t5,
                     sfa_src_hi, sfa_src_lo, sfb_src_hi, sfb_src_lo);
        commit_consumed(el, smbase, (sf_bar3 + SF_FREE_DELTA) - smbase,
                        commit_mask);
    } else {
        sf_slot  = sf_slot3;
        sf_phase = sf_phase3;
    }
    commit_consumed(el, smbase, (ab_bar1 + AB_FREE_DELTA) - smbase, commit_mask);
    mma_cell(el_c6, d_tmem, ab_desc_lo(smbase, A_OFF, ab_slot2, 2), a_desc_hi,
                            ab_desc_lo(smbase, B_OFF, ab_slot2, 2), b_desc_hi,
             idesc_w0, sfa_t0, sfb_t0, 0xFFFFFFFFu);                   // step 6, atom 2
    mma_cell(el_c7, d_tmem, ab_desc_lo(smbase, A_OFF, ab_slot2, 5), a_desc_hi,
                            ab_desc_lo(smbase, B_OFF, ab_slot2, 5), b_desc_hi,
             idesc_w1, sfa_w1, sfb_w1, 0xFFFFFFFFu);                   // step 7, atom 5
    commit_consumed(el, smbase, (ab_bar2 + AB_FREE_DELTA) - smbase, commit_mask);

    return MainloopState{d_parity, ab_phase, ab_slot, sf_phase, sf_slot};
}

}  // namespace nvfp4

// ============================================================================
// The kernel. 224 threads = 7 warps, four jobs (blog: the warp table):
//   warps 0-3  epilogue: drain TMEM, convert to FP16, store straight to global
//   warp 4     the MMA warp (even CTA of each pair only): SF copies + MMAs
//   warp 5     TMA producer: A and B tiles into the 6-slot ring
//   warp 6     TMA producer: scale factors into the 7-slot ring
// ============================================================================
namespace nvfp4 {

// Promise ptxas the exact block shape and express the launch grid in cluster
// units. The launch helper below owns the corresponding grid convention.
#define NVFP4_BLOCK_ATTR __block_size__((TB_SIZE, 1, 1))
constexpr bool GRID_IN_CLUSTERS = true;

__global__
NVFP4_BLOCK_ATTR
__cluster_dims__(2, 1, 1)
__launch_bounds__(TB_SIZE, 1)
void nvfp4_gemm_kernel(const __grid_constant__ CUtensorMap A_tmap,
                       const __grid_constant__ CUtensorMap B_tmap,
                       const __grid_constant__ CUtensorMap SFA_tmap,
                       const __grid_constant__ CUtensorMap SFB_tmap,
                       __half* C_ptr, int M, int N, int K,
                       int table_offset
                       ) {
    extern __shared__ __align__(1024) char smem_buf[];
    SmemCD& smem = *reinterpret_cast<SmemCD*>(smem_buf);

    const int tid       = threadIdx.x;
    const int warp_id   = tid >> 5;
    const int lane_id   = tid & 31;
    const uint32_t crank  = ptx::cluster_rank();
    const uint32_t m_pair = crank & 1u;   // 0 = the pair's even CTA

    const int cluster_grid_m = (M + CLUSTER_M - 1) / CLUSTER_M;
    const int grid_n         = (N + BLOCK_N - 1) / BLOCK_N;
    const int total_tiles    = cluster_grid_m * grid_n;

    // Exact-K tail geometry (the exact-K feed, Other shapes; the host gates K % 16 == 0).
    const int full_groups = K / 768;
    const int tail_cells  = (K - full_groups * 768 + (MMA_K - 1)) / MMA_K;
    const int t_sf        = (tail_cells + 1) / 2;   // live SF slots in the tail
    const int num_groups  = full_groups + (tail_cells ? 1 : 0);
    // K64-dispatch class (banner): r = a x 96 + b x 64, minimal b in {0,1,2};
    // active only for EVEN a (r mod 192 in {64,128}: the with-id SF word is
    // architecturally sf_id=0-only at k_dim=0, and even a also excludes the
    // window-crossing decomposition). Cell count a + b equals tail_cells and
    // t_sf is unchanged, so only the A/B window count differs on the K64 path.
    const int k_rem   = K - full_groups * 768;
    const int b64_try = (k_rem % 96 == 64) ? 1 : (k_rem % 96 == 32) ? 2 : 0;
    const int a64     = tail_cells - b64_try;
    const int b64     = (b64_try != 0 && k_rem % 32 == 0 && a64 >= 0 &&
                         !(a64 & 1)) ? b64_try : 0;
    // Live A/B tail windows on the K64 path (b == 0 keeps all 3).
    const int t_win   = b64 ? (k_rem / 2 + 127) / 128 : 3;

    // Persistent cluster index. The device always sees the grid in CTA units
    // (2, 1, clusters) regardless of the host-side launch-unit regime.
    const bool grid_z      = (gridDim.z > 1);
    const int num_clusters = grid_z ? int(gridDim.z) : int(gridDim.x) / 2;
    const int cluster_id   = grid_z ? int(blockIdx.z) : int(blockIdx.x) / 2;

    // Placement audit (blog: the L2 trick): one predicated atom.global.add checks
    // this SM's censused side against the plan's side for this cluster slot,
    // on every launch, timed runs included. A nonzero counter voids the run.
    {
        unsigned smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        const unsigned bad = unsigned(tid == 0) &
                unsigned(l2a_sm_side[smid] != l2a_cluster_side[cluster_id]);
        asm volatile(
            "{\n\t.reg .pred p;\n\t.reg .u32 old;\n\t"
            "setp.ne.u32 p, %1, 0;\n\t"
            "@p atom.global.add.u32 old, [%0], 1;\n\t}"
            :: "l"(&l2a_placement_errors), "r"(bad) : "memory");
    }

    // Which output tile does persistent work item `t` process?
    auto tile_of = [&](int t) -> int {
        // The ownership schedule is data: a small constant table built on the
        // host. to_uniform keeps its decode on the uniform register datapath.
        return int(to_uniform(uint32_t(l2a_route_tables[table_offset + t])));
    };

    auto producer_tile_of = [&](int t) -> int {
        // This loop runs only on lane 0, so its route-table load is already
        // scalar. tile_of() instead performs a warp-wide uniform reduction.
        return int(l2a_route_tables[table_offset + t]);
    };

    // ---- mbarrier init (warp 0, lane 0) ----
    if (warp_id == 0 && ptx::lane_id() == 0) {
        // Arrival counts are derived from the topology, never tuned:
        // "ready" = one producer expect_tx per slot (bit-24 routes both
        // peers' bytes to the even CTA's barrier); "free" = one consumer
        // commit per slot reuse.
        #pragma unroll
        for (int s = 0; s < CD_AB_SLOTS; ++s) {
            ptx::mbar_init(&smem.hdr.ab_ready[s], 1);
            ptx::mbar_init(&smem.hdr.ab_free[s], 1);
        }
        #pragma unroll
        for (int s = 0; s < CD_SF_SLOTS; ++s) {
            ptx::mbar_init(&smem.hdr.sf_ready[s], 1);
            ptx::mbar_init(&smem.hdr.sf_free[s], 1);
        }
        ptx::mbar_init(&smem.hdr.acc_ready, 1);
        ptx::mbar_init(&smem.hdr.acc_free, N_EPI_WARPS * CTA_GROUP);   // 4 x 2 = 8
        ptx::mbar_init(&smem.hdr.dealloc, 32);   // one warp's lanes, cross-peer
        ptx::fence_mbarrier_init_release_cluster();
    }
    ptx::cluster_sync_rel_acq();

    // TMEM allocation: epilogue warp 0 allocates for the pair; warps 0-4
    // rendezvous before reading the granted base. The TMA producer warps
    // never touch TMEM and proceed straight to their loops.
    uint32_t taddr = 0;
    if (warp_id <= 4) {
        if (warp_id == 0)
            ptx::tcgen05_alloc_2sm(ptx::to_shared(&smem.hdr.tmem_addr),
                                   N_ALLOC_COLS);
        ptx::bar_sync(2, 160);
        taddr = smem.hdr.tmem_addr;
    }

    const int wg = warp_id >> 2;   // 0 = epilogue warps, 1 = producers + MMA
    if (wg == 1) {
        const int pw = warp_id - 4;
        if (pw == 1 && ptx::lane_id() == 0) {
            // ================================================================
            // Warp 5: lane 0 owns the complete A/B TMA producer loop.
            // ================================================================
            // free_phase starts at 1: a fresh mbarrier is phase 0, so the
            // first ring's worth of "free" waits falls through.
            int slot = 0, free_phase = 1;
            const uint32_t base = ptx::to_shared(&smem);
            for (int t = cluster_id; t < total_tiles; t += num_clusters) {
                const int tile  = producer_tile_of(t);
                const int m_row = tile % cluster_grid_m;
                const int n_grp = tile / cluster_grid_m;
                // The pair splits the 256x256 tile: even CTA holds A rows
                // [0,128) and B cols [0,128); odd CTA the other halves.
                const int m_blk = m_row * CTA_GROUP + int(m_pair);
                const int n_blk = n_grp * CTA_GROUP + int(m_pair);
                #pragma unroll 1
                for (int g = 0; g < num_groups; ++g) {
                    // Three full 128-byte windows cover the packed K=768 group.
                    #pragma unroll
                    for (int sub = 0; sub < AB_TMA_SUBS; ++sub) {
                        // The K64-dispatch tail ends at logical K: windows
                        // past t_win are never read — skip them entirely (no
                        // wait, no expect_tx, no cursor advance; the consumer
                        // skips symmetrically). Inactive remainders keep
                        // t_win == 3, so this never fires for them.
                        if (g == full_groups && sub >= t_win) break;
                        ptx::mbar_wait_parity(&smem.hdr.ab_free[slot], free_phase);
                        const uint32_t a_s =
                                base + CD_A_OFF + uint32_t(slot) * CD_AB_STRIDE;
                        const uint32_t b_s =
                                base + CD_B_OFF + uint32_t(slot) * CD_AB_STRIDE;
                        uint64_t* bar = &smem.hdr.ab_ready[slot];
                        if (m_pair == 0)
                            ptx::mbar_arrive_expect_tx_cluster(bar, AB_TX);
                        // A rides the multicast form with a self mask so the
                        // completion routes to the even peer's barrier.
                        const uint16_t mask_a = uint16_t(1u << m_pair);
                        // Read the same row-major FP4 allocation as cuBLAS.
                        // K192 uses four 96-byte boxes; K256 uses three
                        // 128-byte boxes. The tensor map zero-fills past K.
                        const CUtensorMap* a_map = &A_tmap;
                        const CUtensorMap* b_map = &B_tmap;
                        const int x = g * 384 + sub * AB_LOAD_BYTES;
                        const int y_a = m_blk * BLOCK_M;
                        const int y_b = n_blk * BLOCK_N_PEER;
                        const int z_a = 0;
                        const int z_b = 0;
                        ptx::cp_async_bulk_tensor_3d_load_multicast(
                                a_s, a_map, x, y_a, z_a,
                                bar, mask_a
                                );
                        ptx::cp_async_bulk_tensor_3d_load_2sm_bit24(
                                b_s, b_map, x, y_b, z_b,
                                bar
                        );
                        if (++slot == CD_AB_SLOTS) { slot = 0; free_phase ^= 1; }
                    }
                }
            }
            // Drain: wait for the last-filled slot to be consumed, then hand
            // it one clean-shutdown expect_tx.
            {
                const int t_slot  = (slot == 0) ? CD_AB_SLOTS - 1 : slot - 1;
                const int t_phase = (slot == 0) ? free_phase : (free_phase ^ 1);
                ptx::mbar_wait_parity(&smem.hdr.ab_free[t_slot],
                                              uint32_t(t_phase));
                if (m_pair == 0)
                    ptx::mbar_arrive_expect_tx_cluster(
                            &smem.hdr.ab_ready[t_slot], AB_TX);
            }
        } else if (pw == 2 && ptx::lane_id() == 0) {
            // ================================================================
            // Warp 6: scale-factor TMA producer. SFA is per-peer (each CTA's
            // own M half); SFB is the one genuinely multicast operand — the
            // cta_group::2 copy reads BOTH peers' smem as one logical
            // operand, so both copies must be byte-identical (blog: the scale-factor section).
            // ================================================================
            int slot = 0, free_phase = 1;
            const uint32_t base = ptx::to_shared(&smem);
            for (int t = cluster_id; t < total_tiles; t += num_clusters) {
                const int tile  = producer_tile_of(t);
                const int m_row = tile % cluster_grid_m;
                const int n_grp = tile / cluster_grid_m;
                const int m_blk = m_row * CTA_GROUP + int(m_pair);
                for (int g = 0; g < num_groups; ++g) {
                    // 4 SF ring slots per 768-K group (2 K=96 steps each).
                    #pragma unroll
                    for (int sub = 0; sub < 4; ++sub) {
                        // The tail group stores only its live SF slots; dead
                        // slots are skipped entirely (no wait, no arrive, no
                        // cursor advance — the consumer skips symmetrically).
                        if (g == full_groups && sub >= t_sf) break;
                        ptx::mbar_wait_parity(&smem.hdr.sf_free[slot], free_phase);
                        const uint32_t sfa_s =
                                base + CD_SFA_OFF + uint32_t(slot) * CD_SFA_STRIDE;
                        const uint32_t sfb_s =
                                base + CD_SFB_OFF + uint32_t(slot) * CD_SFB_STRIDE;
                        uint64_t* bar = &smem.hdr.sf_ready[slot];
                        if (m_pair == 0)
                            ptx::mbar_arrive_expect_tx_cluster(bar, SF_TX);
                        // VEC16 is [outer-block][K-group-of-4][128 rows x 4].
                        // Three adjacent groups are the 128 x 12 scale slot.
                        const int sf_group = (g * 4 + sub) * 3;
                        ptx::cp_async_bulk_tensor_4d_load_multicast(
                                sfa_s, &SFA_tmap, 0, 0, sf_group, m_blk, bar,
                                uint16_t(1u << m_pair)
                                );
                        ptx::cp_async_bulk_tensor_4d_load_multicast(
                                sfb_s + m_pair * uint32_t(CD_SFB_STRIDE / 2),
                                &SFB_tmap, 0, 0, sf_group,
                                n_grp * CTA_GROUP + int(m_pair), bar,
                                uint16_t(0x3)
                                );
                        if (++slot == CD_SF_SLOTS) { slot = 0; free_phase ^= 1; }
                    }
                }
            }
            {
                const int t_slot  = (slot == 0) ? CD_SF_SLOTS - 1 : slot - 1;
                const int t_phase = (slot == 0) ? free_phase : (free_phase ^ 1);
                ptx::mbar_wait_parity(&smem.hdr.sf_free[t_slot],
                                              uint32_t(t_phase));
                if (m_pair == 0)
                    ptx::mbar_arrive_expect_tx_cluster(
                            &smem.hdr.sf_ready[t_slot], SF_TX);
            }
        } else if (pw == 0 && m_pair == 0 && ptx::lane_id() == 0) {
            // ================================================================
            // Warp 4: the MMA warp — issued once per cluster, on the even
            // CTA. The odd CTA's warp 4 does nothing all kernel (blog: the warp table).
            // The mainloop needs no tile coordinates: operands arrive through
            // the rings; this warp only needs its share of the trip count.
            // ================================================================
            const uint32_t smbase = ptx::to_shared(&smem);
            const uint16_t commit_mask = 0x3;   // both CTAs of this pair
            MainloopState st{1u, 0u, 0u, 0u, 0u};
            for (int t = cluster_id; t < total_tiles; t += num_clusters) {
                st = mainloop_tile(taddr, smbase, st, commit_mask,
                                   uint32_t(full_groups),
                                   uint32_t(tail_cells),
                                   uint32_t(b64), uint32_t(t_win)
                                   );
                // The outer role branch already leaves exactly one issuer.
                asm volatile(
                    "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
                    ".shared::cluster.multicast::cluster.b64 [%0], %1;"
                    :: "r"(smbase + uint32_t(MB_ACC_READY)),
                       "h"(uint16_t(0x3)) : "memory");
            }
            // Drain the final tile's buffer-free arrive before exiting.
            mbar_wait_relaxed(smbase + uint32_t(MB_ACC_FREE), st.d_parity);
        }
    } else {
        // ====================================================================
        // Warps 0-3: the epilogue (blog: the epilogue). Each warp owns a 32-row band
        // and drains a tile as eight 32-column tcgen05.ld loads, converts
        // FP32->FP16 in packed pairs, and stores straight to global — no
        // shared-memory bounce.
        // ====================================================================
        const int epi_warp = warp_id;
        const uint32_t taddr_lane = uint32_t(epi_warp * 32) << 16;
        const uint32_t acc_free_rank = crank & ~1u;   // the pair's even CTA
        int d_db = 0;
        int acc_ready_phase = 0;
        for (int t = cluster_id; t < total_tiles; t += num_clusters) {
            const int tile  = tile_of(t);
            const int m_row = tile % cluster_grid_m;
            const int n_grp = tile / cluster_grid_m;
            const uint32_t tmem_base = taddr + uint32_t(d_db) * D_STRIDE;
            mbar_wait_relaxed(ptx::to_shared(&smem.hdr.acc_ready),
                              uint32_t(acc_ready_phase));
            acc_ready_phase ^= 1;

            const int off_m = (m_row * CTA_GROUP + int(m_pair)) * BLOCK_M;
            const int off_n = n_grp * BLOCK_N;
            const int row = off_m + epi_warp * 32 + lane_id;
            const bool row_in = (row < M);
            constexpr int EPI_BANDS = BLOCK_N / EPI_CHUNK;   // 8
            #pragma unroll 1
            for (int k = 0; k < EPI_BANDS; ++k) {
                // Buffer 0 drains high-to-low and buffer 1 low-to-high, so
                // their shared columns [220,256) drain first for both.
                const int parity = d_db & 1;
                const int band = (parity == 0) ? (EPI_BANDS - 1 - k) : k;
                const uint32_t taddr_n =
                        tmem_base + uint32_t(band) * 32 + taddr_lane;
                uint32_t w[32];
                ptx::tcgen05_ld_32x32b_x32(taddr_n, w);
                // Return the old buffer as soon as its shared edge is out.
                if (k == 1) {
                    ptx::tcgen05_wait_ld_then_arrive_release(
                            &smem.hdr.acc_free, acc_free_rank,
                            /*do_arrive=*/ptx::lane_id() == 0);
                }
                if (!row_in) continue;
                const int col_base = off_n + band * 32;
                uint32_t h[16];
                #pragma unroll
                for (int j = 0; j < 16; ++j) {
                    h[j] = ptx::cvt_pack_f16x2(__int_as_float(w[2 * j + 1]),
                                               __int_as_float(w[2 * j]));
                }
                const size_t row_base = size_t(row) * size_t(N);
                const bool aligned32 = ((row_base & 15u) == 0u) &&
                                       ((uintptr_t(C_ptr) & 31u) == 0u);
                if (aligned32 && (col_base & 15) == 0) {
                    // Precondition here: row_base is 16-element (32-byte)
                    // aligned — the aligned32 guard owns it. Restating it to
                    // ptxas via __builtin_assume compiled this file
                    // SASS-identical (and measured perf-null on the
                    // production kernel), so the guard stands alone.
                    #pragma unroll
                    for (int q = 0; q < 2; ++q) {
                        const int c0 = col_base + q * 16;
                        uint16_t* cdst =
                                reinterpret_cast<uint16_t*>(C_ptr) + row_base + c0;
                        if (c0 + 16 <= N) {
                            uint4 lo{h[q*8+0], h[q*8+1], h[q*8+2], h[q*8+3]};
                            uint4 hi{h[q*8+4], h[q*8+5], h[q*8+6], h[q*8+7]};
                            ptx::st_global_na_ef_b256(cdst, lo, hi);
                        } else {
                            // N tail: 128-bit then scalar.
                            #pragma unroll
                            for (int p = 0; p < 2; ++p) {
                                const int cq = c0 + p * 8;
                                if (cq + 8 <= N) {
                                    uint4 v{h[q*8+p*4+0], h[q*8+p*4+1],
                                            h[q*8+p*4+2], h[q*8+p*4+3]};
                                    ptx::st_global_na_b128(
                                            reinterpret_cast<uint4*>(cdst + p * 8), v);
                                } else {
                                    const uint16_t* hb =
                                            reinterpret_cast<const uint16_t*>(
                                                    &h[q * 8 + p * 4]);
                                    #pragma unroll
                                    for (int c = 0; c < 8; ++c)
                                        if (cq + c < N) cdst[p * 8 + c] = hb[c];
                                }
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int q = 0; q < 4; ++q) {
                        const int c0 = col_base + q * 8;
                        uint16_t* cdst =
                                reinterpret_cast<uint16_t*>(C_ptr) + row_base + c0;
                        if (c0 + 8 <= N) {
                            uint4 v{h[q*4+0], h[q*4+1], h[q*4+2], h[q*4+3]};
                            ptx::st_global_na_b128(
                                    reinterpret_cast<uint4*>(cdst), v);
                        } else {
                            const uint16_t* hb =
                                    reinterpret_cast<const uint16_t*>(&h[q * 4]);
                            #pragma unroll
                            for (int c = 0; c < 8; ++c)
                                if (c0 + c < N) cdst[c] = hb[c];
                        }
                    }
                }
            }
            if (++d_db == D_DB) d_db = 0;
        }
    }

    // Teardown: only the epilogue warps rendezvous per CTA; warp 0 then runs
    // the cross-peer dealloc handshake. Producer and MMA warps exit freely —
    // their drains above already order their work.
    if (wg == 0) {
        ptx::bar_sync(3, 128);
        if (warp_id == 0) {
            ptx::tcgen05_relinquish_2sm();
            const uint32_t peer_rank = crank ^ 1u;
            // All 32 lanes arrive the PEER's count-32 barrier = one flip.
            ptx::mbar_arrive_cluster_release(&smem.hdr.dealloc, peer_rank);
            ptx::mbar_wait_parity(&smem.hdr.dealloc, 0);
            ptx::tcgen05_dealloc_2sm(taddr, N_ALLOC_COLS);
        }
    }
}

// Cluster->SM census probe: same cluster shape, block shape, and full smem
// footprint as the GEMM kernel, so the persistent grid lands on the same
// physical SM assignment the timed launches will get.
__global__
NVFP4_BLOCK_ATTR
__cluster_dims__(2, 1, 1)
__launch_bounds__(TB_SIZE, 1)
void l2a_cluster_probe(unsigned* smids) {
    extern __shared__ char pad[];
    if (threadIdx.x != 0) return;
    pad[0] = 0;
    unsigned rank, smid;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    const int cluster = gridDim.z > 1 ? int(blockIdx.z) : int(blockIdx.x) / 2;
    smids[cluster * CTA_GROUP + int(rank)] = smid;
}

// Keep the host launch convention paired with the kernel's block attribute.
static inline dim3 launch_grid(int clusters) {
    return GRID_IN_CLUSTERS ? dim3(1, 1, unsigned(clusters))
                            : dim3(2, 1, unsigned(clusters));
}

}  // namespace nvfp4

// ============================================================================
// Host-side TMA descriptor support. The shared input and benchmark code lives
// in main.cu.
// ============================================================================
namespace tmap {

// Thin wrappers over cuTensorMapEncodeTiled. Dim 0 is the innermost
// (stride-1) axis; strides for dim 1+ are in bytes.
inline CUtensorMap encode_tiled_3d_box(void* global_ptr, CUtensorMapDataType dtype,
        uint64_t global_depth, uint64_t global_rows, uint64_t global_cols,
        uint64_t row_stride_bytes, uint64_t depth_stride_bytes,
        uint32_t box_depth, uint32_t box_rows, uint32_t box_cols,
        CUtensorMapSwizzle swizzle) {
    cuuint64_t global_dim[3]     = { global_cols, global_rows, global_depth };
    cuuint64_t global_strides[2] = { row_stride_bytes, depth_stride_bytes };
    cuuint32_t box_dim[3]        = { box_cols, box_rows, box_depth };
    cuuint32_t element_strides[3] = { 1, 1, 1 };
    CUtensorMap m{};
    CU_CHECK(cuTensorMapEncodeTiled(
        &m, dtype, 3, global_ptr, global_dim, global_strides, box_dim,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return m;
}

inline CUtensorMap encode_tiled_4d_box(void* global_ptr, CUtensorMapDataType dtype,
        uint64_t global_slots, uint64_t global_tiles,
        uint64_t global_rows, uint64_t global_cols,
        uint64_t row_stride_bytes, uint64_t tile_stride_bytes,
        uint64_t slot_stride_bytes,
        uint32_t box_tiles, uint32_t box_rows, uint32_t box_cols,
        CUtensorMapSwizzle swizzle) {
    cuuint64_t global_dim[4]     = { global_cols, global_rows, global_tiles,
                                     global_slots };
    cuuint64_t global_strides[3] = { row_stride_bytes, tile_stride_bytes,
                                     slot_stride_bytes };
    cuuint32_t box_dim[4]        = { box_cols, box_rows, box_tiles, 1u };
    cuuint32_t element_strides[4] = { 1, 1, 1, 1 };
    CUtensorMap m{};
    CU_CHECK(cuTensorMapEncodeTiled(
        &m, dtype, 4, global_ptr, global_dim, global_strides, box_dim,
        element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
    return m;
}

}  // namespace tmap

namespace host {

using namespace nvfp4;

// ---- implementation-specific tensor maps ---------------------------------

inline CUtensorMap make_ab_tmap(void* ptr, int rows, int K,
                                int row_stride) {
    return tmap::encode_tiled_3d_box(
            ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            /*depth=*/1, uint64_t(rows), /*cols=*/uint64_t(K / 2),
            /*row_stride=*/uint64_t(row_stride),
            /*depth_stride=*/uint64_t(rows) * uint64_t(row_stride),
            /*box=*/1, BLOCK_M, AB_LOAD_BYTES, CU_TENSOR_MAP_SWIZZLE_128B);
}

inline CUtensorMap make_sf_tmap(uint8_t* ptr, int outer, int K) {
    const uint64_t sf_inner = (uint64_t(K) + 63) / 64 * 4;
    const uint64_t outer_blocks = (uint64_t(outer) + 127) / 128;
    return tmap::encode_tiled_4d_box(
            ptr, CU_TENSOR_MAP_DATA_TYPE_UINT8,
            /*slots=*/outer_blocks, /*tiles=*/sf_inner / 4,
            /*rows=*/4, /*cols=*/128,
            /*row_stride=*/128, /*tile_stride=*/512,
            /*slot_stride=*/sf_inner * 128,
            /*box_tiles=*/3, /*box_rows=*/4, /*box_cols=*/128,
            CU_TENSOR_MAP_SWIZZLE_NONE);
}
}  // namespace host

// ============================================================================
// Runtime L2-side map (blog: "Interlude: your L2 is two L2s"). Everything
// here is measured at runtime — nothing hardcodes a split or trusts the
// address hash without checking it. The probing approach follows ademeure's
// QuickRunCUDA side_aware experiment (credited in the post).
//   P1  census: one dependent-load atomic round-trip per SM against a single
//       probe line; the bimodal split classifies each SM's side.
//   P2  address hash: walk single address bits; on GB300 the model is
//       parity(phys_addr & 0x1EF000) XOR a per-2MiB-slab phase.
//   P3  model check: predicted vs measured side over representative slots —
//       any mismatch fails closed (the harness then runs natural raster).
// ============================================================================
namespace l2side {

inline constexpr uint64_t kSlotBytes = 4096;      // hash granule (bit 12 is live)
inline constexpr uint64_t kSlabBytes = 2ull << 20;
inline constexpr uint32_t kExpectedHash = 0x1EF000u;
inline constexpr double kTimingResolutionNs = 1.0;

inline int popcount64(uint64_t v) { return __builtin_popcountll(v); }

struct RuntimeMap {
    int nsm = 0;
    std::array<int, 2> side_counts{};
    std::array<int, 256> sm_side{};
    std::vector<uint8_t> slab_phase;
    double threshold = 0.0;
    int near_smid = -1;
    double near_ns = 0.0;
    double far_ns = 0.0;
    double classification_gap_ns = 0.0;
    double minimum_sample_margin_ns = 0.0;
    double maximum_effective_jitter_ns = 0.0;
    double minimum_confidence_ns = 0.0;
    int stability_repeats = 1;
    uint32_t hash = 0;
    int model_checks = 0;
    int model_mismatches = 0;
    // Every side verdict and its distance from the threshold, in probe
    // order — the stability pass requires each margin to beat its own
    // observed jitter.
    std::vector<uint8_t> classification_bits;
    std::vector<double> classification_margins_ns;
};

namespace detail {

inline constexpr size_t kFatSmem = 130u << 10;   // one resident block per SM

[[noreturn]] inline void fail(const char* phase, const std::string& receipt) {
    throw std::runtime_error(std::string("l2side ") + phase + " failed: " + receipt);
}

inline void cuda_check(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        fail("CUDA", std::string(operation) + ": " + cudaGetErrorString(error));
}

template <class T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) {
        cuda_check(cudaMalloc(&ptr_, count * sizeof(T)), "cudaMalloc scratch");
    }
    ~DeviceBuffer() { if (ptr_ != nullptr) cudaFree(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    T* get() const { return ptr_; }
private:
    T* ptr_ = nullptr;
};

static __device__ __forceinline__ unsigned probe_smid() {
    unsigned value;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
    return value;
}

static __device__ __forceinline__ unsigned long long globaltimer_ns() {
    unsigned long long value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
    return value;
}

// Dependent-load chain of atomics: each result selects the next address, so
// latency can't hide behind parallelism. Probe words are pre-cleared, so
// value>>31 stays zero while still defeating unrolling.
static __device__ __forceinline__ unsigned long long
rtt_chain(unsigned* probe, int reps) {
    unsigned value = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) value = atomicAdd(probe + (value >> 31), 1u);
    const unsigned long long begin = globaltimer_ns();
    for (int i = 0; i < reps; ++i)
        value = atomicAdd(probe + (value >> 31), 1u);
    const unsigned long long end = globaltimer_ns();
    return (end - begin) / static_cast<unsigned>(reps) + (value >> 31);
}

// Fat dynamic smem keeps exactly one block resident per SM; the turn counter
// serializes them so every SM times the same line in isolation.
static __global__ void census_kernel(unsigned* probe, volatile unsigned* turn,
                                     unsigned long long* out_ns,
                                     unsigned* out_smid, int reps) {
    extern __shared__ char pad[];
    if (threadIdx.x != 0) return;
    pad[0] = 0;
    const int block = static_cast<int>(blockIdx.x);
    while (*turn != static_cast<unsigned>(block)) {}
    out_ns[block] = rtt_chain(probe, reps);
    out_smid[block] = probe_smid();
    __threadfence();
    *turn = static_cast<unsigned>(block + 1);
}

static __global__ void clear_probe_words(char* base,
                                         const uint64_t* offsets, int n) {
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    unsigned* probe = reinterpret_cast<unsigned*>(base + offsets[i]);
    probe[0] = 0;
    probe[1] = 0;
}

// Exactly one block, pinned to the near-side SM, sweeps the offset list.
static __global__ void probe_kernel(char* base, const uint64_t* offsets,
                                    int n, unsigned long long* out_ns,
                                    int target_smid, int reps, int* hit) {
    extern __shared__ char pad[];
    if (threadIdx.x != 0) return;
    pad[0] = 0;
    if (probe_smid() != static_cast<unsigned>(target_smid)) return;
    if (atomicCAS(hit, 0, 1) != 0) return;
    for (int i = 0; i < n; ++i)
        out_ns[i] = rtt_chain(reinterpret_cast<unsigned*>(base + offsets[i]), reps);
}

inline double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

inline std::vector<unsigned long long>
probe_offsets(char* arena, const std::vector<uint64_t>& offsets,
              int nsm, int near_smid) {
    const int n = static_cast<int>(offsets.size());
    DeviceBuffer<uint64_t> d_offsets(offsets.size());
    DeviceBuffer<unsigned long long> d_ns(offsets.size());
    DeviceBuffer<int> d_hit(1);
    cuda_check(cudaMemcpy(d_offsets.get(), offsets.data(),
                          offsets.size() * sizeof(offsets[0]),
                          cudaMemcpyHostToDevice), "upload probe offsets");
    cuda_check(cudaMemset(d_ns.get(), 0,
                          offsets.size() * sizeof(unsigned long long)),
               "clear probe timings");
    cuda_check(cudaMemset(d_hit.get(), 0, sizeof(int)), "clear probe hit");
    clear_probe_words<<<(n + 255) / 256, 256>>>(arena, d_offsets.get(), n);
    cuda_check(cudaGetLastError(), "launch clear_probe_words");
    probe_kernel<<<nsm, 32, kFatSmem>>>(arena, d_offsets.get(), n,
                                        d_ns.get(), near_smid, 48, d_hit.get());
    cuda_check(cudaGetLastError(), "launch probe_kernel");
    cuda_check(cudaDeviceSynchronize(), "synchronize probe_kernel");

    int hit = 0;
    cuda_check(cudaMemcpy(&hit, d_hit.get(), sizeof(hit),
                          cudaMemcpyDeviceToHost), "copy probe hit");
    if (hit != 1) fail("probe", "target smid " + std::to_string(near_smid) +
                                " was not scheduled");
    std::vector<unsigned long long> timings(offsets.size());
    cuda_check(cudaMemcpy(timings.data(), d_ns.get(),
                          timings.size() * sizeof(timings[0]),
                          cudaMemcpyDeviceToHost), "copy probe timings");
    return timings;
}

}  // namespace detail

// One full probe pass: census + address-hash walk + model check. Mutates
// probe words throughout `arena` (2 MiB-aligned, a multiple of 2 MiB long).
inline RuntimeMap probe(char* arena, uint64_t bytes) {
    if (arena == nullptr) detail::fail("input", "arena is null");
    if ((reinterpret_cast<uintptr_t>(arena) & (kSlabBytes - 1)) != 0)
        detail::fail("input", "arena is not 2 MiB aligned");
    if (bytes == 0 || (bytes & (kSlabBytes - 1)) != 0)
        detail::fail("input", "byte size is not a nonzero multiple of 2 MiB");
    const uint64_t nslabs64 = bytes / kSlabBytes;
    if (nslabs64 > 0x7fffffffu) detail::fail("input", "arena has too many slabs");

    int device = 0;
    detail::cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cudaDeviceProp prop{};
    detail::cuda_check(cudaGetDeviceProperties(&prop, device),
                       "cudaGetDeviceProperties");
    if (prop.major != 10 || prop.minor != 3)
        detail::fail("input", "device is not an sm_103 (GB300/B300) part");
    if (prop.multiProcessorCount <= 0 || prop.multiProcessorCount > 256)
        detail::fail("input", "nsm=" + std::to_string(prop.multiProcessorCount));
    if (prop.sharedMemPerBlockOptin < detail::kFatSmem)
        detail::fail("input", "opt-in shared memory is below 130 KiB");

    RuntimeMap result;
    result.nsm = prop.multiProcessorCount;
    result.sm_side.fill(-1);

    detail::cuda_check(cudaMemset(arena, 0, 2 * sizeof(unsigned)),
                       "clear census probe");
    detail::DeviceBuffer<unsigned> d_turn(1);
    detail::DeviceBuffer<unsigned long long> d_ns(result.nsm);
    detail::DeviceBuffer<unsigned> d_smid(result.nsm);
    detail::cuda_check(cudaMemset(d_turn.get(), 0, sizeof(unsigned)),
                       "clear census turn");
    detail::cuda_check(cudaFuncSetAttribute(
                           detail::census_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(detail::kFatSmem)),
                       "set census dynamic shared memory");
    detail::cuda_check(cudaFuncSetAttribute(
                           detail::probe_kernel,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(detail::kFatSmem)),
                       "set probe dynamic shared memory");
    detail::census_kernel<<<result.nsm, 32, detail::kFatSmem>>>(
        reinterpret_cast<unsigned*>(arena), d_turn.get(), d_ns.get(),
        d_smid.get(), 64);
    detail::cuda_check(cudaGetLastError(), "launch census_kernel");
    detail::cuda_check(cudaDeviceSynchronize(), "synchronize census_kernel");

    std::vector<unsigned long long> census_ns(result.nsm);
    std::vector<unsigned> census_smid(result.nsm);
    detail::cuda_check(cudaMemcpy(census_ns.data(), d_ns.get(),
                                  census_ns.size() * sizeof(census_ns[0]),
                                  cudaMemcpyDeviceToHost), "copy census timings");
    detail::cuda_check(cudaMemcpy(census_smid.data(), d_smid.get(),
                                  census_smid.size() * sizeof(census_smid[0]),
                                  cudaMemcpyDeviceToHost), "copy census smids");

    // Split the bimodal distribution at the largest gap near the middle.
    std::vector<double> sorted(census_ns.begin(), census_ns.end());
    std::sort(sorted.begin(), sorted.end());
    int cut = result.nsm / 2 - 1;
    double largest_gap = -1.0;
    for (int i = result.nsm / 4; i < 3 * result.nsm / 4 && i + 1 < result.nsm;
         ++i) {
        const double gap = sorted[i + 1] - sorted[i];
        if (gap > largest_gap) {
            largest_gap = gap;
            cut = i;
        }
    }
    result.threshold = (sorted[cut] + sorted[cut + 1]) * 0.5;
    result.classification_gap_ns = sorted[cut + 1] - sorted[cut];
    result.near_ns = detail::median(
        std::vector<double>(sorted.begin(), sorted.begin() + cut + 1));
    result.far_ns = detail::median(
        std::vector<double>(sorted.begin() + cut + 1, sorted.end()));

    const auto record_classification = [&](double timing_ns) {
        const int side = timing_ns > result.threshold;
        const double margin = side ? timing_ns - result.threshold
                                   : result.threshold - timing_ns;
        result.classification_bits.push_back(static_cast<uint8_t>(side));
        result.classification_margins_ns.push_back(margin);
        return side;
    };

    unsigned long long best_near = ~0ull;
    std::array<double, 256> sm_ns{};
    for (int block = 0; block < result.nsm; ++block) {
        const unsigned sm = census_smid[block];
        if (sm >= result.sm_side.size())
            detail::fail("P1", "smid=" + std::to_string(sm) + " exceeds 255");
        if (result.sm_side[sm] != -1)
            detail::fail("P1", "duplicate smid=" + std::to_string(sm));
        const int side = census_ns[block] > result.threshold;
        result.sm_side[sm] = side;
        sm_ns[sm] = static_cast<double>(census_ns[block]);
        ++result.side_counts[side];
        if (side == 0 && census_ns[block] < best_near) {
            best_near = census_ns[block];
            result.near_smid = static_cast<int>(sm);
        }
    }
    for (int sm = 0; sm < result.nsm; ++sm) {
        if (record_classification(sm_ns[sm]) != result.sm_side[sm])
            detail::fail("P1", "SM classification receipt mismatch");
    }
    // Splits are not always even — 72/80 and 74/78 boots exist — but a split
    // outside these bounds means the probe read something else entirely.
    const double ratio = result.far_ns / result.near_ns;
    const bool p1_ok = ratio >= 1.3 && result.side_counts[0] >= 60 &&
                       result.side_counts[0] <= 92 &&
                       result.side_counts[1] >= 60 &&
                       result.side_counts[1] <= 92 && result.near_ns >= 80.0 &&
                       result.near_ns <= 500.0 && result.near_smid >= 0;
    if (!p1_ok) {
        detail::fail("P1", "nsm=" + std::to_string(result.nsm) +
                           " split=" + std::to_string(result.side_counts[0]) +
                           "/" + std::to_string(result.side_counts[1]) +
                           " near_ns=" + std::to_string(result.near_ns) +
                           " far_ns=" + std::to_string(result.far_ns) +
                           " ratio=" + std::to_string(ratio));
    }

    // P2: which address bits flip the side? Walk bits 4..20 one at a time.
    std::vector<uint64_t> offsets{0};
    for (int bit = 4; bit <= 20; ++bit) offsets.push_back(1ull << bit);
    const auto hash_ns =
        detail::probe_offsets(arena, offsets, result.nsm, result.near_smid);
    const bool base_far = record_classification(hash_ns[0]);
    for (int bit = 4; bit <= 20; ++bit) {
        if (record_classification(hash_ns[bit - 3]) != base_far)
            result.hash |= 1u << bit;
    }
    if (result.hash != kExpectedHash) {
        detail::fail("P2", "hash=" + std::to_string(result.hash) +
                           " expected=" + std::to_string(kExpectedHash));
    }

    // Per-slab phase: allocator slab placement changes per boot, so measure.
    const int nslabs = static_cast<int>(nslabs64);
    offsets.clear();
    offsets.reserve(nslabs);
    for (int slab = 0; slab < nslabs; ++slab)
        offsets.push_back(static_cast<uint64_t>(slab) * kSlabBytes);
    const auto slab_ns =
        detail::probe_offsets(arena, offsets, result.nsm, result.near_smid);
    result.slab_phase.resize(nslabs);
    for (int slab = 0; slab < nslabs; ++slab)
        result.slab_phase[slab] = record_classification(slab_ns[slab]);

    // P3: the model must predict every representative slot. One miss = fail.
    std::vector<int> representative_slabs;
    const auto add_slab = [&](int slab) {
        if (std::find(representative_slabs.begin(), representative_slabs.end(),
                      slab) == representative_slabs.end())
            representative_slabs.push_back(slab);
    };
    add_slab(0);
    add_slab(std::min(1, nslabs - 1));
    add_slab(nslabs / 2);
    add_slab(nslabs - 1);
    constexpr std::array<int, 6> slots{1, 2, 3, 5, 8, 17};
    offsets.clear();
    for (int slab : representative_slabs)
        for (int slot : slots)
            offsets.push_back(static_cast<uint64_t>(slab) * kSlabBytes +
                              static_cast<uint64_t>(slot) * kSlotBytes);
    const auto model_ns =
        detail::probe_offsets(arena, offsets, result.nsm, result.near_smid);
    size_t i = 0;
    for (int slab : representative_slabs) {
        for (int slot : slots) {
            const auto within = static_cast<uint64_t>(slot) * kSlotBytes;
            const int expected = (popcount64(within & kExpectedHash) & 1) ^
                                 result.slab_phase[slab];
            const int observed = record_classification(model_ns[i++]);
            ++result.model_checks;
            result.model_mismatches += observed != expected;
        }
    }
    if (result.model_mismatches != 0) {
        detail::fail("P3", "model mismatches=" +
                           std::to_string(result.model_mismatches) + "/" +
                           std::to_string(result.model_checks));
    }
    result.minimum_sample_margin_ns = *std::min_element(
        result.classification_margins_ns.begin(),
        result.classification_margins_ns.end());
    result.minimum_confidence_ns = result.minimum_sample_margin_ns;
    return result;
}

// Repeat the whole probe; every SM/hash/slab/model verdict must hold, and
// each sample's minimum margin must exceed its own observed jitter (floored
// at the timer resolution). Any disagreement fails closed — no majority vote.
inline RuntimeMap probe_stable(char* arena, uint64_t bytes, int repeats = 3) {
    if (repeats < 2) detail::fail("stability", "repeats must be at least two");
    RuntimeMap result = probe(arena, bytes);
    double min_gap = result.classification_gap_ns;
    std::vector<double> minimum_margins = result.classification_margins_ns;
    std::vector<double> maximum_margins = result.classification_margins_ns;
    for (int repeat = 1; repeat < repeats; ++repeat) {
        RuntimeMap observed = probe(arena, bytes);
        if (observed.nsm != result.nsm ||
            observed.side_counts != result.side_counts ||
            observed.sm_side != result.sm_side ||
            observed.hash != result.hash ||
            observed.slab_phase != result.slab_phase ||
            observed.classification_bits != result.classification_bits ||
            observed.classification_margins_ns.size() !=
                result.classification_margins_ns.size() ||
            observed.model_mismatches != 0) {
            detail::fail("stability",
                         "SM or address-side classification changed on repeat " +
                             std::to_string(repeat + 1));
        }
        min_gap = std::min(min_gap, observed.classification_gap_ns);
        for (size_t i = 0; i < minimum_margins.size(); ++i) {
            minimum_margins[i] = std::min(
                minimum_margins[i], observed.classification_margins_ns[i]);
            maximum_margins[i] = std::max(
                maximum_margins[i], observed.classification_margins_ns[i]);
        }
    }
    double minimum_margin = minimum_margins[0];
    double maximum_effective_jitter = 0.0;
    double minimum_confidence = minimum_margins[0];
    for (size_t i = 0; i < minimum_margins.size(); ++i) {
        const double observed_jitter = maximum_margins[i] - minimum_margins[i];
        const double effective_jitter =
                std::max(observed_jitter, kTimingResolutionNs);
        const double confidence = minimum_margins[i] - effective_jitter;
        if (!(confidence > 0.0)) {
            detail::fail("stability",
                         "classification sample " + std::to_string(i) +
                         " margin_ns=" + std::to_string(minimum_margins[i]) +
                         " observed_jitter_ns=" + std::to_string(observed_jitter) +
                         " effective_jitter_ns=" + std::to_string(effective_jitter));
        }
        minimum_margin = std::min(minimum_margin, minimum_margins[i]);
        maximum_effective_jitter =
                std::max(maximum_effective_jitter, effective_jitter);
        minimum_confidence = std::min(minimum_confidence, confidence);
    }
    result.classification_gap_ns = min_gap;
    result.minimum_sample_margin_ns = minimum_margin;
    result.maximum_effective_jitter_ns = maximum_effective_jitter;
    result.minimum_confidence_ns = minimum_confidence;
    result.stability_repeats = repeats;
    return result;
}

}  // namespace l2side

// ============================================================================
// The schedule is data (blog Levers 2-3). The host builds a per-shape table
// mapping each persistent work slot to an output tile, proves it covers every
// tile exactly once and honors the ownership contract, uploads it to constant
// memory, and reads it back. Tiles encode m_row + n_col * mblocks (M fast).
// ============================================================================
namespace sched {

using nvfp4::L2A_ROUTE_WORK_CAP;

struct Census {
    int clusters = 0;
    int side_clusters[2]{};
    std::vector<int> cluster_side;
    std::array<std::vector<int>, 2> clusters_by_side;
};

// Classify the cluster->SM census against the probed side map: every SM seen
// exactly once, both CTAs of each cluster on one side, per-side SM counts
// consistent. A failure here is a bug or a mid-run reschedule — abort loudly.
inline Census classify_census(const std::vector<unsigned>& smids,
                              const l2side::RuntimeMap& map, int sms) {
    Census c;
    c.clusters = int(smids.size()) / 2;
    c.cluster_side.assign(c.clusters, -1);
    std::vector<unsigned char> seen(sms, 0);
    for (int cluster = 0; cluster < c.clusters; ++cluster) {
        int side = -1;
        for (int rank = 0; rank < 2; ++rank) {
            const unsigned smid = smids[size_t(cluster) * 2 + rank];
            if (smid >= unsigned(sms) || seen[smid]++) {
                std::fprintf(stderr, "cluster census coverage failed\n");
                std::abort();
            }
            const int actual = map.sm_side[smid];
            if (side < 0) side = actual;
            if (actual != side) {
                // Every census we've taken places both CTAs of a cluster on
                // one side; a mixed pair would invalidate the whole plan.
                std::fprintf(stderr, "mixed-side physical cluster %d\n", cluster);
                std::abort();
            }
        }
        c.cluster_side[cluster] = side;
        ++c.side_clusters[side];
        c.clusters_by_side[side].push_back(cluster);
    }
    if (std::find(seen.begin(), seen.end(), 0) != seen.end() ||
        2 * c.side_clusters[0] != map.side_counts[0] ||
        2 * c.side_clusters[1] != map.side_counts[1]) {
        std::fprintf(stderr, "cluster/SM side-count gate failed\n");
        std::abort();
    }
    return c;
}

// Per-cluster trip counts under the persistent stride, plus the proof that
// they spread by at most one (the table preserves each cluster's trip count,
// so a wrong spread would make quota assignment impossible).
inline std::vector<int> cluster_trips(int clusters, int total_work) {
    std::vector<int> trips(clusters, 0);
    for (int cluster = 0; cluster < clusters; ++cluster)
        for (int work = cluster; work < total_work; work += clusters)
            ++trips[cluster];
    const int trip_floor = total_work / clusters;
    const int long_trips = total_work % clusters;
    const int trip_ceil = trip_floor + (long_trips != 0);
    const int observed_long = trip_ceil == trip_floor
        ? 0 : int(std::count(trips.begin(), trips.end(), trip_ceil));
    const int observed_short =
        int(std::count(trips.begin(), trips.end(), trip_floor));
    if (trip_ceil - trip_floor > 1 || observed_long != long_trips ||
        observed_short != clusters - long_trips) {
        std::fprintf(stderr, "one-trip-spread proof failed\n");
        std::abort();
    }
    return trips;
}

// The owned M-row partition: split the M rows between the two sides in
// proportion to each side's work share, with at most one boundary row shared
// (split along N). Every re-read of a given A row then comes from one side
// only — you can't make the buffer side-pure (the hash reaches bit 12), but
// you CAN own the reuse.
struct OwnedPartition {
    int side_work[2]{};
    int boundary_row = -1;
    int boundary_n = 0;
    std::array<std::vector<int>, 2> pool_tiles;   // M-fast order within pools
};

inline bool make_owned_partition(const Census& census,
                                 const std::vector<int>& trips,
                                 int mblocks, int nblocks, OwnedPartition* out) {
    OwnedPartition p;
    for (int side = 0; side < 2; ++side)
        for (const int cluster : census.clusters_by_side[side])
            p.side_work[side] += trips[cluster];
    const int total_work = mblocks * nblocks;
    if (p.side_work[0] + p.side_work[1] != total_work) std::abort();
    p.boundary_row = p.side_work[0] / nblocks;
    p.boundary_n = p.side_work[0] % nblocks;
    if (p.boundary_row <= 0 || p.boundary_row >= mblocks ||
        (p.boundary_n != 0 && p.boundary_row + 1 >= mblocks)) {
        return false;   // grid too small to own — the caller falls back
    }
    for (int n = 0; n < nblocks; ++n) {
        for (int mrow = 0; mrow < p.boundary_row; ++mrow)
            p.pool_tiles[0].push_back(mrow + n * mblocks);
        if (p.boundary_n != 0 && n < p.boundary_n)
            p.pool_tiles[0].push_back(p.boundary_row + n * mblocks);
        if (p.boundary_n != 0 && n >= p.boundary_n)
            p.pool_tiles[1].push_back(p.boundary_row + n * mblocks);
        for (int mrow = p.boundary_row + (p.boundary_n != 0); mrow < mblocks;
             ++mrow)
            p.pool_tiles[1].push_back(mrow + n * mblocks);
    }
    if (int(p.pool_tiles[0].size()) != p.side_work[0] ||
        int(p.pool_tiles[1].size()) != p.side_work[1]) {
        std::fprintf(stderr, "owned pool-size mismatch\n");
        std::abort();
    }
    *out = std::move(p);
    return true;
}

// Fill one side of the table from its pool, preserving every cluster's trip
// count exactly (clusters sorted by trips, pool quotas by rank).
inline void assign_owned_pool(std::vector<int>& table, int side,
                              const Census& census,
                              const std::vector<int>& trips,
                              const std::vector<int>& pool) {
    const int clusters = census.clusters;
    const int total_work = int(table.size());
    const int count = int(census.clusters_by_side[side].size());
    const int work_count = int(pool.size());
    std::vector<int> ranks(count);
    std::iota(ranks.begin(), ranks.end(), 0);
    std::sort(ranks.begin(), ranks.end(), [&](int a, int b) {
        const int na = (work_count + count - 1 - a) / count;
        const int nb = (work_count + count - 1 - b) / count;
        return na != nb ? na > nb : a < b;
    });
    std::vector<int> ids = census.clusters_by_side[side];
    std::sort(ids.begin(), ids.end(), [&](int a, int b) {
        return trips[a] != trips[b] ? trips[a] > trips[b] : a < b;
    });
    for (int i = 0; i < count; ++i) {
        const int cluster = ids[i];
        const int rank = ranks[i];
        const int need = (work_count + count - 1 - rank) / count;
        if (trips[cluster] != need) {
            std::fprintf(stderr, "cannot preserve cluster trip count\n");
            std::abort();
        }
        int trip = 0;
        for (int work = cluster; work < total_work;
             work += clusters, ++trip) {
            const int pool_index = rank + trip * count;
            if (pool_index >= work_count) std::abort();
            table[work] = pool[pool_index];
        }
    }
}

// Exactly-once coverage (every table, every time).
inline bool coverage_ok(const std::vector<int>& t, int total_work) {
    std::vector<unsigned char> seen(total_work, 0);
    for (int w = 0; w < total_work; ++w)
        if (t[w] < 0 || t[w] >= total_work || seen[t[w]]++) return false;
    return true;
}

inline bool verify_identity(const std::vector<int>& table) {
    if (!coverage_ok(table, int(table.size()))) return false;
    for (int w = 0; w < int(table.size()); ++w)
        if (table[w] != w) return false;
    return true;
}

// The ownership contract: exactly-once coverage; every M row's requesters
// side-pure (split at the boundary row); every N column visited by both
// sides (B stays shared by everyone).
inline bool verify_owned(const std::vector<int>& table, const Census& census,
                         int mblocks, int nblocks, const OwnedPartition& part) {
    const int total_work = mblocks * nblocks;
    std::vector<unsigned char> visits(total_work, 0);
    std::vector<unsigned> m_masks(mblocks, 0), n_masks(nblocks, 0);
    for (int cluster = 0; cluster < census.clusters; ++cluster) {
        const unsigned side_bit = 1u << census.cluster_side[cluster];
        for (int work = cluster; work < total_work; work += census.clusters) {
            const int tile = table[work];
            if (tile < 0 || tile >= total_work || visits[tile]++) return false;
            m_masks[tile % mblocks] |= side_bit;
            n_masks[tile / mblocks] |= side_bit;
        }
    }
    if (std::find(visits.begin(), visits.end(), 0) != visits.end())
        return false;
    for (int mrow = 0; mrow < mblocks; ++mrow) {
        unsigned want = mrow < part.boundary_row ? 1u : 2u;
        if (part.boundary_n != 0 && mrow == part.boundary_row) want = 3u;
        if (m_masks[mrow] != want) return false;
    }
    return std::find_if(n_masks.begin(), n_masks.end(),
                        [](unsigned mask) { return mask != 3u; }) ==
           n_masks.end();
}

// Matched ownership control: every A row must be requested from both sides,
// while every B column remains shared exactly as in the owned table.
inline bool verify_blind(const std::vector<int>& table, const Census& census,
                         int mblocks, int nblocks) {
    const int total_work = mblocks * nblocks;
    std::vector<unsigned char> visits(total_work, 0);
    std::vector<unsigned> m_masks(mblocks, 0), n_masks(nblocks, 0);
    for (int cluster = 0; cluster < census.clusters; ++cluster) {
        const unsigned side_bit = 1u << census.cluster_side[cluster];
        for (int work = cluster; work < total_work; work += census.clusters) {
            const int tile = table[work];
            if (tile < 0 || tile >= total_work || visits[tile]++) return false;
            m_masks[tile % mblocks] |= side_bit;
            n_masks[tile / mblocks] |= side_bit;
        }
    }
    if (std::find(visits.begin(), visits.end(), 0) != visits.end())
        return false;
    return std::find_if(m_masks.begin(), m_masks.end(),
                        [](unsigned mask) { return mask != 3u; }) ==
               m_masks.end() &&
           std::find_if(n_masks.begin(), n_masks.end(),
                        [](unsigned mask) { return mask != 3u; }) ==
               n_masks.end();
}

// Derive a blind twin from an owned table. Rotating M by an N-dependent
// amount preserves each work slot's N coordinate, so the B/SFB request stream,
// cluster trip counts, kernel instructions, and table lookup are unchanged.
inline std::vector<int> build_blind_twin(const std::vector<int>& owned,
                                         const Census& census, int mblocks,
                                         int nblocks) {
    const int total_work = mblocks * nblocks;
    std::vector<int> blind(total_work, -1);
    for (int stride = 1; stride < mblocks; ++stride) {
        for (int work = 0; work < total_work; ++work) {
            const int tile = owned[work];
            const int mrow = tile % mblocks;
            const int n = tile / mblocks;
            blind[work] = (mrow + n * stride) % mblocks + n * mblocks;
        }
        if (!verify_blind(blind, census, mblocks, nblocks)) continue;
        for (int work = 0; work < total_work; ++work)
            if (blind[work] / mblocks != owned[work] / mblocks)
                std::abort();
        std::printf("SCHEDULE_BLIND stride=%d same_n_slot=PASS\n", stride);
        return blind;
    }
    std::fprintf(stderr, "found no row rotation for a blind schedule twin\n");
    std::abort();
}

// ---- visit orders (blog: the L2 trick, part two) --------------------------------------------

// Standard iterative Hilbert d2xy over an npow2 x npow2 grid.
inline void hilbert_d2xy(int npow2, int d, int& x, int& y) {
    x = 0;
    y = 0;
    for (int s = 1; s < npow2; s *= 2) {
        const int rx = 1 & (d / 2);
        const int ry = 1 & (d ^ rx);
        if (ry == 0) {
            if (rx == 1) {
                x = s - 1 - x;
                y = s - 1 - y;
            }
            const int t = x;
            x = y;
            y = t;
        }
        x += s * rx;
        y += s * ry;
        d /= 4;
    }
}

// Hilbert order over a rows x cols sub-grid whose first M row is row0; the
// curve covers the next power of two and out-of-grid points are skipped.
inline std::vector<int> hilbert_order(int rows, int cols, int row0, int mblocks) {
    int npow2 = 1;
    while (npow2 < rows || npow2 < cols) npow2 *= 2;
    std::vector<int> order;
    order.reserve(size_t(rows) * cols);
    for (int d = 0; d < npow2 * npow2; ++d) {
        int x, y;
        hilbert_d2xy(npow2, d, x, y);
        if (x < rows && y < cols)
            order.push_back(row0 + x + y * mblocks);
    }
    return order;
}

// Hilbert WITHIN each side's owned row range: the curve provides locality,
// the pools keep every A re-read on the requester's own side. This is the
// order that beats every other reorder at every deep-K shape in the post.
inline std::array<std::vector<int>, 2> hilbertown_pools(
        const OwnedPartition& part, int mblocks, int nblocks) {
    const int rows0 = part.boundary_row + (part.boundary_n != 0);
    std::array<std::vector<int>, 2> hpool;
    for (const int tile : hilbert_order(rows0, nblocks, 0, mblocks)) {
        const int mrow = tile % mblocks, n = tile / mblocks;
        if (mrow < part.boundary_row ||
            (part.boundary_n != 0 && n < part.boundary_n))
            hpool[0].push_back(tile);
    }
    for (const int tile : hilbert_order(mblocks - part.boundary_row, nblocks,
                                        part.boundary_row, mblocks)) {
        const int mrow = tile % mblocks, n = tile / mblocks;
        if (mrow > part.boundary_row || part.boundary_n == 0 ||
            n >= part.boundary_n)
            hpool[1].push_back(tile);
    }
    if (int(hpool[0].size()) != part.side_work[0] ||
        int(hpool[1].size()) != part.side_work[1]) {
        std::fprintf(stderr, "hilbertown pool-size mismatch\n");
        std::abort();
    }
    return hpool;
}

// Pocket order: walk the tile grid in R x C rectangles, finishing each
// pocket before the next, pocket bands serpentine. 8x4 won the shape sweep
// on big squares (and bought the ~10% traffic cut the power governor repaid
// in clock — the blog's strangest section).
inline std::vector<int> pocket_order(int R, int C, int mblocks, int nblocks) {
    std::vector<int> order;
    order.reserve(size_t(mblocks) * nblocks);
    const int prows = (mblocks + R - 1) / R;
    const int pcols = (nblocks + C - 1) / C;
    for (int pi = 0; pi < prows; ++pi)
        for (int t = 0; t < pcols; ++t) {
            const int pj = pi % 2 ? pcols - 1 - t : t;
            const int m1 = std::min((pi + 1) * R, mblocks);
            const int n1 = std::min((pj + 1) * C, nblocks);
            for (int m = pi * R; m < m1; ++m)
                for (int n = pj * C; n < n1; ++n)
                    order.push_back(m + n * mblocks);
        }
    return order;
}

// Split a full-grid visit order into the two owned pools, preserving the
// per-side order — the pocket/side composition step.
inline std::array<std::vector<int>, 2> side_filter(
        const std::vector<int>& order, const OwnedPartition& part, int mblocks) {
    std::array<std::vector<int>, 2> pools;
    for (const int tile : order) {
        const int mrow = tile % mblocks, n = tile / mblocks;
        const int side = mrow < part.boundary_row ? 0
                       : mrow > part.boundary_row ? 1
                       : (part.boundary_n != 0 && n < part.boundary_n) ? 0 : 1;
        pools[side].push_back(tile);
    }
    if (int(pools[0].size()) != part.side_work[0] ||
        int(pools[1].size()) != part.side_work[1]) {
        std::fprintf(stderr, "side-filter pool-size mismatch\n");
        std::abort();
    }
    return pools;
}

enum class ScheduleMode {
    AUTO,
    RASTER,
    POCKET_8x4,
    POCKET_8x8,
    HILBERT,
    BLIND_PLAIN,
    BLIND_POCKET_8x4,
    BLIND_POCKET_8x8,
    HILBERT_IN_BLIND,
    OWNED_PLAIN,
    OWNED_POCKET_8x4,
    OWNED_POCKET_8x8,
    HILBERT_IN_OWNED,
};

inline const char* schedule_name(ScheduleMode mode) {
    switch (mode) {
        case ScheduleMode::AUTO:               return "auto";
        case ScheduleMode::RASTER:             return "raster";
        case ScheduleMode::POCKET_8x4:         return "pocket-8x4";
        case ScheduleMode::POCKET_8x8:         return "pocket-8x8";
        case ScheduleMode::HILBERT:             return "hilbert";
        case ScheduleMode::BLIND_PLAIN:         return "blind-plain";
        case ScheduleMode::BLIND_POCKET_8x4:    return "blind-pocket-8x4";
        case ScheduleMode::BLIND_POCKET_8x8:    return "blind-pocket-8x8";
        case ScheduleMode::HILBERT_IN_BLIND:    return "hilbert-in-blind";
        case ScheduleMode::OWNED_PLAIN:         return "owned-plain";
        case ScheduleMode::OWNED_POCKET_8x4:    return "owned-pocket-8x4";
        case ScheduleMode::OWNED_POCKET_8x8:    return "owned-pocket-8x8";
        case ScheduleMode::HILBERT_IN_OWNED:    return "hilbert-in-owned";
    }
    return "?";
}

inline bool parse_schedule_mode(const std::string& name, ScheduleMode* mode) {
    static const std::array<ScheduleMode, 13> modes = {
        ScheduleMode::AUTO,
        ScheduleMode::RASTER,
        ScheduleMode::POCKET_8x4,
        ScheduleMode::POCKET_8x8,
        ScheduleMode::HILBERT,
        ScheduleMode::BLIND_PLAIN,
        ScheduleMode::BLIND_POCKET_8x4,
        ScheduleMode::BLIND_POCKET_8x8,
        ScheduleMode::HILBERT_IN_BLIND,
        ScheduleMode::OWNED_PLAIN,
        ScheduleMode::OWNED_POCKET_8x4,
        ScheduleMode::OWNED_POCKET_8x8,
        ScheduleMode::HILBERT_IN_OWNED,
    };
    for (const ScheduleMode candidate : modes) {
        if (name == schedule_name(candidate)) {
            *mode = candidate;
            return true;
        }
    }
    return false;
}

inline bool uses_side_ownership(ScheduleMode mode) {
    return mode == ScheduleMode::OWNED_PLAIN ||
           mode == ScheduleMode::OWNED_POCKET_8x4 ||
           mode == ScheduleMode::OWNED_POCKET_8x8 ||
           mode == ScheduleMode::HILBERT_IN_OWNED;
}

inline bool uses_blind_twin(ScheduleMode mode) {
    return mode == ScheduleMode::BLIND_PLAIN ||
           mode == ScheduleMode::BLIND_POCKET_8x4 ||
           mode == ScheduleMode::BLIND_POCKET_8x8 ||
           mode == ScheduleMode::HILBERT_IN_BLIND;
}

inline bool requires_side_census(ScheduleMode mode) {
    return uses_side_ownership(mode) || uses_blind_twin(mode);
}

inline const char* requester_assignment(ScheduleMode mode) {
    return uses_side_ownership(mode) ? "owned"
         : uses_blind_twin(mode) ? "blind"
                                 : "none";
}

// Own the sides always. The 32x32 grid uses the measured 8x8 winner. Other
// grids below the reorder knee keep the plain owned order; past it, pick by
// aspect. The knee sits between the measured ~99 MB still-loses and ~159 MB
// wins points; 128 MiB (both L2 sides) is the threshold this file ships.
inline ScheduleMode pick_schedule(size_t operand_read_bytes, int mblocks,
                                  int nblocks) {
    constexpr size_t REORDER_KNEE = size_t(128) << 20;
    if (operand_read_bytes < REORDER_KNEE) {
        if (mblocks == 32 && nblocks == 32)
            return ScheduleMode::OWNED_POCKET_8x8;
        return ScheduleMode::OWNED_PLAIN;
    }
    const int lo = std::min(mblocks, nblocks);
    const int hi = std::max(mblocks, nblocks);
    if (lo >= 48 && hi * 4 <= lo * 5)
        return ScheduleMode::OWNED_POCKET_8x4;   // aspect <= 1.25
    return ScheduleMode::HILBERT_IN_OWNED;
}

// Full-grid controls: these change only tile visit order. They deliberately do
// not probe, classify, or partition the two L2 sides.
inline std::vector<int> build_unowned_schedule(ScheduleMode mode, int mblocks,
                                               int nblocks) {
    const int total_work = mblocks * nblocks;
    std::vector<int> table;
    switch (mode) {
        case ScheduleMode::RASTER:
            table.resize(total_work);
            std::iota(table.begin(), table.end(), 0);
            break;
        case ScheduleMode::POCKET_8x4:
            table = pocket_order(8, 4, mblocks, nblocks);
            break;
        case ScheduleMode::POCKET_8x8:
            table = pocket_order(8, 8, mblocks, nblocks);
            break;
        case ScheduleMode::HILBERT:
            table = hilbert_order(mblocks, nblocks, 0, mblocks);
            break;
        default:
            std::abort();
    }
    if (!coverage_ok(table, total_work)) {
        std::fprintf(stderr, "unowned schedule failed exactly-once coverage\n");
        std::abort();
    }
    return table;
}

// Build + prove one schedule table. Returns empty on an unbuildable
// partition (the caller falls back to natural raster and says so).
inline std::vector<int> build_schedule(const Census& census, int mblocks,
                                       int nblocks,
                                       ScheduleMode mode
                                       ) {
    const int total_work = mblocks * nblocks;
    const std::vector<int> trips = cluster_trips(census.clusters, total_work);
    OwnedPartition part;
    if (!make_owned_partition(census, trips, mblocks, nblocks, &part))
        return {};
    std::array<std::vector<int>, 2> pools;
    switch (mode) {
        case ScheduleMode::BLIND_PLAIN:
        case ScheduleMode::OWNED_PLAIN:
            pools = part.pool_tiles;
            break;
        case ScheduleMode::BLIND_POCKET_8x4:
        case ScheduleMode::OWNED_POCKET_8x4:
            pools = side_filter(pocket_order(8, 4, mblocks, nblocks), part,
                                mblocks);
            break;
        case ScheduleMode::BLIND_POCKET_8x8:
        case ScheduleMode::OWNED_POCKET_8x8:
            pools = side_filter(pocket_order(8, 8, mblocks, nblocks), part,
                                mblocks);
            break;
        case ScheduleMode::HILBERT_IN_BLIND:
        case ScheduleMode::HILBERT_IN_OWNED:
            pools = hilbertown_pools(part, mblocks, nblocks);
            break;
        default:
            std::abort();
    }
    std::vector<int> table(total_work, -1);
    assign_owned_pool(table, 0, census, trips, pools[0]);
    assign_owned_pool(table, 1, census, trips, pools[1]);
    if (!verify_owned(table, census, mblocks, nblocks, part)) {
        std::fprintf(stderr, "schedule table failed the ownership contract\n");
        std::abort();
    }
    if (uses_blind_twin(mode))
        return build_blind_twin(table, census, mblocks, nblocks);
    return table;
}

// Upload one table slot; read it back and require equality.
inline void upload_table(const std::vector<int>& table, int mode) {
    const int total_work = int(table.size());
    if (total_work > L2A_ROUTE_WORK_CAP) std::abort();
    const size_t offset = size_t(mode) * L2A_ROUTE_WORK_CAP * sizeof(int);
    CUDA_CHECK(cudaMemcpyToSymbol(nvfp4::l2a_route_tables, table.data(),
                                  total_work * sizeof(int), offset,
                                  cudaMemcpyHostToDevice));
    std::vector<int> roundtrip(total_work, -1);
    CUDA_CHECK(cudaMemcpyFromSymbol(roundtrip.data(), nvfp4::l2a_route_tables,
                                    total_work * sizeof(int), offset,
                                    cudaMemcpyDeviceToHost));
    if (roundtrip != table) {
        std::fprintf(stderr, "constant table roundtrip failed mode=%d\n", mode);
        std::abort();
    }
}

inline void upload_side_constants(const l2side::RuntimeMap& map,
                                  const Census& census) {
    CUDA_CHECK(cudaMemcpyToSymbol(nvfp4::l2a_sm_side, map.sm_side.data(),
                                  sizeof(nvfp4::l2a_sm_side)));
    CUDA_CHECK(cudaMemcpyToSymbol(nvfp4::l2a_cluster_side,
                                  census.cluster_side.data(),
                                  census.clusters * sizeof(int)));
    const unsigned zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(nvfp4::l2a_placement_errors, &zero,
                                  sizeof(zero)));
}

// The fallback: identity tables, all sides zeroed (the placement audit is
// inert — every SM trivially matches side 0).
inline void upload_fallback_sides() {
    static const std::array<int, nvfp4::L2A_SM_CAP> zeros_sm{};
    static const std::array<int, nvfp4::L2A_CLUSTER_CAP> zeros_cl{};
    CUDA_CHECK(cudaMemcpyToSymbol(nvfp4::l2a_sm_side, zeros_sm.data(),
                                  sizeof(nvfp4::l2a_sm_side)));
    CUDA_CHECK(cudaMemcpyToSymbol(nvfp4::l2a_cluster_side, zeros_cl.data(),
                                  sizeof(nvfp4::l2a_cluster_side)));
    const unsigned zero = 0;
    CUDA_CHECK(cudaMemcpyToSymbol(nvfp4::l2a_placement_errors, &zero,
                                  sizeof(zero)));
}

}  // namespace sched
