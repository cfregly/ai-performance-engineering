#pragma once
// r2: stage K256 A/B windows and release each after its last reader.
// This is a complete, directly editable ladder snapshot.
// It intentionally duplicates the implementation in the other gemm*.cuh files.

// Conditional branches are already materialized; edit this implementation directly.
#define NVFP4_BUILD_NAME "r2"
// Read-only metadata for main.cu; these do not select code in this header.
#define NVFP4_HAS_K64_TAIL 0
#define NVFP4_HAS_SCHEDULE 0

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

// Baseline output store: use the default cache policy.
static __device__ __forceinline__ void st_global_b128(uint4* ptr, uint4 v) {
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                 :: "l"(ptr), "r"(v.x), "r"(v.y), "r"(v.z), "r"(v.w));
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
// Bring-up layout: one accumulator followed immediately by the scale factors.
constexpr int D_STRIDE     = BLOCK_N;
constexpr int D_DB         = 1;
constexpr int N_D_COLS     = BLOCK_N;
constexpr int SF_TMEM_BASE = BLOCK_N;
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

// One output tile: `full_groups` complete 768-element K-groups. Coordinates
// past logical K are zero-filled by TMA from the unchanged canonical inputs.
static __device__ __forceinline__ MainloopState mainloop_tile(
        uint32_t taddr, uint32_t smbase, MainloopState in, uint16_t commit_mask,
        uint32_t full_groups
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
    // Single buffer: every tile writes buf0; d_parity still toggles because
    // it carries the acc_free handshake parity.
    const uint32_t d_tmem = taddr;

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

// Baseline launch: plain launch bounds and a grid expressed in CTA units.
#define NVFP4_BLOCK_ATTR
constexpr bool GRID_IN_CLUSTERS = false;

__global__
NVFP4_BLOCK_ATTR
__cluster_dims__(2, 1, 1)
__launch_bounds__(TB_SIZE, 1)
void nvfp4_gemm_kernel(const __grid_constant__ CUtensorMap A_tmap,
                       const __grid_constant__ CUtensorMap B_tmap,
                       const __grid_constant__ CUtensorMap SFA_tmap,
                       const __grid_constant__ CUtensorMap SFB_tmap,
                       __half* C_ptr, int M, int N, int K
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

    // Baseline rungs execute a whole K=768 group at the edge. Tensor-map OOB
    // fill supplies zeros past logical K without changing the input buffers.
    const int full_groups = (K + 767) / 768;
    const int num_groups  = full_groups;

    // Persistent cluster index. The device always sees the grid in CTA units
    // (2, 1, clusters) regardless of the host-side launch-unit regime.
    const bool grid_z      = (gridDim.z > 1);
    const int num_clusters = grid_z ? int(gridDim.z) : int(gridDim.x) / 2;
    const int cluster_id   = grid_z ? int(blockIdx.z) : int(blockIdx.x) / 2;


    // Which output tile does persistent work item `t` process?
    auto tile_of = [&](int t) -> int {
        return t;   // natural raster: M fast, N slow
    };

    auto producer_tile_of = [&](int t) -> int {
        return tile_of(t);
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
                                   uint32_t(full_groups)
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
                const int band = k;
                const uint32_t taddr_n =
                        tmem_base + uint32_t(band) * 32 + taddr_lane;
                uint32_t w[32];
                ptx::tcgen05_ld_32x32b_x32(taddr_n, w);
                // One buffer cannot be reused until all eight bands are out.
                if (k == EPI_BANDS - 1) {
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
                            // Bring-up form: twice the store requests,
                            // twice the L2 store traffic (the ladder's
                            // tensor-activity 82% -> 42% row).
                            ptx::st_global_b128(
                                    reinterpret_cast<uint4*>(cdst), lo);
                            ptx::st_global_b128(
                                    reinterpret_cast<uint4*>(cdst) + 1, hi);
                        } else {
                            // N tail: 128-bit then scalar.
                            #pragma unroll
                            for (int p = 0; p < 2; ++p) {
                                const int cq = c0 + p * 8;
                                if (cq + 8 <= N) {
                                    uint4 v{h[q*8+p*4+0], h[q*8+p*4+1],
                                            h[q*8+p*4+2], h[q*8+p*4+3]};
                                    ptx::st_global_b128(
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
                            ptx::st_global_b128(reinterpret_cast<uint4*>(cdst), v);
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
