# GB300 blackwell_matmul host-overhead patch (Front D2, 2026-06-11)

## Verdict: WIN

`labs/blackwell_matmul:blackwell_matmul_tcgen05` optimized time **0.2250 ms -> 0.1505-0.1781 ms**
(median of 3 after-runs ~0.155 ms, ~1.45x on the measured target), verify PASS on **both** labs that
share `capstone_kernels_tcgen05.cu`. Outputs are **bit-identical** to the pre-patch kernel for both
CTA1 and CTA2 variants (torch.equal == True on seeded 2048^3 inputs).

## Before / after (harness, GPU 1, --profile none --single-gpu, iterations=5 warmup=5)

| Target | Before | After (3 runs) | Verify |
|---|---|---|---|
| labs/blackwell_matmul:blackwell_matmul_tcgen05 | 0.22504 ms (69.61x vs 15.66 ms naive) | 0.15530 / 0.17815 / 0.15045 ms (101.10x / 88.02x / 104.18x) | PASS all, max_diff 0.125 (atol 5.0) |
| labs/fullstack_cluster:cluster_gemm_tcgen05 | n/a (baseline IS the patched CTA1 kernel) | base 0.1435 -> opt 0.1350 ms (1.06x) | PASS |
| labs/fullstack_cluster:cluster_gemm_tcgen05_cta2 | n/a | base 0.1468 -> opt 0.1314 ms (1.12x) | PASS |

Standalone serial probe (sync per call, mirrors harness contract): 134.4 us -> 71.1 us warm;
156.8 us -> 91.9 us with cold L2 (harness clears L2 each iteration).

## Profile split (nsys, 50 serial iterations, CTA1)

Front D's prime suspect was **refuted**: `cuTensorMapEncodeTiled` is only **0.24 us/call** (0.7% of
API time). The real per-call costs, before -> after:

| Component | Before | After |
|---|---|---|
| gemm_device_variant kernel | 66.2 us | **38.9 us** (epilogue C-load removal exposed ~27 us of serial gmem->reg load latency + L2 pollution) |
| fill kernel (torch::zeros memset) | 3.3 us GPU + ~28 us host (aten::zeros) | **eliminated** |
| fp32->fp16 copy kernel (.to) | 4.0 us GPU + ~20 us host | unchanged (kept: output dtype contract) |
| cudaFuncSetAttribute | 4.5 us/call host | **eliminated** (once per device) |
| cuTensorMapEncodeTiled x2 | 0.5 us/call | **eliminated** (TMA atom cache; CuTe host-side atom construction also skipped) |
| cudaLaunchKernel | 2 calls/iter | 1 call/iter |

## Patch (labs/fullstack_cluster/capstone_kernels_tcgen05.cu; backup at /tmp/frontD2/capstone_kernels_tcgen05.cu.bak)

1. Device epilogue: dropped `copy(tDgC, tDrC)` and the `+ beta * tDrC(i)` term (all call sites
   hardcode Alpha(1.0f)/Beta(0.0f); bit-identical since C was zeros).
2. Host: `torch::zeros` C + `empty_like` D -> single `torch::empty` D aliased as C.
3. Host: TMA atoms cached in a leaked `static thread_local` cache keyed on
   (a_ptr, b_ptr, m, n, k, device) - correct for arbitrary inputs because a CUtensorMap encodes
   address/layout, never content (verified: in-place content mutation at same pointer still correct).
   Atoms are copied out of the cache (value semantics) so `decltype` in the kernel template
   instantiation stays a value type.
4. Host: `cudaFuncSetAttribute` behind a once-per-device `static thread_local` guard.

Full diff appended below.

## Validation

- Bit-exactness vs pre-patch outputs (seed 42, 2048^3): CTA1 True, CTA2 True (max diff 0.0).
- Fresh-tensor and mutated-content-same-pointer cases: max diff 0.125 vs fp32 torch.matmul reference
  (same as pre-patch; fp16 rounding).
- Harness verify PASS on all 3 after-runs + both fullstack_cluster targets.
- Extension rebuilt from mtime (build fingerprint 03:12 UTC, after patch push 03:10 UTC).

## Next lever

**fp16-direct epilogue**: convert the accumulator to fp16 in the epilogue registers and store to an
fp16 D buffer, killing the `.to(kFloat16)` (~3.9 us copy kernel + ~20 us host aten::_to_copy path,
plus halves the D store traffic). After that, remaining host gap is ~28 us/call (pybind + torch::empty
+ launch); a CUDA-graph path in the harness would absorb it, but per-call empty() allocation must be
made graph-safe first. Kernel itself is now 38.9 us = ~440 TFLOP/s fp16, still well under GB300
dense-fp16 SoL - tile-pipelining depth (k-stage double buffering) is the kernel-side breakthrough
candidate.

## Diff

```diff
--- /tmp/frontD2/capstone_kernels_tcgen05.cu.bak	2026-06-11 02:56:43.235977497 +0000
+++ labs/fullstack_cluster/capstone_kernels_tcgen05.cu	2026-06-11 03:10:12.875441117 +0000
@@ -7,6 +7,7 @@
 
 #include <cooperative_groups.h>
 #include <type_traits>
+#include <utility>
 
 #include <cutlass/arch/barrier.h>
 #include <cutlass/cluster_launch.hpp>
@@ -261,9 +262,12 @@
   auto tiled_t2r_copy = make_tmem_copy(SM100_TMEM_LOAD_32dp32b1x{}, tCtAcc);
   auto thr_t2r_copy = tiled_t2r_copy.get_slice(threadIdx.x);
 
+  // Host-overhead fix: every call site hardcodes Alpha(1.0f), Beta(0.0f), so
+  // the C operand is never observable in the output. Drop the global->register
+  // load of C and the beta multiply (bit-identical: C was zero-filled before).
+  (void)beta;
   Tensor tDgC = thr_t2r_copy.partition_D(tCgC);
   Tensor tDrC = make_fragment_like(tDgC);
-  copy(tDgC, tDrC);
 
   Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
   Tensor tDgD = thr_t2r_copy.partition_D(tCgD);
@@ -272,7 +276,7 @@
 
   CUTE_UNROLL
   for (int i = 0; i < size(tDrC); ++i) {
-    tDrC(i) = alpha * tDrAcc(i) + beta * tDrC(i);
+    tDrC(i) = alpha * tDrAcc(i);
   }
   copy(tDrC, tDgD);
 
@@ -300,8 +304,11 @@
   auto n = b_contig.size(0);
 
   auto options = a.options().dtype(torch::kFloat32);
-  auto c_buffer = torch::zeros({m, n}, options);
-  auto d_buffer = torch::empty_like(c_buffer);
+  // Host-overhead fix: the device epilogue no longer reads C (alpha=1, beta=0
+  // at every call site), so no zero-filled C operand is needed. Alias C to D
+  // and skip the per-call 16MB memset kernel + extra allocation.
+  auto d_buffer = torch::empty({m, n}, options);
+  auto c_buffer = d_buffer;
 
   auto cluster_shape = Variant::cluster_shape();
   auto tiled_mma = make_tiled_mma(typename Variant::Mma{});
@@ -353,7 +360,7 @@
                    make_tile(typename decltype(tiled_mma)::AtomThrID{}));
 
   // Build TMA atoms. 2SM variants must use SM100 TMA atoms + multicast semantics.
-  auto tma_atom_A = [&] {
+  auto build_tma_atom_A = [&] {
     if constexpr (Variant::kClusterM == 2) {
       return make_tma_atom_A_sm100(
           SM100_TMA_2SM_LOAD_MULTICAST{},
@@ -369,9 +376,9 @@
           sA_layout,
           make_shape(size<0>(mma_tiler), size<2>(mma_tiler)));
     }
-  }();
+  };
 
-  auto tma_atom_B = [&] {
+  auto build_tma_atom_B = [&] {
     if constexpr (Variant::kClusterM == 2) {
       return make_tma_atom_B_sm100(
           SM100_TMA_2SM_LOAD_MULTICAST{},
@@ -387,7 +394,47 @@
           sB_layout,
           make_shape(size<1>(mma_tiler), size<2>(mma_tiler)));
     }
-  }();
+  };
+
+  // Host-overhead fix: cache the TMA atoms across calls. A CUtensorMap encodes
+  // only the base address, shape, and layout of the operand -- never tensor
+  // contents -- so a rebuild is needed only when the data pointers, problem
+  // shape, or device change. Cache is per template instantiation per thread;
+  // intentionally leaked (the atoms hold no CUDA resources requiring teardown).
+  struct TmaAtomCacheKey {
+    const void* a_ptr;
+    const void* b_ptr;
+    int64_t m, n, k;
+    int device;
+    bool matches(const void* ap, const void* bp, int64_t mm, int64_t nn,
+                 int64_t kk, int dev) const {
+      return a_ptr == ap && b_ptr == bp && m == mm && n == nn && k == kk &&
+             device == dev;
+    }
+  };
+  using TmaAtomPair = std::pair<decltype(build_tma_atom_A()),
+                                decltype(build_tma_atom_B())>;
+  struct TmaAtomCache {
+    TmaAtomCacheKey key;
+    TmaAtomPair atoms;
+  };
+  static thread_local TmaAtomCache* tma_cache = nullptr;
+  const void* a_ptr = a_contig.data_ptr();
+  const void* b_ptr = b_contig.data_ptr();
+  const int device = static_cast<int>(a_contig.get_device());
+  if (tma_cache == nullptr ||
+      !tma_cache->key.matches(a_ptr, b_ptr, m, n, k, device)) {
+    auto* fresh = new TmaAtomCache{
+        TmaAtomCacheKey{a_ptr, b_ptr, m, n, k, device},
+        TmaAtomPair{build_tma_atom_A(), build_tma_atom_B()}};
+    delete tma_cache;
+    tma_cache = fresh;
+  }
+  // Copy (not reference): keeps decltype(tma_atom_A) a value type for the
+  // kernel template instantiation below. The atom is a small trivially
+  // copyable descriptor; the expensive part is the build, not the copy.
+  auto tma_atom_A = tma_cache->atoms.first;
+  auto tma_atom_B = tma_cache->atoms.second;
 
   auto cluster_m_tiles = size<1>(cluster_layout_vmnk);
   auto cluster_n_tiles = size<2>(cluster_layout_vmnk);
@@ -416,8 +463,14 @@
       decltype(tma_atom_A), decltype(tma_atom_B),
       Alpha, Beta>;
 
-  AT_CUDA_CHECK(cudaFuncSetAttribute(
-      kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
+  // Host-overhead fix: the max-dynamic-smem attribute is sticky per
+  // (function, device). Set it once per device instead of on every call.
+  static thread_local int attr_set_for_device = -1;
+  if (attr_set_for_device != device) {
+    AT_CUDA_CHECK(cudaFuncSetAttribute(
+        kernel_ptr, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes));
+    attr_set_for_device = device;
+  }
 
   cutlass::ClusterLaunchParams params{
       dimGrid, dimBlock, dimCluster, smem_bytes};
```
