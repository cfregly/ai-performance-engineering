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


---

# D3 section: fp16-direct epilogue (Front D3, 2026-06-11)

## Step 1 verdict: WIN (bit-identical, all targets PASS)

`labs/blackwell_matmul:blackwell_matmul_tcgen05` optimized time **0.14949 ms -> 0.13005-0.13093 ms**
(median of 3 after-runs 0.13035 ms, ~1.147x on the measured target; speedup vs naive 104.94x -> 119.7-120.7x).
Outputs **bit-identical** to the B36 kernel for both CTA1 and CTA2 (torch.equal == True, max diff 0.0,
seeded 2048^3 inputs): the fp32-accumulator -> fp16 conversion in epilogue registers is the same single
round-to-nearest-even rounding the old fp32-store + host `.to(kFloat16)` performed.

## Before / after (harness, GPU 1, --profile none, iterations=5 warmup=5)

| Target | Before (B36) | After (3 runs) | Verify |
|---|---|---|---|
| labs/blackwell_matmul:blackwell_matmul_tcgen05 | 0.14949 ms (104.94x) | 0.13093 / 0.13005 / 0.13035 ms (119.72x / 120.67x / 120.21x) | PASS all, max_diff 0.125 (atol 5.0), unchanged vs B36 |
| labs/fullstack_cluster:cluster_gemm_tcgen05 | base 0.1435 / opt 0.1350 (1.06x) | base 0.12961 / opt 0.13090 (0.99x) | PASS |
| labs/fullstack_cluster:cluster_gemm_tcgen05_cta2 | base 0.1468 / opt 0.1314 (1.12x) | base 0.12574 / opt 0.13091 (0.96x) | PASS |

Note on fullstack_cluster ratios: baseline (CTA1) and optimized (CTA2) are BOTH this kernel (B28 finding),
so both sides got faster and the ratio stays ~1.0 by construction; harness status `succeeded` on both.

Standalone serial probe (sync per call): 75.7 us -> **64.5 us** warm per call; CUDA-event window
59.8 us -> 41.5 us.

## Profile split (nsys, 50 serial CTA1 iterations, after)

| Component | B36 | D3 |
|---|---|---|
| gemm_device_variant kernel | 38.9 us | **37.6 us** (D store halved: 16MB fp32 -> 8MB fp16) |
| fp32->fp16 copy kernel (`.to`) | 4.0 us GPU + ~20 us host `aten::_to_copy` | **eliminated** (zero non-gemm kernels in trace) |
| cudaLaunchKernelExC | 2 launches/iter | 1 launch/iter, 8.5 us avg |
| Residual host gap per serial call | ~16 us | ~27 us measured (64.5 total - 37.6 kernel); pybind + TORCH_CHECK/contiguous + torch::empty + launch API |

## Patch (vs B36 state; backup at /tmp/frontD3/capstone_kernels_tcgen05.cu.B36)

1. `TypeD = float` -> `cutlass::half_t`. TypeC stays `float` (it is the UMMA accumulator type).
2. Device epilogue: `tDrC` fp32 staging fragment replaced by an fp16 `tDrD` fragment;
   `tDrD(i) = static_cast<DType>(alpha * tDrAcc(i))` then `copy(tDrD, tDgD)`.
3. Host: `d_buffer` allocated as kFloat16; `mC` (never dereferenced post-B36, layout plumbing only)
   points at the D allocation reinterpreted as `TypeC*`; `return d_buffer;` (no `.to`).
4. Blast radius: the shared epilogue template serves only CTA1/CTA2 in this file; both harness labs
   exercise both variants and PASS; returned dtype (fp16) is unchanged for all wrappers.

## Step 2 verdict: SKIPPED (CUDA-graphing the residual host gap is not clean under the contract)

Evidence:
- The sanctioned graph path is harness-level `BenchmarkConfig.enable_cuda_graph` + declared
  `graph_capture_enabled` in the verification signature, audited by `GraphCaptureCheatDetector`
  (capture/replay ratio <= 10x, capture-memory <= 100MB) — the pattern labs/parameterized_cuda_graphs
  uses (captures in setup(), declares the flag, patches memcpy-node params per replay).
- `GraceBlackwellMatmulBenchmark` does NOT declare graph capture; flipping `enable_cuda_graph` in its
  shared config would change the timing mode of every variant in BOTH labs (baseline included) —
  a measurement-contract change, not an op speedup; banked numbers would no longer be comparable.
- An extension-internal stream-capture graph (keyed on ptrs, like the TMA-atom cache) would be
  undeclared graph capture under a signature that says none, bypassing the exact machinery the
  harness uses to police graph timing -> ambiguous-to-illegitimate; per guardrail, not shipped.
- Payoff ceiling is small anyway: a graph launch can only replace the 8.5 us cudaLaunchKernelExC
  (-> ~4-5 us cudaGraphLaunch); the remaining ~18 us (pybind, TORCH_CHECKs, contiguous(),
  torch::empty, stream query) is host code a device graph cannot absorb. ~4 us on a 130 us
  measured iteration (~3%) does not justify undeclared graph state in a shared extension.

## Next lever

Kernel-side, declared and clean: **k-stage pipelining (double-buffered smem + TMA prefetch)** in
gemm_device_variant. The mainloop is serial TMA->wait->MMA->wait per k-tile; 37.6 us = ~457 TFLOP/s
fp16, still well under GB300 dense-fp16 SoL. Overlapping TMA of tile k+1 with MMA of tile k is the
standard CUTLASS sm100 pattern and attacks the largest remaining term (the kernel itself).
Host-side, if ever needed: make the benchmark declare `graph_capture_enabled` explicitly and use the
harness path, accepting the contract change.

## Diff (exact, vs B36)

```diff
diff --git a/code/labs/fullstack_cluster/capstone_kernels_tcgen05.cu b/code/labs/fullstack_cluster/capstone_kernels_tcgen05.cu
index 7d3188e2..89c2059e 100644
--- a/code/labs/fullstack_cluster/capstone_kernels_tcgen05.cu
+++ b/code/labs/fullstack_cluster/capstone_kernels_tcgen05.cu
@@ -26,7 +26,11 @@ using namespace cute;
 using TypeA = cutlass::half_t;
 using TypeB = cutlass::half_t;
 using TypeC = float;
-using TypeD = float;
+// fp16-direct epilogue: D is written as fp16 straight from the epilogue
+// registers (fp32 accumulator -> one round-to-nearest-even conversion), so the
+// host-side d_buffer.to(kFloat16) pass disappears. TypeC stays float: it is
+// the UMMA accumulator type and (post C-load removal) only feeds layout math.
+using TypeD = cutlass::half_t;
 using Accumulator = float;
 using Alpha = float;
 using Beta = float;
@@ -266,19 +270,25 @@ __global__ void gemm_device_variant(ATensor mA,
   // the C operand is never observable in the output. Drop the global->register
   // load of C and the beta multiply (bit-identical: C was zero-filled before).
   (void)beta;
-  Tensor tDgC = thr_t2r_copy.partition_D(tCgC);
-  Tensor tDrC = make_fragment_like(tDgC);
 
   Tensor tDtAcc = thr_t2r_copy.partition_S(tCtAcc);
   Tensor tDgD = thr_t2r_copy.partition_D(tCgD);
   Tensor tDrAcc = make_tensor<Accumulator>(shape(tDgD));
   copy(tiled_t2r_copy, tDtAcc, tDrAcc);
 
+  // fp16-direct epilogue: convert the fp32 accumulator to fp16 in registers
+  // and store straight to the fp16 D buffer. This is the same single
+  // round-to-nearest-even conversion of the same fp32 value that the old
+  // fp32-store + host .to(kFloat16) path performed, so the output is
+  // bit-identical while the separate cast kernel (and its host dispatch)
+  // disappears and the D store traffic halves.
+  using DType = typename DTensor::value_type;
+  Tensor tDrD = make_tensor<DType>(shape(tDgD));
   CUTE_UNROLL
-  for (int i = 0; i < size(tDrC); ++i) {
-    tDrC(i) = alpha * tDrAcc(i);
+  for (int i = 0; i < size(tDrD); ++i) {
+    tDrD(i) = static_cast<DType>(alpha * tDrAcc(i));
   }
-  copy(tDrC, tDgD);
+  copy(tDrD, tDgD);
 
   __syncthreads();
   if (elect_one_warp) {
@@ -303,12 +313,11 @@ torch::Tensor run_tcgen05_variant(torch::Tensor a, torch::Tensor b) {
   auto k = a_contig.size(1);
   auto n = b_contig.size(0);
 
-  auto options = a.options().dtype(torch::kFloat32);
   // Host-overhead fix: the device epilogue no longer reads C (alpha=1, beta=0
-  // at every call site), so no zero-filled C operand is needed. Alias C to D
-  // and skip the per-call 16MB memset kernel + extra allocation.
-  auto d_buffer = torch::empty({m, n}, options);
-  auto c_buffer = d_buffer;
+  // at every call site), so no zero-filled C operand is needed. fp16-direct
+  // epilogue: the kernel writes fp16 D, so allocate the output at its final
+  // dtype and return it without a .to(kFloat16) pass.
+  auto d_buffer = torch::empty({m, n}, a.options().dtype(torch::kFloat16));
 
   auto cluster_shape = Variant::cluster_shape();
   auto tiled_mma = make_tiled_mma(typename Variant::Mma{});
@@ -348,11 +357,15 @@ torch::Tensor run_tcgen05_variant(torch::Tensor a, torch::Tensor b) {
       make_gmem_ptr(reinterpret_cast<TypeB const*>(
           b_contig.data_ptr<at::Half>())),
       make_layout(make_shape(n, k), make_stride(k, Int<1>{})));
+  // mC is layout-plumbing only: the epilogue dropped the C load/store, so this
+  // pointer is never dereferenced (it exists to type partition_C /
+  // make_fragment_C with the fp32 accumulator type). Point it at the D
+  // allocation rather than allocating dead fp32 memory.
   Tensor mC = make_tensor(
-      make_gmem_ptr(c_buffer.data_ptr<TypeC>()),
+      make_gmem_ptr(reinterpret_cast<TypeC*>(d_buffer.data_ptr<at::Half>())),
       make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
   Tensor mD = make_tensor(
-      make_gmem_ptr(d_buffer.data_ptr<TypeD>()),
+      make_gmem_ptr(reinterpret_cast<TypeD*>(d_buffer.data_ptr<at::Half>())),
       make_layout(make_shape(m, n), make_stride(n, Int<1>{})));
 
   auto cluster_layout_vmnk =
@@ -485,7 +498,8 @@ torch::Tensor run_tcgen05_variant(torch::Tensor a, torch::Tensor b) {
   TORCH_CHECK(status == cutlass::Status::kSuccess,
               "tcgen05 inline kernel launch failed");
 
-  return d_buffer.to(torch::kFloat16);
+  // fp16-direct epilogue: D is already fp16; no conversion pass.
+  return d_buffer;
 }
 
 torch::Tensor optimized_matmul_tcgen05(torch::Tensor a, torch::Tensor b) {
```
