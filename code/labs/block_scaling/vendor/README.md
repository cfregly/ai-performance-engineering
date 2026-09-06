# Vendored CUTLASS blockscaled GEMM examples

## sm_100 (B200)

`sm100_dense_blockscaled_gemm_persistent.py` is vendored verbatim from the
NVIDIA CUTLASS v4.5.2 release, whose source matches the pinned
`nvidia-cutlass-dsl[cu13]==4.5.2` runtime. The BSD-3-Clause header is preserved.

| Field | Value |
| --- | --- |
| Source | `examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py` |
| Upstream repo | `NVIDIA/cutlass` |
| Upstream tag | `v4.5.2` |
| Tag commit | `db1c288993354c88e551c40c19a8fb93a774a241` |
| Source URL | `https://raw.githubusercontent.com/NVIDIA/cutlass/db1c288993354c88e551c40c19a8fb93a774a241/examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py` |
| SHA-256 | `98e9974d42e888a27f02f0c52582c218d922b6cd748d78e7e62af19923efd7d8` |
| Lines | 3152 |

The repository's CUTLASS submodule is v4.1.0. Its sm_100 example is not
compatible with DSL 4.5.2: compilation reaches the removed
`cutlass.cute.arch.ProxyKind` API. Vendoring the release-matched example keeps
the kernel and DSL on one public API version instead of patching process-global
CuTe symbols. `block_scaling_common.py` adapts the v4.5.2 example's pointer ABI
to the lab's existing eager and CUDA-graph call shape.

## sm_103 (GB300 / Blackwell Ultra)

`sm103_dense_blockscaled_gemm_persistent.py` is vendored verbatim (byte-identical)
from NVIDIA CUTLASS `main`, license BSD-3-Clause (header preserved in the file).

| Field | Value |
| --- | --- |
| Source | `examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/sm103_dense_blockscaled_gemm_persistent.py` |
| Upstream repo | `NVIDIA/cutlass` |
| Upstream commit | `1fc71b3ed1cab3541f7482c68ee19d...` |
| sha256 (first 16) | `ae48c7eb4d99ac7a` |
| Lines | 3039 |

## Why this is vendored (not the submodule)

The pinned `third_party/cutlass` submodule is v4.1.0, which predates Blackwell
Ultra (sm_103 / compute capability 10.3). Two problems on GB300:

1. The v4.1.0 DSL example (`blackwell/dense_blockscaled_gemm_persistent.py`) targets
   the `Sm100BlockScaledPersistentDenseGemmKernel` and uses CuTe-DSL APIs that were
   removed in `nvidia-cutlass-dsl` 4.5.x.
2. The 4.3.0 CuTe-DSL `Arch` enum has no `sm_103*` entry at all, and its NVFP4
   blockscaled MMA op is arch-locked to `sm_100a` (will not load on sm_103).

`nvidia-cutlass-dsl[cu13]>=4.5.2` (pinned in `requirements_latest.txt`) adds
`sm_103`/`sm_103a`/`sm_103f` to the `Arch` enum, and this example is the matching
sm_103 kernel. It imports only from the pip `cutlass` package
(`cutlass.utils.blackwell_helpers`, `cutlass.utils.blockscaled_layout`), so it has
no sibling-file dependency on the cutlass source tree and runs standalone.

The C++ CUTLASS path (the `nvfp4_*_sm103` binaries, nvcc `-arch=sm_103a`) already
supports sm_103a and is unaffected; only this Python CuTe-DSL JIT example needed a
newer source than the v4.1.0 submodule ships.

`block_scaling_common.py` selects this file automatically on Blackwell Ultra
(via `_resolve_cutlass_example_path()`) and falls back to the submodule sm_100
example elsewhere.

## Validation

On a GB300 node (sm_103) with `nvidia-cutlass-dsl[cu13]==4.5.2`, the 8192x8192x1024
NVFP4 blockscaled GEMM runs and passes the example's internal reference check
(`relL2` within tolerance) at ~45 us.

## Refreshing

Re-copy the file from a newer cutlass checkout and update the commit/sha above.
Keep it byte-identical to upstream so it diffs cleanly.
