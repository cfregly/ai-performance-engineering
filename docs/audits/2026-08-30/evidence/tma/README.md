# W1-001 / W1-047 TMA descriptor evidence

Status: source corrections and host contract checks pass. CUDA compilation,
full-output GPU correctness, Compute Sanitizer, and performance acceptance have
not run on this macOS arm64 host. These findings are not GPU-validated closures.

## Source changes

The shared FP32 row-major descriptor now encodes dimensions as `(width, height)`
and its transfer box as `(box_width, box_height)`, with `ld * sizeof(float)` as
the row stride. A small CUDA-free production helper supplies those exact arrays
to `cuTensorMapEncodeTiled`; host tests compile and execute this helper.

All six active shared-helper callers now provide device coordinates in
`(column, row)` order:

| Caller | Corrected operation |
| --- | --- |
| `code/ch07/async_prefetch_2d_demo.cu` | 2D load and store |
| `code/ch07/optimized_tma_bulk_tensor_2d.cu` | 2D load and store |
| `code/ch07/optimized_tma_copy.cu` | Descriptor neighbor-copy load and store |
| `code/ch10/tma_2d_pipeline_blackwell.cu` | Chunk load and store |
| `code/ch10/tma_multicast_cluster.cu` | B[K,N] multicast coordinates for both SM100 and SM103 paths |
| `code/core/common/headers/cuda13_demos.cuh` | Demo load and store |

Both multicast GEMM launchers now round the M tile count up to complete CTA
clusters. Their existing output bounds checks keep padded CTAs from storing
outside C. Padded CTAs still participate in cluster synchronization.

The pipeline's header and final summary now describe the actual disabled
swizzle and L2-promotion settings (W1-047). The shared CUDA13 demo's inaccurate
feature summary was corrected at the same call site. Neither feature was
enabled by this change. The ch07/ch10 Makefiles track the new metadata header.

Other references were inventoried with `rg`. The private descriptor in
`code/ch10/optimized_flash_attn_tma_micro_pipeline.cu` already uses column/row
order; its unused `using` declaration is not a shared-helper call. The private
descriptor in `code/labs/blackwell_matmul/grace_blackwell_kernels.cu` also already
uses that order. Unrelated 1D helpers are unchanged.

## Executed evidence

- `host-before.txt` and `host-before.json`: preserving the original reversed
  dimensions/box in the factored production helper produced **3 failures and
  1 pass**. All three rectangular metadata cases exposed the reversal.
- `host-after.txt`: **31 passed**, comprising the new host descriptor, unavailable
  compiler, and build-dependency checks plus existing ch10 Makefile contracts.
  The host probe checks every logical address for three rectangular shapes,
  including padded row strides and partial tiles. It does not emulate CUDA.
- `diff-check.txt`: scoped `git diff --check` exit 0.
- `cuda-gate-unavailable.json` and its log: real CUDA runner **SKIPPED, exit 3**
  because `nvcc` is absent. No CUDA command executed and no GPU pass was claimed.
- `validation-receipts.json`: exact commands, interpreter, platform, Git base,
  source SHA-256 hashes, exit codes, and explicit acceptance boundaries.

Host command, from `code/`:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/miniconda3/bin/python -m pytest -q -p no:cacheprovider tests/test_tma_2d_layout.py tests/test_ch10_makefile_contract.py
```

## Prepared CUDA gate; not compiled or executed here

`code/tests/cuda/tma_2d_layout_validation.cu` includes the actual production
kernel definitions. The companion runner compiles seven separate cases and
requires full-output success markers; it never substitutes another kernel.
Across all six pipeline configurations, all three demo configurations, and
the other callers plus multicast baseline, it prepares **42 shape/configuration
checks**. Each check compares every output element and exact allocation/padding
canaries against independent host expectations, with asymmetric input values.

Copy shapes are `(height, width, ld) = (256,384,384), (259,196,208),
(129,385,400)`. Multicast and baseline GEMM shapes are `(M,N,K) =
(64,256,384), (67,196,259), (129,388,131)`, including partial M/N/K tiles and
padded clusters. This is bounded correctness coverage, not exhaustive testing.

On a supported, exclusively assigned CUDA 13+ SM100 target, run from `code/`:

```sh
python tests/cuda/run_tma_2d_layout_validation.py --arch sm_100a --output-dir /tmp/tma-layout-sm100-unique --compute-sanitizer
```

Run the corresponding `--arch sm_103a` command on an assigned SM103 target to
validate the separate multicast instruction path. Each output directory must
be new. The runner records source hashes, compiler version, GPU inventory when
available, command logs, exit codes and timeouts. Compilation, runtime output,
and memcheck must all pass; an unsupported or missing tool/device is not a pass.
Any compile/API issue or actual barrier/tail failure remains work to resolve.

## Primary contract sources

- [NVIDIA tensor-map driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html):
  dimension zero is contiguous, and the following dimension uses the supplied
  row stride, including padding. Box axes follow the same dimension order.
- [NVIDIA PTX tensor tiled mode](https://docs.nvidia.com/cuda/parallel-thread-execution/#tensor-tiled-mode):
  tensor coordinates and tiled transfers follow the encoded dimensions.

No benchmark speedup, CUDA ABI compatibility, or GPU completion is inferred
from the host checks.
