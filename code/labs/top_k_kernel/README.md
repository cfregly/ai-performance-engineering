# Lab - GQA Top-K Selection Kernel

## Summary
Benchmarks the block-selection stage used by grouped-query selection attention. The workload now models explicit query-head grouping onto fewer K/V heads, so the benchmark captures the same GQA structure as the slide deck instead of treating every head independently.

The public pair shape stays the same:
- `labs/top_k_kernel:top_k_kernel` compares the exact dense grouped baseline against a Triton grouped block-score kernel.
- `labs/top_k_kernel:top_k_kernel_cuda` compares the same baseline against a CUTLASS-backed CUDA grouped path.

The default harnessed workload is the large forward-routing case used in the slide matrix:
- `mode=forward`
- `batch=4, heads=8, kv_heads=1`
- `q_len=compressed_k_len=32768`
- `head_dim=128, top_k=16, selection_block_size=64, compress_stride=1`

Keep `fwd_bwd` as an explicit override when you want to study backward-path parity instead of the public routing-stage speed story.

## Problem
Selection attention only pays off when block routing is cheaper than evaluating every compressed K position for every query head. In the real algorithm, query heads are grouped onto fewer K/V heads, then reduced head-wise to produce one block selection per K/V head and query position.

This refactor makes that grouping explicit:
- `q`: `[batch, query_heads, q_len, d]`
- `k`: `[batch, kv_heads, compressed_k_len, d]`
- `gqa_size = query_heads // kv_heads`
- output probabilities / indices: `[batch, kv_heads, q_len, top_k]`

## Paths
### Baseline
- exact dense compressed-position scores against the owning K/V head
- block reduction over `selection_block_size // compress_stride`
- head-wise reduction across the query heads in the same GQA group
- `topk + softmax`
- autograd through the exact dense formulation

### Triton
- explicit grouped-query workload
- pre-aggregate each selection block into one block-key vector
- Triton kernel computes grouped block scores directly for each `(batch, kv_head, q_position)`
- grouped backward computes `dQ` / `dK` from the reduced block-score graph

### CUDA / CUTLASS
- explicit grouped-query workload
- pre-aggregate each selection block into one block-key vector
- reduce each GQA query-head group before scoring
- CUTLASS-backed GEMM helper scores reduced query tiles against block keys
- grouped backward reuses the saved per-group `q_sum` and block-key reductions, then computes `dQ` / `dK` from the reduced block-score graph

## Targets
| Target | Path |
| --- | --- |
| `labs/top_k_kernel:top_k_kernel` | Baseline vs Triton benchmark pair |
| `labs/top_k_kernel:top_k_kernel_cuda` | Baseline vs CUTLASS-backed CUDA benchmark pair |

## Repro Commands
```bash
python -m cli.aisp bench list-targets --chapter labs/top_k_kernel
python -m cli.aisp bench run --targets labs/top_k_kernel:top_k_kernel --profile none --single-gpu
python -m cli.aisp bench run --targets labs/top_k_kernel:top_k_kernel_cuda --profile none --single-gpu
```

Useful grouped-shape overrides:
```bash
python -m cli.aisp bench run \
  --targets labs/top_k_kernel:top_k_kernel \
  --profile none \
  --single-gpu \
  --target-extra-arg 'labs_top_k_kernel:top_k_kernel=--mode forward --batch-size 4 --heads 8 --kv-heads 1 --q-len 32768 --compressed-k-len 32768 --head-dim 128 --top-k 16 --selection-block-size 64 --compress-stride 1'

python -m cli.aisp bench run \
  --targets labs/top_k_kernel:top_k_kernel \
  --profile none \
  --single-gpu \
  --target-extra-arg 'labs_top_k_kernel:top_k_kernel=--mode fwd_bwd --batch-size 4 --heads 16 --kv-heads 1 --q-len 1024 --compressed-k-len 1024 --head-dim 128 --top-k 16 --selection-block-size 64 --compress-stride 1'

python -m cli.aisp bench run \
  --targets labs/top_k_kernel:top_k_kernel_cuda \
  --profile none \
  --single-gpu \
  --target-extra-arg 'labs_top_k_kernel:top_k_kernel_cuda=--mode forward --batch-size 4 --heads 8 --kv-heads 1 --q-len 32768 --compressed-k-len 32768 --head-dim 128 --top-k 16 --selection-block-size 64 --compress-stride 1'

python -m cli.aisp bench run \
  --targets labs/top_k_kernel:top_k_kernel_cuda \
  --profile none \
  --single-gpu \
  --target-extra-arg 'labs_top_k_kernel:top_k_kernel_cuda=--mode fwd_bwd --batch-size 4 --heads 16 --kv-heads 1 --q-len 1024 --compressed-k-len 1024 --head-dim 128 --top-k 16 --selection-block-size 64 --compress-stride 1'
```

Direct Triton-vs-CUDA matrix runner:
```bash
python -m labs.top_k_kernel.compare_top_k_matrix
```

## Current Host Snapshot
The numbers below come from harnessed runs on the current virtualized B200 host with the portable validity profile and locked app clocks. They are useful for directional comparison, but they are not canonical publish-grade numbers.

| Mode | Shape | Baseline (ms) | Triton (ms) | CUDA (ms) | CUDA vs Triton |
| --- | --- | ---: | ---: | ---: | ---: |
| `forward` | `batch=4, heads=8, kv_heads=1, q_len=32768, d=128, top_k=16, block=64` | `228.91` | `53.30` | `10.22` | `5.21x` |
| `fwd_bwd` | `batch=16, heads=16, kv_heads=1, q_len=1024, d=128, top_k=16, block=64` | `4.07` | `1.78` | `2.76` | `0.64x` |

## Notes
- The optimized paths keep the reduced block-score interface explicit. This lab does **not** implement the full selected-attention `P @ V` stage yet; it isolates the grouped Top-K routing stage.
- `--heads` remains the total query-head count. `--kv-heads` is the owning K/V-head count. `gqa_size` is derived internally.
- `topk.packed_q_tiles` reports the grouped query tile size used by the optimized paths.
- `topk.packed_kv_blocks` currently reports the number of materialized selection blocks packed together in the optimized paths. For the current block-score-only refactor this remains `1`, because the position dimension is collapsed into block-key vectors before the optimized kernel launch.
- The direct matrix runner is intentionally lightweight and does not go through the benchmark harness. Use the harnessed `bench run` commands for clock-locked pair results and artifact capture.

## Directory Layout
| Path | Description |
| --- | --- |
| `baseline_top_k_kernel.py` | Conservative grouped dense baseline benchmark wrapper. |
| `optimized_top_k_kernel.py` | Triton grouped block-score benchmark wrapper. |
| `optimized_top_k_kernel_cuda.py` | CUTLASS-backed CUDA grouped benchmark wrapper. |
| `top_k_kernel_common.py` | Shared workload config, grouped baseline, Triton path, CUDA path, and benchmark class. |
| `top_k_kernel_extension.py` | Local CUTLASS CUDA extension loader for the grouped CUDA path. |
| `top_k_kernel_cuda.cu` | CUTLASS GEMM helper used by the CUDA variant. |
| `compare_top_k_matrix.py` | Non-public Triton-vs-CUDA comparison runner for the slide-style shape matrix. |
