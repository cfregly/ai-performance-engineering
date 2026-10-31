# A traffic bound for one causal attention partial-buffer design

This reproducible analytical example helps screen a materialized attention-gradient dataflow before implementing it. It contains no GPU run, attention kernel, measured runtime or speedup. All counts cover one sequence (batch size 1), one attention layer backward and one device.

## Assumptions and calculation

The synthetic workload has 4,096 positions, 16 query heads sharing one KV head, key and value dimensions of 128, and 128-row tiles. Each query head produces a distinct FP32 dK/dV partial pair for each causal query/key tile pair. The producer completes every uncompressed partial before a separate reducer starts. Streaming consumption and recomputation are excluded.

At most 128 MiB of partials may survive on chip across that complete phase boundary. This is an illustrative total retention allowance, not a GPU cache-size claim. The unretained bytes require at least one HBM write and one reread:

```text
32 tiles per sequence dimension
16 × (32 × 33 / 2) = 8,448 partial pairs
128 × (128 + 128) × 4 = 131,072 bytes per pair
S = 1,107,296,256 bytes = 1,056 MiB
C = 134,217,728 bytes = 128 MiB
mandatory partial HBM traffic ≥ 2 × max(0, S − C)
                             = 1,946,157,056 bytes = 1,856 MiB
```

TB/s means 10¹² bytes per second. MiB means 2²⁰ bytes. Bandwidth here is an assumed aggregate payload ceiling shared by reads and writes.

| Assumed bandwidth | Transfer floor | Retention giving exactly a 0.200 ms transfer floor |
| --- | ---: | ---: |
| 8 TB/s | 0.243269632 ms | 307,296,256 bytes |
| 6.4 TB/s | 0.30408704 ms | 467,296,256 bytes |

An illustrative 0.200 ms target rejects this contract at 128 MiB retention under either bandwidth assumption. At 128 MiB retention, the bandwidth giving equality is 9.73078528 TB/s. Equal or lower floors do not establish feasible runtime. NVIDIA lists 8 TB/s nominal B200 bandwidth in its [STAC-AI report](https://developer.nvidia.com/blog/nvidia-blackwell-sets-stac-ai-record-for-llm-inference-in-finance/). Nominal bandwidth is not sustained workload throughput.

The bound excludes attention arithmetic, other tensors, final-gradient traffic, reducer work, launches, synchronization, transaction granularity and bandwidth inefficiency. It applies only to this representation and phase contract. It is not a universal attention lower bound or a model/serving result.

## What changes with smaller batches

[Cache sensitivity](evidence/derivation/cache_sensitivity.csv) changes retention while preserving the complete phase boundary. [Batch sensitivity](evidence/derivation/reduction_batch_sensitivity.csv) instead produces and reduces consecutive query-tile batches. Two query tiles per batch reduce maximum partial scratch from 1,056 MiB to 126 MiB. Total produced partial volume stays 1,056 MiB. Final FP32 dK/dV occupy another 4 MiB logically, with residency unspecified.

Materialized partials give simple producer ownership at the cost of workspace, writes, rereads and a separate reduction. Smaller batches change the contract and add accumulation, scheduling and synchronization responsibilities. Prefer the simpler design when its costs fit the budget. Evaluate the complete changed path before choosing a faster implementation. No complete implementation fit or performance follows from these calculations.

For the larger placement and update decision, see [A whole-system budget for a hypothetical training update](https://github.com/cfregly/ai-performance-engineering/tree/60f8aea572304ddbb3b9b9cc254c7c15cddb7337/research/2026-10-29-distributed-training-budget). It uses a different synthetic workload. Applying this attention check there requires recomputing per-rank shapes, partials, phase order, retention and bandwidth rather than transferring this example's bytes or milliseconds.

## Reproduce

Python 3.10 or newer and the standard library are sufficient. From this directory, write the outputs, check them and run the CPU tests:

```sh
python3 code/roofline.py --output-dir reproduction-output
python3 code/roofline.py --output-dir reproduction-output --check
python3 -m unittest discover -s code -p 'test_*.py' -v
```

The first command writes three deterministic JSON/CSV files. The check reports missing or changed outputs without rewriting them. The tests also cover a modified CSV that the check must reject and bind the [retained evidence values](evidence/analytical-results.json) to the calculator and retained results. [RESULTS.json](evidence/derivation/RESULTS.json) records the full assumptions and exclusions. [Validation](evidence/derivation/VALIDATION.md) describes the focused checks. The tiny CPU toy uses artificial integer and exactly representable dyadic values, preserves reduction order and checks results against an independent cell-wise oracle. It does not implement attention derivatives, model GPU memory behavior or establish equivalence for reordered floating-point sums.

## Related primary sources

The arithmetic above follows from the declared synthetic contract. These sources provide established algorithm and hardware context:

- [FlashAttention](https://arxiv.org/abs/2205.14135v2): IO-aware exact attention reduces transfers between HBM and on-chip SRAM.
- [FlashAttention-4](https://arxiv.org/abs/2603.05451) and the [authors' explanation](https://www.together.ai/blog/flashattention-4): uneven Blackwell resource scaling, shared-memory traffic, tensor memory and reduction choices.
- [NVIDIA PTX ISA, tensor memory](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory): distinct TMEM storage, allocation, movement and synchronization semantics on datacenter Blackwell GPUs that support tcgen05.
- [NVIDIA Blackwell STAC-AI report](https://developer.nvidia.com/blog/nvidia-blackwell-sets-stac-ai-record-for-llm-inference-in-finance/): nominal B200 memory bandwidth, separate from the assumed sustained payload ceilings used here.

No published speedup from those sources is adopted as a result of this example.

## Visual explanations

These diagrams follow the same synthetic workload, phase contract and analytical transfer floors described above.

### Memory paths and traffic

![Even if it fits, count the traffic.](images/fun-square.png)

![Follow every operand](images/scientific/02-operand-paths.png)

![Scratch can become the bottleneck](images/scientific/03-scratch-traffic.png)

![A floor can reject a target](images/scientific/04-target-rejection.png)

![Test bottlenecks independently](images/scientific/05-independent-budgets.png)

![Compare the complete path](images/scientific/06-reduction-comparison.png)

### Reduction choices

![Write now. Read later. Pay twice.](images/a-square.png)

![Transfer floor: 0.243 ms](images/b-square.png)

![Check where partials live](images/c-square.png)

### Ways to apply the calculation

![The desk and the warehouse](images/aud-curious-square.png)

![Choose the reduction plan early](images/aud-ml-engineer-square.png)

![Budget bytes as well as FLOPs](images/aud-performance-engineer-square.png)

![Capacity is a lifetime question](images/aud-kernel-engineer-square.png)

![Validate the whole training step](images/aud-platform-operations-square.png)

![Check the budget before building](images/aud-business-capacity-square.png)
