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

Materialized partials give each producer its own output region at the cost of workspace, writes, rereads and a separate reduction. Smaller batches change the contract and add accumulation, scheduling and synchronization responsibilities. Prefer the simpler design when its costs fit the budget. Evaluate the complete changed path before choosing a faster implementation. No complete implementation fit or performance follows from these calculations.

For the larger placement and update decision, see [A whole-system budget for a hypothetical training update](../2026-10-29-distributed-training-budget). It uses a different synthetic workload. Applying this attention check there requires recomputing per-rank shapes, partials, phase order, retention and bandwidth rather than transferring this example's bytes or milliseconds.

## Reproduce

Python 3.10 or newer and the standard library are sufficient. From a fresh public checkout:

```sh
git clone https://github.com/cfregly/ai-performance-engineering.git
cd ai-performance-engineering/research/2026-10-29-attention-roofline
```

Reproduce the fixed example, check the written outputs, then run its CPU tests:

```sh
python3 code/roofline.py --output-dir reproduction-output
python3 code/roofline.py --output-dir reproduction-output --check
python3 -m unittest discover -s code -p 'test_*.py' -v
```

The write command reports `Wrote 3 deterministic files to reproduction-output`. The check reports `Reproduction check passed for 3 files`. Successful tests finish with `OK`. Each command exits zero. These are success signals for reproducing this fixed example, not a general shape-configurable calculator. To use the check for another workload, adapt its assumptions and tests, then recompute the counts. The check reports missing or changed outputs without rewriting them. The tests also cover a modified CSV that the check must reject and bind the [retained evidence values](evidence/analytical-results.json) to the calculator and retained results. [RESULTS.json](evidence/derivation/RESULTS.json) records the full assumptions and exclusions. [Validation](evidence/derivation/VALIDATION.md) describes the focused checks. The tiny CPU toy uses artificial integer and exactly representable dyadic values, preserves reduction order and checks results against an independent cell-wise oracle. It does not implement attention derivatives, model GPU memory behavior or establish equivalence for reordered floating-point sums.

## Related primary sources

The arithmetic above follows from the declared synthetic contract. These sources provide established algorithm and hardware context:

- [FlashAttention](https://arxiv.org/abs/2205.14135v2): IO-aware exact attention reduces transfers between HBM and on-chip SRAM.
- [FlashAttention-4](https://arxiv.org/abs/2603.05451) by Ted Zadouri, Tri Dao and coauthors and the [authors' explanation](https://www.together.ai/blog/flashattention-4): uneven Blackwell resource scaling, shared-memory traffic, tensor memory and reduction choices.
- [NVIDIA PTX ISA, tensor memory](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensor-memory): distinct TMEM storage, allocation, movement and synchronization semantics on datacenter Blackwell GPUs that support tcgen05.
- [NVIDIA Blackwell STAC-AI report](https://developer.nvidia.com/blog/nvidia-blackwell-sets-stac-ai-record-for-llm-inference-in-finance/): nominal B200 memory bandwidth, separate from the assumed sustained payload ceilings used here.

No published speedup from those sources is adopted as a result of this example.

## Visual explanations

These diagrams follow the same synthetic workload, phase contract and analytical transfer floors described above.

### Memory paths and traffic

![A small workspace, memory path and stored partials: write results out to memory, then reread them. Fitting workspace does not establish a fast kernel.](images/fun-square.png)

![Ink arrows trace HBM to shared memory and back, shared-memory operands to tensor cores, tensor-core reads and writes to TMEM, and transfers between TMEM and thread registers. A may also come from TMEM.](images/scientific/02-operand-paths.png)

![One layer and one device produce 1,056 MiB of FP32 partials. With at most 128 MiB retained on chip at the full-producer boundary, at least 928 MiB is written out and at least 928 MiB reread for the later reduction.](images/scientific/03-scratch-traffic.png)

![Calculated transfer floor at least 0.243 ms versus a 0.200 ms target for one layer, one device and FP32 results at assumed 8 TB/s read-plus-write bandwidth. Compute and other traffic are excluded.](images/scientific/04-target-rejection.png)

![The same 1,856 MiB (1.946 GB) gives calculated floors at least 0.243 ms at assumed 8 TB/s and 0.304 ms at assumed 6.4 TB/s, against a 0.200 ms target. The zero-based time bars share one scale.](images/scientific/05-independent-budgets.png)

![Three reduction designs: keep each producer’s output separate, combine smaller batches, or accumulate directly. Each changes scratch, coordination and synchronization responsibilities.](images/scientific/06-reduction-comparison.png)

### Concept cards

![Producers finish, write scratch to off-chip memory (HBM), and the reducer rereads it. Writing now and reading later pays two transfers.](images/a-square.png)

![One layer and one device: the calculated 0.243 ms transfer floor exceeds the 0.200 ms target at assumed 8 TB/s.](images/b-square.png)

![For each intermediate, count bytes, lifetime, writers and readers. Ask whether reduction can start sooner, then profile the proposed dataflow.](images/c-square.png)

### Ways to apply the calculation

![Write temporary results from on-chip workspace to off-chip memory (HBM), then reread them before the final combination.](images/aud-curious-square.png)

[Landscape card](images/aud-curious.png) · [Editable square diagram](images/aud-curious-square.svg) · [Editable landscape diagram](images/aud-curious.svg)

![Keep each producer’s output separate for a simpler schedule, or combine smaller batches to reduce scratch while adding coordination.](images/aud-ml-engineer-square.png)

[Landscape card](images/aud-ml-engineer.png) · [Editable square diagram](images/aud-ml-engineer-square.svg) · [Editable landscape diagram](images/aud-ml-engineer.svg)

![Check work divided by compute rate and traffic divided by bandwidth separately. Dependencies can push real time above both floors.](images/aud-performance-engineer-square.png)

[Landscape card](images/aud-performance-engineer.png) · [Editable square diagram](images/aud-performance-engineer-square.svg) · [Editable landscape diagram](images/aud-performance-engineer.svg)

![Qualitative buffer lifetimes overlap. Add the live buffers’ bytes during overlap and reuse storage only after its last reader.](images/aud-kernel-engineer-square.png)

[Landscape card](images/aud-kernel-engineer.png) · [Editable square diagram](images/aud-kernel-engineer-square.svg) · [Editable landscape diagram](images/aud-kernel-engineer.svg)

![Compare designs at equal math, quality, load and resources, then include the complete update path.](images/aud-platform-operations-square.png)

[Landscape card](images/aud-platform-operations.png) · [Editable square diagram](images/aud-platform-operations-square.svg) · [Editable landscape diagram](images/aud-platform-operations.svg)

![If the necessary floor exceeds the target, reject the design. If it does not, implement and measure. Passing the screen does not guarantee runtime.](images/aud-business-capacity-square.png)

[Landscape card](images/aud-business-capacity.png) · [Editable square diagram](images/aud-business-capacity-square.svg) · [Editable landscape diagram](images/aud-business-capacity.svg)

## Source snapshot checksums

`evidence/derivation/SHA256SUMS` retains the historical October 29 snapshot. For the maintained tree, run `shasum -a 256 -c evidence/derivation/SHA256SUMS-current`. Its scope excludes both checksum ledgers and the validation note to avoid self-reference.

## Rebuild the reference diagrams and audience cards

The reference paths and the six audience cards are editable SVG diagrams. The square and landscape cards have separate layouts. Python 3.10+ uses only its standard library, with existing `rsvg-convert`, ImageMagick (`magick`) and Fontconfig (`fc-match`). The retained Inter font and license are in `images/attention-scenes/`. No installation or network access occurs during rendering.

```sh
python3 code/render_attention_reference_figures.py --check --receipt reference-check.json
```

A passing check exits zero and writes `status: PASS`. It checks the retained input hashes, actual weighted Inter glyph bounds, directed arrows, exact zero-based bar geometry and two byte-identical fresh renders. It does not overwrite the retained figures. Omit `--check` to replace the reference diagrams and audience cards, or choose a separate package-shaped tree with `--output-dir /path/to/output`. SVG, PNG and text sidecars are written together after validation, with atomic replacement per file. The set of files is not one filesystem transaction. PNG byte identity depends on the installed rendering stack.

The calculator inputs and results are unchanged. The floor bars render the retained values 0.243269632 and 0.30408704 ms. The buffer-lifetime and scheduling diagrams are qualitative, with no measured time or unknown byte amount encoded as a bar length.

## Rebuild the selected figures

The selected three cards use Inter labels of at least 52 px on a 1200 px canvas. Rebuilding requires the retained font and artwork, rsvg-convert 2.62.3, ImageMagick 7.1.2-26 and Fontconfig (`fc-match`). SVG artwork and font links are relative. An isolated Fontconfig configuration resolves the retained Inter font. The selected renderer preserves its existing default Pango/Cairo backend. The receipt records available rendering-library versions. A failed or interrupted invocation replaces any prior PASS receipt with FAIL or INTERRUPTED, then propagates the error.

```sh
python3 code/render_attention_figures.py --check --receipt selected-check.json
```

A successful check prints a JSON receipt with `status: PASS` and exits zero. With `--receipt selected-check.json`, it also writes that receipt. Without `--receipt`, no receipt file is written. The receipt starts at `INCOMPLETE`, so an abruptly killed run is not a PASS. The check regenerates into temporary storage and compares the retained files. It does not overwrite them. PNG byte equality depends on the rendering stack. To regenerate intentionally, omit `--check` or use `--output-dir` for a separate directory. This renders retained analytical values and performs no GPU experiment.

<details>
<summary>Artwork inputs and compact exports</summary>

![Source card supplying the cropped transfer-scene illustration](images/attention-scenes/factory-source.png)

![Source card supplying the cropped warehouse illustration](images/attention-scenes/warehouse-source.png)

![Scratch traffic comparison at 297 pixels](images/scientific/03-scratch-traffic-297.png)

![Scratch traffic comparison at 312 pixels](images/scientific/03-scratch-traffic-312.png)

![Calculated transfer floor and target at 297 pixels](images/scientific/04-target-rejection-297.png)

![Calculated transfer floor and target at 312 pixels](images/scientific/04-target-rejection-312.png)

![Three accumulation schedules at 297 pixels](images/scientific/06-reduction-comparison-297.png)

![Three accumulation schedules at 312 pixels](images/scientific/06-reduction-comparison-312.png)

</details>


<details>
<summary>Landscape diagrams and compact previews</summary>

The audience cards above also have landscape layouts. These retained landscape images and 400-pixel previews are produced by the reference renderer, with the same assumptions as their square counterparts.

![Write and reread materialized partials at 400 pixels](images/a-400.png)

![Producers finish, write scratch to off-chip memory (HBM), and the reducer rereads it. Writing now and reading later pays two transfers.](images/a.png)

![Reject a design or build and measure it at 400 pixels](images/aud-business-capacity-400.png)

![If the necessary floor exceeds the target, reject the design. If it does not, implement and measure. Passing the screen does not guarantee runtime.](images/aud-business-capacity.png)

![On-chip storage and HBM traffic at 400 pixels](images/aud-curious-400.png)

![Write temporary results from on-chip workspace to off-chip memory (HBM), then reread them before the final combination.](images/aud-curious.png)

![Buffer lifetimes and storage reuse at 400 pixels](images/aud-kernel-engineer-400.png)

![Buffer lifetimes and storage reuse, landscape layout](images/aud-kernel-engineer.png)

![Materialize partials or combine smaller batches at 400 pixels](images/aud-ml-engineer-400.png)

![Materialize partials or combine smaller batches, landscape layout](images/aud-ml-engineer.png)

![Screen compute and traffic separately at 400 pixels](images/aud-performance-engineer-400.png)

![Screen compute and traffic separately, landscape layout](images/aud-performance-engineer.png)

![Compare the complete update path at 400 pixels](images/aud-platform-operations-400.png)

![Compare the complete update path, landscape layout](images/aud-platform-operations.png)

![Compare the transfer floor with the target at 400 pixels](images/b-400.png)

![One layer and one device: the calculated 0.243 ms transfer floor exceeds the 0.200 ms target at assumed 8 TB/s.](images/b.png)

</details>

The three selected figures are also available as a [three-page document](images/Attention-three-cards.pdf). Its [composition record](images/Attention-three-cards.composition.json) identifies the source images and their placement.
