# Project through the legal token rows

A synthetic output-head benchmark compares two ways to return legal token scores and their greedy winner. The dense baseline computes `H @ W.T`, gathers the legal scores, and selects their maximum. The candidate gathers legal weight rows first, computes `H @ W[S, :].T`, and uses the same selection path.

One NVIDIA B200 ran strict FP32 with TF32 disabled, batch 8, hidden width 2,048 and vocabulary 131,072. Each step shared one sorted, noncontiguous legal set across all batch rows. A changing set differed at each of 16 steps; the sequence was reused across warmup and timing.

| Legal IDs | Set behavior | Dense ms/block | Selected rows ms/block | Observed speedup | Setup ms |
|---:|---|---:|---:|---:|---:|
| 16 | Static | 5.489 | 0.334 | 16.43x | 0.035 |
| 16 | Changing | 5.494 | 0.560 | 9.81x | 0.015 |
| 64 | Static | 5.409 | 0.349 | 15.48x | 3.252 |
| 64 | Changing | 5.399 | 0.421 | 12.84x | 0.004 |
| 256 | Static | 5.466 | 0.350 | 15.64x | 0.022 |
| 256 | Changing | 5.471 | 0.421 | 12.99x | 0.004 |

Times are medians of three matched warm blocks, each containing 16 projection/filter/selection steps and final device synchronization. Pair order was dense-first, candidate-first, then dense-first after two warmup blocks per path. Every one of the 18 pairs favored selected rows. The nearest-rank p95 of three samples is the largest block, rather than request tail latency.

Static mode gathers rows once before warmup and timing. Its setup cost is outside the warm-block ratios above; the 64-ID setup took 3.252 ms. Changing mode gathers rows in every timed step. Its setup column records synchronization around an empty setup branch. This test did not measure a continuous first-use interval or an observed payback point.

Both paths checked 86,016 legal logits against an FP64 reference with `rtol=1e-4` and `atol=1e-5`. Maximum absolute errors were approximately `2.44e-6` for dense and `1.01e-6` for selected rows. All 768 selected-token rows per path matched the reference argmax. All-equal and two-way-maximum fixtures chose the smallest legal ID. These checks cover the tested inputs and explicit ties; trained-model near-ties need separate validation.

- [Runnable source and reproduction commands](code/README.md)
- [All six results and paired samples](evidence/recomputed-results.json)
- [Original raw timing, legal IDs and correctness records](evidence/raw.jsonl)
- [Native worker summary](evidence/worker_result.json)
- [Correctness records](evidence/correctness.json) and [timing method](evidence/timing-mode.json)
- [Recorded environment](evidence/environment.json), [measurement metadata](evidence/measurement.json) and [file hashes](evidence/hashes.json)

Recompute and test the retained evidence without a GPU from this directory:

```sh
python3 code/analyze.py
python3 -m unittest discover -s code -p 'test_analyze.py' -v
python3 code/screen.py --self-check
```

The baseline is eager PyTorch CUDA `torch.mm` with preallocated buffers. The observations cover this synthetic stage and input shape. Model computation, grammar transitions, tokenization, sorting and transferring legal IDs, request scheduling, independent request grammars, other precisions, quantized or sharded heads and end-to-end generation were outside the timing. No profiler capture or measured memory peak is included.

API contracts: [PyTorch matrix product](https://docs.pytorch.org/docs/2.14/generated/torch.mm.html), [linear projection](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.linear.html). The [vLLM structured-output interface](https://docs.vllm.ai/en/latest/features/structured_outputs/) provides integration context; this test did not run vLLM.

## Trade-offs

- Allowed tokens must be known before projection. This test starts with sorted, device-resident IDs, excluding grammar/tokenizer work, ID production and transfer.
- Changing-set weight gathering is additional work and is included in its timings. Static reuse pays separate setup outside warm ratios, including the retained 3.252 ms interval at 64 IDs. Cold-start benefit and break-even were not measured.
- All batch rows shared one allowed set. Different sets per request, near-full-vocabulary sets and other precisions remain untested here.
- Only legal scores and their greedy winner are returned. The full weight tensor remains resident. No memory saving or complete-request gain is established.

## Prior work

[Underpinning study by Murali Manohar](https://gitlostmurali.com/blog/structured-generation-optimizations/) explains allowed-token weight slicing and batching limitations. The [GSP working paper](https://github.com/StealthEyeLLC/grammar-sparse-projection/blob/30a9d0ed82f26dc53ec74b4f0985e1d213a18625/paper/draft.md) develops packed-mask projection, heterogeneous batching and an end-to-end evaluation. Those authors’ results are separate from this controlled B200 reproduction. The core mechanism is established.

## Concise stage comparison

The complete configuration and methods are above. This figure summarizes the same measured changing-set case.

![Output-head stage comparison with weight gathering included](images/gamma-explainer/result-social.png)

## Figures

The cover illustration is conceptual. Tile counts and drawn proportions do not encode the benchmark dimensions. Numerical figures state their conditions and describe observed block timings.

### Choose first. Multiply less.

![Choose first. Multiply less.](images/fun.png)

<details>
<summary>Square version</summary>

![Choose first. Multiply less. — square](images/fun-square.png)

</details>

### Select rows before multiplying

![Select rows before multiplying](images/a.png)

<details>
<summary>Square version</summary>

![Select rows before multiplying — square](images/a-square.png)

</details>

### Six measured output-head comparisons

![Six measured output-head comparisons](images/b.png)

<details>
<summary>Square version</summary>

![Six measured output-head comparisons — square](images/b-square.png)

</details>

### Where to inspect a decoder

![Where to inspect a decoder](images/c.png)

<details>
<summary>Square version</summary>

![Where to inspect a decoder — square](images/c-square.png)

</details>

### A smaller set of possible answers

![A smaller set of possible answers](images/aud-curious.png)

<details>
<summary>Square version</summary>

![A smaller set of possible answers — square](images/aud-curious-square.png)

</details>

### The output contract matters

![The output contract matters](images/aud-ml-engineer.png)

<details>
<summary>Square version</summary>

![The output contract matters — square](images/aud-ml-engineer-square.png)

</details>

### Read the paired block timings

![Read the paired block timings](images/aud-performance-engineer.png)

<details>
<summary>Square version</summary>

![Read the paired block timings — square](images/aud-performance-engineer-square.png)

</details>

### Avoid projecting unused rows

![Avoid projecting unused rows](images/aud-kernel-engineer.png)

<details>
<summary>Square version</summary>

![Avoid projecting unused rows — square](images/aud-kernel-engineer-square.png)

</details>

### Where integration time can return

![Where integration time can return](images/aud-platform-operations.png)

<details>
<summary>Square version</summary>

![Where integration time can return — square](images/aud-platform-operations-square.png)

</details>

### Stage gains and whole-system gains

![Stage gains and whole-system gains](images/aud-business-capacity.png)

<details>
<summary>Square version</summary>

![Stage gains and whole-system gains — square](images/aud-business-capacity-square.png)

</details>

### Changing legal sets: the observed stage gain

![Changing legal sets: the observed stage gain](images/comparison-detail.png)

<details>
<summary>Square version</summary>

![Changing legal sets: the observed stage gain — square](images/comparison-detail-square.png)

</details>

## Visual walkthrough

These figures explain the illustrative allowed-choice example and the retained output-head measurements. Model and serving performance were not measured.

![Illustrative allowed choices](images/gamma-explainer/example.png)

![Final scoring paths](images/gamma-explainer/mechanism.png)

![Measured stage comparison](images/gamma-explainer/result.png)

![All six measured cases](images/gamma-explainer/matrix.png)

![Static selection setup](images/gamma-explainer/setup.png)

![Applicability and limits](images/gamma-explainer/boundaries.png)
