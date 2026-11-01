# A whole-system budget for a hypothetical training update

This Python workbook follows one complete distributed training update: model work, token ownership, attention and expert layouts, communication, pipeline lifetimes, optimizer ownership and global clipping. It helps reject a design whose necessary resource floor already exceeds its budget. It performs no GPU execution and supplies no measured runtime, speedup or MFU.

The same synthetic work is assigned to two placements of 64 generic GPUs. The primary plan chooses full data-parallel allreduce before the norm, replica 0 ownership of every unique norm term, and FP32 whole-matrix attention Muon on one canonical stage owner. Under those policies, the barrier-ordered floors are **240.63 ms for placement A and 158.93 ms for placement B. Both exceed the illustrative 150 ms budget.** The busiest rank's FP32 Muon work is 51.54 ms for its own experts plus 15.63 ms for gathered attention matrices at the assumed 100 TFLOP/s, totaling 67.17 ms. A separate reduce-scatter-first alternative has different dependency floors, described below. None is a measured runtime or speedup.

## Declared teaching example

| Input | Assumption |
| --- | --- |
| Physical system | 64 GPUs, eight eight-GPU nodes, two racks |
| Training layout | Two data replicas, four pipeline stages per replica, one node per stage |
| Layout inside each stage | Attention TP4 × CP2 and MoE EP8 reuse the same eight ranks |
| Model | 12 blocks, hidden width 2,048, 16 query heads, two K/V heads, head dimension 128 |
| Experts | 16 top-2, two-matrix 2,048 → 4,096 → 2,048 GELU experts per block |
| Step | 2,048 tokens per microbatch, eight microbatches per replica, 32,768 tokens globally |
| Vocabulary | 32,768, with untied input embedding and output head |

The model has 3,469,133,824 unique parameters. Full shapes, precision choices, tensor copies, reductions and exclusions are defined in the [scientific description](code/README.md) and [derivations](evidence/model-derivations.md). The [complete workbook output](evidence/system-budget.json) retains per-rank ledgers. A [compact summary](evidence/reader-summary.json) supplies diagram labels.

All service rates are hypothetical inputs. Per GPU, the model assumes 500 TFLOP/s for BF16 training GEMMs, 100 TFLOP/s for FP32 Muon GEMMs, 3 TB/s HBM and 160 GB/s intra-node service in each direction. Each node has 40 GB/s NIC service in each direction. Each rack has 160 GB/s aggregate intra-rack injected-payload capacity and 80 GB/s uplink capacity in each direction. Links are full duplex. The intra-node fabric has no additional shared bottleneck in this model. Decimal GB/TB and TFLOP/s use powers of 1,000. These are neither hardware specifications nor achieved throughput.

## Dependencies change the rejection test

The primary plan completes local CP/TP sums and full data-parallel allreduce before one global L2 norm. Replica 0 contributes every unique Muon and AdamW gradient norm term, while replica 1 contributes zero. Both optimizers use the resulting clipping scalar. Updated weights and optimizer state must be published before the next step. Full allreduce before the norm and replica 0 norm ownership are chosen policies. Global clipping alone does not require them.

The workbook grants arbitrary overlap by taking the maximum resource floor within each phase. It then adds the pre-clipping and post-clipping maxima across the global barrier:

| Placement | Pre-clipping floor | Post-clipping floor | Ordered floor |
| --- | ---: | ---: | ---: |
| A | 173.456691200 ms | 67.172617421 ms | 240.629308621 ms |
| B | 91.761664000 ms | 67.172617421 ms | 158.934281421 ms |

Norm-scalar service is omitted from this ordered sum and retained in the primary full-step payload screens. Logical directed payload is counted once, without adding received bytes again. Transport headers, retries and physical switch-hop expansion are excluded. Compute floors omit scalar work and software overhead. These optimistic floors reject both placements at 150 ms under the primary DP schedule and FP32 single-owner attention Muon policy. Passing a different budget would establish only that these screens do not reject the design.


## A separate reduce-scatter-first alternative

The alternative replaces each DP allreduce with reduce-scatter before the norm and allgather after clipping. Each directed DP payload is split into equal halves. Both replicas contribute disjoint reduced-gradient norm terms, covering 1,734,566,912 unique parameter elements each. Total bytes and mathematical gradient, norm and update semantics stay the same. The dependency schedule changes.

The post-clipping maximum grants optimistic overlap between allgather and optimizer work:

| Alternative placement | Necessary floor | Screen at 150 ms |
| --- | ---: | --- |
| A | 173.456691200 ms | Excluded |
| B | 113.892310221 ms | Not ruled out |

Norm-scalar service is omitted from all alternative screens. An optimizer must wait for every gradient it consumes, so whole-matrix dependencies and the complete execution DAG can raise these floors. B is merely not ruled out. No feasible schedule, attainable runtime or speedup is established. This alternative does not replace the primary plan.

## What the choices cost

The declared layout changes send 10 MiB per rank, per layer and per microbatch across forward and backward. MiB means `2^20` bytes. Expert dispatch and combine traffic is counted separately.

Placement A keeps each replica within one rack, avoiding cross-rack pipeline transfers while sending all data-parallel gradient pairs across racks. Placement B keeps those gradient pairs within a rack and moves the middle pipeline boundary across racks. It reduces rack pressure but leaves node NIC service and optimizer work. A real comparison needs measured collectives, contention and a dependency-respecting schedule.

The explicit Muon policy gathers whole attention matrices to a canonical stage owner for five finite FP32 Newton–Schulz iterations, then scatters and publishes state. Whole expert matrices already reside on their owners. This creates owner imbalance and communication. Its FP32 recurrence differs from the pinned Muon reference's BF16 iteration. Another optimizer precision, ownership or distributed recurrence requires a new calculation. Muon FLOPs remain separate from training FLOPs and cannot inflate an MFU numerator.

The pipeline illustration uses equal unit forward/backward service and zero handoff latency. Its 22 slots are not milliseconds. The actual output head creates stage imbalance. Chunked expert routing bounds transient staging, while retained backward tensors and extra serial work under skew remain. No tokens are dropped. Across the four stages, the declared totals range from 2.13 to 2.59 GB per rank under balanced routing and 2.59 to 3.41 GB per rank under worst routing skew. GB means `10^9` bytes here, not GiB. These are the stage-wise minimum and maximum ledger totals, rounded to two decimals, not measured peaks. The allocation ledger excludes framework contexts, allocator reserve and fragmentation, extra library/transport workspaces and checkpoint buffers, so it does not prove implementation fit on an 80 GB device.

Use the simpler placement and execution policy when it meets the required budget. Change a binding resource, ownership policy or dependency only after accounting for the costs that the change introduces.

## Reproduce

Python 3.10 or later and the standard library are sufficient. From this directory:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s code -p 'test_*.py' -v
PYTHONDONTWRITEBYTECODE=1 python3 code/system_budget.py --output evidence/system-budget.json
PYTHONDONTWRITEBYTECODE=1 python3 code/reader_summary.py --output evidence/reader-summary.json
PYTHONDONTWRITEBYTECODE=1 python3 code/verify_analytical_results.py
git diff --exit-code evidence/
```

The tests check conservation, unique gradient ownership, rank mapping, whole-matrix optimizer behavior, clipping dependencies, allocation ledgers and the two placement decisions. These local checks do not validate a distributed training implementation or model quality.

To regenerate the six SVG scientific diagram components:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 code/visual_elements.py
PYTHONDONTWRITEBYTECODE=1 python3 code/visual_elements.py --check
```

This uses the retained summary and scientific description to write `images/elements/`. It regenerates the six components, not the complete designed slide deck.

## Primary-source context

The model, traffic choices and service assumptions above are a declared teaching construction. They do not claim to reproduce a framework's default policy or adopt another work's performance results.

- [Megatron Core MoE guide](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/moe.html) and [MoE Parallel Folding](https://arxiv.org/abs/2504.14960v3) describe differing attention and expert parallel mappings.
- [Megatron-LM](https://arxiv.org/abs/2104.04473v5) provides pipeline and tensor-parallel training context.
- Pinned [gradient finalization](https://github.com/NVIDIA/Megatron-LM/blob/53944a33c6e2b440b1ef67ca2ac12f8017cf77ea/megatron/core/distributed/finalize_model_grads.py) and [optimizer ordering](https://github.com/NVIDIA/Megatron-LM/blob/53944a33c6e2b440b1ef67ca2ac12f8017cf77ea/megatron/core/optimizer/optimizer.py) provide source-level context for gradient completion and update dependencies.
- The pinned [Muon implementation](https://github.com/KellerJordan/Muon/blob/f98f1cacc0263b04290753e32be8d498c1efc806/muon.py) is the reference for the finite matrix transform, with the precision difference stated above.
- [Megatron Core distributed optimizer](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/features/dist_optimizer.html) documents an elementwise state-sharding design. Its ownership cannot silently substitute for this workbook's whole-matrix Muon policy.

## Scientific visuals

These are the complete reader cards for the same hypothetical construction. The charts are derived quantities and resource bounds, not measured runtimes.

### Same GPUs. Different traffic.

![Commentary cover with Fregly's take branding, Same GPUs. Different traffic., and a conceptual rack-and-cable illustration](images/fun-square.png)

### Place the whole training graph

![Place the whole training graph](images/article/02-topology-a.png)

R0 and R1 are the two data replicas, one per rack in A. P0–P3 are their pipeline stages.

### Eight GPUs. Two execution views.

![Eight GPUs. Two execution views.](images/article/aud-curious-square.png)

### The full head adds work

![The full head adds work](images/article/aud-ml-engineer-square.png)

### Pay for changing layouts

![Pay for changing layouts](images/article/a-square.png)

### Keep tokens. Count overflow work.

![Keep tokens. Count overflow work.](images/article/06-routing-skew.png)

### Context parallelism moves data too

![Context parallelism moves data too](images/article/07-context-policy.png)

### Pipeline slots are not milliseconds

![Pipeline slots are not milliseconds](images/article/aud-performance-engineer-square.png)

### Memory depends on what stays live

![Memory depends on what stays live](images/article/b-square.png)

### Model state is only the start

![Model state is only the start](images/article/aud-platform-operations-square.png)

### A ready bucket can start reducing

![A ready bucket can start reducing](images/article/11-gradient-barrier.png)

### Same norm. Different waiting.

![Same norm. Different waiting.](images/article/c-square.png)

### Muon needs the whole matrix

![Muon needs the whole matrix](images/article/aud-kernel-engineer-square.png)

### Placement B: pair corresponding stages within racks

![Placement B: pair corresponding stages within racks](images/scientific/14-topology-b.png)

### Placement moves rack traffic

![Placement moves rack traffic](images/scientific/15-tier-traffic.png)

### Two ordered lower bounds

![Two ordered lower bounds](images/scientific/16-ordered-floors.png)

### Decide what to build next

![Decide what to build next](images/article/aud-business-capacity-square.png)


## Rebuild the selected figures

The three selected figures above carry the complete captions, conditions and larger labels. A keeps each data replica in one rack. B pairs corresponding stages within racks. In the ordered-floor chart, blue is before clipping and orange after clipping.

With Pillow, librsvg and Fontconfig (`fc-match`) already installed, validate the selected SVG and PNG exports:

```sh
python3 code/scientific_figures.py --check
```

The renderer reads retained calculations, artwork and Inter font assets. It performs no GPU experiment.

<details>
<summary>Retained barrier illustration used by the selected chart</summary>

![Gradient-barrier conveyor illustration](images/scientific/ordered-barrier-scene.png)

</details>

Selected figure rebuilding was checked with Python 3.12.2, Pillow 11.3.0 and rsvg-convert 2.62.3. Exact PNG bytes depend on the rendering stack and retained Inter font. A PNG mismatch on another build can occur even when SVG and text agree. Inspect those sources before regenerating the PNG. Editable SVGs require their relative artwork assets.

<details>
<summary>Artwork input</summary>

![Conceptual rack-routing illustration cropped into the traffic comparison](images/scientific/rack-traffic-scene.png)

</details>


## Rebuild the article annotations

The article figures use the retained designed PNGs as source artwork. The annotation renderer removes isolated carousel numbers and corrects the pipeline idle legend. Its SVGs link to those original PNGs and keep the annotation shapes and text editable. It does not reconstruct the scientific chart marks. The [input declaration](images/article/render-inputs.json) and [render receipt](images/article/render-receipt.json) bind the source images and the fifteen outputs.

From this package root, check the retained annotation outputs:

```sh
python3 code/article_figures.py --check
```

The retained annotation outputs were generated with Pillow 12.3.0, rsvg-convert 2.62.3 and Fontconfig with `fc-match`. The check reports environment-version differences without failing when all source, SVG and PNG hashes still match. It resolves the packaged Inter Regular font. These presentation tools are separate from the standard-library analytical workbook and the selected-figure rendering stack above. To render a separate preview tree, use `python3 code/article_figures.py --write --output-root /tmp/main-article-figures`. A successful check prints `"status": "CHECKED"`, fifteen figures and one phone derivative. This changes annotations only and performs no scientific calculation.

<details>
<summary>Source artwork for article annotation rendering</summary>

The figures above use these retained inputs. Numbered deck images supply the underlying chart pixels. The original pipeline inputs supply the artwork before its idle-legend annotation is applied.

![Changing layouts source artwork](images/a-square.png)

![Original pipeline schedule source artwork](images/article-sources/aud-performance-engineer-square.png)

![Original landscape pipeline source artwork](images/article-sources/aud-performance-engineer.png)

![Build decision source artwork](images/aud-business-capacity-square.png)

![Eight-rank execution views source artwork](images/aud-curious-square.png)

![Whole-matrix optimizer source artwork](images/aud-kernel-engineer-square.png)

![Full output-head work source artwork](images/aud-ml-engineer-square.png)

![Model-state allocation source artwork](images/aud-platform-operations-square.png)

![Live-allocation source artwork](images/b-square.png)

![Norm-schedule source artwork](images/c-square.png)

![Placement A source artwork](images/scientific/02-topology-a.png)

![Expert-routing source artwork](images/scientific/06-routing-skew.png)

![Context-parallel source artwork](images/scientific/07-context-policy.png)

![Gradient-barrier source artwork](images/scientific/11-gradient-barrier.png)

</details>

### Corrected pipeline outputs

These are annotation-renderer outputs, not the original source artwork. The square and landscape legends use the same text scale as their other legend labels. The phone export is derived from the corrected square.

![Pipeline schedule with a blank-idle legend](images/aud-performance-engineer-square.png)

![Landscape pipeline schedule with a blank-idle legend](images/aud-performance-engineer.png)

The 400 px export below is produced from the corrected square by the same annotation renderer.

![400 px pipeline phone export](images/aud-performance-engineer-400.png)
