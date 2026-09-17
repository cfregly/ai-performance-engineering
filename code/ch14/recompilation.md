# Recompilation and real-time inference

`torch.compile` can improve steady-state execution while a new input signature
causes a long synchronous compile. A fixed-shape benchmark cannot establish an
online service's tail latency. This chapter companion makes the distinction
explicit and supplies a runnable [diagnostic](recompilation_demo.py).

Credit: Chaim Rand's [Overcoming the PyTorch Recompilation Dilemma in Real-Time
Inference Workloads](https://chaimrand.medium.com/overcoming-the-pytorch-recompilation-dilemma-in-real-time-inference-workloads-069fac5485cd)
(September 8, 2026) motivates this review. His six mitigation experiments are
illustrations, not performance evidence for this repository. Our example is an
independent implementation. The article uses PyTorch 2.13. The repository's CPU
validation environment uses 2.9.1, so validate APIs against your installed version.

## Choose a serving contract

| Approach | Appropriate use | Remaining obligation |
| --- | --- | --- |
| Restore compiler cache artifacts | Reduce startup/code-generation work for previously compiled graphs | Cache artifacts do not restore every Dynamo guard or eliminate new signatures. Retain warmup and miss detection. |
| Shape policy and representative warmup | Known input families or padded/bucketed serving | Cover dtype, device, layout/stride, scalar values, zero/one sizes, and distinct control-flow paths. Warmup iteration count alone is insufficient. |
| Scoped `fail_on_recompile` after warmup | Strict compiled measurements and acceptance tests | Reject an unexpected signature visibly. Define admission/coverage before serving. This is the default in the diagnostic. |
| Explicit `eager_on_recompile` | A service deliberately accepts eager latency on a guard miss | Count eager requests, validate their outputs, and include their latency. It does not compile new variants in the background. |
| Background compilation | A separately engineered concurrency experiment | Budget GIL, compiler CPU, GPU memory/stream interference, request ownership, queue bounds, worker errors, shutdown, and state publication. |
| `torch.export` plus AOTInductor | Export-compatible models with an explicit input contract | Validate shape constraints including edge cases and reject out-of-contract inputs. Build/load artifacts for the target runtime. Export alone is not optimized target machine code. |

Do not increase the recompilation limit as a general latency fix. It permits more
variants and associated work. Hitting the limit can produce eager execution (or
an error under stricter settings). This must not masquerade as compiled benchmark
success. `fullgraph=True` detects graph breaks. It does not promise one graph for
all future inputs. `dynamic=True` reduces shape specialization but does not
remove guards on layout, dtype, Python values, or all zero/one cases.

Avoid adopting the article's illustrative worker as a production wrapper. It
changes shared compiler settings from multiple threads. Moving compilation to a
thread also does not establish a latency bound. A separate process cannot simply
publish its in-memory Dynamo state into the serving process. Similarly, private
duck-shape/unbacked controls are version-dependent experiments, not repository
defaults. Never disable guard evaluation to avoid guard failures.

## Review and measure the actual request path

1. Freeze an eager reference with identical weights, inputs, dtype, `eval()` and
   `inference_mode()` settings. Test complete outputs, including misses.
2. Record guard reasons with `TORCH_LOGS=recompiles,graph_breaks,dynamic`. Distinguish
   initial compilation, additional graph submissions, graph breaks, and fallback. A graph counter is not a general count of requests or a pure compile timer.
3. Warm up in the same execution context as serving. Test independent held-out
   signatures, alternate input order, empty/singleton sizes, noncontiguous layouts,
   changing scalar values, and shape relationships. Use tensor scalars where the
   value is data. Keep Python constants when specialization is intentional.
4. Separate startup/warmup from serving, but retain both. Include every serving
   request in p50/p95/p99/max and SLO violations. Never drop compile spikes or eager
   misses from the headline distribution. Report sample count and cache state.
5. Measure host-visible latency with CUDA synchronization when appropriate. CUDA
   events alone omit Python/compiler stalls. Per-request synchronization itself
   changes overlap. Production validation also needs a realistic arrival/queueing
   load test. Diagnostic timings are not canonical benchmark speedups.
6. Repeat after model, runtime, input-contract, and hardware changes. Use fresh
   processes for cold-cache comparisons and distinguish persistent kernel caches
   from in-process Dynamo state. Preserve user caches rather than deleting them.

## Repository audit and placement

Chapter 14 owns compiler guards and policies. Chapter 16 owns serving admission,
load, and SLO validation. The decode lab owns its fixed decode workload. Existing
fixed-shape pairs remain useful steady-state comparisons and should not silently
switch to a variable-shape or hybrid serving contract.

The review found a concrete problem in `torch_compiler_examples.py`: configuration
installed `eager_on_recompile` globally before warmup and set
`TRITON_ALWAYS_COMPILE=1`. The former could suppress compilation/fall back while
reporting compiled results. The latter undermined cache reuse. The example now
preserves compiler policy during configuration, explicitly warms up under the
default stance, and measures under a scoped `fail_on_recompile` stance. It exposes
warmup wall time and uses the same inference/attention context in both phases.

The NanoChat serving `Engine` also used an unsupported compile mode and swallowed
construction errors. It now uses the supported `max-autotune` mode, checks the
model's actual device, reports explicit eligibility/configuration/runtime status,
and propagates constructor and lazy execution failures. Call
`engine.get_compile_diagnostics()` outside the request hot path to inspect status
and process-wide Dynamo graph counts. A completed call is only `runtime_observed`. It does not establish that later cache positions avoid recompilation.

NanoChat's Python KV-cache position still changes during decode. Its compiled
serving path needs real Blackwell multi-token validation. The fixed-input inference
benchmark pair does not exercise that engine contract. The opt-in test covers one-
and two-layer models, warms prefill and the first decode, then checks six more
positions under `fail_on_recompile`. It compares complete logits and populated KV
contents at every step and verifies that preallocated cache storage stays stable.
The opt-in test is:

```bash
NANOCHAT_RUN_BLACKWELL_COMPILE_TESTS=1 python -m pytest labs/nanochat_fullstack/tests/test_engine_compile_policy.py -q
```

The [B200 follow-up](../../docs/audits/2026-09-17/recompilation-gpu-validation.md)
found that sliced rotary `out=` writes prevented full-graph tracing. The compiled
path now uses functional rotary operations and leaves buffer reuse to Inductor.
Standalone CUDA rotary checks pass, but the tested PyTorch 2.13 runtime still
fails during the first full-model decode. That failure also reproduces with fresh
caches and without repository compiler patches. NanoChat's compiled serving path
remains unqualified on that runtime. The follow-up includes separate B200
latency measurements for this chapter's small recompilation diagnostic, with
rejections, fallbacks, warmup cost, and profiler evidence. Those measurements do
not establish NanoChat performance or a benchmark speedup. The follow-up also
reports three matched eager-versus-compiled Transformer runs, with an observed
1.0204× ratio that falls below the harness's 1.05× acceptance threshold.

The general harness also does not enforce zero compilation across every target's
measurement phase. Use the scoped strict example and the diagnostic to establish
that property for a workload. Do not interpret a harness pass alone as proof of
recompilation-free serving.

The new diagnostic exercises real Dynamo guards with correctness checks. An
`eager` backend tests compiler routing without generating optimized kernels. Use
the explicit `inductor` backend for code-generation experiments. Neither mode
replaces the harness's hardware, clock, profiler, and repeated-comparison gates.

## API references

Run from `code/` (PyTorch 2.9+):

```bash
python -m cli.aisp demos ch14-recompilation -- --json /tmp/recompilation-strict.json
python -m cli.aisp demos ch14-recompilation -- --policy eager_on_recompile --json /tmp/recompilation-hybrid.json
TORCH_LOGS=recompiles,graph_breaks,dynamic python -m cli.aisp demos ch14-recompilation -- --device cuda --backend inductor --json /tmp/recompilation-cuda.json
python -m pytest tests/test_recompilation_demo.py tests/test_compiler_example_policy.py -q
```

The default strict experiment intentionally injects one late unseen signature and
checks that it is rejected before compilation. A successful diagnostic run means
the expected policy held, not that every request was served. Inspect rejection
and fallback counts alongside the all-attempt latency distribution. Each scenario
runs in a fresh process. Persistent compiler caches may still be warm.

References:

- [PyTorch 2.9 compiler stances](https://docs.pytorch.org/docs/2.9/generated/torch.compiler.set_stance.html)
- [PyTorch recompilation troubleshooting](https://docs.pytorch.org/docs/2.9/compile/programming_model.recompilation.html)
- [Compiler cache artifacts](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
- [Export and AOTInductor](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html)
