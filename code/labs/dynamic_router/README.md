# Lab - Dynamic Prefill/Decode Router

## Summary
Simulates and benchmarks dynamic routing policies for large-scale inference: split GPUs into prefill/decode pools, monitor TTFT/TPOT, honor KV locality, and migrate traffic only when the score gap warrants it.

## Learning Goals
- Compare naive round-robin routing with telemetry-driven policies that stabilize TTFT.
- Prototype migration budgets, KV-locality boosts, and per-pool thresholds.
- Drive the router against synthetic workloads or real vLLM engines.
- Export detailed metrics (TTFT, TPOT, queue depth) for visualization.

## Directory Layout
| Path | Description |
| --- | --- |
| `router_round_robin.py`, `router_policy.py`, `driver.py`, `eval_stack.py` | Core router logic plus a synthetic simulator for deterministic comparisons. |
| `baseline_dynamic_router_vllm.py`, `optimized_dynamic_router_vllm.py`, `vllm_runner.py` | Integrations for running the routing policy against vLLM instances. |
| `baseline_dual_pool_vllm.py`, `optimized_dual_pool_vllm.py` | Shared-pool vs dual-pool TTFT benchmarks that reuse `vllm_runner.py`. |
| `topology.py`, `topology_probe.py` | NUMA-aware GPU mapping helpers and a target that emits topology JSON under `artifacts/topology/` for routing hints. |

## Running the Benchmarks
Use the benchmark harness for quick comparisons or drive the Typer CLI when you need repeatable artifact capture.
```bash
python -m cli.aisp bench list-targets --chapter labs/dynamic_router
python -m cli.aisp bench run --targets labs/dynamic_router --profile minimal
```
- Targets follow the `labs/dynamic_router:<workload>` naming convention listed by `list-targets`.
- Use `--target-extra-arg labs/dynamic_router:<workload>="--flag value"` to sweep schedule knobs.
- Benchmark validity profile defaults to strict. Virtualization is warning-only; use `--validity-profile portable` for broader compatibility on hardware-limited environments.
- Portable runs do not write expectation files unless `--allow-portable-expectations-update` is also provided.

## Validation Checklist
- `python labs/dynamic_router/driver.py --mode baseline` vs `--mode optimized` shows lower TTFT variance and higher TPOT for the optimized policy.
- `python -m cli.aisp bench run --targets labs/dynamic_router --profile minimal` records artifacts comparing baseline/optimized harness runs.
- `python -m cli.aisp bench run --targets labs/dynamic_router:dynamic_router_vllm --target-extra-arg labs/dynamic_router:dynamic_router_vllm="--model /path/to/model --decode-gpus 0,1"` succeeds on hosts with at least two GPUs and a local model copy.
- `VLLM_BATCH_INVARIANT=1 python -m cli.aisp bench run --targets labs/dynamic_router:dynamic_router_vllm --target-extra-arg labs/dynamic_router:dynamic_router_vllm="--model /path/to/model --decode-gpus 0,1 --attention-backend TRITON_ATTN --routing-arrival-profile two-wave-imbalance --req-count 16 --long-prompt-tokens 4096 --short-prompt-tokens 128 --prefill-burst 4 --decode-requests 4 --continue-requests 8 --max-tokens 16"` runs the opt-in delayed-arrival comparison. Use the same model, GPU visibility and order, clocks, attention backend, warmups, and steady-state iterations for both arms.
- `VLLM_BATCH_INVARIANT=1 python -m cli.aisp bench run --targets labs/dynamic_router:dual_pool_vllm --launch-via python --target-extra-arg labs/dynamic_router:dual_pool_vllm="--model /path/to/model --prefill-gpus 0 --decode-gpus 1 --attention-backend TRITON_ATTN"` contrasts shared versus dual pools with exact token verification and emits per-pool TTFT, queue depth, and per-GPU admission counts.
- `python -m cli.aisp bench run --targets labs/dynamic_router:topology_probe` captures GPU↔NUMA mappings and distance matrices for consumption by the router.

## Notes
- The dual-pool policies produce different batch shapes. On the pinned vLLM 0.16 stack with GPT-OSS-20B, the default backend produced different greedy tokens for identical prompts, including across repeated optimized runs. The explicit batch-invariant Triton configuration above matched all 1,734 output elements on 2×B200. Apply the same backend and environment to both arms; the option does not change the default backend for other workloads. Other models and stacks still require their own correctness check.
- The pinned vLLM 0.16 engine path accepts an unset `CUDA_VISIBLE_DEVICES` or numeric physical indices such as `0,1`; whole-GPU UUID tokens such as `GPU-...` raise a launch-configuration error before engine construction. An allocation launcher that receives whole-GPU UUIDs must resolve those exact assigned devices to numeric physical indices and preserve their order when exporting `CUDA_VISIBLE_DEVICES`; it must not substitute different GPUs. This lab does not support `MIG-...` UUID visibility: keep a MIG allocation unchanged and do not replace it with its parent GPU; obtain a supported whole-GPU allocation before running the lab.
- The vLLM benchmarks construct each model engine once in `setup()`, execute the harness-required five full-workload warmups, and reuse the idle engines for exactly three steady-state iterations. Every invocation clears completed-request bookkeeping, uses a fresh request-id generation, and still verifies every generated token. Custom metrics report engine startup, warmup request processing, and steady-state request processing separately. Teardown emits a `vllm_engine_lifecycle` JSON record with teardown and end-to-end wall time, so moving engine construction outside the steady-state timer cannot be presented as an end-to-end speedup. Prefix caching remains disabled so every request processes its full declared prompt.
- `dynamic_router_vllm` keeps `--routing-arrival-profile all-upfront` as its default: it admits the complete request set before the first engine step. Its optimized arm feeds exact run-local admission counts into the policy's existing smoothed queue-depth metric before each subsequent placement. TTFT and tokens-per-step are retained as output diagnostics; they cannot affect requests that were already admitted. Use the synthetic simulator for continuous arrivals, migration, and KV-locality policy experiments.
- `--routing-arrival-profile two-wave-imbalance` is a controlled two-GPU experiment. Both arms first place `--prefill-burst` long background requests on the first shared GPU and `--decode-requests` short background requests on the second. The runner steps both real engines in stable order until it observes the second engine at queue depth zero while the first remains busy, then admits `--continue-requests` short foreground requests. Baseline uses round robin starting on the loaded GPU; optimized uses the existing Router with the queue depth, TTFT, and output tokens observed from those engine steps. The runner does not sleep to create the arrival state and fails if the required state never occurs. It exports the observed gate depths, per-cohort TTFT, and per-cohort GPU admissions, and still retains every generated token in request order across both waves.
- The fixed background placement is identical in baseline and optimized so only foreground placement differs. The opt-in runner updates exact queue counts immediately after each admission while keeping latency and token-activity smoothing; the all-upfront and sampled-telemetry defaults are unchanged. This workload may trade foreground latency against background completion or total throughput. Treat the comparison as measured only after both arms pass exact-token verification on the same runtime; the opt-in workload does not imply an improvement or a universal acceptance ratio.
- The repeated 2×B200 delayed-arrival run at `15978d45efb6c11b731ed3aae1074f35ad5910ec` passes all 48 full-output checks but shows no useful win over round robin: median completion is 1,285.304 versus 1,283.833 ms, and foreground p50 is 69.480 versus 58.872 ms. Routing foreground p95 varies from 57.893 to 81.749 ms; its 69.794 ms median does not establish a stable improvement over round robin's 70.612 ms. The policy's weighted historical feedback can still send most or all new requests to one engine. Treat this as a measured policy limitation, not an optimized serving recommendation. See the [complete routing evidence](../../../docs/reviews/2026-09-08-b200-remaining-followthrough.md#delayed-arrival-routing-results) for workload, repetitions, exact-output checks, and retained earlier results.
- `dual_pool_vllm` also admits its complete workload before the first engine step. Both arms update run-local queue depth after every successful admission. With the default 102-request workload and `--long-spillover-limit 0`, the shared two-GPU pool admits 51 requests per GPU, while the dedicated layout admits six long-prefill requests to the prefill GPU and 96 short requests to the decode GPU. A positive spillover limit lets the dedicated layout admit at most that many long-prefill requests to a decode-only GPU; requested and actual spillover plus per-class, per-GPU admission counts are exported. This may trade tail balance against short-request TTFT and throughput, so compare the same GPUs and workload with exact token verification and separate engine startup, warmup, and steady-state timings. All counts reset on every reused-engine invocation.
- The harness prepares live CPU prompt IDs during setup. The topology-aware runner requires this input; standalone entrypoints create default prompts before calling it. Conversion to the Python token lists required by vLLM remains part of request admission, and GPU-resident prompt inputs fail explicitly before conversion.
- `driver.py` accepts knobs such as `--prefill-gpus`, `--decode-gpus`, and `--migration-budget` to stress different regimes.
- vLLM integration now takes flags (`--model`, `--prefill-gpus`, `--decode-gpus`, etc.) plus locally available tokenizer/model weights.
- Router scoring incorporates pinned-host KV slab availability and NUMA-locality bias; feed it real topology via `topology_probe.py` or NVML when available.


## B200 placement tradeoff

A six-pair run with GPT-OSS-20B, batch-invariant Triton attention, two B200s
at 1500/3996 MHz, and the default 102-request mix measured these medians.
Engine startup is excluded; each arm reuses its two engines after five warmups.

| Placement | Completion | Short-request TTFT p50 | Long-request TTFT p50 |
| --- | ---: | ---: | ---: |
| Shared | 1330 ms | 921 ms | 592 ms |
| Dedicated (default) | 1633 ms | 474 ms | 947 ms |
| Dedicated, `--long-spillover-limit 1` | 1395 ms | 710 ms | 710 ms |

One long spillover raises throughput about 17% over dedicated placement and
preserves exact generated tokens in this run. It increases short-request
latency, so it is opt-in. Choose shared placement for aggregate throughput,
dedicated placement for the lowest short-request latency, or test spillover
when both matter. Other workloads need their own measurements. The full
[validation report](../../../docs/reviews/2026-09-08-b200-remaining-followthrough.md)
retains p95 latency, source/runtime details, and all six outcomes.
