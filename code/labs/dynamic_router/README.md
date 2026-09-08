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
- `VLLM_BATCH_INVARIANT=1 python -m cli.aisp bench run --targets labs/dynamic_router:dual_pool_vllm --launch-via python --target-extra-arg labs/dynamic_router:dual_pool_vllm="--model /path/to/model --prefill-gpus 0 --decode-gpus 1 --attention-backend TRITON_ATTN"` contrasts shared versus dual pools with exact token verification and emits per-pool TTFT and queue depth.
- `python -m cli.aisp bench run --targets labs/dynamic_router:topology_probe` captures GPU↔NUMA mappings and distance matrices for consumption by the router.

## Notes
- The dual-pool policies produce different batch shapes. On the pinned vLLM 0.16 stack with GPT-OSS-20B, the default backend produced different greedy tokens for identical prompts, including across repeated optimized runs. The explicit batch-invariant Triton configuration above matched all 1,734 output elements on 2×B200. Apply the same backend and environment to both arms; the option does not change the default backend for other workloads. Other models and stacks still require their own correctness check.
- The vLLM benchmarks construct each model engine once in `setup()`, execute the harness-required five full-workload warmups, and reuse the idle engines for exactly three steady-state iterations. Every invocation clears completed-request bookkeeping, uses a fresh request-id generation, and still verifies every generated token. Custom metrics report engine startup, warmup request processing, and steady-state request processing separately. Teardown emits a `vllm_engine_lifecycle` JSON record with teardown and end-to-end wall time, so moving engine construction outside the steady-state timer cannot be presented as an end-to-end speedup. Prefix caching remains disabled so every request processes its full declared prompt.
- `dynamic_router_vllm` admits its complete request set before the first engine step. Its optimized arm feeds exact run-local admission counts into the policy's existing smoothed queue-depth metric before each subsequent placement. TTFT and tokens-per-step are retained as output diagnostics; they cannot affect requests that were already admitted. Use the synthetic simulator for continuous arrivals, TTFT/TPOT feedback, migration, and KV-locality policy experiments.
- The harness prepares live CPU prompt IDs during setup. The topology-aware runner requires this input; standalone entrypoints create default prompts before calling it. Conversion to the Python token lists required by vLLM remains part of request admission, and GPU-resident prompt inputs fail explicitly before conversion.
- `driver.py` accepts knobs such as `--prefill-gpus`, `--decode-gpus`, and `--migration-budget` to stress different regimes.
- vLLM integration now takes flags (`--model`, `--prefill-gpus`, `--decode-gpus`, etc.) plus locally available tokenizer/model weights.
- Router scoring incorporates pinned-host KV slab availability and NUMA-locality bias; feed it real topology via `topology_probe.py` or NVML when available.
