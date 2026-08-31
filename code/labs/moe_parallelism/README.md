# Lab - MoE Parallelism Planner

## Summary
Scenario planning tool for mixture-of-experts clusters: memory budgeting, network affinity, parallelism breakdown, and pipeline schedules.

## Learning Goals
- Quantify memory budgets for experts, routers, and KV caches before deploying models.
- Explore different grouping strategies (hashing, topology-aware) and their throughput impact.
- Model network affinity to decide where experts should live in an NVLink/NVSwitch fabric.
- Simulate pipeline schedules to identify bottlenecks before touching production systems.

## Directory Layout
| Path | Description |
| --- | --- |
| `run_lab.py`, `scenarios.py`, `plan.py` | Tool entry point + canonical scenario definitions and sizing model. |
| `benchmarking.py` | Optional harness-compatible wrapper for ad-hoc integration; not currently a public baseline/optimized benchmark pair. |

## Running the Tool
Use the tool entrypoint or the module command when you want reproducible scenario comparisons. This lab does not currently expose public baseline/optimized harness targets.
```bash
python -m cli.aisp tools moe-parallelism -- --scenario memory_budget
python -m cli.aisp tools moe-parallelism -- --scenario gpt_gb200
python -m labs.moe_parallelism.run_lab --scenario deepseek_gb200
```
- `python -m cli.aisp bench list-targets --chapter labs/moe_parallelism` intentionally returns no benchmark pairs today.
- If this lab is later promoted into harness targets, add explicit `baseline_*.py` and `optimized_*.py` entrypoints instead of implying them in the README.

## Validation Checklist
- `python -m cli.aisp tools moe-parallelism -- --scenario memory_budget` runs a single scenario via the tool registry.
- `python -m cli.aisp tools moe-parallelism -- --scenario gpt_gb200` runs a larger cluster scenario.
- `python -m labs.moe_parallelism.run_lab --scenario deepseek_gb200` runs the planner directly (without aisp).

## Notes
- This is a workflow-oriented scenario and playbook lab, not a benchmark-pair lab.
- Baseline vs optimized here are *planning* scenarios (different designs), not comparable performance benchmarks.
- The first five scenarios use the 128-GPU DGX A100 sizing preset; `gpt_gb200` and `deepseek_gb200` keep their separate 576-GPU presets. Cluster and model values are planning assumptions, not validated deployment specifications.
- `scenarios.py` binds each design to its cluster/model preset; `plan.py` supplies the sizing model. EP=1 has no expert all-to-all, and network messages use the configured interconnect and rates.
