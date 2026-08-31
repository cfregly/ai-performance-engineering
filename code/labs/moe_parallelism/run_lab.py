"""MoE parallelism planner (tool; not a benchmark pair)."""

from __future__ import annotations


import argparse

from labs.moe_parallelism.plan import PlanEvaluator, format_report  # noqa: E402
from labs.moe_parallelism.scenarios import get_scenario_pairs  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MoE parallelism planner (tool).")
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="Scenario name to run (repeatable). Default: run all scenarios.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    scenarios = dict(get_scenario_pairs())
    selected = args.scenario or list(scenarios.keys())

    for name in selected:
        if name not in scenarios:
            choices = ", ".join(sorted(scenarios.keys()))
            raise ValueError(f"Unknown scenario '{name}'. Choices: {choices}")

        scenario = scenarios[name]
        evaluator = PlanEvaluator(scenario.cluster, scenario.model)
        baseline = evaluator.analyze(scenario.baseline)
        optimized = evaluator.analyze(scenario.optimized)

        step_speedup = (
            baseline.estimated_step_ms / optimized.estimated_step_ms
            if optimized.estimated_step_ms
            else float("inf")
        )
        throughput_ratio = (
            optimized.throughput_tokens_per_s / baseline.throughput_tokens_per_s
            if baseline.throughput_tokens_per_s
            else float("inf")
        )

        print(f"\n=== {name} ===")
        print(f"Cluster: {scenario.cluster.name} ({scenario.cluster.gpus_total} GPUs)")
        print(f"Model assumptions: {scenario.model.name}")
        print("Planning estimates only; these are not measured performance results.")
        print(format_report(baseline))
        print("---")
        print(format_report(optimized))
        print(f"\nModeled comparison: step-time ratio={step_speedup:.2f}x  throughput ratio={throughput_ratio:.2f}x")


if __name__ == "__main__":
    main()
