#!/usr/bin/env python3
"""Revalidate the captured four-factory CPU/Gloo result matrix."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from labs.train_distributed.zero2_child_protocol import validate_zero2_result_bundle


ROOT = Path(__file__).resolve().parent
CASES = {
    "baseline-single": ("current-baseline-single.json", "current-baseline-single"),
    "optimized-single": (
        "current-optimized-single.json",
        "current-optimized-single",
    ),
    "baseline-multigpu": (
        "current-baseline-multigpu.json",
        "current-baseline-multigpu",
    ),
    "optimized-multigpu": (
        "current-optimized-multigpu.json",
        "current-optimized-multigpu",
    ),
}


def main() -> None:
    bundles = {}
    for label, (context_name, result_name) in CASES.items():
        context = json.loads((ROOT / context_name).read_text())
        bundles[label] = validate_zero2_result_bundle(
            ROOT / result_name,
            run_id=context["run_id"],
            mode=context["mode"],
            variant=context["variant"],
            world_size=context["world_size"],
            launch_wall_ns=context["launch_wall_ns"],
            launch_monotonic_ns=context["launch_monotonic_ns"],
            finish_wall_ns=context["finish_wall_ns"],
            finish_monotonic_ns=context["finish_monotonic_ns"],
            profile_kind="verification-only",
        )

    pair_diffs = {}
    for variant in ("single", "multigpu"):
        baseline = bundles[f"baseline-{variant}"]["verify_output"]
        optimized = bundles[f"optimized-{variant}"]["verify_output"]
        if set(baseline) != set(optimized):
            raise RuntimeError(f"{variant} output tensor names do not match")
        diffs = {}
        for name in baseline:
            torch.testing.assert_close(
                baseline[name],
                optimized[name],
                rtol=1.0e-5,
                atol=1.0e-6,
            )
            diffs[name] = (
                float((baseline[name] - optimized[name]).abs().max())
                if baseline[name].numel()
                else 0.0
            )
        pair_diffs[variant] = max(diffs.values())

    print(
        json.dumps(
            {
                "status": "PASS_ALL_FOUR_ZERO2_CPU_GLOO_VERIFICATION_ONLY",
                "performance_claim": False,
                "cases": sorted(bundles),
                "pair_max_abs_diff": pair_diffs,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
