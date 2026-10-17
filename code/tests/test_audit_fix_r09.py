"""Audit fix r09: the Ozaki README must describe the pair checksum comparison as it behaves.

These checks run the real policy loader, benchmark wrappers and harness comparator on CPU.
They do not run the CUDA executable, so they do not qualify GPU accuracy.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

CODE = Path(__file__).resolve().parents[1]
OZAKI_DIR = CODE / "labs" / "ozaki_scheme"


def _checksum_std(m: int, n: int, k: int, input_scale: float) -> float:
    # Standard deviation of sum(A @ B) for independent uniform(-s, s) entries.
    return math.sqrt(m * n * k) * input_scale**2 / 3.0


def test_checksum_std_formula_matches_sampled_uniform_gemms() -> None:
    m, n, k, scale, trials = 24, 20, 28, 1e-3, 4000
    generator = torch.Generator().manual_seed(2026)
    a = (torch.rand(trials, m, k, dtype=torch.float64, generator=generator) * 2 - 1) * scale
    b = (torch.rand(trials, k, n, dtype=torch.float64, generator=generator) * 2 - 1) * scale
    checksums = torch.bmm(a, b).sum(dim=(1, 2))
    sampled = float(checksums.std())
    assert sampled == pytest.approx(_checksum_std(m, n, k, scale), rel=0.1)


def test_ozaki_readme_states_pair_checksum_is_not_accuracy_evidence(monkeypatch, tmp_path) -> None:
    from core.benchmark.quarantine import QuarantineManager
    from core.benchmark.verify_runner import VerifyRunner
    from labs.ozaki_scheme.accuracy_policy import (
        DEFAULT_POLICY_PATH,
        WORKLOAD,
        load_accuracy_policy,
    )
    from labs.ozaki_scheme.baseline_ozaki_scheme import get_benchmark as baseline_benchmark
    from labs.ozaki_scheme.optimized_ozaki_scheme_dynamic import get_benchmark as dynamic_benchmark
    from labs.ozaki_scheme.optimized_ozaki_scheme_fixed import get_benchmark as fixed_benchmark

    monkeypatch.setenv("AISP_OZAKI_ACCURACY_POLICY", str(DEFAULT_POLICY_PATH))
    policy = load_accuracy_policy(DEFAULT_POLICY_PATH)
    tolerances = {
        "baseline": baseline_benchmark().get_output_tolerance(),
        "dynamic": dynamic_benchmark().get_output_tolerance(),
        "fixed": fixed_benchmark().get_output_tolerance(),
    }
    for variant in ("dynamic", "fixed"):
        limits = policy["variants"][variant]
        assert tolerances[variant] == (limits["checksum_rtol"], limits["checksum_atol"])

    expected_std = _checksum_std(WORKLOAD["m"], WORKLOAD["n"], WORKLOAD["k"], WORKLOAD["input_scale"])
    smallest_atol = min(atol for _rtol, atol in tolerances.values())
    assert smallest_atol > 100 * expected_std

    runner = VerifyRunner(
        cache_dir=tmp_path / "golden",
        quarantine_manager=QuarantineManager(cache_dir=tmp_path),
    )
    native_checksum = torch.tensor([3 * expected_std], dtype=torch.float32)
    zero_checksum = torch.zeros(1, dtype=torch.float32)
    for name, tolerance in tolerances.items():
        comparison = runner.compare_perf_outputs(native_checksum, zero_checksum, tolerance)
        assert comparison.passed, f"{name} tolerance {tolerance} rejected an all-zero checksum"
    # Negative control: the same comparator does reject the zero checksum under a tight bound.
    assert not runner.compare_perf_outputs(native_checksum, zero_checksum, (0.0, expected_std)).passed

    readme = " ".join((OZAKI_DIR / "README.md").read_text(encoding="utf-8").split())
    assert "that comparison is not accuracy evidence" in readme
    assert f"`{tolerances['dynamic'][1]:.2f}` for dynamic" in readme
    assert f"`{tolerances['fixed'][1]:.2f}` for fixed" in readme
    assert f"standard deviation of only about `{expected_std:.2f}`" in readme
    assert "accepts an all-zero output" in readme
    assert "Only the executable's full-array check gates accuracy." in readme


def test_env_preset_collective_math_matches_the_plan_evaluator() -> None:
    from dataclasses import replace

    from labs.moe_parallelism.moe_env_presets import (
        bytes_to_ms,
        estimate_collective,
        estimate_payload_bytes,
    )
    from labs.moe_parallelism.plan import PlanEvaluator
    from labs.moe_parallelism.scenarios import get_scenario_pairs

    # Network rates are decimal GB/s, and a zero-byte payload takes no time.
    assert bytes_to_ms(1e9, 100) == pytest.approx(10.0)
    assert bytes_to_ms(0, 0) == 0.0

    checked = 0
    for pair in get_scenario_pairs().values():
        for plan in (pair.baseline, pair.optimized, replace(pair.optimized, cross_node_ep=True)):
            report = PlanEvaluator(pair.cluster, pair.model).analyze(plan)
            payload, base_bw, effective_bw, collective_ms = estimate_collective(
                plan, pair.model, pair.cluster, efficiency=1.0, handshake_ms=0.0
            )
            assert payload == estimate_payload_bytes(plan, pair.model)
            assert effective_bw == base_bw
            assert collective_ms == pytest.approx(report.ep_time_ms, rel=1e-12, abs=0.0)
            checked += 1
    assert checked == 3 * len(get_scenario_pairs())

    # EP=1 runs no all-to-all, so it has no payload and pays no handshake.
    pair = get_scenario_pairs()["gpt_gb200"]
    single = replace(pair.optimized, dp=1, pp=1, tp=1, ep=1, stage_layers=None)
    assert estimate_payload_bytes(single, pair.model) == 0.0
    payload, _base_bw, _effective_bw, collective_ms = estimate_collective(
        single, pair.model, pair.cluster, efficiency=0.8, handshake_ms=5.0
    )
    assert payload == 0.0
    assert collective_ms == 0.0
    assert PlanEvaluator(pair.cluster, pair.model).analyze(single).ep_time_ms == 0.0

    # A real collective still pays the handshake once.
    payload, _base_bw, effective_bw, collective_ms = estimate_collective(
        pair.optimized, pair.model, pair.cluster, efficiency=0.8, handshake_ms=5.0
    )
    assert payload > 0
    assert collective_ms == pytest.approx(bytes_to_ms(payload, effective_bw) + 5.0)


def test_ozaki_policy_comment_matches_the_readme_and_the_checksum_math() -> None:
    from labs.ozaki_scheme.accuracy_policy import ENGINEERING_CEILINGS, WORKLOAD

    expected_std = _checksum_std(WORKLOAD["m"], WORKLOAD["n"], WORKLOAD["k"], WORKLOAD["input_scale"])
    dynamic_atol = ENGINEERING_CEILINGS["dynamic"]["checksum_atol"]
    fixed_atol = ENGINEERING_CEILINGS["fixed"]["checksum_atol"]
    assert round(dynamic_atol / expected_std, -2) == 24600
    assert round(fixed_atol / expected_std, -1) == 190

    source = (OZAKI_DIR / "accuracy_policy.py").read_text(encoding="utf-8")
    comment = " ".join(
        line.strip().lstrip("#").strip()
        for line in source.split("ENGINEERING_CEILINGS = {", 1)[0].splitlines()
        if line.strip().startswith("#")
    )
    assert f"({dynamic_atol:.2f} dynamic, {fixed_atol:.2f} fixed)" in comment
    assert "about 24,600 and 190 times the standard deviation" in comment
    assert f"which is about {expected_std:.2f}" in comment
    assert "Even the fixed bound accepts an all-zero output" in comment
    assert "The executable's full-array gate is the accuracy check." in comment
    assert "four orders of magnitude" not in comment
    assert ";" not in comment
