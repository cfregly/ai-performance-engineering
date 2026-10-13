"""Regressions for W2-081 (frozen speedup bar for an unattributable SDPA pair)."""

from __future__ import annotations

import json
import re
import shlex
from functools import lru_cache
from pathlib import Path

import pytest
import torch

from core.benchmark.expectations import ExpectationsStore
from core.discovery import discover_benchmarks
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness
from core.harness.run_benchmarks import expectation_example_key
from core.utils.chapter_compare_template import get_last_load_error, load_benchmark

LAB_SLUG = "labs/cudnn_sdpa_bench"
LAB_DIR = Path(__file__).resolve().parents[1] / LAB_SLUG
CHECKED_IN_HARDWARE_KEYS = {"b200", "4x_gb200"}
_DOCUMENTED_OVERRIDE = re.compile(r'--target-extra-arg (\S+?)="([^"]*)"')
# The harness passes one target's --target-extra-arg to both variants, so a
# shared "--backend X" can turn the pair into X vs X even when no README
# command shows it. Probe every backend name the lab has accepted.
_SHARED_BACKEND_PROBES = ("auto", "cudnn", "flash", "math")
# Sides each backend flag selects, as (baseline, optimized) indexes. A README
# command that uses any other flag ending in "backend" fails the test, so a
# renamed flag has to be added here before its commands are trusted.
_BACKEND_FLAG_SIDES = {
    "--backend": (0, 1),
    "--baseline-backend": (0,),
    "--optimized-backend": (1,),
}
_SDPA_OPS = {
    "cudnn": "aten::_scaled_dot_product_cudnn_attention",
    "flash": "aten::_scaled_dot_product_flash_attention",
    "efficient": "aten::_scaled_dot_product_efficient_attention",
    "math": "aten::_scaled_dot_product_attention_math",
}


@lru_cache(maxsize=1)
def _discovered_pair() -> tuple[str, Path, Path]:
    pairs = discover_benchmarks(LAB_DIR)
    assert len(pairs) == 1, pairs
    baseline_path, optimized_paths, example_name = pairs[0]
    assert len(optimized_paths) == 1, optimized_paths
    return example_name, baseline_path, optimized_paths[0]


def _target_label() -> str:
    return f"{LAB_SLUG}:{_discovered_pair()[0]}"


def _run_configurations() -> dict[str, tuple[list[str] | None, bool]]:
    """Return the default run, every README override, and the shared-flag probes.

    Each value is ``(argv, must_apply)``. A README command whose override the
    hook rejects is a broken doc, so it must apply. A probe is a shared
    --backend that both hooks reject, and the harness then fails the target,
    so a probe times nothing.
    """
    target = _target_label()
    readme = (LAB_DIR / "README.md").read_text(encoding="utf-8")
    documented = [args for label, args in _DOCUMENTED_OVERRIDE.findall(readme) if label == target]
    assert documented, "README documents no per-target override for the pair"
    # Every literal override for this target must parse, so a quoting change cannot hide one.
    assert len(documented) == readme.count(f"--target-extra-arg {target}="), documented
    configurations: dict[str, tuple[list[str] | None, bool]] = {"default": (None, True)}
    for args in documented:
        configurations[args] = (shlex.split(args), True)
    for backend in _SHARED_BACKEND_PROBES:
        configurations.setdefault(f"--backend {backend}", (["--backend", backend], False))
    return configurations


def _load_variants(
    argv: list[str] | None, *, must_apply: bool = True
) -> tuple[BaseBenchmark, BaseBenchmark]:
    """Load both variants and apply ``argv`` through the harness hook a real run uses.

    A probe (must_apply=False) must make the harness raise the hook's
    ValueError, which fails the target instead of running the defaults.
    """
    _, baseline_path, optimized_path = _discovered_pair()
    variants = []
    for path in (baseline_path, optimized_path):
        benchmark = load_benchmark(path)
        assert benchmark is not None, get_last_load_error()
        if argv is not None:
            target = _target_label()
            config = BenchmarkConfig(target_label=target, target_extra_args={target: argv})
            harness = BenchmarkHarness(config=config)
            if must_apply:
                harness._apply_target_overrides(benchmark, config)
                applied = getattr(benchmark, "_applied_target_override_signature", None)
                assert applied == (target, tuple(argv)), (path.name, argv)
            else:
                with pytest.raises(ValueError, match="--baseline-backend and --optimized-backend"):
                    harness._apply_target_overrides(benchmark, config)
        variants.append(benchmark)
    return variants[0], variants[1]


def _assert_documented_backends_reached(argv: list[str], resolved: tuple[str, str]) -> None:
    """Check that each backend a README command requests reaches the side it names.

    Both hooks reject a flag they do not define, so this check guards the side
    mapping. Without it, a README command could promise cuDNN on one side and
    run the default backend there.
    """
    tokens = [part for token in argv for part in token.split("=", 1)]
    for flag, value in zip(tokens, tokens[1:], strict=False):
        if not (flag.startswith("--") and flag.endswith("backend")):
            continue
        assert flag in _BACKEND_FLAG_SIDES, f"unknown backend flag {flag} in {argv}"
        for side in _BACKEND_FLAG_SIDES[flag]:
            assert resolved[side] == value.lower(), (argv, flag, value, resolved)


def _attributable(baseline_backend: str, optimized_backend: str) -> bool:
    # "auto" lets PyTorch pick cuDNN, Flash, or efficient attention, and equal
    # backends compare nothing, so neither pair can explain a stored speedup.
    return "auto" not in {baseline_backend, optimized_backend} and (
        baseline_backend != optimized_backend
    )


def _expectation_stores() -> list[ExpectationsStore]:
    stores = []
    for path in sorted(LAB_DIR.glob("expectations_*.json")):
        hardware_key = path.stem.removeprefix("expectations_")
        # ExpectationsStore treats malformed JSON as an empty file, so parse strictly here.
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["hardware_key"] == hardware_key, path.name
        stores.append(ExpectationsStore(LAB_DIR, hardware_key))
    assert {store.hardware_key for store in stores} >= CHECKED_IN_HARDWARE_KEYS
    return stores


def test_sdpa_expectation_bar_requires_every_documented_run_to_be_attributable() -> None:
    example_name = _discovered_pair()[0]
    example_key = expectation_example_key(example_name, "python")
    configurations = _run_configurations()
    resolved = {}
    for description, (argv, must_apply) in configurations.items():
        baseline, optimized = _load_variants(argv, must_apply=must_apply)
        if not must_apply:
            # The harness fails a rejected probe, so it cannot write a bar.
            continue
        resolved[description] = (baseline.backend, optimized.backend)
        if argv is not None:
            _assert_documented_backends_reached(argv, resolved[description])
    # An expectation entry records neither the override argv nor the backend
    # each side ran, so a stored bar could come from any of these runs.
    unattributable = {
        description: pair for description, pair in resolved.items() if not _attributable(*pair)
    }

    for store in _expectation_stores():
        report = store.validate_all()
        assert not report.has_issues, (store.path.name, report.to_dict())
        if not unattributable:
            continue
        entry = store.get_entry(example_key)
        assert entry is None, (
            f"{store.path.name} stores a {entry.best_speedup:.4f}x bar, but these runs "
            f"cannot be attributed to a backend: {unattributable}"
        )
        # The lab has one pair, so any other key would also hold this pair's bar.
        assert store.data["examples"] == {}, store.path.name


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="SKIPPED: SDPA kernel dispatch can only be observed on a CUDA GPU",
)
def test_timed_body_dispatches_the_kernel_each_side_reports(record_property) -> None:
    # The attribution rule above trusts each side's backend string. Check that
    # the timed body dispatches exactly that kernel for every run it resolves.
    for description, (argv, must_apply) in _run_configurations().items():
        variants = _load_variants(argv, must_apply=must_apply)
        for role, benchmark in zip(("baseline", "optimized"), variants, strict=True):
            backend = benchmark.backend
            benchmark.setup()
            try:
                benchmark.benchmark_fn()
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as prof:
                    benchmark.benchmark_fn()
                    torch.cuda.synchronize()
            finally:
                benchmark.teardown()
            keys = {event.key for event in prof.key_averages()}
            dispatched = {name for name, op in _SDPA_OPS.items() if op in keys}
            record_property(f"{description}|{role}", f"{backend}->{','.join(sorted(dispatched))}")
            if backend == "auto":
                assert len(dispatched) == 1, (description, role, sorted(keys))
                assert dispatched <= {"cudnn", "flash", "efficient"}, (description, role, dispatched)
            else:
                assert dispatched == {backend}, (description, role, backend, sorted(keys))


def test_default_and_documented_runs_pin_two_different_backends() -> None:
    # Applies once the lab pins each side. Every run the README tells readers
    # to make must time two explicit, different backends.
    for description, (argv, must_apply) in _run_configurations().items():
        if not must_apply:
            continue
        baseline, optimized = _load_variants(argv)
        pair = (baseline.backend, optimized.backend)
        assert _attributable(*pair), (description, pair)
    # The harness gives both variants the same argv, so a shared --backend has
    # to be rejected by both hooks rather than applied or silently dropped.
    target = _target_label()
    for backend in _SHARED_BACKEND_PROBES:
        argv = ["--backend", backend]
        for benchmark in _load_variants(argv, must_apply=False):
            with pytest.raises(ValueError, match="--baseline-backend and --optimized-backend"):
                benchmark.apply_target_overrides(argv)
            assert getattr(benchmark, "_applied_target_override_signature", None) != (target, tuple(argv))
