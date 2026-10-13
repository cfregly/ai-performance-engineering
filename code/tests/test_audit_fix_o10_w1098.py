"""Regressions for appendix NVLink bandwidth wording (W1-098) and the
Chapter 8 threshold pipeline references next to the cp.async fix (W1-097).

The appendix describes `code/ch08/threshold_tma_kernel.cuh`, which cannot be
compiled on a CPU-only host. These checks compare the documented facts with
the real kernel source and run the real benchmark gate the appendix cites.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
APPENDIX = REPO_ROOT / "docs" / "appendix.md"
CODE_ROOT = REPO_ROOT / "code"
THRESHOLD_KERNEL = CODE_ROOT / "ch08" / "threshold_tma_kernel.cuh"
THRESHOLD_COMMON = CODE_ROOT / "ch08" / "threshold_common.cuh"
MARKDOWN_LINK = r"\[[^\]]*\]\([^)]*\)"

# One unique phrase from each appendix paragraph this unit corrected.
CORRECTED_PARAGRAPH_MARKERS = (
    "instead of PCIe-only connectivity",
    "Scaling pays only when communication is fast",
    "Always colocate multi-GPU jobs",
    "130 TB/s",
    "consider fine-grained model parallelism",
    "At cluster scale, the network",
    "Ampere (compute capability 8.0)",
    "pick the pipeline scope",
    "caps the grid",
    "Gate each async-copy feature",
    "highlights the divergence you are fixing",
    "Keep every harness knob",
    "Wrap NVTX ranges behind a helper",
    "ValuesPerThread",
)


def _appendix_text() -> str:
    return APPENDIX.read_text(encoding="utf-8")


def _kernel_text() -> str:
    return THRESHOLD_KERNEL.read_text(encoding="utf-8")


def _paragraph(marker: str) -> str:
    matches = [line for line in _appendix_text().splitlines() if marker in line]
    assert len(matches) == 1, (
        f"expected one appendix paragraph containing {marker!r}, found {len(matches)}"
    )
    return matches[0]


def _prose(paragraph: str) -> str:
    """Drop inline code spans and link targets so only prose punctuation remains."""
    without_code = re.sub(r"`[^`]*`", "", paragraph)
    return re.sub(r"\]\([^)]*\)", "]", without_code)


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text) if s]


def _int_constant(source: str, name: str) -> int:
    match = re.search(rf"constexpr int {name} = (\d+);", source)
    assert match, f"{name} not found"
    return int(match.group(1))


def test_nvlink_vs_pcie_comparison_is_repaired() -> None:
    paragraph = _paragraph("instead of PCIe-only connectivity")

    assert '14"' not in paragraph
    assert "GPU-to-GPU bandwidth" not in paragraph
    assert "per GPU" in paragraph
    assert "PCIe Gen5 x16 link" in paragraph

    nvlink = re.search(r"up to ([\d.]+) TB/s of bidirectional bandwidth per GPU", paragraph)
    pcie = re.search(r"roughly (\d+) GB/s bidirectional bandwidth of a PCIe Gen5 x16", paragraph)
    ratio = re.search(r"more than (\d+) times", paragraph)
    assert nvlink and pcie and ratio, paragraph
    # PCIe Gen5 is 32 GT/s per lane, so x16 carries 64 GB/s per direction.
    assert int(pcie.group(1)) == 2 * 32 * 16 // 8
    assert float(nvlink.group(1)) * 1000 / int(pcie.group(1)) > int(ratio.group(1))


def test_nvl72_aggregate_is_the_per_gpu_sum_not_bisection() -> None:
    paragraph = _paragraph("130 TB/s")

    per_direction = re.search(r"(\d+) GB/s per GPU per direction \(([\d.]+) TB/s bidirectional\)", paragraph)
    assert per_direction, paragraph
    assert 2 * int(per_direction.group(1)) == pytest.approx(float(per_direction.group(2)) * 1000)

    total = re.search(r"(\d+) GPUs × ([\d.]+) TB/s = ([\d.]+) TB/s", paragraph)
    assert total, paragraph
    gpus, per_gpu, aggregate = int(total.group(1)), float(total.group(2)), float(total.group(3))
    assert gpus == 72
    assert per_gpu == float(per_direction.group(2))
    assert gpus * per_gpu == pytest.approx(aggregate)
    assert round(aggregate) == 130

    assert "does not identify it as bisection bandwidth" in paragraph
    assert "total bisection bandwidth" not in paragraph


def test_per_gpu_bandwidth_is_not_described_as_a_per_peer_rate() -> None:
    for sentence in _sentences(_appendix_text()):
        assert not ("TB/s" in sentence and "to every other" in sentence), sentence

    paragraph = _paragraph("consider fine-grained model parallelism")
    assert "total across all peers, not a rate to each peer" in paragraph
    assert "without hitting network limits" not in paragraph


def test_nvlink_domains_do_not_promise_near_linear_scaling() -> None:
    claims = [s for s in _sentences(_appendix_text()) if re.search(r"near[- ]linear", s)]
    assert claims
    for sentence in claims:
        assert "measure scaling efficiency" in sentence, sentence

    # Linear scaling is the ceiling, not the default that the network then breaks.
    assert "break linear scaling" not in _appendix_text()
    assert "the network can push scaling efficiency far below linear" in _paragraph("At cluster scale, the network")


def test_external_citations_close_their_sentence() -> None:
    for line in _appendix_text().splitlines():
        # A citation link followed directly by a capitalized word is a missing period.
        assert not re.search(r"\]\(https?://[^)]*\)\s+[A-Z]", line), line

    for marker in CORRECTED_PARAGRAPH_MARKERS:
        # A later sentence made only of citation links has lost the sentence it supports.
        for sentence in _sentences(re.sub(r"`[^`]*`", "", _paragraph(marker)))[1:]:
            if "](http" in sentence:
                assert re.sub(MARKDOWN_LINK, "", sentence).strip(" ,.()"), sentence

    for marker in ("instead of PCIe-only connectivity", "130 TB/s"):
        paragraph = _paragraph(marker)
        assert "([NVIDIA NVLink specifications](https://www.nvidia.com/en-us/data-center/nvlink/))." in paragraph

    paragraph = _paragraph("Ampere (compute capability 8.0)")
    assert "`cp.async` ([NVIDIA Ampere tuning guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html))." in paragraph
    assert (
        "further capabilities on Blackwell "
        "([NVIDIA Hopper tuning guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html))."
    ) in paragraph


@pytest.mark.parametrize("marker", CORRECTED_PARAGRAPH_MARKERS)
def test_corrected_paragraphs_have_no_prose_semicolons(marker: str) -> None:
    assert ";" not in _prose(_paragraph(marker))


def test_repo_file_references_resolve_without_line_ranges() -> None:
    text = _appendix_text()
    assert "ch8/threshold_tma_kernel.cuh" not in text
    assert "common/python/" not in text
    assert "`ch6/" not in text
    assert not re.search(r"\(lines? \d", text)

    linked = re.findall(r"\[`([^`]+)`\]\((\.\./code/[^)\s]+)\)", text)
    assert linked
    for label, target in linked:
        assert (APPENDIX.parent / target).resolve() == (REPO_ROOT / label).resolve(), (label, target)
        assert (REPO_ROOT / label).is_file(), label

    for target in re.findall(r"\]\((\.\./code/[^)\s]+)\)", text):
        assert (APPENDIX.parent / target).resolve().is_file(), target

    # Every backticked source path must be a link, so readers can follow it.
    linked_labels = {label for label, _ in linked}
    for path in re.findall(r"`([\w./-]+\.(?:py|cu|cuh))`", text):
        assert path in linked_labels, path


def test_pipeline_scope_description_matches_kernel() -> None:
    paragraph = _paragraph("pick the pipeline scope")
    source = _kernel_text()

    assert "threshold_tma_pipeline_kernel` in [" in paragraph
    assert "uses the thread-scoped form" in paragraph
    assert "auto pipe = cuda::make_pipeline();" in source
    assert "pipeline_shared_state" not in source
    assert "__global__ void threshold_tma_pipeline_kernel(" in source
    assert "bool stage_ready[stages]" in source
    for call in ("producer_acquire", "producer_commit", "consumer_wait", "consumer_release"):
        assert f"`{call}`" in paragraph
        assert f"pipe.{call}()" in source
    assert "cuda::memcpy_async(" in source
    assert "not bulk tensor TMA" in paragraph
    assert "cp.async.bulk" not in source


def test_grid_cap_description_matches_launcher() -> None:
    paragraph = _paragraph("caps the grid")
    source = _kernel_text()

    cap = re.search(r"constexpr int kMaxPipelineBlocks = (\d+);", source)
    assert cap, "launcher grid cap not found"
    assert f"caps the grid at a fixed {int(cap.group(1)):,} blocks" in paragraph
    assert "if (grid_x < 1) {\n        grid_x = 1;" in source
    assert "const size_t shared_bytes = stages * tile_span * sizeof(float);" in source
    assert "const int tile_span = block.x * values_per_thread;" in source
    assert "`stages × tile_span × sizeof(float)`" in paragraph
    assert "fits in SRAM" not in paragraph
    assert "inline cudaError_t launch_threshold_tma_pipeline_variant(" in source


def test_shared_memory_check_names_the_per_block_launch_limit() -> None:
    paragraph = _paragraph("caps the grid")
    source = _kernel_text()

    assert "check it against the per-SM shared-memory budget" not in paragraph
    assert "more than 48 KB of dynamic shared memory per block fails" in paragraph
    assert "`cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes)`" in paragraph
    assert "`cudaDevAttrMaxSharedMemoryPerBlockOptin`" in paragraph

    # The paragraph says this launcher never opts in, so its largest staging must fit the default.
    assert "This launcher makes no such call." in paragraph
    assert "cudaFuncSetAttribute" not in source
    threads = _int_constant(THRESHOLD_COMMON.read_text(encoding="utf-8"), "kThresholdOptimizedThreads")
    assert "const dim3 block(kThresholdOptimizedThreads);" in source
    stages = _int_constant(source, "stages")
    candidates = re.search(r"constexpr int candidates\[\] = \{([\d, ]+)\};", source)
    assert candidates, "autotune candidate list not found"
    max_values = max(int(value) for value in candidates.group(1).split(","))
    float_bytes = 4
    largest = threads * max_values * stages * float_bytes
    assert largest <= 48 * 1024
    assert (
        f"{threads} threads × {max_values} values × {stages} stages × {float_bytes} bytes = "
        f"{largest // 1024} KB per block"
    ) in paragraph
    assert "per-SM shared-memory budget to see how many CTAs can be resident" in paragraph


def test_guard_description_matches_kernel_guards() -> None:
    paragraph = _paragraph("Gate each async-copy feature")
    source = _kernel_text()

    assert "fall back cleanly" not in paragraph
    for guard in re.findall(r"`(#if [^`]+)`", paragraph):
        assert guard in source, guard
    assert "`#if CUDART_VERSION >= 12000`" in paragraph
    assert "return cudaErrorNotSupported;" in source

    arch_guard = "#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900"
    assert f"`{arch_guard}`" in paragraph
    pre_hopper_branch = re.search(
        re.escape(arch_guard) + r".*?\n#else\n(.*?)\n#endif", source, re.DOTALL
    )
    assert pre_hopper_branch, "kernel arch guard has no #else branch"
    body = [line.strip() for line in pre_hopper_branch.group(1).splitlines() if line.strip()]
    assert body and all(re.fullmatch(r"\(void\)\w+;", line) for line in body), body
    assert "the kernel body there is empty" in paragraph


def test_cp_async_only_kernels_are_gated_at_compute_capability_8() -> None:
    paragraph = _paragraph("Gate each async-copy feature")
    source = _kernel_text()

    # The old opening presented this cp.async-only kernel as the TMA gating model.
    assert not paragraph.startswith("TMA and warp-specialized features")
    assert "Hardware `cp.async` copies arrived with compute capability 8.0 (Ampere)." in paragraph
    assert "`__CUDA_ARCH__ >= 800` guard and a host check for compute capability 8.0 are enough" in paragraph
    assert "need compute capability 9.0 (Hopper) or newer" in paragraph
    assert "its gates are stricter than its copies need" in paragraph
    assert "the kernel's only async operation is `cuda::memcpy_async`" in paragraph

    # The kernel issues only cuda::memcpy_async through a thread-scoped pipeline,
    # so nothing in it needs more than compute capability 8.0.
    kernel_body = source.split("__global__ void threshold_tma_pipeline_kernel(", 1)[1]
    kernel_body = kernel_body.split("#endif  // CUDART_VERSION >= 12000", 1)[0]
    assert "cuda::memcpy_async(" in kernel_body
    for tma_or_hopper_only in (
        "cp.async.bulk",
        "memcpy_async_tx",
        "cuda::barrier",
        "CUtensorMap",
        "tensormap",
        "wgmma",
        "setmaxnreg",
        "asm(",
        "asm volatile",
    ):
        assert tma_or_hopper_only not in kernel_body, tma_or_hopper_only


def test_thresholdtma_benchmark_reports_skipped_on_unsupported_device() -> None:
    from ch08.optimized_thresholdtma import get_benchmark
    from core.harness.cuda_capabilities import blackwell_tma_support_status

    paragraph = _paragraph("Gate each async-copy feature")
    assert "report `SKIPPED:` on any GPU that is not Blackwell" in paragraph
    assert "not because the copies need Blackwell" in paragraph

    supported, reason = blackwell_tma_support_status()
    if supported:
        pytest.skip(f"SKIPPED: host has a supported Blackwell GPU ({reason}), so the unsupported-device gate cannot fire")

    with pytest.raises(RuntimeError, match=r"^SKIPPED: .*requires Blackwell"):
        get_benchmark().setup()


def test_autotune_description_matches_launcher() -> None:
    paragraph = _paragraph("ValuesPerThread")
    source = _kernel_text()

    candidates = re.search(r"constexpr int candidates\[\] = \{([\d, ]+)\};", source)
    assert candidates, "autotune candidate list not found"
    values = [value.strip() for value in candidates.group(1).split(",")]
    assert f"({', '.join(values[:-1])}, and {values[-1]})" in paragraph

    assert "inline cudaError_t launch_threshold_tma_pipeline(" in source
    assert "`launch_threshold_tma_pipeline` in [" in paragraph
    for slot in ("static int cached_device", "static long long cached_count", "static int cached_variant"):
        assert slot in source
    assert "caches a single choice keyed by `(device, count)`" in paragraph
    assert source.count("cudaEventRecord(start, stream);") == 1
    assert "times one launch of each" in paragraph


class _StoppedAtComplianceGateError(Exception):
    """Raised at the compliance gate, the first step after the harness config merge."""


RECORDED_FIELDS = ("iterations", "warmup", "adaptive_iterations", "seed", "deterministic")


def _harness_run_config(monkeypatch: pytest.MonkeyPatch, harness_config: object, benchmark: object) -> dict:
    """Run the real BenchmarkHarness.benchmark config merge and stop before setup.

    Returns the fields in RECORDED_FIELDS from the config the harness would run with.
    """
    import core.benchmark.quarantine as quarantine
    from core.harness import benchmark_harness

    # The harness deep-copies its config before merging, so the copy keeps this
    # marker. The merge skips private keys, so the benchmark cannot overwrite it.
    harness_config._doc_probe = True
    seen: list[dict] = []
    real_sync = benchmark_harness.BenchmarkConfig._sync_execution_mode

    def recording_sync(self: object) -> None:
        if getattr(self, "_doc_probe", False):
            seen.append({name: getattr(self, name) for name in RECORDED_FIELDS})
        real_sync(self)

    def stop_at_compliance_gate(_benchmark: object) -> None:
        raise _StoppedAtComplianceGateError

    with monkeypatch.context() as patch:
        patch.setattr(benchmark_harness.BenchmarkConfig, "_sync_execution_mode", recording_sync)
        patch.setattr(quarantine, "detect_skip_flags", stop_at_compliance_gate)
        harness = benchmark_harness.BenchmarkHarness(
            mode=benchmark_harness.BenchmarkMode.CUSTOM, config=harness_config
        )
        with pytest.raises(_StoppedAtComplianceGateError):
            harness.benchmark(benchmark)
    # The last marked sync is the one BenchmarkHarness.benchmark runs on its merged config.
    assert seen, "BenchmarkHarness.benchmark never synced the merged config"
    return seen[-1]


def _runner_merged_config(
    benchmark: object,
    *,
    iterations: int | None = None,
    warmup: int | None = None,
    reproducible: bool = False,
    **run_settings: object,
) -> tuple[object, object]:
    """Build the run config the way _test_chapter_impl does and merge it with the real runner code.

    `iterations` and `warmup` stand for the CLI values. None means the CLI omitted them.
    Returns (run-level base config, config passed to BenchmarkHarness).
    """
    from core.benchmark.defaults import get_defaults
    from core.harness import run_benchmarks
    from core.harness.benchmark_harness import BenchmarkConfig

    base = BenchmarkConfig(
        iterations=20 if iterations is None else iterations,
        warmup=5 if warmup is None else warmup,
        enable_memory_tracking=True,
        seed=42 if reproducible else None,
        deterministic=reproducible,
        **run_settings,
    )
    locked = run_benchmarks._compute_locked_fields(
        base_config=base,
        cli_iterations_provided=iterations is not None,
        cli_warmup_provided=warmup is not None,
        enable_profiling=False,
        reproducible=reproducible,
    )
    merged = run_benchmarks._merge_benchmark_config(
        base_config=base,
        benchmark_obj=benchmark,
        defaults_obj=get_defaults(),
        locked_fields=locked,
    )
    return base, merged


def _bench_run_options() -> dict:
    """Return the `aisp bench run` typer options (default and help text), keyed by parameter name."""
    import inspect

    from core.benchmark import bench_commands

    commands = [c for c in bench_commands.app.registered_commands if c.callback.__name__ == "run"]
    assert len(commands) == 1, commands
    parameters = inspect.signature(commands[0].callback).parameters
    return {name: parameters[name].default for name in ("iterations", "warmup", "reproducible")}


def test_runner_base_config_matches_test_chapter_impl() -> None:
    """_runner_merged_config copies these settings from _test_chapter_impl. Keep them in step."""
    import inspect

    from core.harness import run_benchmarks

    source = inspect.getsource(run_benchmarks._test_chapter_impl)
    for setting in (
        "cli_iterations_provided = iterations is not None",
        "cli_warmup_provided = warmup is not None",
        "iterations = 20",
        "warmup = 5",
        "enable_memory_tracking=True",
        "seed=42 if reproducible else None",
        "deterministic=reproducible",
        "base_config = BenchmarkConfig(**config_kwargs)",
        "locked_fields = _compute_locked_fields(",
        "reproducible=reproducible,",
        "merged = _merge_benchmark_config(",
        "BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=merged)",
    ):
        assert setting in source, setting

    options = _bench_run_options()
    assert options["iterations"].default is None
    assert options["warmup"].default is None
    assert options["reproducible"].default is False


def test_runner_applies_warp_divergence_iteration_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    from ch06 import baseline_warp_divergence_ilp, optimized_warp_divergence_ilp

    paragraph = _paragraph("highlights the divergence you are fixing")
    # Earlier wording said the harness always replaced these counts, which is false under `aisp bench run`.
    assert "the harness supplies those counts itself" not in paragraph
    assert "uses the run-level counts instead" not in paragraph
    assert "Both files return the same `BenchmarkConfig` from `get_config`." in paragraph
    asked = re.search(
        r"It asks for (\d+) timed and (\d+) warmup iterations, taken from `workload\.ilp_iterations` "
        r"and `workload\.ilp_warmup`, and turns off adaptive iterations\.",
        paragraph,
    )
    assert asked, paragraph
    documented = (int(asked.group(1)), int(asked.group(2)))
    assert (
        "Under `aisp bench run`, `_merge_benchmark_config` in "
        "[`code/core/harness/run_benchmarks.py`](../code/core/harness/run_benchmarks.py) "
        "applies those counts to the run."
    ) in paragraph
    assert "Passing `--iterations` or `--warmup` locks that count at the value you give instead." in paragraph
    assert (
        "On either path both variants run the same warmup and timed iteration counts, "
        "and adaptive iterations stay off."
    ) in paragraph

    cli_iterations, cli_warmup = 7, 6
    own_configs = []
    for module in (baseline_warp_divergence_ilp, optimized_warp_divergence_ilp):
        benchmark = module.get_benchmark()
        own = benchmark.get_config()
        own_configs.append((own.iterations, own.warmup, own.adaptive_iterations))
        assert (own.iterations, own.warmup) == documented
        assert (benchmark.workload.ilp_iterations, benchmark.workload.ilp_warmup) == documented
        assert own.adaptive_iterations is False

        # No CLI counts: the pair's own counts reach the harness and survive its merge.
        base, merged = _runner_merged_config(benchmark)
        # The check is only meaningful if the run-level counts differ from the pair's.
        assert (base.iterations, base.warmup) != documented
        final = _harness_run_config(monkeypatch, merged, benchmark)
        assert (final["iterations"], final["warmup"]) == documented, (module.__name__, final)
        assert final["adaptive_iterations"] is False, (module.__name__, final)

        # Both CLI counts: the run-level values are locked.
        _, merged = _runner_merged_config(benchmark, iterations=cli_iterations, warmup=cli_warmup)
        final = _harness_run_config(monkeypatch, merged, benchmark)
        assert (final["iterations"], final["warmup"]) == (cli_iterations, cli_warmup), (module.__name__, final)
        assert final["adaptive_iterations"] is False, (module.__name__, final)

        # One CLI count locks only that count.
        _, merged = _runner_merged_config(benchmark, iterations=cli_iterations)
        final = _harness_run_config(monkeypatch, merged, benchmark)
        assert (final["iterations"], final["warmup"]) == (cli_iterations, documented[1]), (module.__name__, final)

    assert own_configs[0] == own_configs[1]


def test_direct_harness_call_keeps_harness_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    from ch06 import baseline_warp_divergence_ilp, optimized_warp_divergence_ilp
    from core.harness.benchmark_harness import BenchmarkConfig, _is_chapter_or_labs_benchmark

    paragraph = _paragraph("highlights the divergence you are fixing")
    assert (
        "A direct call to `BenchmarkHarness.benchmark` in "
        "[`code/core/harness/benchmark_harness.py`](../code/core/harness/benchmark_harness.py) "
        "works differently."
    ) in paragraph
    assert (
        "It ignores a chapter or lab benchmark's own counts and keeps the counts in the harness config."
    ) in paragraph

    for module in (baseline_warp_divergence_ilp, optimized_warp_divergence_ilp):
        benchmark = module.get_benchmark()
        assert _is_chapter_or_labs_benchmark(benchmark)
        own = benchmark.get_config()
        harness_config = BenchmarkConfig()
        harness_counts = (harness_config.iterations, harness_config.warmup)
        assert (own.iterations, own.warmup) != harness_counts
        final = _harness_run_config(monkeypatch, harness_config, benchmark)
        assert (final["iterations"], final["warmup"]) == harness_counts, (module.__name__, final)
        assert final["adaptive_iterations"] is False, (module.__name__, final)


def test_warp_divergence_example_matches_the_ch06_pair() -> None:
    import inspect

    from ch06 import baseline_warp_divergence_ilp, optimized_warp_divergence_ilp

    paragraph = _paragraph("highlights the divergence you are fixing")
    assert "torch.cuda.synchronize()" not in paragraph

    baseline_loop = inspect.getsource(baseline_warp_divergence_ilp.BaselineWarpDivergenceILPBenchmark.benchmark_fn)
    assert "self._nvtx_range(" in baseline_loop
    assert "synchronize" not in baseline_loop
    for step in ("mask = activations > 0.5", "positive = result[mask]", "result[~mask] = negative"):
        assert step in baseline_loop, step
    assert "splits the elements with a boolean mask" in paragraph

    optimized_kernel = inspect.getsource(optimized_warp_divergence_ilp._fused_branchless_kernel)
    assert "torch.where(mask, positive, negative, out=result)" in optimized_kernel
    assert "computes both branches for every element and selects the result with `torch.where`" in paragraph


def test_reproducible_flag_description_matches_runner_merge(monkeypatch: pytest.MonkeyPatch) -> None:
    from ch06 import baseline_warp_divergence_ilp, optimized_warp_divergence_ilp
    from core.benchmark.defaults import get_defaults
    from core.harness.benchmark_harness import BenchmarkConfig

    paragraph = _paragraph("Keep every harness knob")
    assert "takes its defaults from `BenchmarkDefaults`" in paragraph
    assert BenchmarkConfig().iterations == get_defaults().iterations
    assert BenchmarkConfig().deterministic == get_defaults().deterministic

    assert "turns on `deterministic` for that run" not in paragraph
    assert (
        "Under `aisp bench run`, `_merge_benchmark_config` in "
        "[`code/core/harness/run_benchmarks.py`](../code/core/harness/run_benchmarks.py) "
        "copies the benchmark's `get_config` values over the run config."
    ) in paragraph
    assert (
        "Locked fields, such as counts passed with `--iterations` or `--warmup`, and a few run-level "
        "policy settings, such as the validity profile, keep the run's values."
    ) in paragraph
    assert "`deterministic` and `seed` are not among them." not in paragraph
    assert "carries every default, including `deterministic=False`" in paragraph
    assert get_defaults().deterministic is False
    assert (
        "`--reproducible` asks for seed 42 and deterministic algorithms, and it locks `seed` and "
        "`deterministic`, so those defaults cannot undo it."
    ) in paragraph
    assert (
        "The ch06 warp-divergence pair above then runs with seed 42 and deterministic algorithms on."
    ) in paragraph
    assert (
        "Without the flag both fields come from the benchmark's config, so the pair runs with "
        "deterministic algorithms off."
    ) in paragraph
    assert "Its seed is 42 either way, because 42 is also the `BenchmarkDefaults` seed." in paragraph
    assert get_defaults().seed == 42
    reproducible_help = _bench_run_options()["reproducible"].help
    assert "seeds to 42" in reproducible_help and "deterministic algorithms" in reproducible_help

    for module in (baseline_warp_divergence_ilp, optimized_warp_divergence_ilp):
        benchmark = module.get_benchmark()

        base, merged = _runner_merged_config(benchmark, reproducible=True, validity_profile="portable")
        assert (base.seed, base.deterministic) == (42, True)
        # The validity profile is one of the run-level policy settings the runner keeps.
        assert merged.validity_profile == "portable"
        # --reproducible locks both fields, so the pair's own config cannot reset them.
        assert (merged.seed, merged.deterministic) == (42, True)
        final = _harness_run_config(monkeypatch, merged, benchmark)
        assert (final["seed"], final["deterministic"]) == (42, True), (module.__name__, final)

        base, merged = _runner_merged_config(benchmark, reproducible=False)
        assert (base.seed, base.deterministic) == (None, False)
        final = _harness_run_config(monkeypatch, merged, benchmark)
        assert (final["seed"], final["deterministic"]) == (42, False), (module.__name__, final)


def test_nvtx_helper_description_matches_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    from ch06.baseline_warp_divergence_ilp import get_benchmark
    from core.harness.benchmark_harness import BenchmarkConfig
    from core.profiling import nvtx_helper

    paragraph = _paragraph("Wrap NVTX ranges behind a helper")
    assert "does nothing when NVTX is disabled or CUDA is unavailable" in paragraph
    assert "calls `get_nvtx_enabled` once for each config object the harness assigns" in paragraph

    entered = []
    with nvtx_helper.nvtx_range("doc_check", enable=False):
        entered.append(True)
    assert entered == [True]

    calls = []
    real_get_nvtx_enabled = nvtx_helper.get_nvtx_enabled

    def counting_get_nvtx_enabled(config: object) -> bool:
        calls.append(config)
        return real_get_nvtx_enabled(config)

    monkeypatch.setattr(nvtx_helper, "get_nvtx_enabled", counting_get_nvtx_enabled)
    benchmark = get_benchmark()
    first = BenchmarkConfig(enable_nvtx=False)
    benchmark._config = first
    for _ in range(3):
        with benchmark._nvtx_range("doc_check"):
            pass
    assert calls == [first]

    second = BenchmarkConfig(enable_nvtx=False)
    benchmark._config = second
    with benchmark._nvtx_range("doc_check"):
        pass
    assert calls == [first, second]
