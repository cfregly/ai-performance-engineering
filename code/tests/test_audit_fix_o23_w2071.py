"""Regressions for W2-071: the ch18 MoE decoder must run the graph mode it reports.

The CLI tests run the real subprocess harness on CPU. The helper forces the CPU
device and use_subprocess, and disables environment validation, profiling and
memory tracking, so the timed child can start on a host without a GPU. The
child still imports the module, applies the state the parent hands off, and
calls the real setup().
"""

from __future__ import annotations

import json

import pytest
import torch

import core.harness.benchmark_harness as harness_module
from ch17.dynamic_routing import Priority, Request
from ch18 import run_vllm_decoder
from ch18.run_vllm_decoder import GraphMode, VLLMMoEInferenceBenchmark

GRAPH_ENV_VARS = ("OPT_MOE_GRAPH_MODE", "OPT_MOE_ENABLE_GRAPHS", "OPT_MOE_MAX_CAPTURE_TOKENS")
# A tiny model keeps any setup that gets past the graph checks fast on CPU.
SMALL_MODEL_ENV = {
    "OPT_MOE_VOCAB": "64",
    "OPT_MOE_HIDDEN": "128",
    "OPT_MOE_FFN": "256",
    "OPT_MOE_LAYERS": "2",
    "OPT_MOE_MOE_LAYERS": "1",
    "OPT_MOE_EXPERTS": "4",
    "OPT_MOE_CONTEXT": "8",
    "OPT_MOE_DECODE": "2",
}


@pytest.fixture(autouse=True)
def _isolated_graph_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in GRAPH_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in SMALL_MODEL_ENV.items():
        monkeypatch.setenv(name, value)


def _use_cpu_subprocess_harness(monkeypatch: pytest.MonkeyPatch, *callers) -> None:
    """Patch the harness class, plus the name bound in each caller module."""
    real_harness = harness_module.BenchmarkHarness

    class CpuSubprocessHarness(real_harness):  # type: ignore[misc, valid-type]
        def __init__(self, *, mode, config) -> None:
            config.device = torch.device("cpu")
            config.use_subprocess = True
            config.enable_profiling = False
            config.enable_memory_tracking = False
            config.enforce_environment_validation = False
            super().__init__(mode=mode, config=config)

    monkeypatch.setattr(harness_module, "BenchmarkHarness", CpuSubprocessHarness)
    for caller in callers:
        monkeypatch.setattr(caller, "BenchmarkHarness", CpuSubprocessHarness)


def _run_cli(*args: str) -> None:
    run_vllm_decoder.main([*args, "--iterations", "1", "--warmup", "0"])


def test_default_graph_mode_is_eager_and_validates() -> None:
    bench = VLLMMoEInferenceBenchmark()

    assert bench.graph_mode is GraphMode.EAGER
    assert bench.enable_graphs is False
    assert bench.max_capture_tokens is None
    assert bench._validate_graph_request() is None


def test_max_capture_tokens_env_must_be_an_integer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPT_MOE_MAX_CAPTURE_TOKENS", "64k")
    with pytest.raises(ValueError, match="OPT_MOE_MAX_CAPTURE_TOKENS must be an integer, got '64k'"):
        VLLMMoEInferenceBenchmark()


def test_nonpositive_max_capture_tokens_fails_before_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPT_MOE_MAX_CAPTURE_TOKENS", "0")
    bench = VLLMMoEInferenceBenchmark()

    with pytest.raises(ValueError, match="max_capture_tokens must be positive, got 0"):
        bench.setup()
    assert bench.model is None


def test_graph_mode_parsing_rejects_unknown_values(monkeypatch: pytest.MonkeyPatch) -> None:
    assert GraphMode.from_str("Full-And-Piecewise") is GraphMode.FULL_AND_PIECEWISE
    with pytest.raises(ValueError, match="Unknown graph mode 'fulll'"):
        GraphMode.from_str("fulll")

    monkeypatch.setenv("OPT_MOE_GRAPH_MODE", "cuda_graphs")
    with pytest.raises(ValueError, match="Unknown graph mode 'cuda_graphs'"):
        VLLMMoEInferenceBenchmark()


def test_enable_graphs_env_accepts_only_zero_or_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPT_MOE_ENABLE_GRAPHS", "true")
    with pytest.raises(ValueError, match="OPT_MOE_ENABLE_GRAPHS must be 0 or 1"):
        VLLMMoEInferenceBenchmark()


@pytest.mark.parametrize(
    ("graph_mode", "enable_graphs"),
    [("full", "0"), ("piecewise", "0"), (None, "1")],
)
def test_conflicting_graph_request_fails_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
    graph_mode: str | None,
    enable_graphs: str,
) -> None:
    if graph_mode is not None:
        monkeypatch.setenv("OPT_MOE_GRAPH_MODE", graph_mode)
    monkeypatch.setenv("OPT_MOE_ENABLE_GRAPHS", enable_graphs)
    bench = VLLMMoEInferenceBenchmark()

    with pytest.raises(ValueError, match="conflicts with graph mode"):
        bench.setup()
    assert bench.model is None
    assert bench.paged_cache is None


@pytest.mark.parametrize("mode", ["full", "piecewise", "full_and_piecewise"])
def test_explicit_graph_mode_fails_before_allocation(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    monkeypatch.setenv("OPT_MOE_GRAPH_MODE", mode)
    bench = VLLMMoEInferenceBenchmark()

    with pytest.raises(RuntimeError, match=f"CUDA graph mode '{mode}' is unsupported"):
        bench.setup()
    assert bench.model is None
    assert bench.draft_model is None
    assert bench.paged_cache is None


def test_graph_mode_round_trips_through_subprocess_state() -> None:
    bench = VLLMMoEInferenceBenchmark()
    bench.graph_mode = json.loads(json.dumps(GraphMode.PIECEWISE))
    bench.enable_graphs = True

    with pytest.raises(RuntimeError, match="CUDA graph mode 'piecewise' is unsupported"):
        bench.setup()
    assert bench.graph_mode is GraphMode.PIECEWISE


def test_benchmark_fn_refuses_eager_timing_under_graph_label(monkeypatch: pytest.MonkeyPatch) -> None:
    bench = VLLMMoEInferenceBenchmark()
    cfg = bench.config
    bench.device = torch.device("cpu")
    bench.model = object()  # type: ignore[assignment]
    bench.prompts = torch.zeros((cfg.batch_size, cfg.context_window), dtype=torch.long)
    bench.paged_cache = object()  # type: ignore[assignment]
    bench.spec_decoder = object()  # type: ignore[assignment]
    bench._router_prefix_cache_lengths = [0] * cfg.batch_size
    bench._router_prefix_count = cfg.batch_size
    bench._router_prompt_stub = [0] * cfg.context_window
    bench._router_prompt_len = cfg.context_window
    bench._router_requests = [
        Request(
            id=f"req-0-{idx}",
            prompt_tokens=bench._router_prompt_stub,
            priority=Priority.STANDARD,
            timestamp=0.0,
            prefix_cached_length=0,
            expected_output_length=cfg.decode_tokens,
        )
        for idx in range(cfg.batch_size)
    ]
    bench._router_request_count = cfg.batch_size
    bench._router_batch_range = range(cfg.batch_size)
    bench.graph_mode = GraphMode.FULL
    bench.enable_graphs = True

    def eager_path_must_not_run() -> None:
        raise AssertionError("benchmark_fn timed the eager path under a graph-mode label")

    monkeypatch.setattr(bench, "_run_eager_path", eager_path_must_not_run)
    with pytest.raises(RuntimeError, match="Graph mode 'full' has no replayable CUDA graph"):
        bench.benchmark_fn()


def test_cli_graph_mode_reaches_child_and_failure_is_not_reported_as_latency(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _use_cpu_subprocess_harness(monkeypatch)

    with pytest.raises(RuntimeError) as excinfo:
        _run_cli("--graph-mode", "piecewise")

    message = str(excinfo.value)
    assert message.startswith("Optimized MoE inference benchmark failed:")
    assert "CUDA graph mode 'piecewise' is unsupported" in message
    assert "mean latency" not in capsys.readouterr().out


def test_cli_eager_and_spec_config_override_reach_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    _use_cpu_subprocess_harness(monkeypatch)
    monkeypatch.setenv("OPT_MOE_GRAPH_MODE", "full")
    missing = tmp_path / "missing_spec.json"

    with pytest.raises(RuntimeError) as excinfo:
        _run_cli("--graph-mode", "eager", "--spec-config", str(missing))

    message = str(excinfo.value)
    assert f"Speculator config not found: {missing}" in message
    assert "graph mode 'full'" not in message


@pytest.mark.parametrize("source", ["cli", "env"])
def test_capture_limit_under_eager_fails_in_child_instead_of_being_ignored(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    source: str,
) -> None:
    _use_cpu_subprocess_harness(monkeypatch)
    cli_args: tuple[str, ...] = ()
    if source == "cli":
        cli_args = ("--max-capture-tokens", "64")
    else:
        monkeypatch.setenv("OPT_MOE_MAX_CAPTURE_TOKENS", "64")

    with pytest.raises(RuntimeError) as excinfo:
        _run_cli(*cli_args)

    message = str(excinfo.value)
    assert message.startswith("Optimized MoE inference benchmark failed:")
    assert "max_capture_tokens=64 has no effect in graph mode 'eager'" in message
    assert "mean latency" not in capsys.readouterr().out


def test_spec_config_sweep_hands_each_config_path_to_the_timed_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    # The sweep tool must pass each config to the subprocess child as a str. A Path
    # attribute is dropped from the child's state, so the child would load the
    # default config. A missing path proves the hand-off: only a child that
    # received it can fail with its name. The failed run must then stop the sweep
    # instead of being recorded as zero metrics.
    from ch18.speculative_decode import spec_config_sweep

    _use_cpu_subprocess_harness(monkeypatch, spec_config_sweep)
    missing = tmp_path / "missing_spec.json"
    artifacts_dir = tmp_path / "artifacts"
    monkeypatch.setattr(
        "sys.argv",
        [
            "spec_config_sweep.py",
            "--configs",
            str(missing),
            "--iterations",
            "1",
            "--warmup",
            "0",
            "--artifacts-dir",
            str(artifacts_dir),
            "--run-id",
            "sweep_handoff",
        ],
    )

    with pytest.raises(RuntimeError) as excinfo:
        spec_config_sweep.main()

    message = str(excinfo.value)
    assert message.startswith(f"Speculator config sweep failed for {missing}")
    assert f"Speculator config not found: {missing}" in message
    assert not list(artifacts_dir.rglob("spec_config_sweep.json"))


def test_env_overrides_fail_on_malformed_values_instead_of_using_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from core.optimization.moe_inference import env_override_float, env_override_int

    monkeypatch.delenv("AISP_TEST_MOE_INT", raising=False)
    monkeypatch.delenv("AISP_TEST_MOE_FLOAT", raising=False)
    assert env_override_int("AISP_TEST_MOE_INT", 7) == 7
    assert env_override_float("AISP_TEST_MOE_FLOAT", 0.5) == 0.5

    monkeypatch.setenv("AISP_TEST_MOE_INT", "12")
    monkeypatch.setenv("AISP_TEST_MOE_FLOAT", "0.25")
    assert env_override_int("AISP_TEST_MOE_INT", 7) == 12
    assert env_override_float("AISP_TEST_MOE_FLOAT", 0.5) == 0.25

    for bad in ("12k", "", "1.5"):
        monkeypatch.setenv("AISP_TEST_MOE_INT", bad)
        with pytest.raises(ValueError, match=f"AISP_TEST_MOE_INT must be an integer, got {bad!r}"):
            env_override_int("AISP_TEST_MOE_INT", 7)
    for bad in ("fast", ""):
        monkeypatch.setenv("AISP_TEST_MOE_FLOAT", bad)
        with pytest.raises(ValueError, match=f"AISP_TEST_MOE_FLOAT must be a number, got {bad!r}"):
            env_override_float("AISP_TEST_MOE_FLOAT", 0.5)


@pytest.mark.parametrize(
    ("name", "value", "kind"),
    [("OPT_MOE_HIDDEN", "1k", "an integer"), ("OPT_MOE_ROUTER_NOISE", "low", "a number")],
)
def test_decoder_rejects_a_malformed_workload_override(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
    kind: str,
) -> None:
    # Before the fix, a typo silently built the default workload size.
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=f"{name} must be {kind}, got '{value}'"):
        VLLMMoEInferenceBenchmark()
