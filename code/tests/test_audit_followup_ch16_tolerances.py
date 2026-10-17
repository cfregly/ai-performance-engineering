"""W2-059 follow-up: more Chapter 16 pairs verify under bounds sized to their outputs.

The runtime_scheduler, piece_graphs and regional_compilation pairs declared
output_tolerance=(0.1, 1.0) on both sides. That bound accepted any output scaled
by 0.95. It also accepted a 0.9x output from the piece_graphs and
regional_compilation pairs, whose values stay below 7. Each pair now declares one
constant from its baseline module, sized from the output scale and the measured
baseline-versus-optimized difference. The dense-attention family keeps its shared
constant, now tightened from (5e-2, 5e-2), which accepted an output whose second
half was zero.

Device coverage:
- runtime_scheduler runs on the CPU and on CUDA. On the CPU the only seam is a
  no-op torch.cuda.synchronize, and each scenario runs one decode step instead of
  sixteen, because CPU bf16 GEMMs here run at about 3 GFLOPS. Every step writes
  the same product into the same buffer, so the captured output is the same.
- piece_graphs and regional_compilation replay CUDA graphs. Their pair tests run
  only on a CUDA device and patch nothing. On the CPU these tests run the shared
  model code at the benchmark's sizes. The regional baseline also runs its real
  setup() and benchmark_fn() at the smallest MODEL_CANDIDATES config.
- The dense test runs the optimized file's production _forward_flash on the CPU
  at the benchmark's sequence length with batch 1. It skips the CUDA-only
  constructor, as tests/test_ch16_dense_attention_regressions.py does.

The jitter tests run the harness's own VerifyRunner._run_live_jitter_check. It
perturbs the declared input in place and reruns benchmark_fn, so a declared copy
of the input leaves the output unchanged and fails. verify_pair runs it only on
the optimized side. On the CPU the dense files run at batch 1 and sequence length
256 with three seams: a no-op CUDA sync, a CPU device for the optimized
constructor, and CPU tensors for the Blackwell variant's hardcoded CUDA setup.
CUDA runs patch nothing.
"""

from __future__ import annotations

import ast
import copy
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.overrides import TorchFunctionMode

from ch16 import (
    baseline_piece_graphs,
    baseline_regional_compilation,
    baseline_runtime_scheduler,
    optimized_dense_attention_flash,
    optimized_piece_graphs,
    optimized_regional_compilation,
    optimized_runtime_scheduler,
)
from ch16.baseline_dense_attention_flash import BaselineDenseAttentionFlashBenchmark
from ch16.baseline_piece_graphs import PIECE_GRAPHS_OUTPUT_TOLERANCE, BaselinePieceGraphsBenchmark
from ch16.baseline_regional_compilation import (
    MODEL_CANDIDATES,
    REGIONAL_COMPILATION_OUTPUT_TOLERANCE,
    BaselineRegionalCompilationBenchmark,
)
from ch16.baseline_runtime_scheduler import (
    RUNTIME_SCHEDULER_OUTPUT_TOLERANCE,
    BaselineRuntimeSchedulerBenchmark,
)
from ch16.dense_attention_accuracy import DENSE_ATTENTION_OUTPUT_TOLERANCE
from ch16.optimized_dense_attention_flash import OptimizedDenseAttentionFlashBenchmark
from ch16.optimized_dense_attention_flash_blackwell_variant import (
    DenseAttentionFlashBlackwellVariantBenchmark,
)
from ch16.optimized_piece_graphs import OptimizedPieceGraphsBenchmark
from ch16.optimized_regional_compilation import (
    OptimizedRegionalCompilationBenchmark,
    RegionalCompilationTransformer,
)
from ch16.optimized_runtime_scheduler import OptimizedRuntimeSchedulerBenchmark
from ch16.piece_graphs_model import RegionalPieceGraph
from core.benchmark import verified_comparison
from core.benchmark.quarantine import QuarantineManager
from core.benchmark.verification import set_deterministic_seeds
from core.benchmark.verify_runner import VerifyConfig, VerifyRunner

PRE_FIX_TOLERANCE = (0.1, 1.0)
PREVIOUS_DENSE_TOLERANCE = (5e-2, 5e-2)
HARNESS_SEED = VerifyConfig().seed
SHARED_TOLERANCES = (
    (baseline_runtime_scheduler, optimized_runtime_scheduler, "RUNTIME_SCHEDULER_OUTPUT_TOLERANCE"),
    (baseline_piece_graphs, optimized_piece_graphs, "PIECE_GRAPHS_OUTPUT_TOLERANCE"),
    (
        baseline_regional_compilation,
        optimized_regional_compilation,
        "REGIONAL_COMPILATION_OUTPUT_TOLERANCE",
    ),
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a CUDA device for the pair's real CUDA graph and cuBLAS paths",
)

# Each dense file names its sequence length differently. CPU jitter runs shorten it.
DENSE_SEQUENCE_ATTRIBUTE = {
    BaselineDenseAttentionFlashBenchmark: "max_seq_len",
    OptimizedDenseAttentionFlashBenchmark: "max_seq_len",
    DenseAttentionFlashBlackwellVariantBenchmark: "seq_length",
}


class _CudaFactoryToCpu(TorchFunctionMode):
    """Create tensors that name a CUDA device on the CPU instead."""

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = dict(kwargs or {})
        device = kwargs.get("device")
        if device is not None and torch.device(device).type == "cuda":
            kwargs["device"] = "cpu"
        return func(*args, **kwargs)


@pytest.fixture
def restore_cudnn_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Restore the process-wide flags that set_deterministic_seeds() changes."""
    monkeypatch.setattr(torch.backends.cudnn, "deterministic", torch.backends.cudnn.deterministic)
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", torch.backends.cudnn.benchmark)


@pytest.fixture
def restore_torch_threads():
    """runtime_scheduler setup() calls torch.set_num_threads(1) for the whole process."""
    threads = torch.get_num_threads()
    yield
    torch.set_num_threads(threads)


@pytest.fixture
def runner(tmp_path: Path) -> VerifyRunner:
    return VerifyRunner(
        cache_dir=tmp_path / "golden",
        quarantine_manager=QuarantineManager(cache_dir=tmp_path),
    )


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=requires_cuda)])
def scheduler_device(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    restore_cudnn_flags: None,
    restore_torch_threads: None,
) -> torch.device:
    """CPU runs skip benchmark_fn's per-step CUDA sync. CUDA runs patch nothing."""
    if request.param == "cpu":
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    return torch.device(request.param)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=requires_cuda)])
def dense_device(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    restore_cudnn_flags: None,
) -> torch.device:
    """CPU runs patch three CUDA-only seams. CUDA runs patch nothing.

    The baseline's setup ends with a CUDA sync. The optimized constructor resolves
    a CUDA device. The Blackwell variant's setup builds every tensor on a hardcoded
    CUDA device, so on the CPU those tensors are created on the CPU, during setup only.
    """
    if request.param == "cpu":
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            optimized_dense_attention_flash, "resolve_device", lambda: torch.device("cpu")
        )
        variant_setup = DenseAttentionFlashBlackwellVariantBenchmark.setup

        def variant_setup_on_cpu(self) -> None:
            with _CudaFactoryToCpu():
                variant_setup(self)

        monkeypatch.setattr(DenseAttentionFlashBlackwellVariantBenchmark, "setup", variant_setup_on_cpu)
    return torch.device(request.param)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=requires_cuda)])
def regional_device(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    restore_cudnn_flags: None,
) -> torch.device:
    """CPU runs skip setup's CUDA sync. CUDA runs patch nothing."""
    if request.param == "cpu":
        monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    return torch.device(request.param)


def _smallest_regional_choice() -> dict:
    """CPU runs use the smallest MODEL_CANDIDATES config, because CPU bf16 work is slow."""
    return min(MODEL_CANDIDATES, key=lambda candidate: candidate["n_layers"] * candidate["d_model"])


def _captured(bench):
    """Seed as the harness does, then run the benchmark's real setup and capture."""
    set_deterministic_seeds(HARNESS_SEED)
    bench.setup()
    bench.benchmark_fn()
    bench.capture_verification_payload()
    return bench


def _scheduler(benchmark_cls, device: torch.device):
    bench = benchmark_cls()
    bench.device = device
    if device.type == "cpu":
        bench.scenarios = tuple(replace(scenario, decode_steps=1) for scenario in bench.scenarios)
    return bench


def _scheduler_workload(bench) -> dict[str, torch.Tensor]:
    workload = bench.workload
    tensors = {
        "cpu_a": workload.cpu_a,
        "cpu_b": workload.cpu_b,
        "send_buffer": workload.send_buffer,
        "send_scale": workload.send_scale,
    }
    for name, (a, b) in workload.gpu_buffers.items():
        tensors[f"{name}.a"] = a
        tensors[f"{name}.b"] = b
    return tensors


def _assert_same_state(first: torch.nn.Module, second: torch.nn.Module) -> None:
    first_state = first.state_dict()
    second_state = second.state_dict()
    assert list(first_state) == list(second_state)
    for name, tensor in first_state.items():
        assert torch.equal(tensor, second_state[name]), name


def _assert_rejects_zero_and_scaled(
    runner: VerifyRunner, expected: torch.Tensor, tolerance: tuple[float, float]
) -> None:
    for name, wrong in {"zeros": torch.zeros_like(expected), "scaled_0.9": expected * 0.9}.items():
        assert not runner.compare_perf_outputs(expected, wrong, tolerance).passed, name


def _declared_tolerances(module: ModuleType) -> list[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    declared = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_set_verification_payload"
        ):
            declared.extend(
                ast.unparse(keyword.value) for keyword in node.keywords if keyword.arg == "output_tolerance"
            )
    return declared


@pytest.mark.parametrize(
    ("baseline_module", "optimized_module", "constant"),
    SHARED_TOLERANCES,
    ids=["runtime_scheduler", "piece_graphs", "regional_compilation"],
)
def test_pair_declares_one_tolerance_from_its_baseline(
    baseline_module: ModuleType, optimized_module: ModuleType, constant: str
) -> None:
    """The harness rejects an optimized bound that is looser than the baseline's."""
    assert getattr(optimized_module, constant) is getattr(baseline_module, constant)
    assert getattr(baseline_module, constant) != PRE_FIX_TOLERANCE
    for module in (baseline_module, optimized_module):
        assert _declared_tolerances(module) == [constant], module.__name__


def test_runtime_scheduler_pair_draws_same_workload_and_verifies(
    scheduler_device: torch.device, runner: VerifyRunner
) -> None:
    baseline = _captured(_scheduler(BaselineRuntimeSchedulerBenchmark, scheduler_device))
    optimized = _captured(_scheduler(OptimizedRuntimeSchedulerBenchmark, scheduler_device))

    baseline_workload = _scheduler_workload(baseline)
    optimized_workload = _scheduler_workload(optimized)
    assert list(baseline_workload) == list(optimized_workload)
    for name, tensor in baseline_workload.items():
        assert torch.equal(tensor, optimized_workload[name]), name
    assert baseline.get_output_tolerance() == RUNTIME_SCHEDULER_OUTPUT_TOLERANCE
    assert optimized.get_output_tolerance() == RUNTIME_SCHEDULER_OUTPUT_TOLERANCE

    receipt = verified_comparison._compare_outputs(baseline, optimized)

    assert receipt["passed"], receipt
    assert (receipt["rtol"], receipt["atol"]) == RUNTIME_SCHEDULER_OUTPUT_TOLERANCE
    expected = baseline.get_verify_output()
    _assert_rejects_zero_and_scaled(runner, expected, RUNTIME_SCHEDULER_OUTPUT_TOLERANCE)
    # The output peaks near 195, so (0.1, 1.0) already rejected zeros and a 0.9x
    # output here. It still accepted any 0.95x output.
    scaled = expected * 0.95
    assert runner.compare_perf_outputs(expected, scaled, PRE_FIX_TOLERANCE).passed
    assert not runner.compare_perf_outputs(expected, scaled, RUNTIME_SCHEDULER_OUTPUT_TOLERANCE).passed


def test_runtime_scheduler_pair_passes_verify_pair(
    scheduler_device: torch.device, runner: VerifyRunner
) -> None:
    baseline = _scheduler(BaselineRuntimeSchedulerBenchmark, scheduler_device)
    optimized = _scheduler(OptimizedRuntimeSchedulerBenchmark, scheduler_device)

    result = runner.verify_pair(baseline, optimized, VerifyConfig(seed=HARNESS_SEED))

    assert result.passed, (result.reason, result.details, result.comparison_details)


def test_piece_graphs_model_split_is_exact_and_bound_admits_fp64_on_cpu(
    restore_cudnn_flags: None, runner: VerifyRunner
) -> None:
    """The optimized side joins its head and tail graphs through an FP16 buffer.

    That buffer holds the same values the whole-model graph passes between the two
    regions, so the split forward equals the whole forward bit for bit. The bound
    admits an FP64 forward of the same weights, which stands in for any other
    correct kernel choice, and rejects an all-zero and a 0.9x output.
    """
    sizes = BaselinePieceGraphsBenchmark()
    set_deterministic_seeds(HARNESS_SEED)
    model = RegionalPieceGraph(
        hidden_dim=sizes.hidden_dim,
        n_layers=sizes.n_layers,
        num_heads=sizes.num_heads,
    ).to(dtype=torch.float16).eval()
    x = torch.randn(1, sizes.seq_len, sizes.hidden_dim, dtype=torch.float16)

    with torch.inference_mode():
        whole = model(x).float()
        head_output = torch.empty_like(x)
        head_output.copy_(model.region_head(model.input_proj(x)))
        split = model.output_proj(model.region_tail(head_output)).float()
        exact = copy.deepcopy(model).to(torch.float64)(x.to(torch.float64)).float()

    assert torch.equal(whole, split)
    assert runner.compare_perf_outputs(whole, exact, PIECE_GRAPHS_OUTPUT_TOLERANCE).passed
    assert runner.compare_perf_outputs(whole, whole * 0.9, PRE_FIX_TOLERANCE).passed
    _assert_rejects_zero_and_scaled(runner, whole, PIECE_GRAPHS_OUTPUT_TOLERANCE)


@requires_cuda
def test_piece_graphs_pair_draws_same_weights_and_verifies(
    restore_cudnn_flags: None, runner: VerifyRunner
) -> None:
    baseline = BaselinePieceGraphsBenchmark()
    baseline.device = torch.device("cuda")
    optimized = OptimizedPieceGraphsBenchmark()
    try:
        _captured(baseline)
        _captured(optimized)
        _assert_same_state(baseline.model, optimized.model)
        assert torch.equal(baseline._verify_input, optimized._verify_input)
        assert baseline.get_output_tolerance() == PIECE_GRAPHS_OUTPUT_TOLERANCE
        assert optimized.get_output_tolerance() == PIECE_GRAPHS_OUTPUT_TOLERANCE

        receipt = verified_comparison._compare_outputs(baseline, optimized)

        assert receipt["passed"], receipt
        expected = baseline.get_verify_output()
        assert runner.compare_perf_outputs(expected, expected * 0.9, PRE_FIX_TOLERANCE).passed
        _assert_rejects_zero_and_scaled(runner, expected, PIECE_GRAPHS_OUTPUT_TOLERANCE)
    finally:
        baseline.teardown()
        optimized.teardown()


@requires_cuda
def test_piece_graphs_pair_passes_verify_pair(restore_cudnn_flags: None, runner: VerifyRunner) -> None:
    baseline = BaselinePieceGraphsBenchmark()
    baseline.device = torch.device("cuda")
    optimized = OptimizedPieceGraphsBenchmark()

    result = runner.verify_pair(baseline, optimized, VerifyConfig(seed=HARNESS_SEED))

    assert result.passed, (result.reason, result.details, result.comparison_details)


def test_regional_compilation_forwards_match_and_bound_rejects_wrong_outputs_on_cpu(
    monkeypatch: pytest.MonkeyPatch, restore_cudnn_flags: None, runner: VerifyRunner
) -> None:
    """The baseline runs its real setup and timed forward at the smallest listed config.

    The optimized model is built in the order its setup() uses and called the way
    its graph capture calls it. Outside torch.compile, nested_compile_region calls
    each layer unchanged, so both forwards run the same kernels.
    """
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    choice = _smallest_regional_choice()
    baseline = BaselineRegionalCompilationBenchmark()
    baseline.device = torch.device("cpu")
    baseline.choice = choice
    try:
        _captured(baseline)
        set_deterministic_seeds(HARNESS_SEED)
        model = RegionalCompilationTransformer(
            n_layers=choice["n_layers"], d_model=choice["d_model"], d_ff=choice["d_ff"]
        ).to(dtype=torch.bfloat16).eval()
        x = torch.randn(1, choice["seq_len"], choice["d_model"], dtype=torch.bfloat16)
        _assert_same_state(baseline.model, model)
        assert torch.equal(baseline.inputs, x)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            optimized_output = model(x).float()

        expected = baseline.get_verify_output()
        assert baseline.get_output_tolerance() == REGIONAL_COMPILATION_OUTPUT_TOLERANCE
        assert torch.equal(expected, optimized_output)
        assert runner.compare_perf_outputs(expected, expected * 0.9, PRE_FIX_TOLERANCE).passed
        _assert_rejects_zero_and_scaled(runner, expected, REGIONAL_COMPILATION_OUTPUT_TOLERANCE)
    finally:
        baseline.teardown()


@requires_cuda
def test_regional_compilation_pair_draws_same_weights_and_verifies(
    restore_cudnn_flags: None, runner: VerifyRunner
) -> None:
    baseline = BaselineRegionalCompilationBenchmark()
    baseline.device = torch.device("cuda")
    optimized = OptimizedRegionalCompilationBenchmark()
    try:
        _captured(baseline)
        _captured(optimized)
        _assert_same_state(baseline.model, optimized.model)
        assert torch.equal(baseline.inputs, optimized._verify_input)
        assert baseline.get_output_tolerance() == REGIONAL_COMPILATION_OUTPUT_TOLERANCE
        assert optimized.get_output_tolerance() == REGIONAL_COMPILATION_OUTPUT_TOLERANCE

        receipt = verified_comparison._compare_outputs(baseline, optimized)

        assert receipt["passed"], receipt
        expected = baseline.get_verify_output()
        assert runner.compare_perf_outputs(expected, expected * 0.9, PRE_FIX_TOLERANCE).passed
        _assert_rejects_zero_and_scaled(runner, expected, REGIONAL_COMPILATION_OUTPUT_TOLERANCE)
    finally:
        baseline.teardown()
        optimized.teardown()


@requires_cuda
def test_regional_compilation_pair_passes_verify_pair(
    restore_cudnn_flags: None, runner: VerifyRunner
) -> None:
    baseline = BaselineRegionalCompilationBenchmark()
    baseline.device = torch.device("cuda")
    optimized = OptimizedRegionalCompilationBenchmark()

    result = runner.verify_pair(baseline, optimized, VerifyConfig(seed=HARNESS_SEED))

    assert result.passed, (result.reason, result.details, result.comparison_details)


def test_regional_compilation_baseline_declares_the_timed_input(
    regional_device: torch.device, runner: VerifyRunner
) -> None:
    """benchmark_fn reads self.inputs, so the payload must declare that tensor.

    The optimized side declares its CUDA graph's static input, and verify_pair
    already jitter-checks that side on CUDA.
    """
    bench = BaselineRegionalCompilationBenchmark()
    if regional_device.type == "cpu":
        bench.device = regional_device
        bench.choice = _smallest_regional_choice()

    passed, message = runner._run_live_jitter_check(bench, VerifyConfig(seed=HARNESS_SEED))

    assert passed, message


@pytest.mark.parametrize(
    "benchmark_cls",
    list(DENSE_SEQUENCE_ATTRIBUTE),
    ids=["baseline", "optimized", "blackwell_variant"],
)
def test_dense_attention_declared_input_is_the_timed_input(
    dense_device: torch.device, runner: VerifyRunner, benchmark_cls
) -> None:
    """Each dense file must declare the tensor its benchmark_fn reads.

    The CPU run uses batch 1 and sequence length 256 because the naive FP16 path
    at 4096 takes minutes there. Which tensor the payload declares does not depend
    on the shape.
    """
    bench = benchmark_cls()
    if dense_device.type == "cpu":
        bench.device = dense_device
        bench.batch_size = 1
        setattr(bench, DENSE_SEQUENCE_ATTRIBUTE[benchmark_cls], 256)

    passed, message = runner._run_live_jitter_check(bench, VerifyConfig(seed=HARNESS_SEED))

    assert passed, message


@requires_cuda
def test_dense_attention_pair_passes_verify_pair(restore_cudnn_flags: None, runner: VerifyRunner) -> None:
    baseline = BaselineDenseAttentionFlashBenchmark()
    baseline.device = torch.device("cuda")
    optimized = OptimizedDenseAttentionFlashBenchmark()

    result = runner.verify_pair(baseline, optimized, VerifyConfig(seed=HARNESS_SEED))

    assert result.passed, (result.reason, result.details, result.comparison_details)


def test_dense_attention_bound_rejects_zeroed_late_rows_on_cpu(
    restore_cudnn_flags: None, runner: VerifyRunner
) -> None:
    """Late causal rows average over thousands of keys, so their outputs stay small.

    At sequence length 4096, rows past 2048 stay below 0.04. The previous
    (5e-2, 5e-2) bound therefore accepted an output whose second half was zero.
    """
    sizes = BaselineDenseAttentionFlashBenchmark()
    seq_len, hidden, dtype = sizes.max_seq_len, sizes.hidden_dim, sizes.dtype
    bench = object.__new__(OptimizedDenseAttentionFlashBenchmark)
    bench.hidden_dim = hidden
    bench.num_heads = sizes.num_heads
    bench.head_dim = sizes.head_dim
    set_deterministic_seeds(HARNESS_SEED)
    bench.qkv_proj = torch.nn.Linear(hidden, hidden * 3, bias=False, dtype=dtype)
    bench.out_proj = torch.nn.Linear(hidden, hidden, bias=False, dtype=dtype)
    bench._qkv_weight_t = bench.qkv_proj.weight.t()
    bench._out_proj_weight_t = bench.out_proj.weight.t()
    bench.inputs = torch.randn(1, seq_len, hidden, dtype=dtype)
    bench._qkv_buffer = torch.empty(1, seq_len, hidden * 3, dtype=dtype)
    bench._attn_merge_buffer = torch.empty(1, seq_len, hidden, dtype=dtype)
    bench._output_buffer = torch.empty(1, seq_len, hidden, dtype=dtype)
    with torch.inference_mode():
        expected = bench._forward_flash().clone()

    second_half_zeroed = expected.clone()
    second_half_zeroed[:, seq_len // 2 :, :] = 0
    last_rows_zeroed = expected.clone()
    last_rows_zeroed[:, -64:, :] = 0

    assert runner.compare_perf_outputs(expected, second_half_zeroed, PREVIOUS_DENSE_TOLERANCE).passed
    for name, wrong in {
        "second_half_zeroed": second_half_zeroed,
        "last_64_rows_zeroed": last_rows_zeroed,
    }.items():
        assert not runner.compare_perf_outputs(expected, wrong, DENSE_ATTENTION_OUTPUT_TOLERANCE).passed, name
    _assert_rejects_zero_and_scaled(runner, expected, DENSE_ATTENTION_OUTPUT_TOLERANCE)
