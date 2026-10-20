"""W2-059: the Chapter 16 attention pairs must verify under bounds sized to their outputs.

The Flash SDP pair's outputs have a standard deviation near 0.04, so the old
(0.1, 1.0) tolerance accepted an all-zero output. It also hid an input mismatch.
The optimized setup drew a random Flash probe tensor from the seeded RNG before
it drew its inputs, and the baseline did not.

The Flash SDP pair tests run on two devices.

On the CPU they run both benchmarks' real setup, timed and capture paths at the
benchmark's shapes, with three CUDA-only seams replaced. The probe sees CUDA as
available, the probe's hardcoded CUDA tensor is made on the CPU, and the CUDA
sync does nothing. One CPU generator then feeds the weights, the inputs and any
probe draw.

On a CUDA device they patch nothing, and they skip when no CUDA device exists.
There the weights come from the CPU generator and the inputs from the CUDA
generator, so these runs check both generators, the real Flash kernel and the
naive cuBLAS path.

The dense-attention test runs only on a CUDA device. Its naive FP16 path at
sequence length 4096 takes minutes on the CPU.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest
import torch
from torch.overrides import TorchFunctionMode

from ch16 import baseline_flash_sdp, optimized_flash_sdp
from ch16.baseline_dense_attention_flash import BaselineDenseAttentionFlashBenchmark
from ch16.baseline_flash_sdp import FLASH_SDP_OUTPUT_TOLERANCE, BaselineFlashSDPBenchmark
from ch16.optimized_flash_sdp import OptimizedFlashSDPBenchmark
from core.benchmark import verified_comparison
from core.benchmark.quarantine import QuarantineManager
from core.benchmark.verification import set_deterministic_seeds
from core.benchmark.verify_runner import VerifyConfig, VerifyRunner

PRE_FIX_TOLERANCE = (0.1, 1.0)
HARNESS_SEED = VerifyConfig().seed

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="needs a CUDA device for the real Flash SDP and cuBLAS paths",
)


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
def cpu_flash_probe(monkeypatch: pytest.MonkeyPatch, restore_cudnn_flags: None) -> None:
    """Run each module's own Flash probe on the CPU and skip setup's device sync."""
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)
    for module in (baseline_flash_sdp, optimized_flash_sdp):
        real_probe = module.ensure_flash_sdp_available

        def probe_on_cpu(real_probe=real_probe) -> None:
            with monkeypatch.context() as probe_patch:
                probe_patch.setattr(torch.cuda, "is_available", lambda: True)
                with _CudaFactoryToCpu():
                    real_probe()

        monkeypatch.setattr(module, "ensure_flash_sdp_available", probe_on_cpu)


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=requires_cuda)])
def flash_device(request: pytest.FixtureRequest, restore_cudnn_flags: None) -> torch.device:
    """CPU runs use the probe seams in cpu_flash_probe. CUDA runs patch nothing."""
    if request.param == "cpu":
        request.getfixturevalue("cpu_flash_probe")
    return torch.device(request.param)


@pytest.fixture
def runner(tmp_path: Path) -> VerifyRunner:
    return VerifyRunner(
        cache_dir=tmp_path / "golden",
        quarantine_manager=QuarantineManager(cache_dir=tmp_path),
    )


def _captured(benchmark_cls, device: torch.device):
    """Seed as the harness does, then run the benchmark's real setup and capture."""
    set_deterministic_seeds(HARNESS_SEED)
    bench = benchmark_cls()
    bench.device = device
    bench.setup()
    bench.benchmark_fn()
    bench.capture_verification_payload()
    return bench


def _rng_states(device: torch.device) -> dict[str, torch.Tensor]:
    states = {"cpu": torch.get_rng_state()}
    if device.type == "cuda":
        states["cuda"] = torch.cuda.get_rng_state(device)
    return states


def _wrong_scale_output(bench: BaselineFlashSDPBenchmark) -> torch.Tensor:
    """Same projections with a 1/head_dim softmax scale instead of 1/sqrt(head_dim)."""
    x = bench.get_verify_inputs()["input"]
    batch, seq_len, hidden = x.shape
    heads = bench.model.num_heads
    head_dim = hidden // heads
    with torch.inference_mode():
        qkv = bench.model.qkv(x).view(batch, seq_len, 3, heads, head_dim)
        q, k, v = (t.transpose(1, 2) for t in qkv.unbind(dim=2))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0 / head_dim)
    return out.transpose(1, 2).reshape(batch, seq_len, hidden)


@pytest.mark.parametrize(
    "module", [baseline_flash_sdp, optimized_flash_sdp], ids=["baseline", "optimized"]
)
def test_flash_sdp_probe_leaves_seeded_rng_untouched(
    flash_device: torch.device, module: ModuleType
) -> None:
    set_deterministic_seeds(HARNESS_SEED)
    before = _rng_states(flash_device)

    module.ensure_flash_sdp_available()

    after = _rng_states(flash_device)
    for generator, state in before.items():
        assert torch.equal(after[generator], state), (
            f"{module.__name__}.ensure_flash_sdp_available advanced the {generator} generator, "
            "so the inputs drawn after it differ from the other side's"
        )


def test_flash_sdp_pair_draws_same_inputs_and_verifies(flash_device: torch.device) -> None:
    baseline = _captured(BaselineFlashSDPBenchmark, flash_device)
    optimized = _captured(OptimizedFlashSDPBenchmark, flash_device)
    try:
        assert baseline.get_output_tolerance() == FLASH_SDP_OUTPUT_TOLERANCE
        assert optimized.get_output_tolerance() == FLASH_SDP_OUTPUT_TOLERANCE
        assert torch.equal(baseline.model.qkv.weight, optimized.model.qkv.weight)
        assert torch.equal(
            baseline.get_verify_inputs()["input"], optimized.get_verify_inputs()["input"]
        )

        receipt = verified_comparison._compare_outputs(baseline, optimized)

        assert receipt["passed"], receipt
        assert (receipt["rtol"], receipt["atol"]) == FLASH_SDP_OUTPUT_TOLERANCE
    finally:
        baseline.teardown()
        optimized.teardown()


def test_flash_sdp_pair_passes_verify_pair(
    flash_device: torch.device, runner: VerifyRunner
) -> None:
    baseline = BaselineFlashSDPBenchmark()
    baseline.device = flash_device
    optimized = OptimizedFlashSDPBenchmark()
    optimized.device = flash_device

    # verify_pair also runs the fresh-input and jitter checks on the optimized side.
    result = runner.verify_pair(baseline, optimized, VerifyConfig(seed=HARNESS_SEED))

    assert result.passed, (result.reason, result.details, result.comparison_details)


@pytest.mark.parametrize(
    "benchmark_cls",
    [BaselineFlashSDPBenchmark, OptimizedFlashSDPBenchmark],
    ids=["baseline", "optimized"],
)
def test_flash_sdp_declared_input_is_the_timed_input(
    flash_device: torch.device, runner: VerifyRunner, benchmark_cls
) -> None:
    """The harness jitter check perturbs the declared input and reruns benchmark_fn.

    verify_pair runs it only on the optimized side, so this test also covers the
    baseline. A declared copy of the input fails because the output never changes.
    """
    bench = benchmark_cls()
    bench.device = flash_device

    passed, message = runner._run_live_jitter_check(bench, VerifyConfig(seed=HARNESS_SEED))

    assert passed, message


@pytest.mark.parametrize("corruption", ["zeros", "scaled_0.9", "wrong_softmax_scale"])
def test_flash_sdp_tolerance_rejects_wrong_outputs(
    cpu_flash_probe: None, runner: VerifyRunner, corruption: str
) -> None:
    bench = _captured(BaselineFlashSDPBenchmark, torch.device("cpu"))
    expected = bench.get_verify_output()
    if corruption == "zeros":
        wrong = torch.zeros_like(expected)
    elif corruption == "scaled_0.9":
        wrong = expected * 0.9
    else:
        wrong = _wrong_scale_output(bench)

    # The pre-fix tolerance accepted every one of these outputs.
    assert runner.compare_perf_outputs(expected, wrong, PRE_FIX_TOLERANCE).passed
    assert not runner.compare_perf_outputs(
        expected, wrong, bench.get_output_tolerance()
    ).passed


@pytest.mark.parametrize("device", [pytest.param("cuda", marks=requires_cuda)])
def test_dense_attention_tolerance_rejects_zeroed_second_half(
    restore_cudnn_flags: None, runner: VerifyRunner, device: str
) -> None:
    """Causal rows late in the sequence average over thousands of keys and stay small.

    A bound that only rejects an all-zero output can still accept an output whose
    second half is zero.
    """
    bench = _captured(BaselineDenseAttentionFlashBenchmark, torch.device(device))
    try:
        expected = bench.get_verify_output()
        wrong = expected.clone()
        wrong[:, expected.shape[1] // 2 :, :] = 0

        assert runner.compare_perf_outputs(expected, wrong, PRE_FIX_TOLERANCE).passed
        assert not runner.compare_perf_outputs(
            expected, wrong, bench.get_output_tolerance()
        ).passed, (
            f"declared dense tolerance {bench.get_output_tolerance()} accepts an output "
            "whose second half is zero"
        )
    finally:
        bench.teardown()
