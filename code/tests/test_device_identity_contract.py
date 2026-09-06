from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness
from core.harness.device_identity_contract import (
    DeviceIdentityContract,
    DeviceIdentityError,
    DeviceIdentityExpectation,
    DeviceIdentityObservation,
    build_device_identity_expectation,
    normalize_compute_capability,
    observe_cuda_device_identity,
    validate_device_identity,
    validate_device_identity_stable,
)

GPU_UUID_A = "GPU-12345678-1234-5678-1234-567812345678"
GPU_UUID_B = "GPU-87654321-4321-8765-4321-876543218765"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ((10, 0), "10.0"),
        ([12, 1], "12.1"),
        ("10.3", "10.3"),
        ("sm_100", "10.0"),
        ("sm_103a", "10.3"),
        ("compute_121", "12.1"),
    ],
)
def test_normalize_compute_capability(value, expected) -> None:
    assert normalize_compute_capability(value) == expected


@pytest.mark.parametrize("value", ["", "10", "sm_1", "sm_10.0", (10,), (10, 0, 1), 100, True])
def test_normalize_compute_capability_rejects_ambiguous_values(value) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalize_compute_capability(value)


def test_expected_device_contract_is_all_or_nothing_and_normalized() -> None:
    bare_uuid = GPU_UUID_A.removeprefix("GPU-").upper()
    expectation = build_device_identity_expectation(bare_uuid, "sm_100")
    assert expectation == DeviceIdentityExpectation(GPU_UUID_A, "10.0")
    assert build_device_identity_expectation(None, None) is None

    with pytest.raises(ValueError, match="configured together"):
        build_device_identity_expectation(GPU_UUID_A, None)
    with pytest.raises(ValueError, match="configured together"):
        build_device_identity_expectation(None, "10.0")
    with pytest.raises(ValueError, match="configured together"):
        BenchmarkConfig(expected_device_uuid=GPU_UUID_A)


def test_benchmark_config_normalizes_and_serializes_expected_device_fields() -> None:
    config = BenchmarkConfig(
        expected_device_uuid=GPU_UUID_A.removeprefix("GPU-"),
        expected_compute_capability="compute_100",
    )
    serialized = {
        key: value
        for key, value in config.__dict__.items()
        if not key.startswith("_") and value is not None
    }
    assert serialized["expected_device_uuid"] == GPU_UUID_A
    assert serialized["expected_compute_capability"] == "10.0"
    assert config.validity.expected_device_uuid == GPU_UUID_A
    assert config.validity.expected_compute_capability == "10.0"


@pytest.mark.parametrize("validity_profile", ["strict", "portable"])
def test_expected_identity_mismatch_is_hard_failure_in_every_validity_profile(
    validity_profile: str,
) -> None:
    config = BenchmarkConfig(
        validity_profile=validity_profile,
        expected_device_uuid=GPU_UUID_A,
        expected_compute_capability="10.0",
    )
    expectation = build_device_identity_expectation(
        config.expected_device_uuid,
        config.expected_compute_capability,
    )
    observation = DeviceIdentityObservation(
        logical_index=0,
        current_logical_index=0,
        cuda_uuid=GPU_UUID_B,
        nvml_uuid=GPU_UUID_B,
        compute_capability="10.0",
    )
    with pytest.raises(DeviceIdentityError, match="Expected GPU UUID mismatch"):
        validate_device_identity(expectation, observation, "before_measurement")


def test_validate_device_identity_accepts_matching_cuda_and_nvml_identity() -> None:
    expectation = DeviceIdentityExpectation(GPU_UUID_A, "10.0")
    observation = DeviceIdentityObservation(
        logical_index=1,
        current_logical_index=1,
        cuda_uuid=GPU_UUID_A,
        nvml_uuid=GPU_UUID_A,
        compute_capability="10.0",
    )
    validate_device_identity(expectation, observation, "after_warmup")


@pytest.mark.parametrize(
    ("observation", "message"),
    [
        (
            DeviceIdentityObservation(0, 1, GPU_UUID_A, GPU_UUID_A, "10.0"),
            "current-device drift",
        ),
        (
            DeviceIdentityObservation(0, 0, GPU_UUID_A, GPU_UUID_B, "10.0"),
            "CUDA/NVML device identity mismatch",
        ),
        (
            DeviceIdentityObservation(0, 0, GPU_UUID_A, GPU_UUID_A, "9.0"),
            "Expected compute capability mismatch",
        ),
    ],
)
def test_validate_device_identity_rejects_each_mismatch(
    observation: DeviceIdentityObservation,
    message: str,
) -> None:
    with pytest.raises(DeviceIdentityError, match=message):
        validate_device_identity(
            DeviceIdentityExpectation(GPU_UUID_A, "10.0"),
            observation,
            "after_setup",
        )


def test_validate_device_identity_uses_nvml_when_cuda_uuid_is_unavailable() -> None:
    observation = DeviceIdentityObservation(
        logical_index=0,
        current_logical_index=0,
        cuda_uuid=None,
        nvml_uuid=GPU_UUID_A,
        compute_capability="10.0",
    )
    validate_device_identity(
        DeviceIdentityExpectation(GPU_UUID_A, "10.0"),
        observation,
        "before_setup",
    )


class _CpuControlBenchmark(BaseBenchmark):
    allow_cpu = True

    def setup(self) -> None:
        self.input = torch.arange(8, dtype=torch.float32)

    def benchmark_fn(self) -> None:
        self.output = self.input + 1

    def get_input_signature(self):
        return {"shape": tuple(self.input.shape), "dtype": str(self.input.dtype)}

    def get_verify_inputs(self):
        return {"input": self.input}

    def get_verify_output(self):
        return self.output

    def get_output_tolerance(self):
        return (0.0, 0.0)


def test_cpu_harness_path_is_unchanged_by_gpu_identity_expectations() -> None:
    config = BenchmarkConfig(
        device=torch.device("cpu"),
        expected_device_uuid=GPU_UUID_A,
        expected_compute_capability="10.0",
        use_subprocess=False,
        iterations=1,
        warmup=5,
        adaptive_iterations=False,
        enforce_environment_validation=False,
        clear_compile_cache=False,
        monitor_backend_policy=False,
        detect_benchmark_fn_sync=False,
        detect_benchmark_fn_antipatterns=False,
    )
    run = BenchmarkHarness(config=config).benchmark_with_manifest(
        _CpuControlBenchmark(),
        run_id="cpu-device-identity-control",
    )
    result = run.result
    assert result.timing.iterations == 1
    assert not result.errors
    assert run.manifest.config is not None
    assert run.manifest.config["expected_device_uuid"] == GPU_UUID_A
    assert run.manifest.config["expected_compute_capability"] == "10.0"


def _live_observation_for_current_device() -> DeviceIdentityObservation:
    if not torch.cuda.is_available():
        pytest.skip("real CUDA device identity validation requires a CUDA device")
    pytest.importorskip("pynvml", reason="real CUDA device identity validation requires pynvml")
    current = int(torch.cuda.current_device())
    return observe_cuda_device_identity(torch.device("cuda", current))


def test_live_cuda_device_identity_contract_matches_observed_device() -> None:
    observed = _live_observation_for_current_device()
    contract = DeviceIdentityContract(
        torch.device("cuda", observed.logical_index),
        DeviceIdentityExpectation(observed.nvml_uuid, observed.compute_capability),
    )
    contract.establish_configured_device()
    try:
        assert contract.check("before_setup") == observed
        assert contract.check("after_teardown").nvml_uuid == observed.nvml_uuid
    finally:
        contract.restore_entry_device()


def test_live_cuda_device_identity_contract_rejects_wrong_expected_uuid() -> None:
    observed = _live_observation_for_current_device()
    wrong_uuid = GPU_UUID_A if observed.nvml_uuid != GPU_UUID_A else GPU_UUID_B
    contract = DeviceIdentityContract(
        torch.device("cuda", observed.logical_index),
        DeviceIdentityExpectation(wrong_uuid, observed.compute_capability),
    )
    contract.establish_configured_device()
    try:
        with pytest.raises(DeviceIdentityError, match="Expected GPU UUID mismatch"):
            contract.check("before_setup")
    finally:
        contract.restore_entry_device()


def test_live_cuda_device_identity_contract_establishes_and_restores_cuda_one() -> None:
    observed = _live_observation_for_current_device()
    if torch.cuda.device_count() < 2:
        pytest.skip("configured cuda:1 establishment requires two visible CUDA devices")
    entry_index = observed.current_logical_index
    target_index = next(index for index in range(torch.cuda.device_count()) if index != entry_index)
    contract = DeviceIdentityContract(torch.device("cuda", target_index))
    contract.establish_configured_device()
    try:
        target_observation = contract.check("before_setup")
        assert target_observation.current_logical_index == target_index
    finally:
        contract.restore_entry_device()
    assert torch.cuda.current_device() == entry_index


class _CudaSetupDriftBenchmark(BaseBenchmark):
    def __init__(self, alternate_index: int) -> None:
        super().__init__()
        self.alternate_index = alternate_index

    def setup(self) -> None:
        torch.cuda.set_device(self.alternate_index)

    def benchmark_fn(self) -> None:
        raise AssertionError("measurement must not run after setup changes the CUDA device")


class _CudaSetupMustNotRunBenchmark(BaseBenchmark):
    def setup(self) -> None:
        raise AssertionError("setup must not run when declared device identity is wrong")

    def benchmark_fn(self) -> None:
        raise AssertionError("measurement must not run when declared device identity is wrong")


@pytest.mark.parametrize("mismatch", ["uuid", "compute_capability"])
def test_live_harness_rejects_wrong_identity_before_setup(mismatch: str) -> None:
    observed = _live_observation_for_current_device()
    wrong_uuid = GPU_UUID_A if observed.nvml_uuid != GPU_UUID_A else GPU_UUID_B
    wrong_capability = "9.0" if observed.compute_capability != "9.0" else "10.0"
    config = BenchmarkConfig(
        device=torch.device("cuda", observed.logical_index),
        expected_device_uuid=(wrong_uuid if mismatch == "uuid" else observed.nvml_uuid),
        expected_compute_capability=(
            wrong_capability if mismatch == "compute_capability" else observed.compute_capability
        ),
        use_subprocess=False,
        iterations=1,
        warmup=5,
        adaptive_iterations=False,
        enforce_environment_validation=False,
        lock_gpu_clocks=False,
        clear_compile_cache=False,
        monitor_gpu_state=False,
    )
    result = BenchmarkHarness(config=config)._benchmark_with_threading(
        _CudaSetupMustNotRunBenchmark(),
        config,
    )
    assert result.timing.iterations == 0
    expected_message = (
        "Expected GPU UUID mismatch"
        if mismatch == "uuid"
        else "Expected compute capability mismatch"
    )
    assert any(expected_message in error for error in result.errors)


def test_live_harness_rejects_setup_device_drift_and_restores_entry_context() -> None:
    observed = _live_observation_for_current_device()
    if torch.cuda.device_count() < 2:
        pytest.skip("real harness context-drift rejection requires two visible CUDA devices")
    entry_index = observed.current_logical_index
    alternate_index = next(index for index in range(torch.cuda.device_count()) if index != entry_index)
    config = BenchmarkConfig(
        device=torch.device("cuda", entry_index),
        expected_device_uuid=observed.nvml_uuid,
        expected_compute_capability=observed.compute_capability,
        use_subprocess=False,
        iterations=1,
        warmup=5,
        adaptive_iterations=False,
        enforce_environment_validation=False,
        lock_gpu_clocks=False,
        clear_compile_cache=False,
        monitor_gpu_state=False,
    )
    harness = BenchmarkHarness(config=config)
    result = harness._benchmark_with_threading(
        _CudaSetupDriftBenchmark(alternate_index),
        config,
    )
    assert result.timing.iterations == 0
    assert any("CUDA current-device drift at after_setup" in error for error in result.errors)
    assert torch.cuda.current_device() == entry_index


def test_observation_fixture_detects_baseline_identity_drift() -> None:
    baseline = DeviceIdentityObservation(0, 0, GPU_UUID_A, GPU_UUID_A, "10.0")
    changed = replace(baseline, compute_capability="10.3")
    with pytest.raises(DeviceIdentityError, match="CUDA device identity drift"):
        validate_device_identity_stable(baseline, changed, "after_setup")
