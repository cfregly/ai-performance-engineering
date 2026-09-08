"""Actual CUDA execution plus explicit runtime-receipt admission controls."""

import pytest
import torch

from core.benchmark.runtime_comparison import compare_executed_runtime_provenance
from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode
from tests.test_evaluation_harness_integration import _benchmark, _harness


def assert_runtime_version_controls(tmp_path, field):
    """Retain live versions; mutate a copied receipt to exercise rejection only.

    This does not install another driver or library, and installed package
    metadata does not claim to identify the native cuBLAS library loaded.
    """
    if not torch.cuda.is_available():
        pytest.skip("real CUDA worker runtime/version admission requires a device")
    config = _harness().config
    config.device = torch.device("cuda", torch.cuda.current_device())
    config.validity_profile = "portable"
    config.allow_virtualization = True
    config.enforce_environment_validation = True
    runs = []
    for arm in ("reference", "candidate"):
        benchmark = _benchmark(tmp_path)
        benchmark.device = config.device
        run = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=config).benchmark_with_manifest(
            benchmark, run_id=f"runtime-version-{arm}",
        )
        assert not run.result.errors, run.result.errors
        torch.testing.assert_close(
            benchmark.get_verify_output(), torch.tensor([2.0], device=config.device),
            rtol=0.0, atol=0.0,
        )
        runs.append(run)
    comparison = compare_executed_runtime_provenance(*runs)
    assert comparison.matches, comparison.model_dump()
    assert comparison.target == "cuda"
    observed = runs[1].result.runtime_provenance
    assert observed.cudnn_version == str(torch.backends.cudnn.version())
    assert observed.driver_version
    assert observed.library_versions_complete and observed.library_versions

    changed = runs[1].model_copy(deep=True)
    if field == "library_versions":
        changed.result.runtime_provenance.library_versions["torch"] += "-receipt-corruption"
    else:
        value = getattr(changed.result.runtime_provenance, field)
        assert value is not None
        setattr(changed.result.runtime_provenance, field, str(value) + "-receipt-corruption")
    changed.manifest.runtime_provenance = changed.result.runtime_provenance.model_copy(deep=True)
    rejected = compare_executed_runtime_provenance(runs[0], changed)
    assert not rejected.matches
    assert field in rejected.runtime_parity.mismatched_fields
