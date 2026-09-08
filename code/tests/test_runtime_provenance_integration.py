"""Runtime identity must belong to the worker that executed the real workload."""

import os
import json
import subprocess
import time

import pytest
from pydantic import ValidationError

from core.benchmark.models import BenchmarkRun
from core.benchmark.run_manifest import require_runtime_provenance_parity
from core.benchmark.runtime_comparison import compare_executed_runtime_provenance
from tests.test_evaluation_harness_integration import _benchmark, _harness


@pytest.mark.parametrize("subprocess_mode", [False, True])
def test_harness_result_and_manifest_retain_execution_process(tmp_path, subprocess_mode):
    run = _harness(subprocess_mode=subprocess_mode).benchmark_with_manifest(_benchmark(tmp_path))
    assert not run.result.errors, run.result.errors
    assert run.result.timing.iterations > 0
    runtime = run.result.runtime_provenance
    assert runtime is not None
    assert run.result.execution_process_ids == {0: runtime.process_id}
    assert run.result.local_world_size == 1
    assert run.result.device == "cpu"
    if subprocess_mode:
        assert runtime.process_id != os.getpid()
    else:
        assert runtime.process_id == os.getpid()
    assert run.manifest is not None
    assert run.manifest.runtime_provenance == runtime
    restored = BenchmarkRun.model_validate_json(run.model_dump_json())
    assert restored.result.local_world_size == 1
    assert restored.manifest.runtime_provenance == runtime
    assert require_runtime_provenance_parity(run.manifest, restored.manifest, target="cpu").matches
    for invalid_count in (True, 0, -1, "1", 1.5):
        corrupted = run.model_dump(mode="json")
        corrupted["result"]["local_world_size"] = invalid_count
        with pytest.raises(ValidationError, match="local_world_size"):
            BenchmarkRun.model_validate_json(json.dumps(corrupted))


def test_pair_gate_uses_real_worker_snapshot_and_rejects_drift(tmp_path):
    baseline = _harness().benchmark_with_manifest(_benchmark(tmp_path))
    candidate = _harness(subprocess_mode=True).benchmark_with_manifest(_benchmark(tmp_path))
    assert not baseline.result.errors and not candidate.result.errors
    assert compare_executed_runtime_provenance(baseline, candidate).matches

    # Explicit receipt corruption exercises the admission policy; these values
    # are never presented as observations from a different installed runtime.
    corrupted = candidate.model_copy(deep=True)
    corrupted.result.runtime_provenance.torch_version += "-unexpected"
    corrupted.manifest.runtime_provenance = corrupted.result.runtime_provenance.model_copy(deep=True)
    parity = compare_executed_runtime_provenance(baseline, corrupted)
    assert not parity.matches
    assert "torch_version" in parity.runtime_parity.mismatched_fields
    assert parity.target == "cpu"

    missing = candidate.model_copy(deep=True)
    missing.manifest.runtime_provenance = None
    missing.result.runtime_provenance = None
    parity = compare_executed_runtime_provenance(baseline, missing)
    assert not parity.matches and parity.integrity_failures

    substituted = candidate.model_copy(deep=True)
    substituted.manifest.runtime_provenance = baseline.manifest.runtime_provenance.model_copy(deep=True)
    assert not compare_executed_runtime_provenance(baseline, substituted).matches


@pytest.mark.parametrize("corruption", ["pid", "device"])
def test_subprocess_transport_rejects_corrupted_real_worker_receipt(tmp_path, monkeypatch, corruption):
    """Corrupt only transport data from an actual completed child invocation."""
    real_popen = subprocess.Popen

    class CorruptingTransport(real_popen):
        def communicate(self, *args, **kwargs):
            stdout, stderr = super().communicate(*args, **kwargs)
            if "core.harness.isolated_runner" not in self.args:
                return stdout, stderr
            start = stdout.index('{"success"')
            record, end = json.JSONDecoder().raw_decode(stdout[start:])
            assert record["success"], record
            result = json.loads(record["result_json"])
            if corruption == "pid":
                result["runtime_provenance"]["process_id"] = os.getpid()
            else:
                result["device"] = "cuda:0"
            record["result_json"] = json.dumps(result)
            return stdout[:start] + json.dumps(record) + stdout[start + end:], stderr

    monkeypatch.setattr(subprocess, "Popen", CorruptingTransport)
    run = _harness(subprocess_mode=True).benchmark_with_manifest(_benchmark(tmp_path))
    assert run.result.errors
    assert run.result.timing.iterations == 0
    assert any(("PID" if corruption == "pid" else "device") in error for error in run.result.errors)


def test_runtime_collection_occurs_after_work_with_separate_grace(tmp_path, monkeypatch):
    from core.benchmark import run_manifest

    benchmark = _benchmark(tmp_path)
    actual_capture = run_manifest.capture_runtime_provenance
    calls = []

    def delayed_capture():
        assert benchmark.benchmark_calls > 5
        assert benchmark.output is not None
        calls.append(benchmark.benchmark_calls)
        time.sleep(0.3)  # Metadata latency, not fabricated benchmark execution.
        return actual_capture()

    monkeypatch.setattr(run_manifest, "capture_runtime_provenance", delayed_capture)
    harness = _harness()
    harness.config.measurement_timeout_seconds = 0.2
    run = harness.benchmark_with_manifest(benchmark)
    assert not run.result.errors, run.result.errors
    assert run.result.timing.iterations > 0
    assert len(calls) == 1


def test_runtime_collection_rejects_an_exceeded_receipt_budget(tmp_path, monkeypatch):
    from core.benchmark import run_manifest
    from core.harness import benchmark_harness

    actual_capture = run_manifest.capture_runtime_provenance

    def delayed_capture():
        time.sleep(0.15)
        return actual_capture()

    monkeypatch.setattr(run_manifest, "capture_runtime_provenance", delayed_capture)
    monkeypatch.setattr(benchmark_harness, "_RUNTIME_PROVENANCE_TIMEOUT_SECONDS", 0.1)
    run = _harness().benchmark_with_manifest(_benchmark(tmp_path))
    assert run.result.timing.iterations == 0
    assert any("Runtime provenance collection exceeded" in error for error in run.result.errors)
