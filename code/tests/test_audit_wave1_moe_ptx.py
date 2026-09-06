"""Full real CPU MoE contracts, with separate real CUDA acceptance cases."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from labs.moe_cuda_ptx.moe_cuda_ptx_common import (
    LayerAccuracyLimits, MoECudaPtxBenchmark, MoECudaPtxWorkload,
    apply_workload_overrides, build_state, load_layer_accuracy_limits,
    measure_layer_output_errors, reference_layer_forward, run_layer_baseline,
    run_layer_cuda, snapshot_layer_inputs,
)


def _workload(dtype=torch.bfloat16, histogram="balanced"):
    return MoECudaPtxWorkload(num_tokens=65, num_experts=4, hidden_dim=48,
        expert_ffn_dim=32, capacity_factor=1.5, dtype=dtype, histogram=histogram)


def _default_workload(dtype=torch.bfloat16, histogram="balanced"):
    return MoECudaPtxWorkload(
        num_tokens=32768,
        num_experts=8,
        top_k=2,
        hidden_dim=7168,
        expert_ffn_dim=2048,
        capacity_factor=1.25,
        mode="forward",
        dtype=dtype,
        histogram=histogram,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA MoE forward execution required")
def test_real_cuda_layer_forward_repacks_current_inputs():
    bench = MoECudaPtxBenchmark(target="moe_layer", backend="cuda", label="current-layer-inputs")
    bench.workload = _workload()
    bench.setup()
    try:
        assert bench.packed is None
        assert bench._grouped_output_buffer is None
        assert bench._padded_tokens_buffer is None
        bench.benchmark_fn()
        bench.capture_verification_payload()
        assert bench.validate_result() is None
        assert torch.count_nonzero(bench.outputs) > 0
        bench.state.expert_indices.add_(1).remainder_(bench.workload.num_experts)
        bench.benchmark_fn()
        expected = reference_layer_forward(snapshot_layer_inputs(bench.state), bench.device)
        limits = load_layer_accuracy_limits(bench.workload.dtype, workload=bench.workload)
        limits.check(measure_layer_output_errors(bench.outputs, expected))
        bench.state.x.zero_()
        bench.benchmark_fn()
        torch.cuda.synchronize(bench.device)
        assert torch.count_nonzero(bench.outputs) == 0
    finally:
        bench.teardown()


def _strict_test_policy(monkeypatch, tmp_path):
    # Zero is a strict test requirement for these bounded CPU fixtures, not a
    # measured or recommended production/GPU error budget.
    path = tmp_path / "strict-unit-policy.json"
    path.write_text(json.dumps({"schema_version": 3, "validated_workloads": [{
        "num_tokens": 65,
        "num_experts": 4,
        "top_k": 2,
        "hidden_dim": 48,
        "expert_ffn_dim": 32,
        "capacity_factor": 1.5,
        "mode": "forward",
        "dtypes": ["fp16", "bf16"],
        "histograms": ["balanced", "skewed"],
    }], "moe_layer_forward": {
        "fp16": {"relative_l2": 0, "normalized_max_abs": 0},
        "bf16": {"relative_l2": 0, "normalized_max_abs": 0},
    }}))
    monkeypatch.setenv("AISP_MOE_PTX_LAYER_ACCURACY_POLICY", str(path))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_old_and_restored_absolute_bounds_accept_real_zero_output(dtype):
    with torch.random.fork_rng():
        torch.manual_seed(314159)
        state = build_state(_workload(dtype), torch.device("cpu"))
    with torch.inference_mode():
        actual = run_layer_baseline(state, _workload(dtype))
    assert actual.count_nonzero() > 0
    assert torch.allclose(torch.zeros_like(actual), actual, rtol=.05, atol=.2)
    assert torch.allclose(torch.zeros_like(actual), actual, rtol=.02, atol=.02)
    with pytest.raises(AssertionError, match="omitted"):
        measure_layer_output_errors(torch.zeros_like(actual), actual)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("histogram", ["balanced", "skewed"])
def test_real_cpu_full_paths_and_independent_snapshot_reference(dtype, histogram):
    workload = _workload(dtype, histogram)
    state = build_state(workload, torch.device("cpu"))
    snapshot = snapshot_layer_inputs(state)
    expected = reference_layer_forward(snapshot, torch.device("cpu"))
    with torch.inference_mode():
        baseline, candidate = run_layer_baseline(state, workload), run_layer_cuda(state, workload)
    for output in (baseline, candidate):
        assert output.shape == (65, 48)
        assert measure_layer_output_errors(output, expected) == {"relative_l2": 0, "normalized_max_abs": 0}
    for key, tensor in snapshot.items():
        original = getattr(state, key)
        assert tensor.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
    state.x.zero_(); state.gate_proj.zero_(); state.expert_indices.zero_()
    torch.testing.assert_close(reference_layer_forward(snapshot, torch.device("cpu")), expected, rtol=0, atol=0)


@pytest.mark.parametrize("backend", ["baseline", "cuda"])
def test_real_cpu_benchmark_payload_is_complete_and_owns_original_inputs(backend, monkeypatch, tmp_path):
    _strict_test_policy(monkeypatch, tmp_path)
    bench = MoECudaPtxBenchmark(target="moe_layer", backend=backend, label="cpu-contract")
    bench.device = torch.device("cpu")  # Execute actual Torch math; not a simulated CUDA result.
    bench.workload = _workload()
    bench.setup()
    try:
        bench.benchmark_fn(); bench.capture_verification_payload()
        assert bench.validate_result() is None
        assert bench.get_verify_output().shape == (65, 48)
        assert bench.get_verify_output().untyped_storage().data_ptr() != bench.outputs.untyped_storage().data_ptr()
        inputs = bench.get_verify_inputs()
        assert inputs["x"].shape == (65, 48)
        assert inputs["expert_indices"].shape == (65, 2)
        assert inputs["gate_proj"].shape == (4, 48, 32)
        before = inputs["x"].clone()
        bench.state.x.zero_()
        torch.testing.assert_close(inputs["x"], before, rtol=0, atol=0)
        # Last output element is beyond both dimensions of the old32x32 slice.
        bench.outputs[-1, -1] += 1
        with pytest.raises(AssertionError, match="accuracy policy"):
            bench.capture_verification_payload()
        assert bench.validate_result() is not None
    finally:
        bench.teardown()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fault", ["zero", "dropped_row", "last_element", "cancel", "half_scale", "nan", "alias", "short"])
def test_production_comparator_rejects_corrupt_full_outputs(fault, dtype, monkeypatch):
    monkeypatch.delenv("AISP_MOE_PTX_LAYER_ACCURACY_POLICY", raising=False)
    limits = load_layer_accuracy_limits(dtype)
    assert limits.relative_l2 == torch.finfo(dtype).eps
    assert limits.normalized_max_abs == 2 * torch.finfo(dtype).eps
    expected = (
        torch.arange(1, 65 * 48 + 1, dtype=torch.float32).reshape(65, 48) * 1e-3
    ).to(dtype)
    actual = expected.clone()
    if fault == "zero": actual.zero_()
    elif fault == "dropped_row": actual[-1].zero_()
    elif fault == "last_element": actual[-1, -1] += 1
    elif fault == "cancel": actual[0, 0] += 1; actual[-1, -1] -= 1
    elif fault == "half_scale": actual.mul_(0.5)
    elif fault == "nan": actual[-1, -1] = float("nan")
    elif fault == "alias": actual = expected
    elif fault == "short": actual = actual[:32, :32]
    with pytest.raises(AssertionError):
        limits.check(measure_layer_output_errors(actual, expected))


def test_packaged_policy_and_invalid_overrides_fail_closed(monkeypatch, tmp_path):
    monkeypatch.delenv("AISP_MOE_PTX_LAYER_ACCURACY_POLICY", raising=False)
    assert load_layer_accuracy_limits(torch.float16) == LayerAccuracyLimits(2**-10, 2**-9)
    assert load_layer_accuracy_limits(torch.bfloat16) == LayerAccuracyLimits(2**-7, 2**-6)
    for dtype in (torch.float16, torch.bfloat16):
        load_layer_accuracy_limits(dtype, workload=_workload(dtype))
    production_default = MoECudaPtxWorkload()
    assert production_default == _default_workload()
    load_layer_accuracy_limits(torch.bfloat16, workload=production_default)
    with pytest.raises(ValueError, match="dtype differs"):
        load_layer_accuracy_limits(torch.float16, workload=_workload(torch.bfloat16))
    with pytest.raises(RuntimeError, match="outside the reviewed numerical accuracy domain"):
        load_layer_accuracy_limits(torch.float16, workload=_default_workload(torch.float16))
    with pytest.raises(RuntimeError, match="outside the reviewed numerical accuracy domain"):
        load_layer_accuracy_limits(
            torch.bfloat16, workload=_default_workload(histogram="skewed")
        )
    near_default = _default_workload()
    near_default.num_tokens -= 1
    with pytest.raises(RuntimeError, match="outside the reviewed numerical accuracy domain"):
        load_layer_accuracy_limits(torch.bfloat16, workload=near_default)
    # Domain entries are exact workload cells, not independently calibrated
    # dimension lists whose untested Cartesian product can be accepted.
    untested_combination = _workload()
    untested_combination.hidden_dim = 128
    untested_combination.expert_ffn_dim = 64
    with pytest.raises(RuntimeError, match="outside the reviewed numerical accuracy domain"):
        load_layer_accuracy_limits(torch.bfloat16, workload=untested_combination)
    _strict_test_policy(monkeypatch, tmp_path)
    assert load_layer_accuracy_limits(torch.float16) == LayerAccuracyLimits(0, 0)
    assert load_layer_accuracy_limits(torch.bfloat16) == LayerAccuracyLimits(0, 0)
    invalid = tmp_path / "old-schema.json"
    invalid.write_text(json.dumps({"schema_version": 2, "moe_layer_forward": {}}))
    monkeypatch.setenv("AISP_MOE_PTX_LAYER_ACCURACY_POLICY", str(invalid))
    with pytest.raises(ValueError, match="schema_version=3"):
        load_layer_accuracy_limits(torch.float16)
    malformed_domain = json.loads(
        (Path(__file__).parents[1] / "labs/moe_cuda_ptx/layer_accuracy_policy.json").read_text()
    )
    malformed_domain["validated_workloads"][0]["num_tokens"] = False
    malformed = tmp_path / "malformed-domain.json"
    malformed.write_text(json.dumps(malformed_domain))
    monkeypatch.setenv("AISP_MOE_PTX_LAYER_ACCURACY_POLICY", str(malformed))
    with pytest.raises(ValueError, match="positive integer"):
        load_layer_accuracy_limits(torch.float16)
    monkeypatch.delenv("AISP_MOE_PTX_LAYER_ACCURACY_POLICY")
    with pytest.raises(ValueError, match="does not cover"):
        load_layer_accuracy_limits(torch.float32)
    for bad in (False, True, -1, 1, float("inf"), float("nan"), "0", None):
        with pytest.raises(ValueError): LayerAccuracyLimits(bad, 0)
    for bad in (False, True, -1, float("inf"), float("nan"), "0", None):
        with pytest.raises(ValueError):
            LayerAccuracyLimits(0.5, 0.5).check(
                {"relative_l2": bad, "normalized_max_abs": 0}
            )
    with pytest.raises(ValueError):
        LayerAccuracyLimits(0, 0).check({})


def test_near_default_unreviewed_domain_fails_before_workload_allocation(monkeypatch):
    import labs.moe_cuda_ptx.moe_cuda_ptx_common as common

    monkeypatch.delenv("AISP_MOE_PTX_LAYER_ACCURACY_POLICY", raising=False)
    monkeypatch.setattr(
        common,
        "build_state",
        lambda *_args, **_kwargs: pytest.fail("unreviewed workload allocated state"),
    )
    benchmark = common.MoECudaPtxBenchmark(
        target="moe_layer", backend="baseline", label="unreviewed-near-default"
    )
    benchmark.workload.num_tokens -= 1
    with pytest.raises(RuntimeError, match="outside the reviewed numerical accuracy domain"):
        benchmark.setup()


def test_cli_rejects_unimplemented_topk_and_explicit_zero_dimensions():
    for top_k in (0, 1, 3):
        with pytest.raises(ValueError, match="top_k=2"):
            apply_workload_overrides(_workload(), ["--top-k", str(top_k)])
    for option in ("--num-tokens", "--num-experts", "--hidden-dim", "--expert-ffn-dim", "--capacity-factor"):
        with pytest.raises(ValueError): apply_workload_overrides(_workload(), [option, "0"])


def test_inputs_follow_harness_seed():
    with torch.random.fork_rng():
        torch.manual_seed(8); first = build_state(_workload(), torch.device("cpu"))
        torch.manual_seed(9); second = build_state(_workload(), torch.device("cpu"))
    assert not torch.equal(first.x, second.x)
    assert not torch.equal(first.gate_proj, second.gate_proj)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA layer math and reviewed policy required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("histogram", ["balanced", "skewed"])
def test_real_cuda_full_layer_with_reviewed_policy(dtype, histogram):
    load_layer_accuracy_limits(dtype)
    workload = MoECudaPtxWorkload(num_tokens=129, num_experts=4, hidden_dim=128,
        expert_ffn_dim=64, capacity_factor=1.5, dtype=dtype, histogram=histogram)
    for seed in (314159, 8675309):
        for backend in ("baseline", "cuda"):
            with torch.random.fork_rng():
                torch.manual_seed(seed)
                bench = MoECudaPtxBenchmark(target="moe_layer", backend=backend, label="cuda-acceptance")
                bench.workload = workload
                bench.setup()
            try:
                for _ in range(3):
                    bench.benchmark_fn(); torch.cuda.synchronize()
                    bench.capture_verification_payload()
                    assert bench.validate_result() is None
                    assert bench.get_verify_output().shape == (129, 128)
                bench.outputs[-1].zero_()
                with pytest.raises(AssertionError, match="omitted"):
                    bench.capture_verification_payload()
            finally:
                bench.teardown()


def test_calibration_retains_invalid_configuration_attempt(tmp_path):
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[2] / 'docs/audits/2026-08-30/evidence/moe-ptx/calibrate_layer_accuracy.py'
    output = tmp_path / 'failed-attempt.json'
    result = subprocess.run([sys.executable, str(script), '--output', str(output), '--num-tokens', '0'],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2, result.stderr
    report = json.loads(output.read_text())
    assert report['status'] == 'FAILED_NOT_ACCEPTED' and report['exception_type'] == 'ValueError'
    assert not report['gpu_executed'] and not report['accepted']
    original = output.read_bytes()
    repeated = subprocess.run([sys.executable, str(script), '--output', str(output), '--num-tokens', '0'],
                              capture_output=True, text=True, timeout=30)
    assert repeated.returncode != 0
    assert output.read_bytes() == original
