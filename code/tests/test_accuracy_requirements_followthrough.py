from __future__ import annotations

import json
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def test_kv_checked_in_policy_uses_format_ceiling_and_rejects_widening(tmp_path: Path) -> None:
    from labs.kv_cache_compression.accuracy import (
        DEFAULT_POLICY_PATH,
        ENGINEERING_CEILINGS,
        load_accuracy_policy,
    )

    policy = load_accuracy_policy(DEFAULT_POLICY_PATH)
    assert policy["variants"]["fp8"]["relative_l2"] == 2**-4
    assert policy["variants"]["nvfp4"]["relative_l2"] == 2**-2

    widened = deepcopy(policy)
    widened["variants"]["fp8"]["relative_l2"] = ENGINEERING_CEILINGS["fp8"].relative_l2 + 0.001
    path = tmp_path / "widened.json"
    path.write_text(json.dumps(widened))
    with pytest.raises(ValueError, match="exceeds the source-defined engineering ceiling"):
        load_accuracy_policy(path)


@pytest.mark.parametrize("variant,corruption", [("fp8", 0.07), ("nvfp4", 0.26)])
def test_kv_format_requirements_reject_meaningful_full_cache_corruption(
    variant: str, corruption: float
) -> None:
    from labs.kv_cache_compression.accuracy import ENGINEERING_CEILINGS, assert_cache_accuracy
    from labs.kv_cache_compression.kv_cache_common import KVCache

    reference = KVCache(torch.ones(2, 4, 2, 8), -torch.ones(2, 4, 2, 8))
    actual = KVCache(reference.cache_k.clone() + corruption, reference.cache_v.clone() - corruption)
    with pytest.raises(AssertionError, match="KV cache accuracy failed"):
        assert_cache_accuracy(actual, reference, ENGINEERING_CEILINGS[variant])


def test_kv_edge_cohorts_preserve_shape_and_create_stress_patterns() -> None:
    from labs.kv_cache_compression.calibrate_accuracy import _apply_input_cohort

    alternating = SimpleNamespace(
        hidden_dim=8,
        tensor_dtype=torch.bfloat16,
        device=torch.device("cpu"),
        prefill_inputs=[torch.empty(2, 3, 8, dtype=torch.bfloat16)],
        decode_inputs=[torch.empty(2, 2, 8, dtype=torch.bfloat16)],
    )
    shapes = [tensor.shape for tensor in alternating.prefill_inputs + alternating.decode_inputs]
    _apply_input_cohort(alternating, "alternating")
    assert [tensor.shape for tensor in alternating.prefill_inputs + alternating.decode_inputs] == shapes
    torch.testing.assert_close(
        alternating.prefill_inputs[0][0, 0],
        -alternating.prefill_inputs[0][0, 1],
        rtol=0,
        atol=0,
    )

    _apply_input_cohort(alternating, "sparse_outlier")
    for tensor in alternating.prefill_inputs + alternating.decode_inputs:
        assert torch.count_nonzero(tensor[..., 2:]) == 0
        assert torch.all(tensor[..., 0] == 1)
        assert torch.all(tensor[..., 1] == -1)


def test_kv_calibration_retains_unsupported_host_failure(tmp_path: Path) -> None:
    if torch.cuda.is_available():
        pytest.skip("This control exercises the unsupported CPU-host receipt")
    output = tmp_path / "failure.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "labs.kv_cache_compression.calibrate_accuracy",
            "--variant",
            "fp8",
            "--cohort",
            "nominal",
            "--seed",
            "2026",
            "--output",
            str(output),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    receipt = json.loads(output.read_text())
    assert receipt["status"] == "failure_not_accepted"
    assert receipt["error_type"] == "RuntimeError"
    assert "actual CUDA" in receipt["error"]


def _kv_receipts(policy: dict) -> list[dict]:
    from labs.kv_cache_compression.accuracy import REFERENCE_ID, WORKLOAD
    from labs.kv_cache_compression.qualify_accuracy import required_cases

    workload = {key: value for key, value in WORKLOAD.items() if key != "storage_dtype"}
    return [{
        "schema_version": 2,
        "status": "measurement_only_not_accepted",
        "variant": variant,
        "cohort": cohort,
        "seed": seed,
        "reference_id": REFERENCE_ID,
        "git_commit": "0123456789abcdef",
        "torch": "2.9.1",
        "cuda": "13.0",
        "transformer_engine": "2.18.0",
        "gpu": "NVIDIA B200",
        "compute_capability": [10, 0],
        "workload": workload,
        "metrics": {
            "cache_k.relative_l2": 0.0,
            "cache_k.normalized_max_abs": 0.0,
            "cache_v.relative_l2": 0.0,
            "cache_v.normalized_max_abs": 0.0,
        },
    } for variant, cohort, seed in sorted(required_cases(policy))]


def test_kv_qualification_requires_all_holdouts_and_retains_failure() -> None:
    from labs.kv_cache_compression.accuracy import DEFAULT_POLICY_PATH, load_accuracy_policy
    from labs.kv_cache_compression.qualify_accuracy import assess_receipts

    policy = load_accuracy_policy(DEFAULT_POLICY_PATH)
    receipts = _kv_receipts(policy)
    assert assess_receipts(policy, receipts)["status"] == "qualified_arithmetic_gate"

    damaged = deepcopy(receipts)
    damaged[-1]["metrics"]["cache_v.normalized_max_abs"] = 0.5
    result = assess_receipts(policy, damaged)
    assert result["status"] == "failed_arithmetic_gate"
    assert any("cache_v.normalized_max_abs" in failure for failure in result["failures"])


def test_ozaki_checked_in_policy_rejects_widening(tmp_path: Path) -> None:
    from labs.ozaki_scheme.accuracy_policy import DEFAULT_POLICY_PATH, load_accuracy_policy

    policy = load_accuracy_policy(DEFAULT_POLICY_PATH)
    assert policy["variants"]["dynamic"]["relative_l2"] == 2**-5
    assert policy["variants"]["fixed"]["relative_l2"] == 2**-12
    widened = deepcopy(policy)
    widened["variants"]["fixed"]["normalized_max_abs"] = 0.001
    path = tmp_path / "widened.json"
    path.write_text(json.dumps(widened))
    with pytest.raises(ValueError, match="exceeds the source-defined engineering ceiling"):
        load_accuracy_policy(path)


@pytest.fixture(scope="module")
def ozaki_long_double_reference_probe(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    directory = tmp_path_factory.mktemp("ozaki_long_double_reference")
    source = directory / "probe.cpp"
    source.write_text(r'''
#include "accuracy.h"
#include <cmath>
#include <iostream>
int main() {
    const double a[] = {1, 2, 3, 4, 5, 6};
    const double b[] = {7, 8, 9, 10, 11, 12};
    const double expected[] = {58, 64, 139, 154};
    std::vector<double> reference;
    try {
        reference = ozaki_scheme::reference_gemm_long_double(a, b, 2, 2, 3);
    } catch (const std::exception& error) {
        std::cout << error.what() << "\n";
        return 4;
    }
    const auto exact = ozaki_scheme::measure_accuracy(reference.data(), expected, 4);
    if (exact.relative_l2 != 0 || exact.normalized_max_abs != 0) return 2;
    auto corrupt = reference;
    corrupt.back() += 1;
    const auto damaged = ozaki_scheme::measure_accuracy(corrupt.data(), reference.data(), 4);
    try {
        ozaki_scheme::assert_accuracy(damaged, 1.0 / 4096.0, 1.0 / 2048.0);
    } catch (const std::exception&) {
        std::cout << "corruption rejected\n";
        return 0;
    }
    return 3;
}
''')
    completed = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-pedantic", "-I",
         str(Path(__file__).parents[1] / "labs" / "ozaki_scheme"), str(source), "-o", str(directory / "probe")],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    return directory / "probe"


def test_ozaki_cpu_long_double_reference_and_corruption_control(
    ozaki_long_double_reference_probe: Path,
) -> None:
    completed = subprocess.run([str(ozaki_long_double_reference_probe)], capture_output=True, text=True)
    assert completed.returncode in (0, 4), completed.stdout + completed.stderr
    if completed.returncode == 0:
        assert completed.stdout.strip() == "corruption rejected"
    else:
        # arm64 Darwin aliases long double to FP64. The reference must fail
        # closed there; x86_64 B200 qualification executes the wider oracle.
        assert completed.stdout.strip() == "CPU long-double reference requires precision wider than FP64"


def _ozaki_log(case: dict, *, relative_l2: float = 0.0) -> str:
    variant = case["variant"]
    knob = (
        "DYNAMIC_MAX_BITS: 16\nDYNAMIC_OFFSET: -56"
        if variant == "dynamic"
        else "FIXED_BITS: 12"
    )
    return f"""VARIANT: ozaki_{variant}
M: {case['m']}
N: {case['n']}
K: {case['k']}
SEED: {case['seed']}
INPUT_SCALE: {case['input_scale']}
INPUT_PATTERN: {case['input_pattern']}
REFERENCE_MODE: {case['reference_mode']}
GPU_NAME: NVIDIA B200
COMPUTE_CAPABILITY: 10.0
CUDA_RUNTIME_VERSION: 13000
CUBLAS_VERSION: 130000
{knob}
EMULATION_STRATEGY: eager
EMULATION_USED: 1
RETAINED_BITS: 12
RELATIVE_L2_ERROR: {relative_l2}
NORMALIZED_MAX_ABS_ERROR: 0
ACCURACY_STATUS: MEASUREMENT_ONLY_NOT_ACCEPTED
"""


def test_ozaki_qualification_requires_independent_edges_and_rejects_corruption() -> None:
    from labs.ozaki_scheme.accuracy_policy import DEFAULT_POLICY_PATH, load_accuracy_policy
    from labs.ozaki_scheme.qualify_accuracy import _required_cases, assess_logs

    policy = load_accuracy_policy(DEFAULT_POLICY_PATH)
    cases = list(_required_cases(policy).values())
    logs = [_ozaki_log(case) for case in cases]
    assert assess_logs(policy, logs)["status"] == "qualified_arithmetic_gate"

    logs[-1] = _ozaki_log(cases[-1], relative_l2=0.1)
    result = assess_logs(policy, logs)
    assert result["status"] == "failed_arithmetic_gate"
    assert any("relative_l2_error" in failure for failure in result["failures"])
