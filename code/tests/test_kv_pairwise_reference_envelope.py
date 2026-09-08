from __future__ import annotations

import json
from copy import deepcopy

import pytest
import torch

from core.benchmark.verification import resolve_output_tolerances
from core.benchmark.verify_runner import VerifyRunner
from labs.kv_cache_compression.accuracy import (
    DEFAULT_POLICY_PATH,
    ENGINEERING_CEILINGS,
    PairwiseEnvelope,
    assert_cache_accuracy_evidence,
    cache_accuracy_evidence,
    load_accuracy_contract,
    load_accuracy_policy,
    pairwise_absolute_tolerance,
)
from labs.kv_cache_compression.kv_cache_common import KVCache


def _cache(k: list[float], v: list[float]) -> KVCache:
    return KVCache(
        torch.tensor(k, dtype=torch.float64),
        torch.tensor(v, dtype=torch.float64),
    )


def _raw(cache: KVCache) -> torch.Tensor:
    return torch.cat((cache.cache_k.reshape(-1), cache.cache_v.reshape(-1)))


def test_policy_derives_pairwise_envelope_from_unchanged_independent_limits(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    policy = load_accuracy_policy(DEFAULT_POLICY_PATH)
    assert "pairwise_rtol" not in policy["variants"]["fp8"]
    assert "pairwise_atol" not in policy["variants"]["nvfp4"]

    monkeypatch.setenv("AISP_KV_CACHE_ACCURACY_POLICY", str(DEFAULT_POLICY_PATH))
    fp8_limits, fp8_envelope = load_accuracy_contract("fp8")
    nvfp4_limits, nvfp4_envelope = load_accuracy_contract("nvfp4")
    assert fp8_limits.normalized_max_abs == 2**-4
    assert nvfp4_limits.normalized_max_abs == 2**-2
    assert fp8_envelope == nvfp4_envelope
    assert fp8_envelope.normalized_max_abs == 2**-4 + 2**-2
    assert fp8_envelope.output_rtol == 0

    tighter = deepcopy(policy)
    tighter["variants"]["fp8"].update(relative_l2=2**-5, normalized_max_abs=2**-5)
    tighter["variants"]["nvfp4"].update(relative_l2=2**-3, normalized_max_abs=2**-3)
    tighter_path = tmp_path / "tighter.json"
    tighter_path.write_text(json.dumps(tighter))
    monkeypatch.setenv("AISP_KV_CACHE_ACCURACY_POLICY", str(tighter_path))
    _, tighter_envelope = load_accuracy_contract("nvfp4")
    assert tighter_envelope.normalized_max_abs == 2**-5 + 2**-3


@pytest.mark.parametrize(
    "mutation,match",
    [
        (("fp8", "pairwise_rtol", 0.25), "obsolete raw allclose fields"),
        (("nvfp4", "normalized_max_abs", False), "must be numeric, not boolean"),
    ],
)
def test_policy_rejects_obsolete_or_non_numeric_pairwise_inputs(
    tmp_path, mutation: tuple[str, str, object], match: str
) -> None:
    policy = json.loads(DEFAULT_POLICY_PATH.read_text())
    variant, name, value = mutation
    policy["variants"][variant][name] = value
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match=match):
        load_accuracy_policy(path)


def test_reference_triangle_envelope_handles_cancellation_and_rejects_corruption() -> None:
    reference = _cache([16.0, 0.0], [8.0, 0.0])
    fp8 = _cache([16.0, 1.0], [8.0, 0.5])
    nvfp4 = _cache([16.0, -4.0], [8.0, -2.0])

    fp8_evidence = assert_cache_accuracy_evidence(
        fp8, reference, ENGINEERING_CEILINGS["fp8"]
    )
    nvfp4_evidence = assert_cache_accuracy_evidence(
        nvfp4, reference, ENGINEERING_CEILINGS["nvfp4"]
    )
    assert fp8_evidence.reference_max_abs == nvfp4_evidence.reference_max_abs == 16
    assert set(fp8_evidence.metrics) == {
        "cache_k.relative_l2",
        "cache_k.normalized_max_abs",
        "cache_v.relative_l2",
        "cache_v.normalized_max_abs",
    }

    raw_fp8, raw_nvfp4 = _raw(fp8), _raw(nvfp4)
    assert raw_fp8.numel() == fp8.cache_k.numel() + fp8.cache_v.numel()
    assert raw_nvfp4.numel() == nvfp4.cache_k.numel() + nvfp4.cache_v.numel()
    assert not torch.allclose(raw_fp8, raw_nvfp4, rtol=0.25, atol=0.0625)

    envelope = PairwiseEnvelope(normalized_max_abs=2**-4 + 2**-2)
    absolute = pairwise_absolute_tolerance(envelope, fp8_evidence.reference_max_abs)
    tolerance_map = {"output": (0.0, absolute)}
    assert absolute == 5.0
    assert VerifyRunner().compare_perf_outputs(raw_fp8, raw_nvfp4, tolerance_map).passed

    corrupted = _cache([16.0, -4.125], [8.0, -2.0])
    with pytest.raises(AssertionError, match="KV cache accuracy failed"):
        assert_cache_accuracy_evidence(
            corrupted, reference, ENGINEERING_CEILINGS["nvfp4"]
        )
    corrupted_raw = _raw(corrupted)
    assert not VerifyRunner().compare_perf_outputs(
        raw_fp8, corrupted_raw, tolerance_map
    ).passed


def test_zero_reference_is_exact_only_and_invalid_scale_fails_closed() -> None:
    reference = _cache([0.0, 0.0], [0.0, 0.0])
    exact = _cache([0.0, 0.0], [0.0, 0.0])
    evidence = cache_accuracy_evidence(exact, reference)
    assert evidence.reference_max_abs == 0
    assert all(value == 0 for value in evidence.metrics.values())

    envelope = PairwiseEnvelope(normalized_max_abs=2**-4 + 2**-2)
    assert pairwise_absolute_tolerance(envelope, 0) == 0
    assert VerifyRunner().compare_perf_outputs(
        _raw(exact), _raw(exact), {"output": (0.0, 0.0)}
    ).passed

    corrupt = _cache([0.0, 0.0], [0.0, 1.0])
    with pytest.raises(AssertionError, match="KV cache accuracy failed"):
        assert_cache_accuracy_evidence(
            corrupt, reference, ENGINEERING_CEILINGS["nvfp4"]
        )
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            pairwise_absolute_tolerance(envelope, bad)


def test_pairwise_tolerance_maps_must_match_exactly() -> None:
    expected = {"output": (0.0, 5.0)}
    assert resolve_output_tolerances(
        expected,
        dict(expected),
        baseline_output_names={"output"},
        optimized_output_names={"output"},
    ) == expected
    with pytest.raises(ValueError, match="maps must be identical"):
        resolve_output_tolerances(
            expected,
            {"output": (0.0, 4.999)},
            baseline_output_names={"output"},
            optimized_output_names={"output"},
        )
