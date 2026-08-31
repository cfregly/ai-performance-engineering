"""CPU contracts/negative controls; CUDA checks explicitly skip without real hardware."""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace
import weakref

import pytest
import torch

CODE = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("fault", ["correct", "zero", "corrupt", "nan", "input_mutation", "shared_workspace"])
def test_gemm_verification_owns_reference_and_snapshot(fault):
    from labs.nvfp4_gemm.local_eval_submission import _verify_submission
    from labs.nvfp4_gemm.utils import make_match_reference

    a = torch.tensor([[1., 2.], [3., -2.]])
    b = torch.tensor([[2., 1.], [4., 3.]])
    data = (a, b, torch.full((2, 2), -999.))
    workspace = torch.empty_like(data[2])

    def reference(sample):
        sample[2].copy_(sample[0] @ sample[1])
        workspace.copy_(sample[2])  # Same global storage a submission could return.
        return sample[2]

    def submission(sample):
        if fault == "input_mutation":
            sample[0].zero_()
        sample[2].copy_(sample[0] @ sample[1])
        if fault in {"zero", "shared_workspace"}:
            sample[2].zero_()
        elif fault == "corrupt":
            sample[2][0, 0] += 3
        elif fault == "nan":
            sample[2][0, 0] = float("nan")
        if fault == "shared_workspace":
            workspace.copy_(sample[2])
            return workspace
        return sample[2]  # Intentional output/input-C alias in every case.

    before = tuple(x.clone() for x in data)
    ok, _reason = _verify_submission(data, SimpleNamespace(custom_kernel=submission),
        SimpleNamespace(check_implementation=make_match_reference(reference, rtol=1e-3, atol=1e-3)))
    assert ok is (fault == "correct")
    for got, old in zip(data, before):
        torch.testing.assert_close(got, old, rtol=0, atol=0)


def test_gemv_real_child_protocol(tmp_path):
    from labs.nvfp4_gemv.local_eval import _run_official_eval
    # A protocol fixture, not an official benchmark or a GPU success result.
    (tmp_path / "eval.py").write_text(
        "import os,sys\nassert sys.argv[1:] == ['leaderboard','benchmarks.txt']\n"
        "assert os.environ['POPCORN_SEED'] == '123'\n"
        "os.write(int(os.environ['POPCORN_FD']), b'audit.protocol=fixture_only\\n')\n"
        "print('child interpreter ok')\n")
    result = _run_official_eval(work_dir=tmp_path, seed=123)
    assert result[0] == 0
    assert "child interpreter ok" in result[1]
    assert "audit.protocol=fixture_only" in result[1]


@pytest.mark.parametrize("entry", ["-m", "script"])
def test_gemv_cli_help(entry):
    path = "labs/nvfp4_gemv/local_eval.py"
    args = ["-m", "labs.nvfp4_gemv.local_eval"] if entry == "-m" else [path]
    result = subprocess.run([sys.executable, *args, "--help"], cwd=CODE, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "--no-lock" in result.stdout


def test_gemv_scale_identity_lifetime_and_mutation():
    from labs.nvfp4_gemv import optimized_submission as mod
    mod._PACKED_SCALE_CACHE.clear()
    a, b = torch.ones(128, 4, 1), torch.full((128, 4, 1), 2.)
    first = mod._get_packed_scales(a, b, 1)
    assert mod._get_packed_scales(a, b, 1)[0][0] is first[0][0]
    a.add_(3)
    changed = mod._get_packed_scales(a, b, 1)
    torch.testing.assert_close(changed[0][0], mod.to_blocked(a[:, :, 0]))
    assert not torch.equal(first[0][0], changed[0][0])
    distinct = torch.full_like(a, 7.)
    packed = mod._get_packed_scales(distinct, b, 1)
    assert torch.all(packed[0][0] == 7)
    kept = weakref.ref(distinct)
    del distinct
    assert kept() is not None  # Strong identity ownership prevents allocator recycling.
    for i in range(12):
        mod._get_packed_scales(torch.full_like(a, float(i)), b, 1)
        assert len(mod._PACKED_SCALE_CACHE) <= mod._PACKED_SCALE_CACHE_LIMIT
    mod._PACKED_SCALE_CACHE.clear()
    assert kept() is None


def test_gemv_versionless_inference_scales_are_not_cached():
    from labs.nvfp4_gemv import optimized_submission as mod
    mod._PACKED_SCALE_CACHE.clear()
    with torch.inference_mode():
        a, b = torch.ones(128, 4, 1), torch.ones(128, 4, 1)
        first = mod._get_packed_scales(a, b, 1)
        a.fill_(9)
        second = mod._get_packed_scales(a, b, 1)
    assert not mod._PACKED_SCALE_CACHE
    assert torch.all(first[0][0] == 1) and torch.all(second[0][0] == 9)


def test_vendor_recipe_default_is_supported_not_fake_calibration():
    path = CODE / "third_party/TransformerEngine/transformer_engine/common/recipe/__init__.py"
    spec = importlib.util.spec_from_file_location("audit_nvfp4_recipe", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        recipe = module.NVFP4BlockScaling()
        unsupported = {"calibration_steps", "amax_history_len", "fp4_tensor_block"}
        assert unsupported.isdisjoint(inspect.signature(module.NVFP4BlockScaling).parameters)
        assert unsupported.isdisjoint(vars(recipe))
        assert recipe.fp4_format is module.Format.E2M1
    finally:
        sys.modules.pop(spec.name, None)


def _kv_pair():
    from labs.kv_cache_compression.kv_cache_common import KVCache
    k = torch.arange(1, 1 + 2 * 7 * 3 * 8, dtype=torch.float32).reshape(2, 7, 3, 8) / 100
    v = k.flip(1).neg()
    return KVCache(k.clone(), v.clone()), KVCache(k, v)


@pytest.mark.parametrize("fault", ["correct", "zero", "tail_corrupt", "nan", "alias", "shape"])
def test_full_cache_actual_comparator_rejects_negative_controls(fault):
    from labs.kv_cache_compression.accuracy import AccuracyLimits, assert_cache_accuracy
    actual, ref = _kv_pair()
    if fault == "zero":
        actual.cache_k.zero_(); actual.cache_v.zero_()
    elif fault == "tail_corrupt":
        actual.cache_v[-1, -1, -1, -1] += 5
    elif fault == "nan":
        actual.cache_k[-1, -1, -1, -1] = float("nan")
    elif fault == "alias":
        actual.cache_k = ref.cache_k
    elif fault == "shape":
        actual.cache_k = actual.cache_k[:, :1]
    # Exact synthetic fixture policy only; deliberately NOT workload calibration.
    limits = AccuracyLimits(0, 0, 0, 0)
    if fault == "correct":
        assert all(value == 0 for value in assert_cache_accuracy(actual, ref, limits).values())
    else:
        with pytest.raises(AssertionError):
            assert_cache_accuracy(actual, ref, limits)


def test_cache_policy_missing_invalid_and_explicit(tmp_path, monkeypatch):
    from labs.kv_cache_compression.accuracy import AccuracyLimits, load_accuracy_limits
    monkeypatch.delenv("AISP_KV_CACHE_ACCURACY_POLICY", raising=False)
    with pytest.raises(RuntimeError, match="uncalibrated"):
        load_accuracy_limits("nvfp4")
    for invalid in (1., float("nan"), float("inf"), -1.):
        with pytest.raises(ValueError):
            AccuracyLimits(invalid, 0, 0, 0)
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"schema_version": 1, "fp8": {
        "relative_l2": 0, "normalized_max_abs": 0, "pairwise_rtol": 0, "pairwise_atol": 0}}))
    monkeypatch.setenv("AISP_KV_CACHE_ACCURACY_POLICY", str(path))
    assert load_accuracy_limits("fp8") == AccuracyLimits(0, 0, 0, 0)


def test_cache_reference_computes_every_token_head_and_channel():
    from labs.kv_cache_compression.accuracy import reference_cache
    from labs.kv_cache_compression.kv_cache_common import KVCache
    generator = torch.Generator().manual_seed(17)
    model = SimpleNamespace(hidden_dim=8, num_heads=2, head_dim=4,
        ln=torch.nn.LayerNorm(8, dtype=torch.bfloat16), qkv=torch.nn.Linear(8, 24, dtype=torch.bfloat16))
    with torch.no_grad():
        model.qkv.weight.copy_(torch.randn(24, 8, generator=generator).to(torch.bfloat16))
        model.qkv.bias.copy_(torch.randn(24, generator=generator).to(torch.bfloat16))
    tokens = torch.randn(2, 260, 8, generator=generator).to(torch.bfloat16)
    cache = KVCache(torch.zeros(2, 260, 2, 4, dtype=torch.bfloat16), torch.zeros(2, 260, 2, 4, dtype=torch.bfloat16))
    reference = reference_cache(model, [(tokens[:, :130], 0), (tokens[:, 130:], 130)], cache)
    with torch.no_grad():
        expected = model.qkv(model.ln(tokens)).reshape(2, 260, 3, 2, 4)
    torch.testing.assert_close(reference.cache_k, expected[:, :, 1], rtol=0, atol=0)
    torch.testing.assert_close(reference.cache_v, expected[:, :, 2], rtol=0, atol=0)
    assert cache.cache_k.count_nonzero() == 0
    with pytest.raises(ValueError, match="entire"):
        reference_cache(model, [(tokens[:, :130], 0)], cache)


def test_cache_metrics_measure_actual_bf16_bytes():
    from labs.kv_cache_compression.baseline_kv_cache import BaselineKVCacheBenchmark
    from labs.kv_cache_compression.optimized_kv_cache_nvfp4 import OptimizedKVCacheNVFP4Benchmark
    actual, _ = _kv_pair()
    actual.cache_k = actual.cache_k.bfloat16(); actual.cache_v = actual.cache_v.bfloat16()
    for cls in (BaselineKVCacheBenchmark, OptimizedKVCacheNVFP4Benchmark):
        bench = cls.__new__(cls)  # Exercise metrics with real CPU tensors; no CUDA setup claim.
        bench.cache = actual; bench._accuracy_metrics = {}; bench.nvfp4_active = True
        bench.batch_size, bench.prefill_seq, bench.decode_seq, bench.decode_steps, bench.hidden_dim = 2, 5, 1, 2, 24
        metrics = bench.get_custom_metrics()
        assert metrics["kv_cache.storage_bytes"] == 2 * actual.cache_k.numel() * 2
        assert metrics["kv_cache.compression_ratio"] == 1
        assert metrics["kv_cache.storage_bits_per_element"] == 16
        assert bench.get_optimization_goal() == "speed"


def _group_fixture():
    # Contains every signed E2M1 nibble and differing block scales. No custom packing helper.
    raw = torch.tensor([0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE] * 2, dtype=torch.uint8)
    a = raw.reshape(1, 16, 1).repeat(2, 1, 1)
    b = raw.flip(0).reshape(1, 16, 1).repeat(3, 1, 1)
    sa = torch.tensor([1., 2.]).reshape(1, 2, 1).repeat(2, 1, 1)
    sb = torch.tensor([2., 1.]).reshape(1, 2, 1).repeat(3, 1, 1)
    return ([(a, b, torch.empty(2, 3, 1, dtype=torch.float16))], [(sa, sb)], [(None, None)], [(2, 3, 32, 1)])


def test_group_independent_fp4_math_matches_scalar_oracle():
    from labs.nvfp4_group_gemm.reference_math import reference_group_gemm
    data = _group_fixture()
    a, b, c = data[0][0]; sa, sb = data[1][0]
    lut = [0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6]
    expected = torch.empty_like(c)
    for row in range(2):
        for col in range(3):
            total = 0.
            for k in range(32):
                ac = (int(a[row, k // 2, 0]) >> (4 * (k % 2))) & 15
                bc = (int(b[col, k // 2, 0]) >> (4 * (k % 2))) & 15
                total += lut[ac] * float(sa[row, k // 16, 0]) * lut[bc] * float(sb[col, k // 16, 0])
            expected[row, col, 0] = total
    result = reference_group_gemm(data, write_output=False)[0]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    assert result.untyped_storage().data_ptr() != c.untyped_storage().data_ptr()


def test_group_payload_covers_fused_requests_not_only_return_value():
    from labs.nvfp4_group_gemm.reference_math import reference_group_gemm
    from labs.nvfp4_group_gemm.nvfp4_group_gemm_common import GroupGemmCase, NVFP4GroupGemmBenchmark
    inputs = [_group_fixture(), _group_fixture()]
    expected = [reference_group_gemm(data, write_output=False)[0] for data in inputs]
    for data in inputs:
        reference_group_gemm(data)
    bench = NVFP4GroupGemmBenchmark(case=GroupGemmCase("cpu_fixture", (2,), (3,), (32,), 1, 17),
        custom_kernel=reference_group_gemm, inputs_per_iteration=2, capture_iter_graph=False)
    bench._canonical_data = inputs
    bench.data_list = [inputs[0]]  # Fused wrapper's compressed Python call list.
    bench._last_output = [inputs[0][0][0][2]]
    bench._reference_outputs = expected
    bench._verify_output = torch.empty(12, dtype=torch.float16)
    bench.capture_verification_payload()
    torch.testing.assert_close(bench.get_verify_output(), torch.cat([r.reshape(-1) for r in expected]))
    inputs[1][0][0][2][-1, -1, -1] += 8  # Outside the returned first request.
    with pytest.raises(AssertionError):
        bench.capture_verification_payload()


def test_all_group_baselines_use_independent_math():
    from labs.nvfp4_group_gemm.reference_math import reference_group_gemm, prepare_reference
    for path in sorted((CODE / "labs/nvfp4_group_gemm").glob("baseline_nvfp4_group_gemm*.py")):
        module = importlib.import_module(f"labs.nvfp4_group_gemm.{path.stem}")
        benchmark = module.get_benchmark()
        assert benchmark._custom_kernel is reference_group_gemm
        assert benchmark._prepare is prepare_reference


@pytest.mark.parametrize("fault", ["correct", "zero", "corrupt", "nan", "alias"])
def test_group_actual_comparator_negative_controls(fault):
    from labs.nvfp4_group_gemm.reference_math import reference_group_gemm, assert_group_outputs
    ref = reference_group_gemm(_group_fixture(), write_output=False)
    actual = [ref[0].clone()]
    assert ref[0].count_nonzero() > 0
    if fault == "zero": actual[0].zero_()
    if fault == "corrupt": actual[0][-1, -1, -1] += 8
    if fault == "nan": actual[0][0, 0, 0] = float("nan")
    if fault == "alias": actual = ref
    if fault == "correct": assert_group_outputs(actual, ref)
    else:
        with pytest.raises(AssertionError): assert_group_outputs(actual, ref)


@pytest.fixture(scope="module")
def ozaki_accuracy_binary(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler: pytest.skip("C++ compiler unavailable")
    path = tmp_path_factory.mktemp("audit_ozaki")
    source = path / "probe.cpp"
    source.write_text(r'''
#include "accuracy.h"
#include <iostream>
#include <string>
int main(int argc, char** argv) {
    const double ref[] = {1e-5, -2e-5, 3e-5, -4e-5};
    double got[] = {1e-5, -2e-5, 3e-5, -4e-5};
    std::string mode = argv[1];
    if (mode == "zero") for (double& x : got) x = 0;
    if (mode == "cancel") { got[0] += 1e-5; got[1] -= 1e-5; }
    if (mode == "nan") got[3] = std::numeric_limits<double>::quiet_NaN();
    try {
        auto metrics = ozaki_scheme::measure_accuracy(mode == "alias" ? ref : got, ref, 4);
        ozaki_scheme::assert_accuracy(metrics, mode == "uncalibrated" ? NAN : 0.0, 0.0);
        std::cout << metrics.relative_l2 << " " << metrics.normalized_max_abs << "\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << e.what() << "\n"; return 1; }
}
''')
    result = subprocess.run([compiler, "-std=c++17", "-Wall", "-Wextra", "-pedantic",
        "-I", str(CODE / "labs/ozaki_scheme"), str(source), "-o", str(path / "probe")], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return path / "probe"


@pytest.mark.parametrize("mode", ["correct", "zero", "cancel", "nan", "alias", "uncalibrated"])
def test_ozaki_real_cpp_comparator(ozaki_accuracy_binary, mode):
    result = subprocess.run([str(ozaki_accuracy_binary), mode], capture_output=True, text=True)
    assert result.returncode == (0 if mode == "correct" else 1), result.stdout + result.stderr


def test_ozaki_policy_is_explicit_and_fail_closed(monkeypatch, tmp_path):
    from labs.ozaki_scheme.accuracy_policy import configured_accuracy
    monkeypatch.delenv("AISP_OZAKI_ACCURACY_POLICY", raising=False)
    assert configured_accuracy("dynamic") == ([], (0, 0))
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"schema_version": 1, "fixed": {
        "relative_l2": 0, "normalized_max_abs": 0, "checksum_rtol": 0, "checksum_atol": 0}}))
    monkeypatch.setenv("AISP_OZAKI_ACCURACY_POLICY", str(policy))
    assert configured_accuracy("fixed")[0] == ["--relative-l2-limit", "0", "--normalized-max-abs-limit", "0"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Actual CUDA required; CPU checks do not qualify GPU numerics")
def test_group_real_cuda_custom_kernel_against_independent_reference(monkeypatch):
    from labs.nvfp4_group_gemm.nvfp4_group_gemm_inputs import generate_input
    from labs.nvfp4_group_gemm.reference_math import reference_group_gemm, assert_group_outputs
    from labs.nvfp4_group_gemm.custom_cuda_submission import prepare_custom_cuda, custom_kernel_custom_cuda
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100 family required for tcgen05 group kernel")
    monkeypatch.setenv("AISP_NVFP4_GROUP_GEMM_FUSE_INPUTS", "0")
    data = generate_input(m=(128, 256), n=(256, 256), k=(256, 256), g=2, seed=2111)
    reference = reference_group_gemm(data, write_output=False)
    prepared = prepare_custom_cuda([data])
    actual = custom_kernel_custom_cuda((prepared or [data])[0])
    torch.cuda.synchronize()
    assert_group_outputs(actual, reference)
