"""Real reference and CUDA paths for foundational kernel mechanisms."""

import importlib
import itertools

import pytest
import torch

from ch09.online_softmax_common import (
    materialized_softmax,
    online_softmax,
    online_softmax_launch_config,
)
from labs.prefix_scan.reference import inclusive_scan


@pytest.mark.parametrize("values", [[7], [1, -3, 7, -9, 0], [2**31 - 1, 1, 1]])
def test_scan_matches_scalar_prefixes_with_int32_wrap(values):
    expected = [((v + 2**31) % 2**32) - 2**31 for v in itertools.accumulate(values)]
    actual = inclusive_scan(torch.tensor(values, dtype=torch.int32))
    assert actual.tolist() == expected


@pytest.mark.parametrize("columns,block", [(1, 16), (17, 16), (257, 32), (1025, 256)])
def test_online_normalizer_matches_double_precision_softmax(columns, block):
    generator = torch.Generator().manual_seed(901)
    values = torch.randn(3, columns, generator=generator, dtype=torch.float64) * 300
    expected = torch.softmax(values, dim=1)
    torch.testing.assert_close(online_softmax(values, block), expected, rtol=1e-12, atol=1e-14)
    torch.testing.assert_close(
        materialized_softmax(values.float()), expected.float(), rtol=2e-4, atol=2e-6
    )


def test_online_normalizer_handles_fully_masked_first_tile():
    x = torch.tensor([[-torch.inf, -torch.inf, 1000.0, 999.0]])
    torch.testing.assert_close(online_softmax(x, 2), torch.softmax(x, 1))


@pytest.mark.parametrize("cols", [257, 4097, 8193])
def test_online_normalizer_selected_tiles_preserve_masked_and_ragged_rows(cols):
    block_size, num_warps = online_softmax_launch_config(cols)
    assert block_size & (block_size - 1) == 0
    assert 256 <= block_size <= 4096
    assert num_warps in {4, 8}
    generator = torch.Generator().manual_seed(217)
    values = torch.randn(2, cols, generator=generator, dtype=torch.float64) * 300
    values[:, : min(cols - 1, block_size)] = -torch.inf

    torch.testing.assert_close(
        online_softmax(values, block_size), torch.softmax(values, dim=1), rtol=1e-12, atol=1e-14
    )


def test_online_normalizer_translation_invariance_and_noncontiguous_input():
    x = torch.arange(56, dtype=torch.float64).reshape(7, 8).T
    assert not x.is_contiguous()
    torch.testing.assert_close(online_softmax(x, 3), online_softmax(x - 10000, 3))


def test_input_contracts():
    with pytest.raises(ValueError):
        inclusive_scan(torch.empty(0, dtype=torch.int32))
    with pytest.raises(ValueError):
        inclusive_scan(torch.ones(5))
    with pytest.raises(ValueError):
        online_softmax(torch.ones(2, 4), 0)


@pytest.mark.parametrize("operation", ["prefix_scan", "online_softmax"])
def test_real_wrappers_and_unsupported_host_diagnostic(operation):
    for arm in ["baseline", "optimized"]:
        owner = "labs.prefix_scan" if operation == "prefix_scan" else "ch09"
        module = importlib.import_module(f"{owner}.{arm}_{operation}")
        benchmark = module.get_benchmark()
        assert benchmark.get_config().iterations > 0
        if not torch.cuda.is_available():
            with pytest.raises(RuntimeError, match="SKIPPED:.*CUDA"):
                benchmark.setup()


@pytest.mark.parametrize("operation", ["prefix_scan", "online_softmax"])
def test_cpu_baseline_publishes_only_computed_results_and_invalidates_stale_payload(operation):
    from ch09.online_softmax_common import OnlineSoftmaxBenchmark
    from labs.prefix_scan.benchmarks import PrefixScanBenchmark

    if operation == "prefix_scan":
        benchmark = PrefixScanBenchmark(False, numel=5)
        values = torch.tensor([2**31 - 1, 1, -2, 4, -7], dtype=torch.int32)
        expected = values.long().cumsum(0).int()
    else:
        benchmark = OnlineSoftmaxBenchmark(False, rows=2, cols=3)
        values = torch.tensor([[-3.0, 0.0, 2.0], [1.0, -1.0, 5.0]])
        expected = torch.softmax(values.double(), 1).float()
    benchmark.device = torch.device("cpu")
    assert benchmark.output is None
    benchmark.x = values.clone()
    benchmark._scratch = torch.empty_like(values)
    with pytest.raises(RuntimeError, match="benchmark_fn"):
        benchmark.capture_verification_payload()

    benchmark.benchmark_fn()
    benchmark.capture_verification_payload()
    torch.testing.assert_close(benchmark.get_verify_output(), expected)
    if operation == "prefix_scan":
        assert benchmark.output.data_ptr() == benchmark._scratch.data_ptr()
    benchmark.x = torch.ones(2, dtype=torch.float64)
    with pytest.raises(ValueError):
        benchmark.benchmark_fn()
    assert benchmark.output is None
    assert not benchmark.executed
    with pytest.raises(RuntimeError):
        benchmark.get_verify_output()
    with pytest.raises(RuntimeError, match="benchmark_fn"):
        benchmark.capture_verification_payload()
    benchmark.teardown()
    assert benchmark.output is benchmark._scratch is None


def test_native_scan_jitter_changes_live_input_and_restores_it(tmp_path):
    from core.benchmark.verify_runner import VerifyConfig, VerifyRunner
    from labs.prefix_scan.benchmarks import PrefixScanBenchmark

    benchmark = PrefixScanBenchmark(False, numel=1025)
    benchmark.device = torch.device("cpu")
    benchmark.x = torch.arange(1025, dtype=torch.int32) - 512
    benchmark.x[0] = 2**31 - 1
    original = benchmark.x.clone()
    benchmark._scratch = torch.empty_like(benchmark.x)
    benchmark.benchmark_fn()
    benchmark.capture_verification_payload()
    inputs = benchmark.get_verify_inputs()
    assert inputs["x"].shape == (1, 1025)
    assert inputs["x"].data_ptr() == benchmark.x.data_ptr()

    passed, reason = VerifyRunner(cache_dir=tmp_path / "cache")._run_jitter_check(
        benchmark, benchmark.get_input_signature(), VerifyConfig()
    )
    assert passed, reason
    assert reason is None
    assert torch.equal(benchmark.x, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real NVIDIA CUDA GPU required")
@pytest.mark.parametrize("size", [1, 17, 1023, 1024, 1025, 1_048_579])
def test_cuda_hierarchical_scan_ragged_tiles_and_replay(size):
    pytest.importorskip("triton")
    from labs.prefix_scan.benchmarks import ScanPlan

    x = torch.randint(-20, 21, (size,), device="cuda", dtype=torch.int32)
    plan = ScanPlan(x)
    for delta in [0, 1]:
        x.add_(delta)
        torch.testing.assert_close(
            plan.run(), x.to(torch.int64).cumsum(0).to(torch.int32), rtol=0, atol=0
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real NVIDIA CUDA GPU required")
@pytest.mark.parametrize("cols", [1, 257, 4097, 8193])
def test_cuda_online_softmax_and_actual_verification_payload(cols):
    pytest.importorskip("triton")
    from ch09.online_softmax_common import OnlineSoftmaxBenchmark
    from core.harness.validity_checks import check_setup_precomputation

    for optimized in [False, True]:
        bench = OnlineSoftmaxBenchmark(optimized, rows=3, cols=cols)
        valid, error = check_setup_precomputation(
            lambda current=bench: {"output": current.output}, bench.setup
        )
        assert valid, error
        assert bench.output is None
        with pytest.raises(RuntimeError, match="benchmark_fn"):
            bench.capture_verification_payload()
        block_size, _ = online_softmax_launch_config(cols)
        bench.x[:, : min(cols - 1, block_size)] = -torch.inf
        bench.benchmark_fn()
        bench.capture_verification_payload()
        torch.testing.assert_close(
            bench.output, torch.softmax(bench.x.double(), 1).float(), rtol=2e-5, atol=2e-7
        )
        bench.teardown()
        with pytest.raises(RuntimeError):
            bench.get_verify_output()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real NVIDIA CUDA GPU required")
@pytest.mark.parametrize("backend", ["baseline", "hierarchical", "lookback"])
def test_cuda_scan_setup_guard_and_actual_replayed_output(backend):
    if backend == "hierarchical":
        pytest.importorskip("triton")
    from core.harness.validity_checks import check_setup_precomputation
    from labs.prefix_scan.benchmarks import PrefixScanBenchmark

    retained = torch.arange(1025, dtype=torch.int32, device="cuda") - 512
    retained[0] = 2**31 - 1
    original = retained.clone()
    bench = PrefixScanBenchmark(
        backend != "baseline",
        numel=retained.numel(),
        lookback=backend == "lookback",
        input_tensors={"x": retained},
    )
    valid, error = check_setup_precomputation(lambda: {"output": bench.output}, bench.setup)
    assert valid, error
    assert bench.output is None
    with pytest.raises(RuntimeError, match="benchmark_fn"):
        bench.capture_verification_payload()
    for delta in [0, 1]:
        bench.x.add_(delta)
        bench.benchmark_fn()
        bench.capture_verification_payload()
        torch.testing.assert_close(
            bench.get_verify_output(), bench.x.long().cumsum(0).int(), rtol=0, atol=0
        )
    assert torch.equal(retained, original)
    bench.teardown()
    assert bench.output is bench._scratch is None
    with pytest.raises(RuntimeError):
        bench.get_verify_output()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA toolkit and GPU required")
def test_cuda_decoupled_lookback_fresh_state_ragged_tiles_and_overflow():
    from labs.prefix_scan.lookback import LookbackPlan

    for size in [1, 257, 65537, 1_048_579]:
        x = torch.randint(-7, 8, (size,), device="cuda", dtype=torch.int32)
        x[0] = 2**31 - 1
        plan = LookbackPlan(x)
        for delta in [0, 1, -2]:
            x.add_(delta)
            torch.testing.assert_close(plan.run(), x.long().cumsum(0).int(), rtol=0, atol=0)
