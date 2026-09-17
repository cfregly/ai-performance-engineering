"""Real reference and CUDA paths for foundational kernel mechanisms."""

import importlib
import itertools

import pytest
import torch

from ch09.online_softmax_common import materialized_softmax, online_softmax
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
@pytest.mark.parametrize("cols", [1, 257, 8193])
def test_cuda_online_softmax_and_actual_verification_payload(cols):
    pytest.importorskip("triton")
    from ch09.online_softmax_common import OnlineSoftmaxBenchmark

    for optimized in [False, True]:
        bench = OnlineSoftmaxBenchmark(optimized, rows=3, cols=cols)
        bench.setup()
        bench.x[:, : min(cols - 1, 256)] = -torch.inf
        bench.benchmark_fn()
        bench.capture_verification_payload()
        torch.testing.assert_close(
            bench.output, torch.softmax(bench.x.double(), 1).float(), rtol=2e-5, atol=2e-7
        )
        bench.teardown()


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
