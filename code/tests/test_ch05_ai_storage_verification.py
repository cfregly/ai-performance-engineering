"""Exercise real storage-backed input mutation and both inference paths."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ch05.baseline_ai import BaselineAIBenchmark
from ch05.optimized_ai import OptimizedAIBenchmark


@pytest.mark.parametrize("benchmark_type", [BaselineAIBenchmark, OptimizedAIBenchmark])
def test_verification_input_changes_the_storage_batch_consumed_on_rerun(benchmark_type):
    benchmark = benchmark_type()
    benchmark.device = torch.device("cpu")
    benchmark.num_blocks = 3
    benchmark.batch = 4
    benchmark.hidden = 8
    benchmark.setup()
    input_path = Path(benchmark.inputs_path)
    try:
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        before = benchmark.get_verify_output()
        declared = benchmark.get_verify_inputs()["inputs"]
        stored_before = np.load(input_path).copy()
        torch.testing.assert_close(declared, torch.from_numpy(stored_before[-1]))

        with torch.no_grad():
            declared.add_(torch.linspace(-1, 1, declared.numel()).reshape_as(declared))
        stored_after = np.load(input_path)
        np.testing.assert_array_equal(stored_after[:-1], stored_before[:-1])
        torch.testing.assert_close(declared, torch.from_numpy(stored_after[-1]))
        assert not np.array_equal(stored_after[-1], stored_before[-1])

        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        after = benchmark.get_verify_output()
        assert not torch.equal(before, after)
        with torch.inference_mode():
            expected = benchmark.block(torch.from_numpy(stored_after[-1])).clone()
        torch.testing.assert_close(after, expected, rtol=0, atol=0)
    finally:
        benchmark.teardown()
    assert not input_path.exists()


def test_storage_pipeline_pair_still_matches_after_changing_the_final_source_batch():
    benchmarks = [BaselineAIBenchmark(), OptimizedAIBenchmark()]
    outputs = []
    try:
        for benchmark in benchmarks:
            torch.manual_seed(42)
            benchmark.device = torch.device("cpu")
            benchmark.num_blocks = 3
            benchmark.batch = 4
            benchmark.hidden = 8
            benchmark.setup()
            benchmark.benchmark_fn()
            benchmark.capture_verification_payload()
            benchmark.get_verify_inputs()["inputs"].mul_(1.1)
            benchmark.benchmark_fn()
            benchmark.capture_verification_payload()
            outputs.append(benchmark.get_verify_output())
        torch.testing.assert_close(outputs[0], outputs[1], rtol=1e-3, atol=1e-3)
    finally:
        for benchmark in benchmarks:
            benchmark.teardown()


@pytest.mark.parametrize("benchmark_type", [BaselineAIBenchmark, OptimizedAIBenchmark])
def test_harness_seed_controls_storage_input_and_model_output(benchmark_type):
    inputs, outputs = [], []
    for seed in (42, 1042):
        torch.manual_seed(seed)
        benchmark = benchmark_type()
        benchmark.device = torch.device("cpu")
        benchmark.num_blocks = 3
        benchmark.batch = 4
        benchmark.hidden = 8
        benchmark.setup()
        try:
            benchmark.benchmark_fn()
            benchmark.capture_verification_payload()
            inputs.append(benchmark.get_verify_inputs()["inputs"].clone())
            outputs.append(benchmark.get_verify_output())
        finally:
            benchmark.teardown()
    assert not torch.equal(inputs[0], inputs[1])
    assert not torch.equal(outputs[0], outputs[1])
