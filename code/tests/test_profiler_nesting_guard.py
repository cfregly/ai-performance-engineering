"""Real profiler contexts must not contaminate harness timing or nesting."""

import pytest
import torch

from tests.protection_test_utils import cpu_harness


@pytest.mark.parametrize("profiler_kind", ["kineto", "autograd"])
@pytest.mark.parametrize("entrypoint", [
    "benchmark", "_benchmark_custom", "_benchmark_pytorch",
    "_benchmark_triton", "_benchmark_with_profiling",
])
def test_enclosing_profiler_rejected_before_work(profiler_kind, entrypoint):
    harness, config = cpu_harness()
    calls = []

    def work():
        calls.append(torch.ones(4) + 1)

    profiler = (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        if profiler_kind == "kineto" else torch.autograd.profiler.profile()
    )
    with profiler:
        with pytest.raises(RuntimeError, match="Active PyTorch profiler"):
            if entrypoint == "benchmark":
                harness.benchmark(work)
            else:
                getattr(harness, entrypoint)(work, config)
        assert calls == []
        # The rejected run must leave the caller's profiler usable.
        torch.ones(4).add_(1)
    assert any(event.key == "aten::add_" for event in profiler.key_averages())
    samples, _ = harness._benchmark_custom(work, config)
    assert len(samples) == len(calls) == config.iterations
    for output in calls:
        torch.testing.assert_close(output, torch.full((4,), 2.0), rtol=0, atol=0)


def test_harness_can_collect_its_own_profile_after_timing(tmp_path):
    import json
    from pathlib import Path

    harness, config = cpu_harness(profiling_output_dir=tmp_path)
    outputs = []

    def work():
        outputs.append(torch.ones(4).add_(1))

    times, _ = harness._benchmark_custom(work, config)
    assert len(times) == config.iterations
    profile_times, traces = harness._benchmark_with_profiling(work, config)
    assert len(profile_times) == config.iterations
    trace = json.loads(Path(traces["pytorch_trace"]).read_text())
    assert any(event.get("name") == "aten::add_" for event in trace["traceEvents"])
    for output in outputs:
        torch.testing.assert_close(output, torch.full((4,), 2.0), rtol=0, atol=0)
