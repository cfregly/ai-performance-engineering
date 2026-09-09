"""CPU controls for the complete MoE communication verification payload."""

from __future__ import annotations

import torch

from ch15.moe_comm_exchange_benchmarks import MoeCommExchangeBenchmark


def _cpu_payload_benchmark() -> MoeCommExchangeBenchmark:
    bench = MoeCommExchangeBenchmark(variant="baseline", label="cpu_control")
    bench.batch = 3
    bench.seq = 3
    bench.hidden_size = 300
    values = torch.arange(bench.batch * bench.seq * bench.hidden_size, dtype=torch.float32)
    bench.inputs = values.reshape(bench.batch, bench.seq, bench.hidden_size).to(torch.bfloat16)
    bench.expert_ids = torch.arange(bench.batch * bench.seq, dtype=torch.int64).reshape(
        bench.batch, bench.seq
    )
    bench.output = (bench.inputs + 1).to(torch.bfloat16)
    return bench


def test_capture_retains_every_timed_output_input_and_route_element() -> None:
    bench = _cpu_payload_benchmark()
    # This element was outside the former [2, 2, 256] verification slice.
    bench.output[-1, -1, -1] = torch.tensor(123, dtype=torch.bfloat16)

    bench.capture_verification_payload()

    payload_inputs = bench.get_verify_inputs()
    payload_output = bench.get_verify_output()
    assert torch.equal(payload_inputs["tokens"], bench.inputs)
    assert torch.equal(payload_inputs["expert_ids"], bench.expert_ids)
    assert payload_inputs["tokens"].numel() == bench.inputs.numel()
    assert payload_inputs["expert_ids"].numel() == bench.expert_ids.numel()
    assert payload_output.shape == bench.output.shape
    assert payload_output.numel() == bench.output.numel()
    assert payload_output[-1, -1, -1] == 123
    assert bench.get_output_tolerance() == (0.0, 0.0)


def test_exact_pair_comparison_detects_corruption_outside_former_sample() -> None:
    baseline = _cpu_payload_benchmark()
    optimized = _cpu_payload_benchmark()
    optimized.output[-1, -1, -1] = -123

    baseline.capture_verification_payload()
    optimized.capture_verification_payload()

    rtol, atol = baseline.get_output_tolerance()
    assert (rtol, atol) == (0.0, 0.0)
    assert not torch.allclose(
        baseline.get_verify_output(),
        optimized.get_verify_output(),
        rtol=rtol,
        atol=atol,
    )


def test_validate_result_checks_complete_shape_and_finiteness() -> None:
    bench = _cpu_payload_benchmark()
    assert bench.validate_result() is None

    bench.output[-1, -1, -1] = float("nan")
    assert bench.validate_result() == "Output contains non-finite values"

    bench.output = bench.output[:-1]
    assert bench.validate_result() == (
        "Output shape (2, 3, 300) does not match (3, 3, 300)"
    )
