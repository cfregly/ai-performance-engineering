"""Regressions for audit findings W2-069 and W2-068 in the Chapter 17 prefill/decode pair.

W2-069: the shared SimpleLLM decayed every prompt to one bias-driven output, and
both arms declared a tolerance of (0.1, 1.0). Verification accepted a zero
output, another prompt's output, and a shortened decode. The first fix also
left the token-feature frequencies in a floating buffer that Module.to cast to
bf16, so half of the 1024 input features repeated another feature.

W2-068 residual: the optimized arm handed a prefill-stream allocation to the
decode stream without recording that use, so the allocator could give the
block to a later prefill while the decode stream still read it.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from ch17.baseline_prefill_decode_disagg import BaselinePrefillDecodeMonolithicBenchmark
from ch17.optimized_prefill_decode_disagg import OptimizedDisaggregatedBenchmark
from ch17.prefill_decode_disagg_monolithic_common import SimpleLLM

# Model shape used by both arms' setup().
HIDDEN_DIM = 1024
NUM_LAYERS = 12
VOCAB = 10_000
ARMS = (BaselinePrefillDecodeMonolithicBenchmark, OptimizedDisaggregatedBenchmark)


def _relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    expected_f = expected.float()
    return float((actual.float() - expected_f).norm() / expected_f.norm())


def _model(seed: int = 42) -> SimpleLLM:
    torch.manual_seed(seed)
    return SimpleLLM(hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(torch.bfloat16).eval()


def _random_prompts(prompt_len: int, count: int, seed: int = 7) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return [torch.randint(0, VOCAB, (1, prompt_len), generator=generator) for _ in range(count)]


def _decode_trace(model: SimpleLLM, prompt: torch.Tensor, steps: int) -> list[torch.Tensor]:
    """Return the prefill state followed by each decoded state, as both arms compute them."""
    with torch.inference_mode():
        state = model.prefill(prompt).clone()
        trace = [state]
        for _ in range(steps):
            state = model.decode_step(state)
            trace.append(state)
    return trace


def _declared_tolerance(bench_cls: type, prompt: torch.Tensor, output: torch.Tensor) -> tuple[float, float]:
    bench = bench_cls()
    bench.prompt = prompt
    bench.output = output
    bench.capture_verification_payload()
    return bench.get_output_tolerance()


def test_w2_069_declared_tolerance_rejects_wrong_outputs() -> None:
    reference = BaselinePrefillDecodeMonolithicBenchmark()
    prompt_len, steps = reference.prefill_seq, reference.decode_seq
    model = _model()
    prompt, other_prompt = _random_prompts(prompt_len, count=2)
    trace = _decode_trace(model, prompt, steps)
    repeat = _decode_trace(model, prompt, steps)
    other = _decode_trace(model, other_prompt, steps)
    expected = trace[-1]

    tolerances = {cls.__name__: _declared_tolerance(cls, prompt, expected) for cls in ARMS}
    assert len(set(tolerances.values())) == 1, tolerances
    rtol, atol = next(iter(tolerances.values()))

    assert torch.allclose(repeat[-1], expected, rtol=rtol, atol=atol)
    wrong_outputs = {
        "zeros": torch.zeros_like(expected),
        "other_prompt": other[-1],
        "one_fewer_decode_step": trace[-2],
        "prefill_state_without_decode": trace[0],
    }
    for name, wrong in wrong_outputs.items():
        assert not torch.allclose(wrong, expected, rtol=rtol, atol=atol), name


def test_w2_069_decoded_state_tracks_prompt_and_decode_steps() -> None:
    reference = BaselinePrefillDecodeMonolithicBenchmark()
    prompt_len, steps = reference.prefill_seq, reference.decode_seq
    model = _model()
    prompt, other_prompt = _random_prompts(prompt_len, count=2)
    trace = _decode_trace(model, prompt, steps)
    other = _decode_trace(model, other_prompt, steps)

    for state in trace:
        assert bool(torch.isfinite(state).all())
        assert float(state.float().abs().max()) > 0.0
    # The old model left different prompts within 0.05% of each other after
    # prefill and within 0.13% after 16 decode steps. This model measures 3.5%,
    # 8.5% and 19% for these three checks. The checks catch a collapse to one
    # output. They do not separate prompts from accumulated bf16 rounding,
    # which is why both arms require exact equality.
    assert _relative_l2(other[0], trace[0]) > 0.02
    assert _relative_l2(other[-1], trace[-1]) > 0.04
    assert _relative_l2(trace[-2], trace[-1]) > 0.1


def _assert_float32_token_features(model: SimpleLLM, prompt: torch.Tensor) -> torch.Tensor:
    with torch.inference_mode():
        state = model.prefill(prompt).clone()
    scale = model._prefill_feature_scale
    assert scale.dtype == torch.float32
    assert scale.device == prompt.device
    assert scale.unique().numel() == HIDDEN_DIM
    seq_len = int(prompt.size(1))
    features = model._prefill_input_buffer[: seq_len * HIDDEN_DIM].view(seq_len, HIDDEN_DIM)
    # No input feature may repeat another one across the whole prompt.
    assert torch.unique(features.t(), dim=0).size(0) == HIDDEN_DIM
    return state


def test_w2_069_token_feature_frequencies_stay_float32_after_bf16_cast() -> None:
    reference = BaselinePrefillDecodeMonolithicBenchmark()
    # Built as both arms build it, including Module.to(torch.bfloat16).
    model = _model()
    (prompt,) = _random_prompts(reference.prefill_seq, count=1)
    first = _assert_float32_token_features(model, prompt)

    # A cast after first use turns the cached frequencies into bf16, which
    # keeps only 512 of the 1024 values distinct. The next prefill must
    # rebuild them in float32.
    model.to(torch.bfloat16)
    assert model._prefill_feature_scale.dtype == torch.bfloat16
    assert model._prefill_feature_scale.unique().numel() < HIDDEN_DIM
    second = _assert_float32_token_features(model, prompt)

    assert torch.equal(first, second)


def test_w2_069_prefill_keeps_nearby_token_ids_distinct() -> None:
    reference = BaselinePrefillDecodeMonolithicBenchmark()
    model = _model()
    (prompt,) = _random_prompts(reference.prefill_seq, count=1)
    low = prompt.clone()
    high = prompt.clone()
    # bf16 rounds both ids to 9984, so a bf16 feature product cannot tell them apart.
    low[0, 0] = 9990
    high[0, 0] = 9999
    with torch.inference_mode():
        low_state = model.prefill(low).clone()
        high_state = model.prefill(high).clone()

    assert not torch.equal(low_state, high_state)


def test_w2_068_decode_marks_handoff_state_before_reading_it() -> None:
    # The allocator race needs CUDA streams, which this host lacks. The
    # CUDA-gated test below exercises the same path on a GPU.
    source = inspect.getsource(OptimizedDisaggregatedBenchmark.benchmark_fn)
    decode_block = source.index("with torch.cuda.stream(self.decode_stream):")
    wait = source.index("self.decode_stream.wait_event(self._decode_ready)")
    mark = source.index("token_output.record_stream(self.decode_stream)")
    first_read = source.index("token_output = self.model.decode_step(token_output)")
    handoff_state = source.index("token_output = self._decode_state")
    release = source.index("self._decode_state = next_decode_state")

    assert source.count("record_stream(") == 1
    assert handoff_state < decode_block < wait < mark < first_read < release


@pytest.mark.skipif(not torch.cuda.is_available(), reason="SKIPPED: CUDA required for the stream handoff path")
def test_w2_068_w2_069_pair_matches_after_unsynchronized_iterations(monkeypatch: pytest.MonkeyPatch) -> None:
    original_record_stream = torch.Tensor.record_stream
    recorded_streams: list[torch.cuda.Stream] = []

    def _recording_record_stream(tensor: torch.Tensor, stream: torch.cuda.Stream) -> None:
        recorded_streams.append(stream)
        original_record_stream(tensor, stream)

    monkeypatch.setattr(torch.Tensor, "record_stream", _recording_record_stream)

    iterations = 8
    outputs: dict[str, torch.Tensor] = {}
    tolerances: dict[str, tuple[float, float]] = {}
    for bench_cls in ARMS:
        torch.manual_seed(42)
        bench = bench_cls()
        bench.setup()
        try:
            recorded_streams.clear()
            # No synchronization between calls, like warmup, so a block freed
            # too early would be reused while the decode stream still reads it.
            for _ in range(iterations):
                bench.benchmark_fn()
            torch.cuda.synchronize(bench.device)
            if bench_cls is OptimizedDisaggregatedBenchmark:
                assert recorded_streams == [bench.decode_stream] * iterations
            scale = bench.model._prefill_feature_scale
            assert scale.dtype == torch.float32
            assert scale.device == bench.prompt.device
            assert scale.unique().numel() == HIDDEN_DIM
            bench.capture_verification_payload()
            outputs[bench_cls.__name__] = bench.get_verify_output()
            tolerances[bench_cls.__name__] = bench.get_output_tolerance()
        finally:
            bench.teardown()

    baseline = outputs[BaselinePrefillDecodeMonolithicBenchmark.__name__]
    optimized = outputs[OptimizedDisaggregatedBenchmark.__name__]
    assert len(set(tolerances.values())) == 1, tolerances
    rtol, atol = next(iter(tolerances.values()))
    assert float(baseline.float().abs().max()) > 0.0
    assert torch.allclose(optimized, baseline, rtol=rtol, atol=atol)
