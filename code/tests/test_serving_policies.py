"""Policy tests use actual queue operations, token transitions, and sampling."""

import random

import pytest
import torch

from ch16.fair_scheduler import Request, VirtualTokenScheduler, dispatch_trace
from labs.decode_optimization.token_grammar import TokenGrammar


@pytest.mark.parametrize("backend", ["linear", "heap"])
def test_fairness_and_fifo_within_client(backend):
    requests = [Request(i, i // 4, 2) for i in range(8)]
    assert dispatch_trace(requests, dict.fromkeys(range(8), 1), backend=backend) == (
        0,
        4,
        1,
        5,
        2,
        6,
        3,
        7,
    )


@pytest.mark.parametrize("backend", ["linear", "heap"])
def test_counter_lift_and_token_feedback(backend):
    scheduler = VirtualTokenScheduler(backend=backend)
    scheduler.enqueue(Request(0, 0, 10))
    scheduler.admit()
    scheduler.record_output(0, 20)
    scheduler.complete(0)
    scheduler.enqueue(Request(1, 1, 1))
    assert scheduler.counters[1] == 50
    scheduler.enqueue(Request(2, 0, 1))
    assert scheduler.admit().request_id == 1
    scheduler.record_output(1, 10)
    assert scheduler.admit().request_id == 2


def test_heap_matches_linear_under_arrival_and_decode_interleavings():
    rng = random.Random(907)
    schedulers = [VirtualTokenScheduler(backend=b) for b in ("linear", "heap")]
    next_id = 0
    running = set()
    for _ in range(1000):
        op = rng.randrange(4)
        if op == 0 or not (len(schedulers[0]) or running):
            request = Request(next_id, rng.randrange(17), rng.randrange(1, 70))
            next_id += 1
            for scheduler in schedulers:
                scheduler.enqueue(request)
        elif op == 1 and len(schedulers[0]):
            assert schedulers[0].peek() == schedulers[1].peek()
            left, right = (s.admit() for s in schedulers)
            assert left == right
            running.add(left.request_id)
        elif running:
            request_id = rng.choice(sorted(running))
            tokens = rng.randrange(5)
            for scheduler in schedulers:
                scheduler.record_output(request_id, tokens)
                if op == 3:
                    scheduler.complete(request_id)
            if op == 3:
                running.remove(request_id)
        assert schedulers[0].counters == schedulers[1].counters
        assert len(schedulers[1].heap) <= 4 * len(schedulers[1].queues) + 65


def test_scheduler_rejects_invalid_and_duplicate_work():
    with pytest.raises(ValueError):
        Request(1, 0, -1)
    with pytest.raises(ValueError):
        VirtualTokenScheduler(input_weight=float("nan"))
    scheduler = VirtualTokenScheduler()
    scheduler.enqueue(Request(1, 0, 3))
    with pytest.raises(ValueError):
        scheduler.enqueue(Request(1, 1, 3))
    scheduler.admit()
    with pytest.raises(ValueError):
        scheduler.enqueue(Request(1, 1, 3))
    with pytest.raises(KeyError):
        scheduler.record_output(123)
    with pytest.raises(IndexError):
        scheduler.admit()


@pytest.mark.parametrize("cache", [False, True])
def test_token_boundaries_eos_and_byte_level_utf8(cache):
    vocabulary = [b"", b"{", b'"ok"', b":", b"true", b"false}", b"}", b"true}garbage"]
    grammar = TokenGrammar.literals(
        [b'{"ok":true}', b'{"ok":false}'], vocabulary, eos_token_id=0, cache=cache
    )
    state = 0
    for token_id in [1, 2, 3]:
        state = grammar.advance(state, token_id)
    assert grammar.allowed_mask(state) == (False, False, False, False, True, True, False, False)
    state = grammar.advance(state, 5)
    assert grammar.advance(state, 0) == -1
    with pytest.raises(ValueError):
        grammar.advance(-1, 0)
    utf8 = TokenGrammar.literals(
        ["é".encode()], [b"", b"\xc3", b"\xa9"], eos_token_id=0, cache=cache
    )
    assert utf8.advance(utf8.advance(utf8.advance(0, 1), 2), 0) == -1


def test_nonproductive_dfa_edges_are_not_valid_tokens():
    grammar = TokenGrammar(
        [{ord("a"): 1, ord("b"): 2}, {}, {ord("b"): 2}], {1}, [b"", b"a", b"b"], eos_token_id=0
    )
    assert grammar.allowed_mask(0) == (False, True, False)


def test_grammar_sampling_preserves_allowed_odds_and_produces_valid_json():
    vocabulary = [b"", b"true", b"false", b"invalid"]
    grammar = TokenGrammar.literals([b"true", b"false"], vocabulary, eos_token_id=0)
    logits = torch.tensor([100.0, 1.0, 2.0, 200.0])
    masked = grammar.mask_logits(logits, 0)
    torch.testing.assert_close(masked[1:3], logits[1:3])
    assert torch.isneginf(masked[[0, 3]]).all()
    generator = torch.Generator().manual_seed(704)
    samples = torch.multinomial(masked.softmax(0), 40, replacement=True, generator=generator)
    assert set(samples.tolist()) == {1, 2}
    for token in samples.tolist():
        assert grammar.advance(grammar.advance(0, token), 0) == -1


def test_cached_masks_match_uncached_for_every_state():
    vocab = [b"", b"a", b"b", b"aa", b"ab", b"ba", b"bb", b"bad"]
    left = TokenGrammar.literals([b"a", b"abba", b"bb"], vocab, eos_token_id=0, cache=False)
    right = TokenGrammar.literals([b"a", b"abba", b"bb"], vocab, eos_token_id=0, cache=True)
    right.precompile()
    for state in range(len(left.transitions)):
        assert left.allowed_mask(state) == right.allowed_mask(state)
    with pytest.raises(ValueError):
        right.allowed_mask(10000)


@pytest.mark.parametrize("operation", ["fair_scheduler", "grammar_mask"])
def test_real_cpu_benchmark_pair_captures_complete_decisions(operation):
    from ch16.fair_scheduler_benchmarks import FairSchedulerBenchmark
    from labs.decode_optimization.grammar_mask_benchmark import GrammarMaskBenchmark

    outputs = []
    for optimized in [False, True]:
        torch.manual_seed(402)
        benchmark_type = (
            FairSchedulerBenchmark if operation == "fair_scheduler" else GrammarMaskBenchmark
        )
        benchmark = benchmark_type(optimized, size=32)
        benchmark.setup()
        benchmark.benchmark_fn()
        benchmark.capture_verification_payload()
        outputs.append(benchmark.get_verify_output())
        assert benchmark.device.type == "cpu"
        benchmark.teardown()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
