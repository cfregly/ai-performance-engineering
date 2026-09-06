"""Input discrimination for real peer-output verification."""

import torch

from ch04.symmetric_memory_perf_common import make_rank_distinct_input


def test_identically_seeded_ranks_still_have_distinct_transfer_inputs() -> None:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        first = make_rank_distinct_input(262144, torch.device("cpu"), 0)
        torch.manual_seed(42)
        second = make_rank_distinct_input(262144, torch.device("cpu"), 1)
        assert torch.initial_seed() == 42
    assert not torch.allclose(first, second, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(second, first + 1, rtol=0, atol=0)
