"""Focused CPU contracts for declared discrete-input jitter domains."""

from __future__ import annotations

import pytest
import torch

from core.benchmark.verification import InputSignature, PrecisionFlags
from core.benchmark.verify_runner import VerifyConfig, VerifyRunner


class _DiscreteWork:
    def __init__(
        self,
        token_ids: torch.Tensor,
        bounds: object = (0, 2),
    ) -> None:
        self.token_ids = token_ids
        self.input_jitter_bounds = {"token_ids": bounds}
        self.output = torch.empty_like(token_ids, dtype=torch.float32)
        self.observed_inputs: list[torch.Tensor] = []

    def benchmark_fn(self) -> None:
        self.observed_inputs.append(self.token_ids.clone())
        self.output.copy_(self.token_ids)

    def get_verify_inputs(self) -> dict[str, torch.Tensor]:
        return {"token_ids": self.token_ids}

    def get_verify_output(self) -> torch.Tensor:
        return self.output

    def get_input_signature(self) -> InputSignature:
        return InputSignature(
            shapes={
                "token_ids": tuple(self.token_ids.shape),
                "output": tuple(self.output.shape),
            },
            dtypes={
                "token_ids": str(self.token_ids.dtype),
                "output": str(self.output.dtype),
            },
            batch_size=1,
            parameter_count=0,
            precision_flags=PrecisionFlags(),
        )


def _run_jitter(tmp_path, work: _DiscreteWork) -> tuple[bool, str | None]:
    work.benchmark_fn()
    runner = VerifyRunner(cache_dir=tmp_path / "cache")
    return runner._run_jitter_check(work, work.get_input_signature(), VerifyConfig())


def test_integer_jitter_uses_declared_domain_and_restores_exactly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    work = _DiscreteWork(torch.ones((1, 8), dtype=torch.int64))
    original = work.token_ids.clone()

    def unchanged_draw(*_args, **_kwargs) -> torch.Tensor:
        return original.clone()

    monkeypatch.setattr(torch, "randint", unchanged_draw)
    passed, reason = _run_jitter(tmp_path, work)

    assert passed, reason
    assert len(work.observed_inputs) == 2
    perturbed = work.observed_inputs[-1]
    assert not torch.equal(perturbed, original)
    assert bool(((perturbed >= 0) & (perturbed < 2)).all())
    assert torch.equal(work.token_ids, original)


def test_integer_jitter_without_declared_domain_is_explicitly_advisory(tmp_path) -> None:
    work = _DiscreteWork(torch.ones((1, 4), dtype=torch.int64))
    del work.input_jitter_bounds
    original = work.token_ids.clone()

    passed, reason = _run_jitter(tmp_path, work)

    assert passed
    assert "requires input_jitter_bounds" in reason
    assert len(work.observed_inputs) == 1
    assert torch.equal(work.token_ids, original)


def test_integer_jitter_reports_declared_domain_draw_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    work = _DiscreteWork(torch.ones((1, 4), dtype=torch.int64))
    original = work.token_ids.clone()

    def rejected_draw(*_args, **_kwargs) -> torch.Tensor:
        raise RuntimeError("unsupported integer range")

    monkeypatch.setattr(torch, "randint", rejected_draw)
    passed, reason = _run_jitter(tmp_path, work)

    assert not passed
    assert "cannot draw valid values" in reason
    assert len(work.observed_inputs) == 1
    assert torch.equal(work.token_ids, original)


@pytest.mark.parametrize(
    ("token_ids", "bounds", "expected"),
    [
        (torch.ones((1, 4), dtype=torch.int64), (0, 1), "at least two values"),
        (torch.ones((1, 4), dtype=torch.int64), (0.0, 2), "pair of integers"),
        (torch.tensor([[0, 2]], dtype=torch.int64), (0, 2), "outside [0, 2)"),
        (torch.ones((1, 4), dtype=torch.uint8), (-1, 2), "does not fit dtype"),
    ],
)
def test_integer_jitter_rejects_invalid_bounds_without_mutation(
    tmp_path,
    token_ids: torch.Tensor,
    bounds: object,
    expected: str,
) -> None:
    work = _DiscreteWork(token_ids.clone(), bounds)
    original = work.token_ids.clone()

    passed, reason = _run_jitter(tmp_path, work)

    assert not passed
    assert expected in reason
    assert len(work.observed_inputs) == 1
    assert torch.equal(work.token_ids, original)
