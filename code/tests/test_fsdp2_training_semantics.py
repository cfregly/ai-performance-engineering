from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.nn import functional

from labs.train_distributed.training_utils.child_result import RESULT_DIR_ENV
from labs.train_distributed.training_utils.fsdp2_training import (
    fsdp2_causal_lm_loss,
    initialize_fsdp2_seed,
    validate_fsdp2_training_args,
)


class _TinyCausalLM(torch.nn.Module):
    """Small trainable LM whose cumulative hidden state is strictly causal."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(11, 6)
        self.lm_head = torch.nn.Linear(6, 11, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        hidden = self.embedding(input_ids)
        if attention_mask is not None:
            hidden = hidden * attention_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        logits = self.lm_head(hidden.cumsum(dim=1)).to(dtype=torch.bfloat16)
        return SimpleNamespace(logits=logits)


def _tiny_causal_model() -> _TinyCausalLM:
    return _TinyCausalLM().eval()


@pytest.mark.parametrize("with_attention_mask", [False, True])
def test_fsdp2_loss_uses_already_shifted_targets_once_and_backpropagates(
    with_attention_mask: bool,
) -> None:
    torch.manual_seed(17)
    model = _tiny_causal_model()
    input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    labels = torch.tensor([[2, 3, 4, -100]], dtype=torch.long)
    batch = {"input_ids": input_ids, "labels": labels}
    if with_attention_mask:
        batch["attention_mask"] = torch.ones_like(input_ids)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=batch.get("attention_mask"),
        ).logits
        log_probabilities = functional.log_softmax(logits.float(), dim=-1)
        valid = labels != -100
        target_log_probabilities = log_probabilities.gather(
            -1, labels.clamp_min(0).unsqueeze(-1)
        ).squeeze(-1)
        manual_loss = -target_log_probabilities[valid].mean()
        double_shifted_loss = functional.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.shape[-1]),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )

    loss = fsdp2_causal_lm_loss(model, batch)

    assert loss.shape == ()
    assert loss.dtype == torch.float32
    torch.testing.assert_close(loss, manual_loss)
    assert not torch.isclose(loss.detach(), double_shifted_loss, rtol=1e-4, atol=1e-4)

    loss.backward()
    gradient = model.lm_head.weight.grad
    assert gradient is not None
    assert torch.count_nonzero(gradient).item() > 0


def test_initialize_fsdp2_seed_preserves_harness_state_and_seeds_direct_execution(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setenv(RESULT_DIR_ENV, str(tmp_path))
    random.seed(3)
    np.random.seed(5)
    torch.manual_seed(1042)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state().clone()

    assert initialize_fsdp2_seed() == 1042
    assert random.getstate() == python_state
    current_numpy_state = np.random.get_state()
    assert current_numpy_state[0] == numpy_state[0]
    assert np.array_equal(current_numpy_state[1], numpy_state[1])
    assert current_numpy_state[2:] == numpy_state[2:]
    assert torch.equal(torch.random.get_rng_state(), torch_state)

    monkeypatch.delenv(RESULT_DIR_ENV)
    assert initialize_fsdp2_seed(fallback=73) == 73
    first = (random.random(), float(np.random.random()), torch.rand(3))

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    assert initialize_fsdp2_seed(fallback=73) == 73
    second = (random.random(), float(np.random.random()), torch.rand(3))

    assert first[:2] == second[:2]
    torch.testing.assert_close(first[2], second[2], rtol=0, atol=0)


def _valid_args() -> SimpleNamespace:
    return SimpleNamespace(
        steps=3,
        sequence_length=8,
        micro_batch_size=2,
        grad_accum=4,
        learning_rate=2e-4,
    )


def test_validate_fsdp2_training_args_preserves_optimizer_update_count() -> None:
    args = _valid_args()
    original = vars(args).copy()

    assert validate_fsdp2_training_args(args) is None
    assert vars(args) == original
    assert args.steps == 3


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("steps", 0),
        ("steps", -1),
        ("sequence_length", 0),
        ("micro_batch_size", 0),
        ("grad_accum", 0),
        ("grad_accum", False),
        ("micro_batch_size", 1.5),
    ],
)
def test_validate_fsdp2_training_args_rejects_nonpositive_or_noninteger_counts(
    name: str,
    value: object,
) -> None:
    args = _valid_args()
    setattr(args, name, value)

    with pytest.raises(ValueError, match=rf"{name} must be a positive integer"):
        validate_fsdp2_training_args(args)


@pytest.mark.parametrize(
    "learning_rate",
    [0.0, -1e-4, float("nan"), float("inf"), float("-inf"), True, "1e-4"],
)
def test_validate_fsdp2_training_args_rejects_invalid_learning_rate(
    learning_rate: object,
) -> None:
    args = _valid_args()
    args.learning_rate = learning_rate

    with pytest.raises(ValueError, match="learning_rate must be finite and positive"):
        validate_fsdp2_training_args(args)
