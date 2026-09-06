from __future__ import annotations

import torch

from labs.train_distributed.training_utils.utils import make_causal_lm_labels


def test_causal_lm_labels_exclude_padding_from_loss_and_gradients() -> None:
    eos_and_pad_token_id = 2
    input_ids = torch.tensor(
        [[7, eos_and_pad_token_id, 4, eos_and_pad_token_id, eos_and_pad_token_id]]
    )
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

    labels = make_causal_lm_labels(input_ids, attention_mask)

    assert labels.tolist() == [[7, eos_and_pad_token_id, 4, -100, -100]]
    assert input_ids.tolist() == [
        [7, eos_and_pad_token_id, 4, eos_and_pad_token_id, eos_and_pad_token_id]
    ]

    torch.manual_seed(0)
    hidden = torch.randn(1, input_ids.size(1), 6)
    lm_head = torch.nn.Linear(6, 11, bias=False)
    logits = lm_head(hidden)
    logits.retain_grad()
    loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        labels[:, 1:].reshape(-1),
    )
    loss.backward()

    assert logits.grad is not None
    assert torch.count_nonzero(logits.grad[:, 0]).item() > 0
    assert torch.count_nonzero(logits.grad[:, 1]).item() > 0
    assert torch.count_nonzero(logits.grad[:, 2:]).item() == 0


def test_causal_lm_labels_reject_mismatched_mask_shape() -> None:
    input_ids = torch.ones((2, 4), dtype=torch.long)
    attention_mask = torch.ones((2, 3), dtype=torch.long)

    try:
        make_causal_lm_labels(input_ids, attention_mask)
    except ValueError as exc:
        assert "identical shapes" in str(exc)
    else:
        raise AssertionError("mismatched attention mask shape was accepted")
