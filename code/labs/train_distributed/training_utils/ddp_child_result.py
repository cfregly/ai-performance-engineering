"""Post-training correctness evidence for the plain DDP benchmark pair."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from core.benchmark.verification import PrecisionFlags
from labs.train_distributed.training_utils.child_result import (
    TorchrunChildResultContract,
    child_result_requested,
    write_training_child_result,
)
from labs.train_distributed.training_utils.utils import make_causal_lm_labels, set_seed

# Start exact and let target-GPU full-output evidence justify any future bound.
# A generic Llama inference tolerance does not establish training equivalence.
DDP_TRAINING_OUTPUT_TOLERANCE = (0.0, 0.0)
DDP_ADAMW_BETAS = (0.9, 0.95)
DDP_ADAMW_WEIGHT_DECAY = 0.1


def make_ddp_adamw(
    parameters: Any, learning_rate: float, *, prefer_fused: bool
) -> torch.optim.AdamW:
    """Construct both arms with identical AdamW math configuration."""

    options = {
        "lr": learning_rate,
        "betas": DDP_ADAMW_BETAS,
        "weight_decay": DDP_ADAMW_WEIGHT_DECAY,
    }
    if prefer_fused:
        try:
            return torch.optim.AdamW(parameters, **options, fused=True)
        except TypeError:
            pass
    return torch.optim.AdamW(parameters, **options, fused=False)


def initialize_ddp_seed(*, fallback: int = 42) -> int:
    """Preserve a harness-owned seed, falling back only for direct scripts."""

    if not child_result_requested():
        set_seed(fallback)
    return int(torch.initial_seed())


def bind_distributed_sampler_seed(dataloader: Any, seed: int) -> None:
    """Make DistributedSampler respond to the active harness seed."""

    sampler = getattr(dataloader, "sampler", None)
    set_epoch = getattr(sampler, "set_epoch", None)
    if callable(set_epoch):
        set_epoch(seed)


def make_ddp_child_result_contract(*, multigpu: bool) -> TorchrunChildResultContract:
    """Describe the bounded validation work produced after timed DDP training."""

    return TorchrunChildResultContract(
        profile=(
            "labs/train_distributed:ddp-multigpu-post-training-v1"
            if multigpu
            else "labs/train_distributed:ddp-post-training-v1"
        ),
        input_names=("input_ids", "attention_mask", "labels"),
        output_names=("logits", "loss"),
        per_rank_batch_size=32 if multigpu else 16,
        parameter_count=None,
        precision_flags=PrecisionFlags(bf16=True, tf32=True),
        output_tolerance=DDP_TRAINING_OUTPUT_TOLERANCE,
        independent_reference="same-trained-weights-unwrapped-eager-forward-only-check",
        collective_type="all_reduce" if multigpu else None,
        collective_algorithm=("ddp-gradient-sum-divide-world-size" if multigpu else None),
        max_rank_payload_bytes=1024 * 1024 * 1024,
    )


def _unwrap_training_module(candidate_model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap compile/DDP shells without copying or rebuilding trained weights."""

    current = candidate_model
    seen: set[int] = set()
    while id(current) not in seen:
        seen.add(id(current))
        next_module = getattr(current, "_orig_mod", None)
        if not isinstance(next_module, torch.nn.Module):
            next_module = getattr(current, "module", None)
        if not isinstance(next_module, torch.nn.Module):
            break
        current = next_module
    return current


def _actual_final_batch(batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    expected = ("input_ids", "attention_mask", "labels")
    if any(name not in batch or not isinstance(batch[name], torch.Tensor) for name in expected):
        raise RuntimeError(
            "DDP child verification requires tensor input_ids, attention_mask and labels "
            "from the final completed training step"
        )
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    if (
        input_ids.ndim != 2
        or attention_mask.shape != input_ids.shape
        or labels.shape != input_ids.shape
        or input_ids.shape[0] < 1
    ):
        raise RuntimeError("DDP child verification final batch has an invalid causal-LM layout")

    result = {name: batch[name].detach().clone() for name in expected}
    expected_labels = make_causal_lm_labels(result["input_ids"], result["attention_mask"])
    if not torch.equal(result["labels"], expected_labels):
        raise RuntimeError(
            "DDP child verification labels do not match the consumed causal-LM input"
        )
    return result


def _sensitivity_input(
    inputs: Mapping[str, torch.Tensor], *, vocab_size: int
) -> dict[str, torch.Tensor]:
    if vocab_size < 2:
        raise RuntimeError("DDP child verification requires a model vocabulary with >=2 tokens")
    changed = {name: value.detach().clone() for name, value in inputs.items()}
    selected: tuple[int, int] | None = None
    for row in range(changed["input_ids"].shape[0]):
        active = torch.nonzero(changed["attention_mask"][row] != 0, as_tuple=False).flatten()
        if active.numel() >= 2:
            selected = (row, int(active[1].item()))
            break
    if selected is None:
        raise RuntimeError("DDP child verification sensitivity input needs two active tokens")
    row, position = selected
    token = int(changed["input_ids"][row, position].item())
    if token < 0 or token >= vocab_size:
        raise RuntimeError("DDP child verification input contains an out-of-vocabulary token")
    changed["input_ids"][row, position] = (token + 1) % vocab_size
    changed["labels"] = make_causal_lm_labels(changed["input_ids"], changed["attention_mask"])
    return changed


def _full_causal_lm_outputs(
    model: torch.nn.Module, inputs: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    result: Any = model(**inputs)
    logits = getattr(result, "logits", None)
    loss = getattr(result, "loss", None)
    if not isinstance(logits, torch.Tensor) or not isinstance(loss, torch.Tensor):
        raise RuntimeError("DDP child verification model must return tensor logits and loss")
    if logits.ndim != 3 or logits.shape[:2] != inputs["input_ids"].shape:
        raise RuntimeError("DDP child verification model returned an invalid full-logit layout")
    if loss.numel() != 1:
        raise RuntimeError("DDP child verification model returned a non-scalar loss")
    return {"logits": logits, "loss": loss.reshape(())}


def publish_ddp_child_result(
    *,
    candidate_model: torch.nn.Module,
    reference_model: torch.nn.Module,
    final_batch: Mapping[str, torch.Tensor] | None,
    completed_iterations: int,
) -> Path | None:
    """Publish bounded validation forwards after, and outside, the timed hot path."""

    if not child_result_requested():
        return None
    if final_batch is None:
        raise RuntimeError("DDP child verification has no completed training batch")
    if _unwrap_training_module(candidate_model) is not reference_model:
        raise RuntimeError(
            "DDP child verification reference must be the exact unwrapped trained model"
        )
    config = getattr(reference_model, "config", None)
    try:
        vocab_size = int(config.vocab_size)
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("DDP child verification model has no valid vocabulary size") from exc

    inputs = _actual_final_batch(final_batch)
    changed_inputs = _sensitivity_input(inputs, vocab_size=vocab_size)
    candidate_training = candidate_model.training
    reference_training = reference_model.training
    try:
        candidate_model.eval()
        reference_model.eval()
        with torch.inference_mode():
            outputs = _full_causal_lm_outputs(candidate_model, inputs)
            reference_outputs = _full_causal_lm_outputs(reference_model, inputs)
            sensitivity_outputs = _full_causal_lm_outputs(candidate_model, changed_inputs)
            sensitivity_reference_outputs = _full_causal_lm_outputs(reference_model, changed_inputs)
    finally:
        candidate_model.train(candidate_training)
        reference_model.train(reference_training)

    parameter_count = sum(parameter.numel() for parameter in reference_model.parameters())
    return write_training_child_result(
        inputs=inputs,
        outputs=outputs,
        reference_outputs=reference_outputs,
        sensitivity_inputs=changed_inputs,
        sensitivity_outputs=sensitivity_outputs,
        sensitivity_reference_outputs=sensitivity_reference_outputs,
        completed_iterations=completed_iterations,
        parameter_count=parameter_count,
    )


__all__ = [
    "DDP_ADAMW_BETAS",
    "DDP_ADAMW_WEIGHT_DECAY",
    "DDP_TRAINING_OUTPUT_TOLERANCE",
    "bind_distributed_sampler_seed",
    "initialize_ddp_seed",
    "make_ddp_adamw",
    "make_ddp_child_result_contract",
    "publish_ddp_child_result",
]
