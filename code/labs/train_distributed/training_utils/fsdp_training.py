"""Shared training semantics for the FSDP benchmark pairs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional

from labs.train_distributed.training_utils.child_result import child_result_requested
from labs.train_distributed.training_utils.utils import set_seed

if TYPE_CHECKING:
    from torch.distributed.fsdp import MixedPrecision


def initialize_fsdp_seed(*, fallback: int = 42) -> int:
    """Preserve a harness-owned seed, falling back only for direct execution."""

    if not child_result_requested():
        set_seed(fallback)
    return int(torch.initial_seed())


def move_fsdp_model_to_device(
    model: torch.nn.Module, device: torch.device | str | int
) -> torch.nn.Module:
    """Move already-typed parameters without rounding FP32 rotary buffers.

    Transformers constructs BF16 parameters and FP32 inverse frequencies.
    Casting the entire model to BF16 rounds those frequencies before RoPE's
    FP32 arithmetic, so the position-dependent error cannot be recovered there.
    """

    return model.to(device=device)


def fsdp1_mixed_precision_policy() -> MixedPrecision:
    """Keep position-frequency buffers in FP32 through FSDP1's forward cast."""

    from torch.distributed.fsdp import MixedPrecision

    return MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.float32,
    )


def shifted_causal_lm_loss(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    """Return FP32 causal-LM loss for labels already shifted to the next token.

    ``batch["labels"]`` must align with the logits at the same positions. The
    model is called without labels so Transformers cannot shift those targets a
    second time.
    """

    input_ids = batch.get("input_ids")
    labels = batch.get("labels")
    attention_mask = batch.get("attention_mask")
    if not isinstance(input_ids, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("FSDP loss requires tensor input_ids and labels")
    if input_ids.ndim != 2 or labels.shape != input_ids.shape:
        raise ValueError("FSDP input_ids and labels must have the same rank-2 shape")
    if labels.dtype != torch.long:
        raise TypeError("FSDP causal-LM labels must have torch.long dtype")
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise TypeError("FSDP attention_mask must be a tensor when provided")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("FSDP attention_mask must match the input_ids shape")

    model_inputs: dict[str, torch.Tensor] = {"input_ids": input_ids}
    if attention_mask is not None:
        model_inputs["attention_mask"] = attention_mask
    result: Any = model(**model_inputs)
    logits = getattr(result, "logits", None)
    if not isinstance(logits, torch.Tensor):
        raise RuntimeError("FSDP causal-LM model must return tensor logits")
    if logits.ndim != 3 or logits.shape[:-1] != labels.shape or logits.shape[-1] < 1:
        raise RuntimeError("FSDP causal-LM model returned an invalid full-logit layout")

    return functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.to(device=logits.device).reshape(-1),
        ignore_index=-100,
    )


def validate_fsdp_training_args(args: Any) -> None:
    """Validate FSDP CLI invariants without changing optimizer-step semantics."""

    for name in ("steps", "sequence_length", "micro_batch_size", "grad_accum"):
        value = getattr(args, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"FSDP {name} must be a positive integer")

    learning_rate = getattr(args, "learning_rate", None)
    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, Real)
        or not math.isfinite(float(learning_rate))
        or learning_rate <= 0
    ):
        raise ValueError("FSDP learning_rate must be finite and positive")


__all__ = [
    "initialize_fsdp_seed",
    "move_fsdp_model_to_device",
    "fsdp1_mixed_precision_policy",
    "shifted_causal_lm_loss",
    "validate_fsdp_training_args",
]
