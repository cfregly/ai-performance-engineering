from __future__ import annotations

import copy
import math
from typing import Any

import pytest
import torch

from labs.train_distributed.training_utils.utils import make_collate_fn


class StrictCpuTokenizer:
    """Small CPU tokenizer pad surface with Hugging Face padding semantics."""

    model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
    pad_token_id = 99
    pad_token_type_id = 0

    def __init__(self, *, padding_side: str, truncation_side: str) -> None:
        self.padding_side = padding_side
        self.truncation_side = truncation_side
        self.last_features = None
        self.last_kwargs: dict[str, Any] = {}

    def pad(
        self,
        features,
        *,
        padding,
        max_length,
        pad_to_multiple_of,
        return_tensors,
    ):
        self.last_features = features
        self.last_kwargs = {
            "padding": padding,
            "max_length": max_length,
            "pad_to_multiple_of": pad_to_multiple_of,
            "return_tensors": return_tensors,
        }
        lengths = [len(feature["input_ids"]) for feature in features]
        if padding == "max_length":
            if any(length > max_length for length in lengths):
                raise ValueError(
                    f"expected sequence of length {max(lengths)} at dim 1 (got {max_length})"
                )
            padded_length = max_length
        else:
            assert padding == "longest"
            assert max_length is None
            padded_length = max(lengths)
        if pad_to_multiple_of is not None:
            padded_length = math.ceil(padded_length / pad_to_multiple_of) * pad_to_multiple_of

        padded: dict[str, list[Any]] = {key: [] for key in features[0]}
        pad_values = {
            "input_ids": self.pad_token_id,
            "attention_mask": 0,
            "token_type_ids": self.pad_token_type_id,
            "special_tokens_mask": 1,
        }
        for feature in features:
            for key, value in feature.items():
                if key == "labels":
                    padded[key].append(value)
                    continue
                pad_count = padded_length - len(value)
                padding_values = [pad_values[key]] * pad_count
                if self.padding_side == "right":
                    padded[key].append([*value, *padding_values])
                else:
                    padded[key].append([*padding_values, *value])
        assert return_tensors == "pt"
        return {key: torch.tensor(value) for key, value in padded.items()}


def _pretokenized_feature(
    token_ids: list[int], *, prepadding_side: str, padded_length: int, label: int
) -> dict[str, Any]:
    pad_count = padded_length - len(token_ids)
    pads = [StrictCpuTokenizer.pad_token_id] * pad_count
    zeros = [0] * pad_count
    ones = [1] * pad_count
    token_types = [100 + token_id for token_id in token_ids]
    special_tokens = [0] * len(token_ids)
    if prepadding_side == "right":
        return {
            "input_ids": [*token_ids, *pads],
            "attention_mask": [1] * len(token_ids) + zeros,
            "token_type_ids": [*token_types, *zeros],
            "special_tokens_mask": [*special_tokens, *ones],
            "labels": label,
        }
    return {
        "input_ids": [*pads, *token_ids],
        "attention_mask": [*zeros, *([1] * len(token_ids))],
        "token_type_ids": [*zeros, *token_types],
        "special_tokens_mask": [*ones, *special_tokens],
        "labels": label,
    }


def _padded(values: list[int], *, side: str, pad_value: int, length: int = 8) -> list[int]:
    pads = [pad_value] * (length - len(values))
    return [*values, *pads] if side == "right" else [*pads, *values]


@pytest.mark.parametrize("prepadding_side", ["left", "right"])
@pytest.mark.parametrize("truncation_side", ["left", "right"])
@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_fixed_length_collator_strips_existing_padding_then_truncates_and_repads(
    prepadding_side: str,
    truncation_side: str,
    padding_side: str,
) -> None:
    tokenizer = StrictCpuTokenizer(padding_side=padding_side, truncation_side=truncation_side)
    batch = [
        _pretokenized_feature(
            list(range(1, 11)), prepadding_side=prepadding_side, padded_length=12, label=7
        ),
        _pretokenized_feature(
            [21, 22, 23], prepadding_side=prepadding_side, padded_length=12, label=8
        ),
    ]
    original = copy.deepcopy(batch)

    result = make_collate_fn(tokenizer, max_length=6)(batch)

    selected = list(range(5, 11)) if truncation_side == "left" else list(range(1, 7))
    expected_tokens = [selected, [21, 22, 23]]
    assert result["input_ids"].tolist() == [
        _padded(values, side=padding_side, pad_value=99) for values in expected_tokens
    ]
    assert result["attention_mask"].tolist() == [
        _padded([1] * len(values), side=padding_side, pad_value=0) for values in expected_tokens
    ]
    assert result["token_type_ids"].tolist() == [
        _padded([100 + value for value in values], side=padding_side, pad_value=0)
        for values in expected_tokens
    ]
    assert result["special_tokens_mask"].tolist() == [
        _padded([0] * len(values), side=padding_side, pad_value=1) for values in expected_tokens
    ]
    assert result["labels"].tolist() == [7, 8]
    assert tokenizer.last_kwargs == {
        "padding": "max_length",
        "max_length": 6,
        "pad_to_multiple_of": 8,
        "return_tensors": "pt",
    }
    assert batch == original


def test_fixed_length_collator_handles_the_131_and_128_token_mrpc_case() -> None:
    tokenizer = StrictCpuTokenizer(padding_side="right", truncation_side="right")
    batch = [
        _pretokenized_feature(
            list(range(1, 132)), prepadding_side="right", padded_length=131, label=1
        ),
        _pretokenized_feature(
            list(range(201, 328)), prepadding_side="right", padded_length=128, label=0
        ),
    ]

    result = make_collate_fn(tokenizer, max_length=128)(batch)

    assert result["input_ids"].shape == (2, 128)
    assert result["input_ids"][0].tolist() == list(range(1, 129))
    assert result["input_ids"][1, :127].tolist() == list(range(201, 328))
    assert result["input_ids"][1, 127].item() == tokenizer.pad_token_id
    assert result["attention_mask"][1].tolist() == [1] * 127 + [0]
    assert result["labels"].tolist() == [1, 0]


@pytest.mark.parametrize("padding_side", ["left", "right"])
@pytest.mark.parametrize("truncation_side", ["left", "right"])
def test_fixed_length_collator_with_real_huggingface_tokenizer(
    padding_side: str,
    truncation_side: str,
) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    # Construct the real tokenizer entirely locally; no model or vocab downloads.
    words = [f"token{index}" for index in range(131)]
    vocab = {"[PAD]": 0, "[UNK]": 1, **{word: index + 2 for index, word in enumerate(words)}}
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="[PAD]",
        unk_token="[UNK]",
        truncation_side=truncation_side,
        padding_side="right" if padding_side == "left" else "left",
    )
    batch = [
        dict(tokenizer(" ".join(words), return_special_tokens_mask=True), labels=1),
        dict(
            tokenizer(
                " ".join(words[:127]),
                padding="max_length",
                max_length=128,
                return_special_tokens_mask=True,
            ),
            labels=0,
        ),
    ]
    original = copy.deepcopy(batch)
    assert [len(feature["input_ids"]) for feature in batch] == [131, 128]
    # Repad on the opposite side to make removal of existing padding observable.
    tokenizer.padding_side = padding_side

    result = make_collate_fn(tokenizer, max_length=128)(batch)

    selected_words = words[-128:] if truncation_side == "left" else words[:128]
    long_ids = tokenizer.convert_tokens_to_ids(selected_words)
    short_ids = tokenizer.convert_tokens_to_ids(words[:127])
    pad_index = 0 if padding_side == "left" else 127
    short_ids.insert(pad_index, tokenizer.pad_token_id)
    short_mask = [1] * 128
    short_mask[pad_index] = 0
    short_special_mask = [0] * 128
    short_special_mask[pad_index] = 1
    assert result["input_ids"].shape == (2, 128)
    assert result["input_ids"].tolist() == [long_ids, short_ids]
    assert result["attention_mask"].tolist() == [[1] * 128, short_mask]
    assert result["token_type_ids"].tolist() == [[0] * 128, [0] * 128]
    assert result["special_tokens_mask"].tolist() == [[0] * 128, short_special_mask]
    assert result["labels"].tolist() == [1, 0]
    assert batch == original


def test_dynamic_padding_passes_original_features_through_without_truncation() -> None:
    tokenizer = StrictCpuTokenizer(padding_side="right", truncation_side="left")
    batch = [
        _pretokenized_feature(
            list(range(1, 11)), prepadding_side="right", padded_length=10, label=7
        ),
        _pretokenized_feature([21, 22, 23], prepadding_side="right", padded_length=3, label=8),
    ]

    result = make_collate_fn(tokenizer)(batch)

    assert tokenizer.last_features is batch
    assert tokenizer.last_kwargs == {
        "padding": "longest",
        "max_length": None,
        "pad_to_multiple_of": 8,
        "return_tensors": "pt",
    }
    assert result["input_ids"].shape == (2, 16)
    assert result["input_ids"][0, :10].tolist() == list(range(1, 11))
