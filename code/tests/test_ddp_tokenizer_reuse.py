from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from labs.train_distributed.training_utils import utils


class _FakeDatasetDict(dict):
    def __init__(self) -> None:
        super().__init__(train=[])
        self.tokenizer_result = None
        self.renamed_columns = None

    def map(self, function, *, batched, remove_columns):
        assert batched is True
        assert remove_columns == ["idx", "sentence1", "sentence2"]
        self.tokenizer_result = function(
            {"sentence1": ["same model"], "sentence2": ["same vocabulary"]}
        )
        return self

    def rename_columns(self, mapping):
        self.renamed_columns = mapping
        return self


def _install_fake_hugging_face(monkeypatch):
    tokenizer_calls: list[tuple[str, str]] = []
    model_calls: list[tuple[str, str]] = []
    tokenizer_uses: list[object] = []
    datasets: list[_FakeDatasetDict] = []

    class FakeTokenizer:
        vocab_size = 32_000
        pad_token = None
        eos_token = "<eos>"

        def __call__(self, sentence1, sentence2, **kwargs):
            tokenizer_uses.append(self)
            assert sentence1 == ["same model"]
            assert sentence2 == ["same vocabulary"]
            assert kwargs == {"truncation": True, "padding": True}
            return {"input_ids": [[1, 2]], "attention_mask": [[1, 1]]}

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(cls, model_id, *, cache_dir):
            tokenizer_calls.append((model_id, cache_dir))
            return FakeTokenizer()

    class FakeAutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, model_id, *, cache_dir, **kwargs):
            model_calls.append((model_id, cache_dir))
            assert kwargs["attn_implementation"] == "eager"
            return SimpleNamespace(config=SimpleNamespace(vocab_size=32_000))

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = FakeAutoTokenizer
    transformers.AutoModelForCausalLM = FakeAutoModelForCausalLM
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    def load_dataset(name, subset):
        assert (name, subset) == ("glue", "mrpc")
        dataset = _FakeDatasetDict()
        datasets.append(dataset)
        return dataset

    datasets_module = ModuleType("datasets")
    datasets_module.load_dataset = load_dataset
    monkeypatch.setitem(sys.modules, "datasets", datasets_module)
    monkeypatch.setattr(utils, "_ensure_writable_hf_cache", lambda: "/tmp/fake-hf-cache")

    return tokenizer_calls, model_calls, tokenizer_uses, datasets


def test_ddp_dataset_reuses_one_tokenizer_with_model_vocabulary(monkeypatch) -> None:
    tokenizer_calls, model_calls, tokenizer_uses, datasets = _install_fake_hugging_face(
        monkeypatch
    )

    tokenizer = utils.build_tokenizer()
    dataset = utils.get_dataset(tokenizer=tokenizer)
    model = utils.build_text_model()

    assert tokenizer_calls == [(utils.MODEL_NAME, "/tmp/fake-hf-cache")]
    assert model_calls == [(utils.MODEL_NAME, "/tmp/fake-hf-cache")]
    assert tokenizer.vocab_size == model.config.vocab_size
    assert tokenizer_uses == [tokenizer]
    assert dataset is datasets[0]
    assert dataset.tokenizer_result == {
        "input_ids": [[1, 2]],
        "attention_mask": [[1, 1]],
    }
    assert dataset.renamed_columns == {"label": "labels"}


def test_get_dataset_still_constructs_a_tokenizer_by_default(monkeypatch) -> None:
    tokenizer_calls, _, tokenizer_uses, _ = _install_fake_hugging_face(monkeypatch)

    utils.get_dataset()

    assert tokenizer_calls == [(utils.MODEL_NAME, "/tmp/fake-hf-cache")]
    assert len(tokenizer_uses) == 1
