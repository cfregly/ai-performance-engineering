"""Lightweight helpers shared by distributed training examples."""

from __future__ import annotations

import os

from core.common.device_utils import resolve_local_rank
import random
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

# Modern, lighter causal LM for training demos (open weights).
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"


def _ensure_writable_hf_cache() -> str:
    """Return a writable Hugging Face cache root and export HF_* env vars.

    Some benchmark hosts have a non-writable shared `~/.cache/huggingface`.
    We fall back to a repo-local cache to avoid hard torchrun failures.
    """
    candidates = []
    if os.environ.get("HF_HOME"):
        candidates.append(Path(os.environ["HF_HOME"]))
    candidates.append(Path.home() / ".cache" / "huggingface")
    candidates.append(Path(__file__).resolve().parents[1] / ".hf_cache")

    last_error: Exception | None = None
    for root in candidates:
        try:
            root.mkdir(parents=True, exist_ok=True)
            if os.access(root, os.W_OK):
                hub_dir = root / "hub"
                hub_dir.mkdir(parents=True, exist_ok=True)
                os.environ["HF_HOME"] = str(root)
                os.environ.setdefault("HF_HUB_CACHE", str(hub_dir))
                return str(root)
        except Exception as exc:  # pragma: no cover - defensive fallback
            last_error = exc
            continue

    raise RuntimeError(
        "Unable to initialize a writable Hugging Face cache directory"
    ) from last_error


def build_text_model(model_id: str = MODEL_NAME, dtype: torch.dtype = torch.bfloat16):
    """Return a small causal LM we can shard/replicate (eager attention)."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("build_text_model() requires the `transformers` package") from exc
    cache_dir = _ensure_writable_hf_cache()
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="eager",
        cache_dir=cache_dir,
    )


def build_text_model_flash(model_id: str = MODEL_NAME, dtype: torch.dtype = torch.bfloat16):
    """Return a small causal LM using flash attention (requires compatible kernels)."""
    try:
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError("build_text_model_flash() requires the `transformers` package") from exc
    cache_dir = _ensure_writable_hf_cache()
    return AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
        cache_dir=cache_dir,
    )


def build_smol_model(model_id: str = MODEL_NAME, dtype: torch.dtype = torch.bfloat16):
    """Alias for ZeRO demos; kept for backward compatibility."""
    return build_text_model(model_id=model_id, dtype=dtype)


def build_smol_model_flash(model_id: str = MODEL_NAME, dtype: torch.dtype = torch.bfloat16):
    """Alias for ZeRO demos using flash attention."""
    return build_text_model_flash(model_id=model_id, dtype=dtype)


def build_tokenizer(model_id: str = MODEL_NAME):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("build_tokenizer() requires the `transformers` package") from exc
    cache_dir = _ensure_writable_hf_cache()
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_smol_tokenizer(model_id: str = MODEL_NAME):
    """Alias for ZeRO demos; kept for backward compatibility."""
    return build_tokenizer(model_id=model_id)


def set_seed(seed: int = 1337) -> None:
    """Force determinism across python/torch/cuda for fair perf comparisons."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_training_matmul_policy() -> None:
    """Enable the shared matmul policy used by the DDP demos."""
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except AttributeError:
        pass


class cache_mesh:
    """Small decorator to reuse the last device mesh/process group when handy."""

    def __init__(self, func: Callable):
        self.func = func
        self._mesh = None

    def __call__(self, key, dm: dist.device_mesh.DeviceMesh | None = None):
        mesh = self._mesh if dm is None else dm
        return self.func(key, mesh)

    def register_mesh(self, mesh: dist.device_mesh.DeviceMesh):
        self._mesh = mesh
        return self


@cache_mesh
def get(key, dm: dist.device_mesh.DeviceMesh | None = None):
    """Convenience lookup for rank/world size with optional device mesh."""
    pg = dm.get_group() if dm else None

    if key == "ws":
        return dist.get_world_size(pg)
    if key in {"rank", "grank"}:
        return dist.get_rank(pg)
    if key == "pg":
        return pg
    if key == "lrank":
        return dm.get_local_rank() if dm else resolve_local_rank()
    raise ValueError(f"Invalid string: {key}")


def get_dataset():
    """Tokenize a tiny MRPC slice for quick training/debug cycles."""
    tokenizer = build_tokenizer()
    try:
        from datasets import load_dataset
        load_err = None
    except ImportError as exc:
        load_dataset = None
        load_err = exc

    def _build_synthetic_dataset():
        # Keep distributed training labs runnable without external dataset pulls.
        # The training examples overwrite `labels` with next-token targets, so
        # only token ids + attention mask are required here.
        try:
            vocab_size = int(getattr(tokenizer, "vocab_size", 0) or len(tokenizer))
        except Exception:
            vocab_size = 32000
        if vocab_size <= 0:
            vocab_size = 32000

        seq_len = 128
        num_train = 2048
        rng = random.Random(0)

        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_id is not None:
            eos_id = int(eos_id)

        def _sample_tokens() -> dict:
            ids = [rng.randrange(vocab_size) for _ in range(seq_len)]
            if eos_id is not None:
                ids[-1] = eos_id
            return {"input_ids": ids, "attention_mask": [1] * seq_len}

        return {"train": [_sample_tokens() for _ in range(num_train)]}

    if load_dataset is None:
        print(
            "[train_distributed] WARNING: datasets package unavailable; "
            f"falling back to synthetic dataset ({load_err}).",
            flush=True,
        )
        return _build_synthetic_dataset()

    try:
        dataset = load_dataset("glue", "mrpc")
    except Exception as exc:
        print(
            "[train_distributed] WARNING: failed to load GLUE/MRPC from Hugging Face; "
            f"falling back to synthetic dataset ({exc}).",
            flush=True,
        )
        return _build_synthetic_dataset()

    def tokenize_func(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            padding=True,
        )

    dataset = dataset.map(
        tokenize_func, batched=True, remove_columns=["idx", "sentence1", "sentence2"]
    )
    dataset = dataset.rename_columns({"label": "labels"})
    return dataset


def make_collate_fn(tokenizer, max_length: int | None = None):
    def collate(batch):
        return tokenizer.pad(
            batch,
            padding="max_length" if max_length is not None else "longest",
            max_length=max_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    return collate


def make_causal_lm_labels(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Clone token ids as causal-LM targets while excluding padded positions."""
    if input_ids.shape != attention_mask.shape:
        raise ValueError(
            "input_ids and attention_mask must have identical shapes for causal-LM labels"
        )
    labels = input_ids.clone()
    labels.masked_fill_(attention_mask == 0, -100)
    return labels


def build_dataloader(
    dataset,
    tokenizer,
    batch_size: int,
    shuffle: bool,
    drop_last: bool,
    num_workers: int = 2,
    prefetch_factor: int | None = 2,
    pin_memory: bool = True,
    distributed: bool = False,
    max_length: int | None = None,
):
    sampler = DistributedSampler(dataset, shuffle=shuffle) if distributed else None
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=shuffle if sampler is None else False,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=num_workers > 0,
        collate_fn=make_collate_fn(tokenizer, max_length=max_length),
    )


def is_namedtuple(data):
    """Heuristic check for namedtuple-like objects."""
    return (
        isinstance(data, tuple)
        and hasattr(data, "_asdict")
        and hasattr(data, "_fields")
    )


def honor_type(obj, generator):
    """Cast a generator output back to the input container type."""
    if is_namedtuple(obj):
        return type(obj)(*list(generator))
    return type(obj)(generator)


def is_torch_tensor(tensor):
    return isinstance(tensor, torch.Tensor)


def recursively_apply(
    func, data, *args, test_type=is_torch_tensor, error_on_other_type=False, **kwargs
):
    """Recursively apply fn to tensors inside nested structures."""
    if isinstance(data, (tuple, list)):
        return honor_type(
            data,
            (
                recursively_apply(
                    func,
                    o,
                    *args,
                    test_type=test_type,
                    error_on_other_type=error_on_other_type,
                    **kwargs,
                )
                for o in data
            ),
        )

    if isinstance(data, dict):
        return {
            k: recursively_apply(
                func,
                v,
                *args,
                test_type=test_type,
                error_on_other_type=error_on_other_type,
                **kwargs,
            )
            for k, v in data.items()
        }

    if test_type(data):
        return func(data, *args, **kwargs)

    if error_on_other_type:
        raise TypeError(
            f"Unsupported type {type(data)} passed to {func.__name__}. "
            f"Only nested containers of {test_type.__name__} objects allowed."
        )

    return data


def gather(tensor: torch.Tensor, pg: dist.ProcessGroup = None):
    """All-gather tensors while preserving structure."""

    def _gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        output_tensors = torch.empty(
            tensor.numel() * get("ws", pg), dtype=tensor.dtype, device=tensor.device
        )
        dist.all_gather(output_tensors, tensor, group=pg)
        return output_tensors.view(-1, *tensor.size()[1:])

    return recursively_apply(_gather_one, tensor, pg=pg, error_on_other_type=True)
