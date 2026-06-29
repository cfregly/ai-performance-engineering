"""Shared helpers for the FSDP training demos."""

from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

if TYPE_CHECKING:  # pragma: no cover
    from datasets import Dataset
    from transformers import PreTrainedTokenizerBase
else:
    Dataset = Any  # type: ignore[misc,assignment]
    PreTrainedTokenizerBase = Any  # type: ignore[misc,assignment]


def setup_tokenizer(model_id: str) -> PreTrainedTokenizerBase:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("setup_tokenizer() requires the `transformers` package") from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


@contextmanager
def _main_process_first(is_main_process: bool):
    """Ensure rank-0 runs the protected block before other ranks."""

    distributed_ready = dist.is_available() and dist.is_initialized()
    if not distributed_ready:
        yield
        return

    if is_main_process:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield


def load_tinystories(
    tokenizer: PreTrainedTokenizerBase,
    seq_len: int,
    *,
    is_main_process: bool,
    split: str | None = None,
) -> Dataset:
    """Tokenize and pack TinyStories into contiguous blocks for LM training."""
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError as exc:
        raise RuntimeError("load_tinystories() requires the `datasets` package") from exc
    def _parse_streaming_limit(dataset_split: str) -> int:
        match = re.search(r":(\\d+)\\]$", dataset_split)
        if match:
            return int(match.group(1))
        return int(os.getenv("AISP_TINYSTORIES_STREAMING_MAX", "512"))

    with _main_process_first(is_main_process):
        local_path = os.getenv("AISP_TINYSTORIES_LOCAL_PATH")
        if local_path:
            path = Path(local_path)
            if not path.exists():
                raise FileNotFoundError(f"TinyStories local path not found: {path}")
            if path.is_dir():
                raw_dataset = load_from_disk(str(path))
            else:
                raw_dataset = load_dataset("json", data_files=str(path), split="train")
        else:
            dataset_split = split or os.getenv("AISP_TINYSTORIES_SPLIT", "train[:5%]")
            streaming = os.getenv("AISP_TINYSTORIES_STREAMING") == "1"
            if streaming:
                limit = _parse_streaming_limit(dataset_split)
                raw_iter = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
                samples = []
                for idx, example in enumerate(raw_iter):
                    if idx >= limit:
                        break
                    samples.append(example)
                from datasets import Dataset

                raw_dataset = Dataset.from_list(samples)
            else:
                raw_dataset = load_dataset("roneneldan/TinyStories", split=dataset_split)

    def tokenize_function(examples):
        tokens = tokenizer(
            examples["text"],
            padding=False,
            truncation=True,
            max_length=seq_len + 1,
            return_tensors=None,
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    with _main_process_first(is_main_process):
        tokenized_dataset = raw_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    def pack_sequences(examples):
        flat_tokens = []
        for input_ids in examples["input_ids"]:
            flat_tokens.extend(input_ids)

        num_sequences = len(flat_tokens) // (seq_len + 1)
        packed_input_ids = []
        packed_labels = []

        for i in range(num_sequences):
            start_idx = i * (seq_len + 1)
            end_idx = start_idx + (seq_len + 1)
            sequence = flat_tokens[start_idx:end_idx]
            packed_input_ids.append(sequence[:-1])
            packed_labels.append(sequence[1:])

        return {"input_ids": packed_input_ids, "labels": packed_labels}

    with _main_process_first(is_main_process):
        packed_dataset = tokenized_dataset.map(
            pack_sequences,
            batched=True,
            remove_columns=tokenized_dataset.column_names,
            batch_size=1000,
        )

    return packed_dataset.shuffle(seed=2025)


def load_tinystories_packed(path: str | Path, seq_len: int, *, is_main_process: bool) -> Dataset:
    """Load a pre-packed TinyStories dataset (input_ids/labels already aligned)."""
    try:
        from datasets import load_dataset, load_from_disk
    except ImportError:
        load_dataset = None
        load_from_disk = None

    packed_path = Path(path)
    if not packed_path.exists():
        raise FileNotFoundError(f"Packed TinyStories path not found: {packed_path}")

    with _main_process_first(is_main_process):
        if packed_path.is_dir():
            if load_from_disk is None:
                raise RuntimeError(
                    "load_tinystories_packed() requires the `datasets` package when loading a directory dataset."
                )
            dataset = load_from_disk(str(packed_path))
        else:
            if load_dataset is None:
                # Minimal fallback to avoid the optional `datasets` dependency for the bundled JSONL.
                samples: list[dict[str, Any]] = []
                with packed_path.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            payload = json.loads(line)
                        except json.JSONDecodeError as exc:
                            raise ValueError(f"Invalid JSON in packed TinyStories file: {packed_path}") from exc
                        if not isinstance(payload, dict):
                            raise ValueError(f"Packed TinyStories rows must be objects (dict), got: {type(payload)}")
                        samples.append(payload)

                class _JsonlPackedDataset(torch.utils.data.Dataset):
                    def __init__(self, rows: list[dict[str, Any]]):
                        self._rows = rows
                        self.column_names = list(rows[0].keys()) if rows else []

                    def __len__(self) -> int:
                        return len(self._rows)

                    def __getitem__(self, idx: int) -> dict[str, Any]:
                        return self._rows[idx]

                    def shuffle(self, seed: int = 0):
                        rng = random.Random(int(seed))
                        indices = list(range(len(self._rows)))
                        rng.shuffle(indices)
                        return _JsonlPackedDataset([self._rows[i] for i in indices])

                dataset = _JsonlPackedDataset(samples)
            else:
                dataset = load_dataset("json", data_files=str(packed_path), split="train")

    columns = set(getattr(dataset, "column_names", []))
    if not {"input_ids", "labels"}.issubset(columns):
        raise ValueError("Packed TinyStories dataset must include input_ids and labels columns.")
    if len(dataset) == 0:
        raise ValueError("Packed TinyStories dataset is empty.")

    sample = dataset[0]
    if len(sample["input_ids"]) != seq_len or len(sample["labels"]) != seq_len:
        raise ValueError(f"Packed TinyStories sequences must match seq_len={seq_len}.")

    return dataset.shuffle(seed=2025)

def create_collate_fn():
    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}

    return collate_fn


def get_model_flops_per_token(config, seq_len: int) -> float:
    """Estimate FLOPs/token using a rough decoder-only formula."""
    head_dim = config.hidden_size // config.num_attention_heads
    mlp = 18 * config.hidden_size * config.intermediate_size
    attn_proj = 12 * head_dim * (config.num_attention_heads + config.num_key_value_heads)
    attn_scores = 12 * config.num_attention_heads * head_dim * seq_len
    return (mlp + attn_proj + attn_scores) * config.num_hidden_layers


class ThroughputTracker:
    """Tracks warmup and steady-state throughput plus peak memory."""

    def __init__(self, warmup_steps: int = 10):
        self.warmup_steps = warmup_steps
        self._empty_metrics: dict = {}
        self._metrics: dict = {}
        self.reset()

    def reset(self):
        self.start_time = None
        self.tokens = 0
        self.steps_seen = 0
        self.in_warmup = True
        self._metrics.clear()

    def step(self, tokens: int, flops_per_token: float | None = None) -> dict:
        self.steps_seen += 1

        if self.steps_seen == self.warmup_steps:
            self.start_time = time.perf_counter()
            self.tokens = 0
            self.in_warmup = False
            return self._empty_metrics

        if self.in_warmup or self.start_time is None:
            return self._empty_metrics

        self.tokens += tokens
        elapsed = time.perf_counter() - self.start_time
        steps = self.steps_seen - self.warmup_steps
        metrics = self._metrics
        metrics["tokens_per_second"] = self.tokens / elapsed
        metrics["steps_per_second"] = steps / elapsed
        metrics["total_tokens"] = self.tokens
        metrics["total_time"] = elapsed

        if flops_per_token is not None:
            metrics["tflops_per_device"] = (flops_per_token * self.tokens) / (elapsed * 1e12)
        else:
            metrics.pop("tflops_per_device", None)

        return metrics

    @staticmethod
    def format(metrics: dict, include_memory: bool = False) -> str:
        msg = (
            f" | steps/s={metrics['steps_per_second']:.2f}"
            f" | toks/s={metrics['tokens_per_second']:.0f}"
        )
        if "tflops_per_device" in metrics:
            msg += f" | TFLOPs/rank={metrics['tflops_per_device']:.2f}"
        if include_memory:
            msg += (
                f" | peak_active={metrics.get('peak_memory_active', 0):.2f}GB"
                f" | peak_reserved={metrics.get('peak_memory_reserved', 0):.2f}GB"
            )
        return msg


def gpu_memory_usage(device=0):
    """Return a small dict of peak memory stats and reset counters."""
    backend = torch.cuda
    dev = torch.device(f"cuda:{device}")
    gib = 1024**3
    stats = backend.memory_stats(dev)
    mem = {
        "peak_memory_active": stats.get("active_bytes.all.peak", 0) / gib,
        "peak_memory_alloc": backend.max_memory_allocated(dev) / gib,
        "peak_memory_reserved": backend.max_memory_reserved(dev) / gib,
    }
    backend.reset_peak_memory_stats(dev)
    return mem
