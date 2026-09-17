"""Exercise the actual FSDP data producers without launching CUDA training."""

import importlib
import json

import pytest
import torch

PRODUCERS = [
    "baseline_fsdp",
    "optimized_fsdp",
    "baseline_fsdp_multigpu",
    "optimized_fsdp_multigpu",
    "baseline_fsdp2",
    "optimized_fsdp2",
    "baseline_fsdp2_multigpu",
    "optimized_fsdp2_multigpu",
]


def build_loader(module, *, rank=0, world_size=2, seed=42):
    kwargs = {"seed": seed}
    if module.__name__.endswith("_multigpu"):
        kwargs.update(steps=2, grad_accum=2)
    loader, sampler = module._build_dataloader(8, 2, rank, world_size, **kwargs)
    # Worker process startup is exercised by the direct GPU entrypoints.
    loader.num_workers = 0
    return loader, sampler


@pytest.mark.parametrize("name", PRODUCERS)
def test_packed_batches_preserve_next_token_targets_and_sampler_seed(name, tmp_path, monkeypatch):
    module = importlib.import_module(f"labs.train_distributed.{name}")
    data = tmp_path / "packed.jsonl"
    rows = []
    for start in range(50):
        tokens = list(range(start, start + 9))
        rows.append({"input_ids": tokens[:-1], "labels": tokens[1:]})
    data.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    monkeypatch.setenv("AISP_TINYSTORIES_PACKED_PATH", str(data))
    monkeypatch.delenv("AISP_FSDP_FAST", raising=False)

    rank0, sampler0 = build_loader(module)
    rank1, sampler1 = build_loader(module, rank=1)
    other_seed, sampler_other = build_loader(module, seed=1042)
    assert set(sampler0).isdisjoint(set(sampler1))
    assert list(sampler0) != list(sampler_other)
    assert sampler0.seed == sampler1.seed == 42
    assert sampler_other.seed == 1042
    for loader in (rank0, rank1, other_seed):
        batches = list(loader)
        assert len(batches) == 12  # 25 rows/rank: no undersized thirteenth batch.
        for batch in batches:
            assert batch["input_ids"].shape == batch["labels"].shape == (2, 8)
            torch.testing.assert_close(batch["labels"], batch["input_ids"] + 1, rtol=0, atol=0)


@pytest.mark.parametrize("name", [name for name in PRODUCERS if name.endswith("_multigpu")])
def test_synthetic_generation_is_shifted_seeded_and_does_not_change_model_rng(name, monkeypatch):
    module = importlib.import_module(f"labs.train_distributed.{name}")
    monkeypatch.setenv("AISP_FSDP_FAST", "1")
    monkeypatch.setenv("AISP_TINYSTORIES_VOCAB", "31")
    torch.manual_seed(7)
    before = torch.get_rng_state().clone()
    loader, _ = build_loader(module, seed=42)
    batch = next(iter(loader))
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(batch["input_ids"][:, 1:], batch["labels"][:, :-1])
    repeat, _ = build_loader(module, seed=42)
    other, _ = build_loader(module, seed=1042)
    assert torch.equal(batch["input_ids"], next(iter(repeat))["input_ids"])
    assert not torch.equal(batch["input_ids"], next(iter(other))["input_ids"])
