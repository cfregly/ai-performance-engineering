"""High-throughput DataLoader patterns for storage-bound training (PyTorch 2.10)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset


@dataclass
class DummyDataset(Dataset):
    length: int = 10_000

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        # Use empty + fill for better performance than randn
        image = torch.empty(3, 224, 224)
        # Simple deterministic pattern based on index for reproducibility
        image.fill_(float((index % 256) / 255.0))
        label = index % 10
        return image, label


def make_dataloader(dataset: Dataset,
                    batch_size: int = 64,
                    num_workers: int = 4,
                    *,
                    pin_memory: bool = True,
                    prefetch_factor: int | None = 4,
                    persistent_workers: bool = True) -> DataLoader:
    if num_workers == 0:
        prefetch_factor = None
        persistent_workers = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        drop_last=True,
    )


def train_epoch(model: torch.nn.Module,
                loader: DataLoader,
                device: torch.device,
                optimizer: torch.optim.Optimizer,
                criterion: torch.nn.Module,
                stream: torch.cuda.Stream | None = None) -> None:
    model.train()
    stream = stream or torch.cuda.Stream(device=device) if device.type == "cuda" else None

    for inputs, targets in loader:
        if stream is not None:
            copy_stream = stream
            with torch.cuda.stream(copy_stream):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                inputs.record_stream(copy_stream)
                targets.record_stream(copy_stream)
            torch.cuda.current_stream(device).wait_stream(copy_stream)
        else:
            inputs = inputs.to(device)
            targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = DummyDataset()
    loader = make_dataloader(dataset, batch_size=64, num_workers=8)
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(3 * 224 * 224, 512),
        torch.nn.ReLU(inplace=True),
        torch.nn.Linear(512, 10),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    start = time.perf_counter()
    train_epoch(model, loader, device, optimizer, criterion)
    torch.cuda.synchronize() if device.type == "cuda" else None
    print(f"Epoch time: {time.perf_counter() - start:.2f}s")


if __name__ == "__main__":
    main()
