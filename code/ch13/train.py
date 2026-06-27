#!/usr/bin/env python3

"""Minimal training loop used in Chapter 13 NUMA examples."""

import torch
from torch import nn, optim
from core.common.device_utils import get_preferred_device


def main() -> None:
    device, cuda_err = get_preferred_device()
    if cuda_err:
        print(f"WARNING: CUDA unavailable ({cuda_err}); using CPU.")
    torch.manual_seed(0)

    model = nn.Sequential(
        nn.Linear(1024, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 1),
    )
    model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.01)
    data = torch.randn(512, 1024, device=device)
    target = torch.randn(512, 1, device=device)

    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        out = model(data)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()

    print("Training loop completed; final loss {:.4f}".format(loss.item()))


if __name__ == "__main__":
    main()
