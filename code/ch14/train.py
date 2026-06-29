#!/usr/bin/env python3

from core.benchmark.utils import scalar_tensor_to_float
from core.utils import compile_utils as _compile_utils_patch  # noqa: F401

"""Minimal torch.compile training harness for Chapter 14."""

import torch
from torch import nn, optim


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.Sequential(nn.Linear(1024, 2048), nn.ReLU(inplace=True), nn.Linear(2048, 2048)).to(device)
    model = torch.compile(model, mode="reduce-overhead")
    opt = optim.AdamW(model.parameters(), lr=1e-3)

    data = torch.randn(16, 1024, device=device)
    target = torch.randn(16, 2048, device=device)

    opt.zero_grad(set_to_none=True)
    out = model(data)
    loss = nn.functional.mse_loss(out, target)
    loss.backward()
    opt.step()

    print("ch14 train.py completed with loss {:.4f}".format(scalar_tensor_to_float(loss)))


if __name__ == "__main__":
    main()
