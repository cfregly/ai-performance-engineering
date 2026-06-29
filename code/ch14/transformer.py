"""Lightweight transformer stub matching Chapter 14 trace paths."""

from core.benchmark.utils import scalar_tensor_to_float

import torch
from torch import nn


class MiniTransformer(nn.Module):
    def __init__(self, hidden: int = 1024) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden, num_heads=8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, hidden * 4),
            nn.GELU(),
            nn.Linear(hidden * 4, hidden),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = x + attn_out
        x = x + self.ffn(x)
        return x


def demo() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniTransformer().to(device)
    data = torch.randn(4, 128, 1024, device=device)
    out = model(data)
    print("transformer.py demo rms:", scalar_tensor_to_float(out.norm()))


if __name__ == "__main__":
    demo()
