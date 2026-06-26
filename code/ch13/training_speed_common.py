"""Shared model/config helpers for the Chapter 13 training-speed benchmark."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TrainingSpeedConfig:
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    seq_len: int = 256
    batch_size: int = 8
    vocab_size: int = 8192


class TrainingSpeedModel(nn.Module):
    """Small fixed-shape transformer used for end-to-end training-speed checks."""

    def __init__(self, cfg: TrainingSpeedConfig):
        super().__init__()
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_embedding = nn.Embedding(cfg.seq_len, cfg.hidden_dim)
        self.register_buffer(
            "_position_ids",
            torch.arange(cfg.seq_len, dtype=torch.long).unsqueeze(0),
            persistent=False,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.hidden_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.ln_f = nn.LayerNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        pos_ids = self._position_ids[:, :seq_len].expand(batch_size, -1)
        x = self.embedding(input_ids) + self.pos_embedding(pos_ids)
        x = self.transformer(x)
        x = self.ln_f(x)
        return self.lm_head(x)


def make_training_batch(
    cfg: TrainingSpeedConfig,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
    targets = torch.randint(0, cfg.vocab_size, (cfg.batch_size, cfg.seq_len), device=device)
    return input_ids, targets
