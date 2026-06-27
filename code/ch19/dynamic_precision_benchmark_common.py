"""Shared helpers for the Chapter 19 dynamic precision benchmark pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ch19.dynamic_precision_switching import compute_entropy, decode_with_dynamic_precision


@dataclass(frozen=True)
class DynamicPrecisionBenchmarkConfig:
    batch_size: int = 4
    prompt_len: int = 32
    max_steps: int = 32
    vocab_size: int = 512
    hidden_dim: int = 256


class HighConfidenceDecoder(nn.Module):
    """Toy decode model with stable top-1 logits across precision modes."""

    def __init__(self, vocab_size: int, hidden_dim: int, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim, device=device, dtype=dtype)
        self.proj_in = nn.Linear(hidden_dim, hidden_dim * 2, device=device, dtype=dtype)
        self.proj_out = nn.Linear(hidden_dim * 2, vocab_size, device=device, dtype=dtype)
        self.register_buffer("_target_boost", torch.tensor(16.0, device=device), persistent=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = x.mean(dim=1)
        x = F.gelu(self.proj_in(x))
        logits = self.proj_out(x).to(torch.float32)
        next_id = (input_ids[:, -1] + 1) % self.vocab_size
        logits.add_(-4.0)
        logits.scatter_add_(1, next_id.unsqueeze(-1), self._target_boost.expand(next_id.size(0), 1))
        return logits


def build_prompt(cfg: DynamicPrecisionBenchmarkConfig, device: torch.device) -> torch.Tensor:
    prompt = torch.arange(cfg.batch_size * cfg.prompt_len, device=device, dtype=torch.int64)
    return (prompt.reshape(cfg.batch_size, cfg.prompt_len) % cfg.vocab_size).contiguous()


def build_model(
    cfg: DynamicPrecisionBenchmarkConfig,
    device: torch.device,
    *,
    dtype: torch.dtype,
) -> HighConfidenceDecoder:
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = HighConfidenceDecoder(cfg.vocab_size, cfg.hidden_dim, dtype=dtype, device=device)
    model.eval()
    return model


@torch.no_grad()
def decode_fixed_precision(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    max_steps: int,
    device: torch.device,
) -> torch.Tensor:
    prompt = tokens.to(device, non_blocking=True)
    batch_size, prompt_len = prompt.shape
    generated = torch.empty(
        (batch_size, prompt_len + max_steps),
        device=device,
        dtype=prompt.dtype,
    )
    generated[:, :prompt_len].copy_(prompt)
    current_len = prompt_len
    for _ in range(max_steps):
        active_tokens = generated[:, :current_len]
        logits = model(input_ids=active_tokens)
        if hasattr(logits, "logits"):
            logits = logits.logits
        last_step_logits = logits if logits.dim() == 2 else logits[:, -1, :]
        next_token = torch.argmax(last_step_logits, dim=-1, keepdim=True)
        generated[:, current_len : current_len + 1].copy_(next_token)
        current_len += 1
    return generated[:, :current_len].contiguous()


@torch.no_grad()
def decode_host_policy_baseline(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    max_steps: int,
    device: torch.device,
) -> torch.Tensor:
    """Naive baseline: fixed precision plus host-visible confidence checks."""
    prompt = tokens.to(device, non_blocking=True)
    batch_size, prompt_len = prompt.shape
    generated = torch.empty(
        (batch_size, prompt_len + max_steps),
        device=device,
        dtype=prompt.dtype,
    )
    generated[:, :prompt_len].copy_(prompt)
    current_len = prompt_len
    for _ in range(max_steps):
        active_tokens = generated[:, :current_len]
        logits = model(input_ids=active_tokens)
        if hasattr(logits, "logits"):
            logits = logits.logits
        last_step_logits = logits if logits.dim() == 2 else logits[:, -1, :]
        # Deliberately conservative baseline: move confidence analysis to host.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        host_logits = last_step_logits.to(torch.float32).cpu()
        policy_metrics = torch.stack(
            (
                compute_entropy(host_logits).mean(),
                torch.softmax(host_logits, dim=-1).max(dim=-1).values.mean(),
                torch.topk(host_logits, k=2, dim=-1).values.mean(),
                torch.sort(host_logits, dim=-1).values[:, -1].mean(),
            )
        )
        _ = policy_metrics.tolist()
        next_token = torch.argmax(last_step_logits, dim=-1, keepdim=True)
        generated[:, current_len : current_len + 1].copy_(next_token)
        current_len += 1
    return generated[:, :current_len].contiguous()


@torch.no_grad()
def decode_dynamic_precision(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    max_steps: int,
    device: torch.device,
) -> Tuple[torch.Tensor, object]:
    return decode_with_dynamic_precision(
        model=model,
        tokens=tokens,
        max_steps=max_steps,
        device=device,
        prefer_bfloat16=True,
        enable_fp8=False,
        enable_fp4=True,
        enter_fp8_threshold=1e9,
        exit_fp8_threshold=1e9,
        enter_fp4_threshold=0.0,
        exit_fp4_threshold=0.0,
        fp4_memory_enter=0.0,
        fp4_memory_exit=0.0,
        reeval_interval=1,
        collect_stats=True,
    )
