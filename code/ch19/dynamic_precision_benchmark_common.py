"""Shared helpers for the Chapter 19 dynamic precision benchmark pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ch19.dynamic_precision_switching import (
    DynamicPrecisionWorkspace,
    compute_entropy,
    decode_with_dynamic_precision,
)


@dataclass(frozen=True)
class DynamicPrecisionBenchmarkConfig:
    batch_size: int = 4
    prompt_len: int = 32
    max_steps: int = 32
    vocab_size: int = 512
    hidden_dim: int = 256


@dataclass
class FixedDecodeWorkspace:
    """Reusable buffers for fixed-precision and host-policy decode loops."""

    generated: torch.Tensor
    next_token: torch.Tensor
    next_token_values: torch.Tensor | None = None
    host_logits_buffer: torch.Tensor | None = None
    policy_metrics_buffer: torch.Tensor | None = None
    policy_metric_values: list[float] | None = None


class HighConfidenceDecoder(nn.Module):
    """Toy decode model with stable top-1 logits across precision modes."""

    def __init__(self, vocab_size: int, hidden_dim: int, *, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_dim, device=device, dtype=dtype)
        self.proj_in = nn.Linear(hidden_dim, hidden_dim * 2, device=device, dtype=dtype)
        self.proj_out = nn.Linear(hidden_dim * 2, vocab_size, device=device, dtype=dtype)
        self.register_buffer("_target_boost", torch.tensor(16.0, device=device), persistent=False)
        self._target_boost_views: dict[tuple[int, torch.device], torch.Tensor] = {}

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = x.mean(dim=1)
        x = F.gelu(self.proj_in(x))
        logits = self.proj_out(x).to(torch.float32)
        next_id = (input_ids[:, -1] + 1) % self.vocab_size
        logits.add_(-4.0)
        boost_key = (int(next_id.size(0)), logits.device)
        target_boost = self._target_boost_views.get(boost_key)
        if target_boost is None:
            target_boost = self._target_boost.expand(next_id.size(0), 1)
            self._target_boost_views[boost_key] = target_boost
        logits.scatter_add_(1, next_id.unsqueeze(-1), target_boost)
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


@torch.inference_mode()
def decode_fixed_precision(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    max_steps: int,
    device: torch.device,
    workspace: FixedDecodeWorkspace | None = None,
) -> torch.Tensor:
    prompt = tokens.to(device, non_blocking=True)
    prompt_device = prompt.device
    batch_size, prompt_len = prompt.shape
    generated_shape = (batch_size, prompt_len + max_steps)
    if workspace is None:
        generated = torch.empty(
            generated_shape,
            device=device,
            dtype=prompt.dtype,
        )
        next_token = torch.empty((batch_size, 1), device=device, dtype=prompt.dtype)
        next_token_values: torch.Tensor | None = None
    else:
        generated = workspace.generated
        next_token = workspace.next_token
        if generated.shape != generated_shape or generated.device != prompt_device or generated.dtype != prompt.dtype:
            raise ValueError("workspace.generated does not match decode shape/device/dtype")
        if next_token.shape != (batch_size, 1) or next_token.device != prompt_device or next_token.dtype != prompt.dtype:
            raise ValueError("workspace.next_token does not match decode shape/device/dtype")
        next_token_values = workspace.next_token_values
    generated[:, :prompt_len].copy_(prompt)
    current_len = prompt_len
    for _ in range(max_steps):
        active_tokens = generated[:, :current_len]
        logits = model(input_ids=active_tokens)
        if hasattr(logits, "logits"):
            logits = logits.logits
        last_step_logits = logits if logits.dim() == 2 else logits[:, -1, :]
        if next_token_values is None:
            next_token_values = torch.empty(
                (batch_size, 1),
                device=last_step_logits.device,
                dtype=last_step_logits.dtype,
            )
            if workspace is not None:
                workspace.next_token_values = next_token_values
        torch.max(last_step_logits, dim=-1, keepdim=True, out=(next_token_values, next_token))
        generated[:, current_len : current_len + 1].copy_(next_token)
        current_len += 1
    return generated[:, :current_len].contiguous()


@torch.inference_mode()
def decode_host_policy_baseline(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    max_steps: int,
    device: torch.device,
    workspace: FixedDecodeWorkspace | None = None,
) -> torch.Tensor:
    """Naive baseline: fixed precision plus host-visible confidence checks."""
    prompt = tokens.to(device, non_blocking=True)
    prompt_device = prompt.device
    batch_size, prompt_len = prompt.shape
    generated_shape = (batch_size, prompt_len + max_steps)
    if workspace is None:
        generated = torch.empty(
            generated_shape,
            device=device,
            dtype=prompt.dtype,
        )
        next_token = torch.empty((batch_size, 1), device=device, dtype=prompt.dtype)
        next_token_values: torch.Tensor | None = None
        host_logits_buffer: torch.Tensor | None = None
        policy_metrics_buffer: torch.Tensor | None = None
        policy_metric_values: list[float] | None = None
    else:
        generated = workspace.generated
        next_token = workspace.next_token
        if generated.shape != generated_shape or generated.device != prompt_device or generated.dtype != prompt.dtype:
            raise ValueError("workspace.generated does not match decode shape/device/dtype")
        if next_token.shape != (batch_size, 1) or next_token.device != prompt_device or next_token.dtype != prompt.dtype:
            raise ValueError("workspace.next_token does not match decode shape/device/dtype")
        next_token_values = workspace.next_token_values
        host_logits_buffer = workspace.host_logits_buffer
        policy_metrics_buffer = workspace.policy_metrics_buffer
        policy_metric_values = workspace.policy_metric_values
    if policy_metric_values is None:
        policy_metric_values = [0.0] * 4
        if workspace is not None:
            workspace.policy_metric_values = policy_metric_values
    generated[:, :prompt_len].copy_(prompt)
    current_len = prompt_len
    for _ in range(max_steps):
        active_tokens = generated[:, :current_len]
        logits = model(input_ids=active_tokens)
        if hasattr(logits, "logits"):
            logits = logits.logits
        last_step_logits = logits if logits.dim() == 2 else logits[:, -1, :]
        # Deliberately conservative baseline: move confidence analysis to host.
        if (
            host_logits_buffer is None
            or tuple(host_logits_buffer.shape) != tuple(last_step_logits.shape)
        ):
            host_logits_buffer = torch.empty(
                tuple(last_step_logits.shape),
                device="cpu",
                dtype=torch.float32,
                pin_memory=device.type == "cuda" and torch.cuda.is_available(),
            )
            if workspace is not None:
                workspace.host_logits_buffer = host_logits_buffer
        host_logits_buffer.copy_(
            last_step_logits,
            non_blocking=host_logits_buffer.is_pinned(),
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        host_logits = host_logits_buffer
        if policy_metrics_buffer is None:
            policy_metrics_buffer = torch.empty(4, device="cpu", dtype=torch.float32)
            if workspace is not None:
                workspace.policy_metrics_buffer = policy_metrics_buffer
        policy_metrics_buffer[0].copy_(compute_entropy(host_logits).mean())
        policy_metrics_buffer[1].copy_(torch.softmax(host_logits, dim=-1).max(dim=-1).values.mean())
        policy_metrics_buffer[2].copy_(torch.topk(host_logits, k=2, dim=-1).values.mean())
        policy_metrics_buffer[3].copy_(torch.amax(host_logits, dim=-1).mean())
        for metric_idx in range(4):
            policy_metric_values[metric_idx] = float(policy_metrics_buffer[metric_idx])
        if next_token_values is None:
            next_token_values = torch.empty(
                (batch_size, 1),
                device=last_step_logits.device,
                dtype=last_step_logits.dtype,
            )
            if workspace is not None:
                workspace.next_token_values = next_token_values
        torch.max(last_step_logits, dim=-1, keepdim=True, out=(next_token_values, next_token))
        generated[:, current_len : current_len + 1].copy_(next_token)
        current_len += 1
    return generated[:, :current_len].contiguous()


@torch.inference_mode()
def decode_dynamic_precision(
    model: nn.Module,
    tokens: torch.Tensor,
    *,
    max_steps: int,
    device: torch.device,
    workspace: DynamicPrecisionWorkspace | None = None,
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
        workspace=workspace,
    )
