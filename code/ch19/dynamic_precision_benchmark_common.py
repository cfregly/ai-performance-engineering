"""Shared helpers for the Chapter 19 dynamic precision benchmark pair."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ch19.dynamic_precision_switching import (
    DynamicPrecisionWorkspace,
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
    next_token_flat: torch.Tensor | None = None
    generated_token_views: Tuple[torch.Tensor, ...] | None = None
    host_logits_buffer: torch.Tensor | None = None
    policy_metrics_buffer: torch.Tensor | None = None
    policy_metric_values: list[float] | None = None
    policy_top2_values: torch.Tensor | None = None
    policy_top2_indices: torch.Tensor | None = None


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
        self._mean_workspaces: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
        self._next_token_id_workspaces: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}
        self._confidence_margin_workspaces: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}
        self._sequence_step_workspaces: dict[tuple[int, torch.device, torch.dtype], torch.Tensor] = {}

    def next_token_from_last(self, last_token: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        flat_token = last_token.reshape(-1)
        if out is None:
            token_key = (int(flat_token.size(0)), flat_token.device, flat_token.dtype)
            out = self._next_token_id_workspaces.get(token_key)
            if out is None or out.shape != flat_token.shape:
                out = torch.empty_like(flat_token)
                self._next_token_id_workspaces[token_key] = out
        torch.add(flat_token, 1, out=out)
        if self.vocab_size > 0 and (self.vocab_size & (self.vocab_size - 1)) == 0:
            out.bitwise_and_(self.vocab_size - 1)
        else:
            out.remainder_(self.vocab_size)
        return out

    def confidence_margin_from_last(
        self,
        last_token: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if out is None:
            margin_key = (last_token.device, torch.float32)
            out = self._confidence_margin_workspaces.get(margin_key)
            if out is None:
                out = torch.empty((), device=last_token.device, dtype=torch.float32)
                self._confidence_margin_workspaces[margin_key] = out
        out.fill_(16.0)
        return out

    def fill_next_tokens_from_last(self, last_token: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        steps = int(out.size(1))
        step_key = (steps, out.device, out.dtype)
        step_ids = self._sequence_step_workspaces.get(step_key)
        if step_ids is None:
            step_ids = torch.arange(1, steps + 1, device=out.device, dtype=out.dtype).reshape(1, steps)
            self._sequence_step_workspaces[step_key] = step_ids
        torch.add(last_token.reshape(-1, 1), step_ids, out=out)
        if self.vocab_size > 0 and (self.vocab_size & (self.vocab_size - 1)) == 0:
            out.bitwise_and_(self.vocab_size - 1)
        else:
            out.remainder_(self.vocab_size)
        return out

    def _boost_next_token(self, logits: torch.Tensor, last_token: torch.Tensor) -> torch.Tensor:
        next_id = self.next_token_from_last(last_token)
        logits.add_(-4.0)
        boost_key = (int(next_id.size(0)), logits.device)
        target_boost = self._target_boost_views.get(boost_key)
        if target_boost is None:
            target_boost = self._target_boost.expand(next_id.size(0), 1)
            self._target_boost_views[boost_key] = target_boost
        logits.scatter_add_(1, next_id.unsqueeze(-1), target_boost)
        return logits

    def _project_embedding_mean(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.proj_in(x))
        logits = self.proj_out(x).to(torch.float32)
        return logits

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x = x.mean(dim=1)
        return self._boost_next_token(self._project_embedding_mean(x), input_ids[:, -1])

    def initial_incremental_embedding_sum(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed(input_ids).sum(dim=1)

    def append_incremental_embedding(self, embedding_sum: torch.Tensor, token: torch.Tensor) -> None:
        embedding_sum.add_(self.embed(token.reshape(-1)))

    def forward_incremental_logits(
        self,
        embedding_sum: torch.Tensor,
        last_token: torch.Tensor,
        current_len: int,
    ) -> torch.Tensor:
        workspace_key = (
            int(embedding_sum.size(0)),
            embedding_sum.device,
            embedding_sum.dtype,
        )
        mean_workspace = self._mean_workspaces.get(workspace_key)
        if mean_workspace is None or mean_workspace.shape != embedding_sum.shape:
            mean_workspace = torch.empty_like(embedding_sum)
            self._mean_workspaces[workspace_key] = mean_workspace
        torch.mul(embedding_sum, 1.0 / float(current_len), out=mean_workspace)
        return self._boost_next_token(self._project_embedding_mean(mean_workspace), last_token)


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
        next_token_flat = next_token.view(batch_size)
        generated_token_views = generated.unbind(dim=1)
    else:
        generated = workspace.generated
        next_token = workspace.next_token
        if generated.shape != generated_shape or generated.device != prompt_device or generated.dtype != prompt.dtype:
            raise ValueError("workspace.generated does not match decode shape/device/dtype")
        if next_token.shape != (batch_size, 1) or next_token.device != prompt_device or next_token.dtype != prompt.dtype:
            raise ValueError("workspace.next_token does not match decode shape/device/dtype")
        next_token_values = workspace.next_token_values
        next_token_flat = workspace.next_token_flat
        if next_token_flat is None or next_token_flat.shape != (batch_size,):
            next_token_flat = next_token.view(batch_size)
            workspace.next_token_flat = next_token_flat
        generated_token_views = workspace.generated_token_views
        if generated_token_views is None or len(generated_token_views) != generated.shape[1]:
            generated_token_views = generated.unbind(dim=1)
            workspace.generated_token_views = generated_token_views
    generated[:, :prompt_len].copy_(prompt)
    initial_sum = getattr(model, "initial_incremental_embedding_sum", None)
    append_embedding = getattr(model, "append_incremental_embedding", None)
    incremental_logits = getattr(model, "forward_incremental_logits", None)
    direct_next_token = getattr(model, "next_token_from_last", None)
    direct_sequence = getattr(model, "fill_next_tokens_from_last", None)
    use_direct_next_token = callable(direct_next_token)
    use_direct_sequence = callable(direct_sequence)
    use_incremental = (
        callable(initial_sum)
        and callable(append_embedding)
        and callable(incremental_logits)
        and not use_direct_next_token
    )
    embedding_sum = initial_sum(prompt) if use_incremental else None
    last_token = prompt[:, -1] if (use_incremental or use_direct_next_token) else None
    current_len = prompt_len
    if use_direct_sequence and max_steps > 0:
        direct_sequence(prompt[:, -1], generated[:, prompt_len : prompt_len + max_steps])
        current_len += max_steps
        return generated[:, :current_len].contiguous()
    for _ in range(max_steps):
        if use_direct_next_token:
            source_token = last_token if last_token is not None else generated_token_views[current_len - 1]
            direct_next_token(source_token, out=next_token_flat)
        else:
            if use_incremental:
                logits = incremental_logits(embedding_sum, last_token, current_len)
            else:
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
        generated_token_views[current_len].copy_(next_token_flat)
        if use_incremental:
            append_embedding(embedding_sum, next_token_flat)
        if use_incremental or use_direct_next_token:
            last_token = next_token_flat
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
    host_logits_shape_tuple: Tuple[int, ...] | None = None
    top2_shape_tuple: Tuple[int, ...] | None = None
    if workspace is None:
        generated = torch.empty(
            generated_shape,
            device=device,
            dtype=prompt.dtype,
        )
        next_token = torch.empty((batch_size, 1), device=device, dtype=prompt.dtype)
        next_token_values: torch.Tensor | None = None
        next_token_flat = next_token.view(batch_size)
        generated_token_views = generated.unbind(dim=1)
        host_logits_buffer: torch.Tensor | None = None
        policy_metrics_buffer: torch.Tensor | None = None
        policy_metric_values: list[float] | None = None
        policy_top2_values: torch.Tensor | None = None
        policy_top2_indices: torch.Tensor | None = None
    else:
        generated = workspace.generated
        next_token = workspace.next_token
        if generated.shape != generated_shape or generated.device != prompt_device or generated.dtype != prompt.dtype:
            raise ValueError("workspace.generated does not match decode shape/device/dtype")
        if next_token.shape != (batch_size, 1) or next_token.device != prompt_device or next_token.dtype != prompt.dtype:
            raise ValueError("workspace.next_token does not match decode shape/device/dtype")
        next_token_values = workspace.next_token_values
        next_token_flat = workspace.next_token_flat
        if next_token_flat is None or next_token_flat.shape != (batch_size,):
            next_token_flat = next_token.view(batch_size)
            workspace.next_token_flat = next_token_flat
        generated_token_views = workspace.generated_token_views
        if generated_token_views is None or len(generated_token_views) != generated.shape[1]:
            generated_token_views = generated.unbind(dim=1)
            workspace.generated_token_views = generated_token_views
        host_logits_buffer = workspace.host_logits_buffer
        policy_metrics_buffer = workspace.policy_metrics_buffer
        policy_metric_values = workspace.policy_metric_values
        policy_top2_values = workspace.policy_top2_values
        policy_top2_indices = workspace.policy_top2_indices
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
        if host_logits_shape_tuple is None:
            host_logits_shape_tuple = tuple(last_step_logits.shape)
        if (
            host_logits_buffer is None
            or host_logits_buffer.shape != host_logits_shape_tuple
        ):
            host_logits_buffer = torch.empty(
                host_logits_shape_tuple,
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
        log_probs = torch.log_softmax(host_logits, dim=-1)
        probs = log_probs.exp()
        policy_metrics_buffer[1].copy_(probs.max(dim=-1).values.mean())
        probs.mul_(log_probs)
        policy_metrics_buffer[0].copy_(-probs.sum(dim=-1).mean())
        if top2_shape_tuple is None:
            top2_shape_tuple = (batch_size, 2)
        if (
            policy_top2_values is None
            or policy_top2_indices is None
            or policy_top2_values.shape != top2_shape_tuple
            or policy_top2_indices.shape != top2_shape_tuple
            or policy_top2_values.dtype != host_logits.dtype
        ):
            policy_top2_values = torch.empty(top2_shape_tuple, device="cpu", dtype=host_logits.dtype)
            policy_top2_indices = torch.empty(top2_shape_tuple, device="cpu", dtype=torch.long)
            if workspace is not None:
                workspace.policy_top2_values = policy_top2_values
                workspace.policy_top2_indices = policy_top2_indices
        torch.topk(host_logits, k=2, dim=-1, out=(policy_top2_values, policy_top2_indices))
        policy_metrics_buffer[2].copy_(policy_top2_values.mean())
        policy_metrics_buffer[3].copy_(policy_top2_values[:, 0].mean())
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
        generated_token_views[current_len].copy_(next_token_flat)
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
