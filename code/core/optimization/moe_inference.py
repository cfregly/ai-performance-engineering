"""Shared Mixture-of-Experts inference helpers for chapter benchmarks.

Provides lightweight GPT-style MoE blocks plus configuration utilities so
baseline/optimized inference demos can share the same synthetic workload.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_DTYPE_MAP = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
}


def resolve_dtype(dtype: torch.dtype | str) -> torch.dtype:
    """Normalize dtype inputs from config/env vars."""
    if isinstance(dtype, torch.dtype):
        return dtype
    key = dtype.lower().strip()
    if key not in _DTYPE_MAP:
        raise ValueError(f"Unsupported dtype '{dtype}'")
    return _DTYPE_MAP[key]


def dtype_bytes(dtype: torch.dtype | str) -> int:
    """Return element size (bytes) for the dtype."""
    dt = resolve_dtype(dtype)
    if dt == torch.float32:
        return 4
    if dt in (torch.float16, torch.bfloat16):
        return 2
    if dt == torch.float64:
        return 8
    return torch.tensor([], dtype=dt).element_size()


def allocate_kv_cache(
    batch: int,
    total_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Allocate KV cache-sized tensor."""
    return torch.empty(batch, total_tokens, hidden_size, dtype=dtype, device=device)


def env_override_int(name: str, default: int) -> int:
    """Read integer override from environment, falling back to default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def env_override_float(name: str, default: float) -> float:
    """Read float override from environment, falling back to default."""
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class MoeInferenceConfig:
    """Synthesizes a GPT-style MoE stack configurable via env vars."""

    vocab_size: int = 32768
    hidden_size: int = 2048
    ffn_size: int = 8192
    num_layers: int = 12
    num_moe_layers: int = 6
    num_experts: int = 16
    top_k: int = 1
    moe_layer_frequency: int = 2
    batch_size: int = 4
    context_window: int = 2048
    decode_tokens: int = 64
    router_noise: float = 0.0
    capacity_factor: float | None = None
    dtype: torch.dtype | str = field(default_factory=lambda: torch.bfloat16)

    def __post_init__(self) -> None:
        self.top_k = max(1, min(self.top_k, self.num_experts))
        self.moe_layer_frequency = max(1, self.moe_layer_frequency)
        self.num_moe_layers = max(0, min(self.num_layers, self.num_moe_layers))

    @property
    def dtype_obj(self) -> torch.dtype:
        if not hasattr(self, "_cached_dtype"):
            self._cached_dtype = resolve_dtype(self.dtype)
        return self._cached_dtype

    @property
    def tokens_per_iteration(self) -> int:
        return self.batch_size * (self.context_window + self.decode_tokens)


class ExpertMLP(nn.Module):
    """Two-layer feed-forward expert block."""

    def __init__(self, hidden: int, ffn: int, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        linear_kwargs = {}
        if device is not None:
            linear_kwargs["device"] = device
        if dtype is not None:
            linear_kwargs["dtype"] = dtype
        self.fc1 = nn.Linear(hidden, ffn, **linear_kwargs)
        self.fc2 = nn.Linear(ffn, hidden, **linear_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.silu(self.fc1(x)))


class DenseFeedForward(nn.Module):
    """Fallback dense FFN when layer does not use MoE."""

    def __init__(self, hidden: int, ffn: int, *, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.net = ExpertMLP(hidden, ffn, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEFeedForward(nn.Module):
    """Top-k router with per-expert FFNs."""

    def __init__(
        self,
        hidden: int,
        ffn: int,
        num_experts: int,
        top_k: int,
        router_noise: float = 0.0,
        capacity_factor: Optional[float] = None,
        *,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        linear_kwargs = {}
        if device is not None:
            linear_kwargs["device"] = device
        if dtype is not None:
            linear_kwargs["dtype"] = dtype
        self.router = nn.Linear(hidden, num_experts, bias=False, **linear_kwargs)
        self.experts = nn.ModuleList([ExpertMLP(hidden, ffn, device=device, dtype=dtype) for _ in range(num_experts)])
        self.top_k = top_k
        self.router_noise = router_noise
        self.capacity_factor = capacity_factor
        self.num_experts = num_experts

    def forward(self, x: torch.Tensor, *, collect_router_stats: bool = False) -> torch.Tensor | Tuple[torch.Tensor, Optional[dict]]:
        batch, seq, hidden = x.shape
        flat = x.reshape(batch * seq, hidden)
        logits = self.router(flat)
        if self.router_noise > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise
        probs = torch.softmax(logits, dim=-1)
        router_entropy = None
        if collect_router_stats:
            with torch.no_grad():
                router_entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        top_scores, top_indices = torch.topk(probs, k=self.top_k, dim=-1)
        drop_mask = None
        overflow_mask = None
        expert_counts = None
        if self.capacity_factor is not None:
            tokens = flat.shape[0]
            avg_tokens_per_expert = max(1, math.ceil((tokens * self.top_k) / max(self.num_experts, 1)))
            capacity = max(1, math.ceil(self.capacity_factor * avg_tokens_per_expert))
            expert_counts = torch.bincount(top_indices.reshape(-1), minlength=self.num_experts)
            overloaded = expert_counts > capacity
            drop_mask = overloaded[top_indices]
            top_scores = top_scores * (~drop_mask).float()
            overflow_mask = drop_mask.any(dim=-1)
        combined = torch.zeros_like(flat)

        for k in range(self.top_k):
            expert_ids = top_indices[:, k]
            weights = top_scores[:, k].unsqueeze(-1)
            for expert_id, expert in enumerate(self.experts):
                mask = expert_ids == expert_id
                if mask.any():
                    indices = mask.nonzero(as_tuple=False).squeeze(-1)
                    expert_input = flat.index_select(0, indices)
                    expert_out = expert(expert_input)
                    selected_weights = weights.index_select(0, indices)
                    if selected_weights.dim() == 1:
                        selected_weights = selected_weights.unsqueeze(-1)
                    selected_weights = selected_weights.to(expert_out.dtype)
                    combined.index_add_(0, indices, expert_out * selected_weights)
        combined = combined.view(batch, seq, hidden)
        if collect_router_stats:
            stats = {
                "expert_indices": top_indices.detach(),
                "overflow_mask": overflow_mask.detach() if overflow_mask is not None else None,
                "expert_counts": expert_counts.detach() if expert_counts is not None else None,
                "router_entropy": float(router_entropy.detach()) if router_entropy is not None else None,
            }
            return combined, stats
        return combined


class MoEFeedForwardNoHostSync(MoEFeedForward):
    """MoEFeedForward variant that avoids host sync in expert dispatch.

    The baseline implementation uses `if mask.any():` inside a Python loop over
    experts. `mask.any()` produces a CUDA tensor, and converting it to a Python
    bool triggers a device sync + D2H transfer (performance bug).

    This variant keeps the same algorithm (top-k routing + per-expert MLPs +
    weighted accumulation) but removes Python boolean control flow by always
    computing `indices = mask.nonzero(...)` and letting empty tensors no-op.
    """

    def forward(self, x: torch.Tensor, *, collect_router_stats: bool = False):  # type: ignore[override]
        batch, seq, hidden = x.shape
        flat = x.reshape(batch * seq, hidden)
        logits = self.router(flat)
        if self.router_noise > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise
        probs = torch.softmax(logits, dim=-1)
        router_entropy = None
        if collect_router_stats:
            with torch.no_grad():
                router_entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        top_scores, top_indices = torch.topk(probs, k=self.top_k, dim=-1)

        drop_mask = None
        overflow_mask = None
        expert_counts = None
        if self.capacity_factor is not None:
            tokens = flat.shape[0]
            avg_tokens_per_expert = max(1, math.ceil((tokens * self.top_k) / max(self.num_experts, 1)))
            capacity = max(1, math.ceil(self.capacity_factor * avg_tokens_per_expert))
            expert_counts = torch.bincount(top_indices.reshape(-1), minlength=self.num_experts)
            overloaded = expert_counts > capacity
            drop_mask = overloaded[top_indices]
            # Avoid host sync from `if drop_mask.any()`; applying a no-op mask is fine.
            top_scores = top_scores * (~drop_mask).float()
            overflow_mask = drop_mask.any(dim=-1)

        combined = torch.zeros_like(flat)

        for k in range(self.top_k):
            expert_ids = top_indices[:, k]
            weights = top_scores[:, k].unsqueeze(-1)
            for expert_id, expert in enumerate(self.experts):
                mask = expert_ids == expert_id
                # IMPORTANT: avoid `if mask.any()` (host sync). Empty indices are fine.
                indices = mask.nonzero(as_tuple=False).squeeze(-1)
                if indices.numel() == 0:
                    continue
                expert_input = flat.index_select(0, indices)
                expert_out = expert(expert_input)
                selected_weights = weights.index_select(0, indices)
                if selected_weights.dim() == 1:
                    selected_weights = selected_weights.unsqueeze(-1)
                selected_weights = selected_weights.to(expert_out.dtype)
                combined.index_add_(0, indices, expert_out * selected_weights)

        combined = combined.view(batch, seq, hidden)
        if collect_router_stats:
            stats = {
                "expert_indices": top_indices.detach(),
                "overflow_mask": overflow_mask.detach() if overflow_mask is not None else None,
                "expert_counts": expert_counts.detach() if expert_counts is not None else None,
                "router_entropy": float(router_entropy.detach()) if router_entropy is not None else None,
            }
            return combined, stats
        return combined


class MoEFeedForwardSortedDispatch(MoEFeedForward):
    """MoEFeedForward that dispatches only active experts via a sorted assignment list.

    Baseline MoEFeedForward scans *every* expert and uses a boolean mask per expert.
    This variant keeps the same math (top-k gating + expert MLP + weighted sum) but:
      - flattens token→expert assignments (tokens * top_k)
      - sorts by expert id once
      - loops only over experts that appear in the assignment list

    This reduces Python overhead and avoids per-expert mask+any() patterns.
    """

    def _token_ids_for(self, tokens: int, device: torch.device) -> torch.Tensor:
        cache_key = (int(tokens), int(self.top_k), device.type, device.index)
        cached = getattr(self, "_token_ids_cache", None)
        if cached is None or getattr(self, "_token_ids_cache_key", None) != cache_key:
            token_ids = torch.arange(tokens * self.top_k, device=device, dtype=torch.long)
            if self.top_k > 1:
                token_ids.div_(self.top_k, rounding_mode="floor")
            self._token_ids_cache = token_ids
            self._token_ids_cache_key = cache_key
            return token_ids
        return cached

    def forward(self, x: torch.Tensor, *, collect_router_stats: bool = False):  # type: ignore[override]
        batch, seq, hidden = x.shape
        flat = x.reshape(batch * seq, hidden)
        logits = self.router(flat)
        if self.router_noise > 0:
            logits = logits + torch.randn_like(logits) * self.router_noise
        probs = torch.softmax(logits, dim=-1)
        router_entropy = None
        if collect_router_stats:
            with torch.no_grad():
                router_entropy = -(probs * torch.log(probs + 1e-9)).sum(dim=-1).mean()
        top_scores, top_indices = torch.topk(probs, k=self.top_k, dim=-1)

        drop_mask = None
        overflow_mask = None
        expert_counts = None
        if self.capacity_factor is not None:
            tokens = flat.shape[0]
            avg_tokens_per_expert = max(1, math.ceil((tokens * self.top_k) / max(self.num_experts, 1)))
            capacity = max(1, math.ceil(self.capacity_factor * avg_tokens_per_expert))
            expert_counts = torch.bincount(top_indices.reshape(-1), minlength=self.num_experts)
            overloaded = expert_counts > capacity
            drop_mask = overloaded[top_indices]
            top_scores = top_scores * (~drop_mask).float()
            overflow_mask = drop_mask.any(dim=-1)

        combined = torch.zeros_like(flat)
        tokens = flat.shape[0]
        token_ids = self._token_ids_for(tokens, flat.device)
        expert_ids = top_indices.reshape(-1).to(dtype=torch.long)
        weights = top_scores.reshape(-1).unsqueeze(-1)

        sorted_expert_ids, perm = torch.sort(expert_ids)
        sorted_token_ids = token_ids.index_select(0, perm)
        sorted_weights = weights.index_select(0, perm)

        unique_experts, counts = torch.unique_consecutive(sorted_expert_ids, return_counts=True)
        # Convert small metadata to CPU for efficient Python looping.
        expert_list = unique_experts.tolist()
        count_list = counts.tolist()

        offset = 0
        for expert_id, count in zip(expert_list, count_list):
            if count <= 0:
                continue
            segment_tokens = sorted_token_ids.narrow(0, offset, count)
            segment_weights = sorted_weights.narrow(0, offset, count)
            offset += count

            expert_input = flat.index_select(0, segment_tokens)
            expert_out = self.experts[int(expert_id)](expert_input)
            segment_weights = segment_weights.to(expert_out.dtype)
            combined.index_add_(0, segment_tokens, expert_out * segment_weights)

        combined = combined.view(batch, seq, hidden)
        if collect_router_stats:
            stats = {
                "expert_indices": top_indices.detach(),
                "overflow_mask": overflow_mask.detach() if overflow_mask is not None else None,
                "expert_counts": expert_counts.detach() if expert_counts is not None else None,
                "router_entropy": float(router_entropy.detach()) if router_entropy is not None else None,
            }
            return combined, stats
        return combined


class SimpleMoEBlock(nn.Module):
    """Attention + (dense or MoE) feed-forward."""

    def __init__(self, config: MoeInferenceConfig, use_moe: bool, device: torch.device):
        super().__init__()
        heads = max(1, config.hidden_size // 128)
        self.attn = nn.MultiheadAttention(
            embed_dim=config.hidden_size,
            num_heads=heads,
            batch_first=True,
            device=device,
            dtype=config.dtype_obj,
        )
        self.ln_attn = nn.LayerNorm(config.hidden_size, device=device, dtype=config.dtype_obj)
        self.ln_mlp = nn.LayerNorm(config.hidden_size, device=device, dtype=config.dtype_obj)
        if use_moe:
            self.ff = MoEFeedForward(
                config.hidden_size,
                config.ffn_size,
                num_experts=config.num_experts,
                top_k=config.top_k,
                router_noise=config.router_noise,
                capacity_factor=config.capacity_factor,
                device=device,
                dtype=config.dtype_obj,
            )
        else:
            self.ff = DenseFeedForward(config.hidden_size, config.ffn_size, device=device, dtype=config.dtype_obj)

    def forward(self, hidden: torch.Tensor, *, collect_router_stats: bool = False) -> torch.Tensor | Tuple[torch.Tensor, Optional[dict]]:
        attn_out, _ = self.attn(self.ln_attn(hidden), self.ln_attn(hidden), self.ln_attn(hidden), need_weights=False)
        hidden = hidden + attn_out
        if collect_router_stats and isinstance(self.ff, MoEFeedForward):
            ff_out, stats = self.ff(self.ln_mlp(hidden), collect_router_stats=True)
        else:
            ff_out = self.ff(self.ln_mlp(hidden))
            stats = None
        hidden = hidden + ff_out
        if collect_router_stats:
            return hidden, stats
        return hidden


class SimpleMoEGPT(nn.Module):
    """Tiny GPT-style stack with configurable MoE frequency."""

    def __init__(self, config: MoeInferenceConfig, *, device: torch.device):
        super().__init__()
        self.config = config
        self.device = device
        self.embed = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            device=device,
            dtype=config.dtype_obj,
        )
        self.layers = nn.ModuleList()
        for idx in range(config.num_layers):
            use_moe = idx < config.num_moe_layers and (idx % config.moe_layer_frequency == 0)
            self.layers.append(SimpleMoEBlock(config, use_moe=use_moe, device=device))
        self.final_norm = nn.LayerNorm(config.hidden_size, device=device, dtype=config.dtype_obj)
        self.lm_head = nn.Linear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            device=device,
            dtype=config.dtype_obj,
        )

    def forward_tokens(self, token_ids: torch.Tensor, *, collect_router_stats: bool = False) -> torch.Tensor | Tuple[torch.Tensor, List[dict]]:
        if token_ids.dtype != torch.long:
            token_ids = token_ids.long()
        hidden = self.embed(token_ids)
        router_stats: List[dict] = []
        for block in self.layers:
            if collect_router_stats:
                hidden, stats = block(hidden, collect_router_stats=True)  # type: ignore[assignment]
                if stats is not None:
                    router_stats.append(stats)
            else:
                hidden = block(hidden)
        hidden = self.final_norm(hidden)
        if collect_router_stats:
            return hidden, router_stats
        return hidden

    def prefill(
        self,
        input_ids: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        cache_start: int = 0,
        output_router_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, List[dict]]:
        if output_router_stats:
            hidden, router_stats = self.forward_tokens(input_ids, collect_router_stats=True)  # type: ignore[misc]
        else:
            hidden = self.forward_tokens(input_ids)  # type: ignore[assignment]
            router_stats = []
        if kv_cache is not None:
            kv_cache[:, cache_start:cache_start + hidden.size(1)].copy_(hidden)
        logits = self.lm_head(hidden)
        if output_router_stats:
            return hidden, logits, router_stats
        return hidden, logits

    def decode(
        self,
        token_ids: torch.Tensor,
        kv_cache: Optional[torch.Tensor] = None,
        position: Optional[int] = None,
        output_router_stats: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, List[dict]]:
        if output_router_stats:
            hidden, router_stats = self.forward_tokens(token_ids, collect_router_stats=True)  # type: ignore[misc]
        else:
            hidden = self.forward_tokens(token_ids)  # type: ignore[assignment]
            router_stats = []
        if kv_cache is not None and position is not None:
            kv_cache[:, position:position + hidden.size(1)].copy_(hidden)
        logits = self.lm_head(hidden)
        if output_router_stats:
            return hidden, logits, router_stats
        return hidden, logits


__all__ = [
    "allocate_kv_cache",
    "dtype_bytes",
    "env_override_float",
    "env_override_int",
    "MoeInferenceConfig",
    "SimpleMoEGPT",
]
