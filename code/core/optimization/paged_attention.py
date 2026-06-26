"""Shared utilities for paged attention benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch


@dataclass
class PagedAttentionConfig:
    batch_size: int
    page_size: int
    num_heads: int
    head_dim: int
    device: torch.device
    dtype: torch.dtype


class PagedKVCache:
    """Page-based KV cache that avoids large contiguous copies."""

    def __init__(self, config: PagedAttentionConfig):
        self.config = config
        self.k_pages: List[torch.Tensor] = []
        self.v_pages: List[torch.Tensor] = []
        self.page_map: List[int] = []

    def _allocate_page(self) -> None:
        page_shape = (
            self.config.batch_size,
            self.config.page_size,
            self.config.num_heads,
            self.config.head_dim,
        )
        self.k_pages.append(
            torch.empty(page_shape, dtype=self.config.dtype, device=self.config.device)
        )
        self.v_pages.append(
            torch.empty(page_shape, dtype=self.config.dtype, device=self.config.device)
        )

    def write(self, pos: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Write a single token (B=..., T=1) into the paged cache."""
        page_idx = pos // self.config.page_size
        offset = pos % self.config.page_size

        while len(self.k_pages) <= page_idx:
            self._allocate_page()

        self.k_pages[page_idx][:, offset : offset + 1, :, :] = k
        self.v_pages[page_idx][:, offset : offset + 1, :, :] = v

        if pos >= len(self.page_map):
            self.page_map.extend([page_idx] * (pos - len(self.page_map) + 1))
        else:
            self.page_map[pos] = page_idx

    def get_kv(self, length: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return concatenated K/V tensors for the first `length` positions."""
        if length == 0 or not self.k_pages:
            empty = torch.empty(
                self.config.batch_size,
                0,
                self.config.num_heads,
                self.config.head_dim,
                device=self.config.device,
                dtype=self.config.dtype,
            )
            return empty, empty

        k_list = []
        v_list = []
        pos = 0
        while pos < length:
            page_idx = self.page_map[pos] if pos < len(self.page_map) else len(self.k_pages) - 1
            offset = pos % self.config.page_size
            take = min(self.config.page_size - offset, length - pos)
            k_list.append(self.k_pages[page_idx][:, offset : offset + take, :, :])
            v_list.append(self.v_pages[page_idx][:, offset : offset + take, :, :])
            pos += take

        return torch.cat(k_list, dim=1), torch.cat(v_list, dim=1)
