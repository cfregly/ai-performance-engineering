"""Optimized paged KV-cache prefetch benchmark (pinned async-stream prefetch).

- Pinned buffers plus an async copy stream prefetch the next page to overlap H2D
  copies with compute.
- GB300 note: the host prefetch THREAD is counterproductive here. On NVLink-C2C
  the coherent H2D copies are already cheap, so the Python/GIL thread overhead
  dominates them and inverts the result to ~0.57x. Async-stream prefetch without
  the thread wins (~1.22x), so use_host_prefetch_thread is False.
- Uses a pinned host cache (memmap disabled) to isolate overlap effects.
"""

from __future__ import annotations

import torch

from core.benchmark.wrapper_utils import attach_benchmark_metadata
from labs.persistent_decode.paged_kv_offload_common import PagedKVConfig, PagedKVOffloadBenchmark


def get_benchmark() -> PagedKVOffloadBenchmark:
    cfg = PagedKVConfig(
        batch_size=4,
        num_heads=16,
        head_dim=128,
        max_seq_len=65536,
        page_tokens=8192,
        decode_tokens=1,
        repeat_pages=128,
        use_pinned_stage=True,
        use_async_stream=True,
        use_memmap=False,
        prefer_fp8=False,
        require_fused_fp8=False,
        fallback_dtype=torch.float16,
        prefetch_next_page=True,
        use_direct_h2d=False,
        use_host_prefetch_thread=False,
    )
    return attach_benchmark_metadata(
        PagedKVOffloadBenchmark(cfg, label="paged_kv_offload_prefetch_optimized"),
        __file__,
    )

