"""Paged/NVMe-style KV-cache offload microbenchmarks with fused FP8 gating.

This module provides a small, GPU-backed benchmark that simulates:
1) A hot KV window that remains resident on the GPU.
2) Cold KV pages that live in either pageable CPU memory or an NVMe-backed
   memmap file.
3) Optional pinned staging + async H2D copies for overlap.
4) FP8 KV usage that is only enabled when a fused FlashAttention-style path is
   likely to exist (B200/GB200 or newer).

The goal is to encode the practical rule from the post:
- Use FP8 KV only when a fused attention kernel is available; otherwise fall
  back to FP16 to avoid paying dequant cost with no speedup.
- Use paged/NVMe-style offload when context length forces it, and measure the
  TTFT impact of pulling pages back in.
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from core.benchmark.verification import PrecisionFlags, simple_signature
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.arch_config import prefer_sdpa_backends
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig

_FLOAT8_E4M3FN = getattr(torch, "float8_e4m3fn", None)
_FLOAT8_E5M2FN = getattr(torch, "float8_e5m2fn", None)
_HAS_FLOAT8_E4M3FN = _FLOAT8_E4M3FN is not None


def _supports_fp8_kv() -> bool:
    """Return True if FP8 KV is even representable in this build of PyTorch."""
    return _HAS_FLOAT8_E4M3FN and torch.cuda.is_available()


def _supports_fused_fp8_attention() -> bool:
    """Return True if this runtime can execute FP8 SDPA for (q, k, v).

    PyTorch builds vary in FP8 kernel availability; rely on a minimal runtime
    probe rather than GPU CC heuristics to avoid hard failures like:
    "No available kernel. Aborting execution."
    """
    if not torch.cuda.is_available():
        return False
    fp8_dtype = _FLOAT8_E4M3FN
    if fp8_dtype is None:
        return False
    try:
        q = torch.randn((1, 1, 16, 64), device="cuda", dtype=torch.float16).to(fp8_dtype)
        ctx = prefer_sdpa_backends() if prefer_sdpa_backends is not None else nullcontext()
        with ctx:
            _ = F.scaled_dot_product_attention(q, q, q)
        return True
    except Exception:
        return False


def _np_dtype_for(torch_dtype: torch.dtype) -> np.dtype:
    """Map a torch dtype to a numpy dtype used for the memmap backing store."""
    if torch_dtype in (_FLOAT8_E4M3FN, _FLOAT8_E5M2FN):
        # memmap sticks to fp16; conversion to fp8 happens during staging/H2D.
        return np.float16
    return torch.empty([], dtype=torch_dtype).numpy().dtype


@dataclass
class PagedKVConfig:
    """Configuration for paged KV-cache simulation."""

    batch_size: int = 2
    num_heads: int = 16
    head_dim: int = 128
    max_seq_len: int = 8192
    page_tokens: int = 512
    decode_tokens: int = 64
    repeat_pages: int = 8
    use_pinned_stage: bool = False
    use_async_stream: bool = False
    use_memmap: bool = False  # When True, store cold pages on disk to mimic NVMe.
    prefer_fp8: bool = True
    require_fused_fp8: bool = False  # If True, FP8 is only used when fused path is present.
    fallback_dtype: torch.dtype = torch.float16
    prefetch_next_page: bool = False
    use_direct_h2d: bool = False
    use_host_prefetch_thread: bool = False
    use_page_major_host_cache: bool = False


class PagedKVOffloadBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Synthetic decode microbenchmark with paged KV offload and FP8 gating."""

    def __init__(self, cfg: Optional[PagedKVConfig] = None, label: str = "paged_kv_offload"):
        super().__init__()
        self.cfg = cfg or PagedKVConfig()
        self.label = label

        self.runtime_dtype: torch.dtype = self.cfg.fallback_dtype
        self.enable_flash: bool = False
        self._fp8_reason: str = ""

        self.hot_k: Optional[torch.Tensor] = None
        self.hot_v: Optional[torch.Tensor] = None
        self.hot_kv_bufs: list[torch.Tensor] = []
        self.hot_k_bufs: list[torch.Tensor] = []
        self.hot_v_bufs: list[torch.Tensor] = []
        self._hot_buffer_count = 0
        self.active_buf_idx: int = 0
        self.prefetch_buf_idx: Optional[int] = None
        self.prefetch_slice_len: Optional[int] = None
        self.prefetch_event: Optional[torch.cuda.Event] = None
        self.staging: Optional[torch.Tensor] = None
        self.prefetch_staging: Optional[torch.Tensor] = None
        self.prefetched_range: Optional[Tuple[int, int]] = None
        self.copy_stream: Optional[torch.cuda.Stream] = None
        self.q: Optional[torch.Tensor] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._prefetch_queue: Optional["queue.Queue[int]"] = None
        self._prefetch_ready: Optional[threading.Event] = None
        self._prefetch_lock = threading.Lock()
        self._prefetch_ready_start: Optional[int] = None
        self._prefetch_stop = threading.Event()

        self.host_cache: Optional[torch.Tensor] = None
        self._host_cache_is_pinned: bool = False
        self.host_memmap: Optional[np.memmap] = None
        self._memmap_path: Optional[Path] = None

        self.page_cursor: int = 0
        self._page_count: int = 0
        self._repeat_pages = 1
        self._repeat_page_range = range(0)
        self._bytes_per_iteration: float = 0.0
        self.output: Optional[torch.Tensor] = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._verify_head_dim = min(8, self.cfg.head_dim)
        self.register_workload_metadata(requests_per_iteration=1.0)

    # -------------------- Setup helpers --------------------

    def _select_runtime_dtype(self) -> torch.dtype:
        if self.cfg.prefer_fp8 and _supports_fp8_kv():
            if not _supports_fused_fp8_attention():
                if self.cfg.require_fused_fp8:
                    raise RuntimeError(
                        "SKIPPED: FP8 KV requested with require_fused_fp8=True, but FP8 SDPA "
                        "kernel is unavailable on this torch/runtime stack."
                    )
                self._fp8_reason = (
                    "FP8 requested but FP8 SDPA kernel unavailable; "
                    f"falling back to {self.cfg.fallback_dtype}."
                )
                return self.cfg.fallback_dtype
            if _FLOAT8_E4M3FN is None:
                self._fp8_reason = (
                    "FP8 requested but torch.float8_e4m3fn is unavailable; "
                    f"falling back to {self.cfg.fallback_dtype}."
                )
                return self.cfg.fallback_dtype
            self._fp8_reason = "Using FP8 KV: FP8 SDPA kernel available."
            return _FLOAT8_E4M3FN
        return self.cfg.fallback_dtype

    def _page_major_shape(self) -> Tuple[int, ...]:
        self._page_count = (self.cfg.max_seq_len + self.cfg.page_tokens - 1) // self.cfg.page_tokens
        return (
            self._page_count,
            2,  # k and v
            self.cfg.batch_size,
            self.cfg.num_heads,
            self.cfg.page_tokens,
            self.cfg.head_dim,
        )

    def _copy_canonical_to_page_major(self, canonical: torch.Tensor, page_major: torch.Tensor) -> None:
        for page_idx in range(self._page_count):
            start = page_idx * self.cfg.page_tokens
            end = min(start + self.cfg.page_tokens, self.cfg.max_seq_len)
            slice_len = end - start
            page_major[page_idx, ..., :slice_len, :].copy_(canonical[..., start:end, :])
            if slice_len < self.cfg.page_tokens:
                page_major[page_idx, ..., slice_len:, :].zero_()

    def _init_host_cache(self, shape: Tuple[int, ...]) -> None:
        generator = torch.Generator().manual_seed(42)
        self._host_cache_is_pinned = False
        if self.cfg.use_memmap:
            np_dtype = _np_dtype_for(self.runtime_dtype)
            tmp_dir = Path(tempfile.mkdtemp(prefix="paged_kv_cache_"))
            self._memmap_path = tmp_dir / "kv_cache.bin"
            memmap_shape = self._page_major_shape() if self.cfg.use_page_major_host_cache else shape
            self.host_memmap = np.memmap(self._memmap_path, mode="w+", dtype=np_dtype, shape=memmap_shape)
            if self.cfg.use_page_major_host_cache:
                canonical = torch.randn(shape, dtype=torch.float16, generator=generator)
                page_major = torch.empty(memmap_shape, dtype=torch.float16)
                self._copy_canonical_to_page_major(canonical, page_major)
                host = page_major.numpy().astype(np_dtype, copy=False)
            else:
                host = torch.randn(shape, dtype=torch.float16, generator=generator).numpy().astype(np_dtype, copy=False)
            self.host_memmap[:] = host
            self.host_cache = torch.from_numpy(self.host_memmap)
        else:
            if self.cfg.use_page_major_host_cache:
                page_major_shape = self._page_major_shape()
                canonical = torch.randn(shape, dtype=torch.float16, generator=generator)
                self.host_cache = torch.empty(
                    page_major_shape,
                    dtype=torch.float16,
                    pin_memory=self.cfg.use_pinned_stage,
                )
                self._copy_canonical_to_page_major(canonical, self.host_cache)
            else:
                self.host_cache = torch.randn(
                    shape,
                    dtype=torch.float16,
                    generator=generator,
                    pin_memory=self.cfg.use_pinned_stage,
                )
            self._host_cache_is_pinned = self.cfg.use_pinned_stage

    def _host_page_view(self, start: int, slice_len: int) -> torch.Tensor:
        if self.host_cache is None:
            raise RuntimeError("Host cache not initialized")
        if self.cfg.use_page_major_host_cache:
            page_idx = start // self.cfg.page_tokens
            return self.host_cache[page_idx, ..., :slice_len, :]
        end = start + slice_len
        return self.host_cache[..., start:end, :]

    def _stage_page(self, start: int, into_prefetch: bool = False) -> Tuple[torch.Tensor, int]:
        end = min(start + self.cfg.page_tokens, self.cfg.max_seq_len)
        slice_len = end - start
        target = self.prefetch_staging if into_prefetch else self.staging
        assert target is not None

        if self.cfg.use_direct_h2d and self._host_cache_is_pinned and self.host_cache is not None:
            return self._host_page_view(start, slice_len), slice_len

        if self.host_memmap is not None:
            target[..., :slice_len, :].copy_(self._host_page_view(start, slice_len))
        elif self.host_cache is not None:
            target[..., :slice_len, :].copy_(self._host_page_view(start, slice_len))
        else:
            raise RuntimeError("Host cache not initialized")
        return target, slice_len

    def _copy_to_device(
        self,
        staged: torch.Tensor,
        slice_len: int,
        buffer_idx: int = 0,
        wait_for_copy: bool = True,
        record_event: Optional[torch.cuda.Event] = None,
        wait_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        if not self.hot_k_bufs or not self.hot_v_bufs:
            raise RuntimeError("Device KV buffers are not initialized")
        target_k = self.hot_k_bufs[buffer_idx]
        target_v = self.hot_v_bufs[buffer_idx]
        target_kv = self.hot_kv_bufs[buffer_idx] if self.hot_kv_bufs else None
        direct_copy = self.cfg.use_direct_h2d and staged.dtype == self.runtime_dtype

        def _copy_planes() -> None:
            if direct_copy and target_kv is not None:
                target_kv[..., :slice_len, :].copy_(
                    staged[..., :slice_len, :],
                    non_blocking=self.cfg.use_pinned_stage,
                )
            elif direct_copy:
                src_k = staged[0, ..., :slice_len, :]
                src_v = staged[1, ..., :slice_len, :]
                target_k[..., :slice_len, :].copy_(src_k, non_blocking=self.cfg.use_pinned_stage)
                target_v[..., :slice_len, :].copy_(src_v, non_blocking=self.cfg.use_pinned_stage)
            else:
                src_k = staged[0, ..., :slice_len, :]
                src_v = staged[1, ..., :slice_len, :]
                target_k[..., :slice_len, :].copy_(src_k, non_blocking=self.cfg.use_pinned_stage)
                target_v[..., :slice_len, :].copy_(src_v, non_blocking=self.cfg.use_pinned_stage)

        label = "transfer_async:h2d" if (self.copy_stream is not None or self.cfg.use_pinned_stage) else "transfer_sync:h2d"
        if self.copy_stream is not None:
            if wait_stream is not None:
                self.copy_stream.wait_stream(wait_stream)
            with torch.cuda.stream(self.copy_stream), self._nvtx_range(label):
                _copy_planes()
                if record_event is not None:
                    record_event.record(self.copy_stream)
            if wait_for_copy:
                consumer_stream = wait_stream or torch.cuda.current_stream()
                consumer_stream.wait_stream(self.copy_stream)
        else:
            with self._nvtx_range(label):
                _copy_planes()
                if record_event is not None:
                    record_event.record(torch.cuda.current_stream())

    def setup(self) -> None:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        self.runtime_dtype = self._select_runtime_dtype()
        self.enable_flash = _supports_fused_fp8_attention() or self.runtime_dtype in (torch.float16, torch.bfloat16)

        head_shape = (
            self.cfg.batch_size,
            self.cfg.num_heads,
            self.cfg.page_tokens,
            self.cfg.head_dim,
        )
        staging_dtype = torch.float16
        buffer_count = 2 if (self.cfg.prefetch_next_page and self.cfg.use_async_stream) else 1
        if self.cfg.use_direct_h2d and self.runtime_dtype == staging_dtype:
            combined_shape = (2, *head_shape)
            self.hot_kv_bufs = [
                torch.empty(combined_shape, device=self.device, dtype=self.runtime_dtype)
                for _ in range(buffer_count)
            ]
            self.hot_k_bufs = [buf[0] for buf in self.hot_kv_bufs]
            self.hot_v_bufs = [buf[1] for buf in self.hot_kv_bufs]
        else:
            self.hot_kv_bufs = []
            self.hot_k_bufs = [torch.empty(head_shape, device=self.device, dtype=self.runtime_dtype) for _ in range(buffer_count)]
            self.hot_v_bufs = [torch.empty_like(self.hot_k_bufs[0]) for _ in range(buffer_count)]
        self._hot_buffer_count = buffer_count
        self.hot_k = self.hot_k_bufs[0]
        self.hot_v = self.hot_v_bufs[0]
        self.active_buf_idx = 0
        self.prefetch_buf_idx = None
        self.prefetch_slice_len = None
        self.prefetch_event = torch.cuda.Event() if buffer_count == 2 else None

        staging_shape = (
            2,  # k and v planes
            self.cfg.batch_size,
            self.cfg.num_heads,
            self.cfg.page_tokens,
            self.cfg.head_dim,
        )
        self.staging = torch.empty(
            staging_shape,
            device="cpu",
            dtype=staging_dtype,
            pin_memory=self.cfg.use_pinned_stage,
        )
        if self.cfg.prefetch_next_page:
            self.prefetch_staging = torch.empty(
                staging_shape,
                device="cpu",
                dtype=staging_dtype,
                pin_memory=self.cfg.use_pinned_stage,
            )
        self.copy_stream = torch.cuda.Stream() if self.cfg.use_async_stream else None

        host_shape = (
            2,  # k and v
            self.cfg.batch_size,
            self.cfg.num_heads,
            self.cfg.max_seq_len,
            self.cfg.head_dim,
        )
        self._init_host_cache(host_shape)

        bytes_per_page = (
            2
            * self.cfg.batch_size
            * self.cfg.num_heads
            * self.cfg.page_tokens
            * self.cfg.head_dim
            * torch.finfo(self.runtime_dtype).bits
            / 8.0
        )
        self._repeat_pages = max(1, self.cfg.repeat_pages)
        self._repeat_page_range = range(self._repeat_pages)
        self._bytes_per_iteration = float(bytes_per_page * self._repeat_pages)
        self.register_workload_metadata(bytes_per_iteration=self._bytes_per_iteration)

        self._start_host_prefetch_thread()

        # Precompute a deterministic query tensor in the runtime dtype.
        # Some PyTorch builds lack RNG kernels for FP8; generate in FP16 and cast.
        q_dtype = self.runtime_dtype
        needs_cast = q_dtype in (_FLOAT8_E4M3FN, _FLOAT8_E5M2FN)
        gen_dtype = torch.float16 if needs_cast else q_dtype
        q = torch.randn(
            self.cfg.batch_size,
            self.cfg.num_heads,
            self.cfg.decode_tokens,
            self.cfg.head_dim,
            device=self.device,
            dtype=gen_dtype,
        )
        if needs_cast:
            q = q.to(dtype=q_dtype)
        self.q = q
        self._verify_output_buffer = torch.empty(
            (self.cfg.batch_size, self.cfg.num_heads, 1, self._verify_head_dim),
            device=self.device,
            dtype=torch.float32,
        )

    # -------------------- Benchmark --------------------

    def _maybe_use_prefetch(self, start: int) -> Optional[Tuple[torch.Tensor, int]]:
        if self.prefetched_range is None or self.prefetch_staging is None:
            return None
        pref_start, pref_end = self.prefetched_range
        if start != pref_start:
            return None
        return self.prefetch_staging, pref_end - pref_start

    def _prefetch_worker(self) -> None:
        if self._prefetch_queue is None or self.prefetch_staging is None:
            return
        while not self._prefetch_stop.is_set():
            try:
                start = self._prefetch_queue.get(timeout=0.05)
            except queue.Empty:
                continue
            if start < 0 or self._prefetch_stop.is_set():
                self._prefetch_queue.task_done()
                break
            _, slice_len = self._stage_page(start, into_prefetch=True)
            with self._prefetch_lock:
                self.prefetched_range = (start, start + slice_len)
                self.prefetch_slice_len = slice_len
                self._prefetch_ready_start = start
            if self._prefetch_ready is not None:
                self._prefetch_ready.set()
            self._prefetch_queue.task_done()

    def _start_host_prefetch_thread(self) -> None:
        if not (self.cfg.use_host_prefetch_thread and self.cfg.prefetch_next_page):
            return
        if self.prefetch_staging is None:
            return
        if self._prefetch_thread is not None:
            return
        self._prefetch_queue = queue.Queue(maxsize=1)
        self._prefetch_ready = threading.Event()
        self._prefetch_stop.clear()
        self._prefetch_thread = threading.Thread(
            target=self._prefetch_worker,
            name="paged_kv_prefetch",
            daemon=True,
        )
        self._prefetch_thread.start()

    def _stop_host_prefetch_thread(self) -> None:
        if self._prefetch_thread is None:
            return
        self._prefetch_stop.set()
        if self._prefetch_queue is not None:
            try:
                self._prefetch_queue.put_nowait(-1)
            except queue.Full:
                pass
        self._prefetch_thread.join(timeout=2.0)
        self._prefetch_thread = None
        self._prefetch_queue = None
        self._prefetch_ready = None
        self._prefetch_ready_start = None

    def _schedule_host_prefetch(self, start: int) -> None:
        if self._prefetch_queue is None:
            return
        if self._prefetch_ready is not None and self._prefetch_ready.is_set():
            return
        try:
            self._prefetch_queue.put_nowait(start)
        except queue.Full:
            return

    def _consume_host_prefetch(self, start: int) -> Optional[Tuple[torch.Tensor, int]]:
        if self._prefetch_ready is None or self.prefetch_staging is None:
            return None
        if not self._prefetch_ready.is_set():
            return None
        with self._prefetch_lock:
            if self._prefetch_ready_start != start:
                return None
            slice_len = self.prefetch_slice_len
            self._prefetch_ready.clear()
            self._prefetch_ready_start = None
        if slice_len is None:
            return None
        return self.prefetch_staging, slice_len

    def _wait_for_host_prefetch(self, start: int, timeout_s: float = 1.0) -> Optional[Tuple[torch.Tensor, int]]:
        if self._prefetch_ready is None:
            return None
        if self._prefetch_ready.is_set():
            return self._consume_host_prefetch(start)
        if self._prefetch_ready.wait(timeout=timeout_s):
            return self._consume_host_prefetch(start)
        return None

    def benchmark_fn(self) -> None:
        repeat_page_range = self._repeat_page_range
        attn_out = None
        use_host_prefetch = bool(self.cfg.use_host_prefetch_thread and self.cfg.prefetch_next_page)
        current_stream = torch.cuda.current_stream() if self.copy_stream is not None else None
        ctx = prefer_sdpa_backends() if prefer_sdpa_backends is not None else nullcontext()
        with ctx:
            for _ in repeat_page_range:
                start = self.page_cursor
                active_idx = self.active_buf_idx
                prefetched = self._consume_host_prefetch(start) if use_host_prefetch else self._maybe_use_prefetch(start)
                if prefetched is not None:
                    staged, slice_len = prefetched
                    if self.prefetch_event is not None and self.prefetch_buf_idx is not None:
                        if current_stream is None:
                            raise RuntimeError("Prefetch event requires an async copy stream")
                        current_stream.wait_event(self.prefetch_event)
                        active_idx = self.prefetch_buf_idx
                    else:
                        self._copy_to_device(
                            staged,
                            slice_len,
                            buffer_idx=active_idx,
                            wait_for_copy=True,
                            wait_stream=current_stream,
                        )
                else:
                    staged, slice_len = self._stage_page(start)
                    self._copy_to_device(
                        staged,
                        slice_len,
                        buffer_idx=active_idx,
                        wait_for_copy=True,
                        wait_stream=current_stream,
                    )

                next_start = (start + self.cfg.page_tokens) % self.cfg.max_seq_len
                if self.cfg.prefetch_next_page and use_host_prefetch:
                    self._schedule_host_prefetch(next_start)

                self.hot_k = self.hot_k_bufs[active_idx]
                self.hot_v = self.hot_v_bufs[active_idx]

                # Simple attention step that will pick flash/mathematics based on dtype/backend.
                q = self.q
                if q is None:
                    raise RuntimeError("Query tensor not initialized")
                k = self.hot_k[..., :slice_len, :]
                v = self.hot_v[..., :slice_len, :]
                attn_out = F.scaled_dot_product_attention(q, k, v)

                if self.cfg.prefetch_next_page:
                    # Launch next-page prefetch so H2D can overlap attention compute.
                    if use_host_prefetch:
                        prefetched = self._wait_for_host_prefetch(next_start)
                        if prefetched is None:
                            staged_prefetch, pref_len = self._stage_page(next_start, into_prefetch=True)
                        else:
                            staged_prefetch, pref_len = prefetched
                    else:
                        staged_prefetch, pref_len = self._stage_page(next_start, into_prefetch=True)
                    self.prefetched_range = (next_start, next_start + pref_len)
                    self.prefetch_slice_len = pref_len
                    if self.copy_stream is not None and self._hot_buffer_count > 1:
                        prefetch_idx = 1 - active_idx
                        self.prefetch_buf_idx = prefetch_idx
                        if self.prefetch_event is None:
                            raise RuntimeError("Prefetch event not initialized for async two-buffer prefetch")
                        self._copy_to_device(
                            staged_prefetch,
                            pref_len,
                            buffer_idx=prefetch_idx,
                            wait_for_copy=False,
                            record_event=self.prefetch_event,
                        )
                    else:
                        self.prefetch_buf_idx = None
                else:
                    self.prefetched_range = None
                    self.prefetch_slice_len = None
                    self.prefetch_buf_idx = None

                self.page_cursor = next_start

        if attn_out is None:
            raise RuntimeError("benchmark_fn() did not produce output")
        # Capture a slice of attention output for verification
        self.output = attn_out[:, :, :1, : self._verify_head_dim]
        fp8_enabled = _FLOAT8_E4M3FN is not None and self.runtime_dtype == _FLOAT8_E4M3FN
        self._payload_fp8_enabled = fp8_enabled
        self._payload_k = k
        self._payload_q = q
        self._payload_v = v

    def capture_verification_payload(self) -> None:
        fp8_enabled = self._payload_fp8_enabled
        k = self._payload_k
        q = self._payload_q
        v = self._payload_v
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"q": q.detach(), "k": k.detach(), "v": v.detach()},
            output=self._verify_output_buffer,
            batch_size=self.cfg.batch_size,
            parameter_count=0,
            precision_flags={
                "fp16": self.runtime_dtype == torch.float16,
                "bf16": self.runtime_dtype == torch.bfloat16,
                "fp8": fp8_enabled,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    # -------------------- Teardown --------------------

    def teardown(self) -> None:
        self._stop_host_prefetch_thread()
        self.hot_k = None
        self.hot_v = None
        self.hot_kv_bufs = []
        self.hot_k_bufs = []
        self.hot_v_bufs = []
        self._hot_buffer_count = 0
        self.active_buf_idx = 0
        self.prefetch_buf_idx = None
        self.prefetch_slice_len = None
        self.prefetch_event = None
        self.q = None
        self.staging = None
        self.prefetch_staging = None
        self.copy_stream = None
        self.host_cache = None
        self._host_cache_is_pinned = False
        self._verify_output_buffer = None
        if self.host_memmap is not None:
            self.host_memmap._mmap.close()  # type: ignore[attr-defined]
        self.host_memmap = None
        if self._memmap_path is not None:
            try:
                os.remove(self._memmap_path)
                os.rmdir(self._memmap_path.parent)
            except OSError:
                pass
        self._memmap_path = None
        self.output = None
        self._repeat_page_range = range(0)
        super().teardown()

    # -------------------- Harness config --------------------

    def get_config(self) -> BenchmarkConfig:
        config = BenchmarkConfig(
            iterations=12,
            warmup=5,
            timeout_seconds=300,
            measurement_timeout_seconds=300,
            deterministic=False,
        )
        # Force single-GPU visibility for consistent runs on multi-GPU hosts.
        config.single_gpu = True
        return config

    def get_input_signature(self) -> dict:
        return simple_signature(
            batch_size=self.cfg.batch_size,
            dtype="float16",
            num_heads=self.cfg.num_heads,
            head_dim=self.cfg.head_dim,
            max_seq_len=self.cfg.max_seq_len,
            page_tokens=self.cfg.page_tokens,
            decode_tokens=self.cfg.decode_tokens,
            repeat_pages=self.cfg.repeat_pages,
            precision_flags=PrecisionFlags(fp16=True, tf32=False),
        ).to_dict()

    def validate_result(self) -> Optional[str]:
        if self._fp8_reason and self.cfg.prefer_fp8:
            # Report the path we took for visibility; not a failure.
            print(f"[{self.label}] {self._fp8_reason}")
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Return paged KV offload performance metrics."""
        return {
            f"{self.label}.bytes_per_iteration": self._bytes_per_iteration,
            f"{self.label}.batch_size": float(self.cfg.batch_size),
            f"{self.label}.num_heads": float(self.cfg.num_heads),
            f"{self.label}.head_dim": float(self.cfg.head_dim),
            f"{self.label}.page_tokens": float(self.cfg.page_tokens),
            f"{self.label}.decode_tokens": float(self.cfg.decode_tokens),
            f"{self.label}.max_seq_len": float(self.cfg.max_seq_len),
            f"{self.label}.use_fp8": float(_FLOAT8_E4M3FN is not None and self.runtime_dtype == _FLOAT8_E4M3FN),
            f"{self.label}.use_pinned": float(self.cfg.use_pinned_stage),
            f"{self.label}.use_async": float(self.cfg.use_async_stream),
        }

 
