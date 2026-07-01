"""Shared helpers for the split Chapter 18 paged-attention benchmarks."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn.functional as F

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range
from ch18.paged_attn_common import compute_paged_attention_metrics


def _set_math_backend() -> None:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    if not torch.backends.cuda.math_sdp_enabled():
        raise RuntimeError("FAIL FAST: Math SDPA backend is not enabled")


def _set_flash_backend() -> None:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(False)
    if not torch.backends.cuda.flash_sdp_enabled():
        raise RuntimeError("FAIL FAST: Flash SDPA backend is not enabled")


def _require_flex_attention() -> tuple[Callable[..., object], Callable[..., torch.Tensor]]:
    try:
        from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    except Exception as exc:  # pragma: no cover - exercised by runtime dogfooding
        raise RuntimeError(
            "FAIL FAST: ch18 paged_attn_layout requires torch.nn.attention.flex_attention"
        ) from exc
    return create_block_mask, flex_attention


def _empty_cpu_staging(shape: torch.Size | tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    try:
        return torch.empty(shape, device="cpu", dtype=dtype, pin_memory=True)
    except RuntimeError:
        return torch.empty(shape, device="cpu", dtype=dtype)


class DensePagedAttnBase(VerificationPayloadMixin, BaseBenchmark):
    """Dense attention benchmark that varies only the SDPA backend."""

    backend = "math"
    nvtx_label = "paged_attn"

    def __init__(self) -> None:
        super().__init__()
        self.qkv: Optional[torch.Tensor] = None
        self._q_view: Optional[torch.Tensor] = None
        self._k_view: Optional[torch.Tensor] = None
        self._v_view: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.batch_size = 4
        self.num_heads = 16
        self.seq_len = 2048
        self.head_dim = 64
        self.block_size = 128
        self.chunk_size = 2048
        self._empty_iteration_result = {}
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.seq_len),
        )
        self._enable_nvtx = False
        self._verify_input_buffers: dict[str, torch.Tensor] = {}
        self._verify_output_buffer: Optional[torch.Tensor] = None

    def _configure_backend(self) -> None:
        if self.backend == "math":
            _set_math_backend()
        elif self.backend == "flash":
            _set_flash_backend()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        if not torch.cuda.is_available():
            raise RuntimeError("FAIL FAST: paged attention benchmark requires CUDA")

        b, h, s, d = self.batch_size, self.num_heads, self.seq_len, self.head_dim
        self.qkv = torch.randn(b, h, s, 3, d, device=self.device, dtype=torch.bfloat16)
        self._configure_backend()
        self._q_view = self.qkv[:, :, :, 0]
        self._k_view = self.qkv[:, :, :, 1]
        self._v_view = self.qkv[:, :, :, 2]
        self._verify_input_buffers = {
            "qkv": _empty_cpu_staging(self.qkv.shape, self.qkv.dtype),
        }
        self._verify_output_buffer = _empty_cpu_staging(self._q_view.shape, self.qkv.dtype)
        for _ in range(8):
            _ = F.scaled_dot_product_attention(self._q_view, self._k_view, self._v_view)
            torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if self.qkv is None or self._q_view is None or self._k_view is None or self._v_view is None:
            raise RuntimeError("FAIL FAST: QKV not initialized")

        with nvtx_range(self.nvtx_label, enable=self._enable_nvtx):
            self.output = F.scaled_dot_product_attention(self._q_view, self._k_view, self._v_view)
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output")
        return self._empty_iteration_result

    def capture_verification_payload(self) -> None:
        if self.qkv is None or self.output is None:
            raise RuntimeError("FAIL FAST: paged_attn verification requires setup() and benchmark_fn()")
        if self._verify_output_buffer is None or "qkv" not in self._verify_input_buffers:
            raise RuntimeError("FAIL FAST: paged_attn verification buffers are not initialized")
        self._verify_input_buffers["qkv"].copy_(self.qkv, non_blocking=False)
        self._verify_output_buffer.copy_(self.output, non_blocking=False)
        self._set_verification_payload(
            # Move verification tensors off GPU so the worker can release device
            # memory before the next phase starts.
            inputs=self._verify_input_buffers,
            output=self._verify_output_buffer,
            batch_size=self.qkv.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": self.qkv.dtype == torch.float16,
                "bf16": self.qkv.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.1, 1.0),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def teardown(self) -> None:
        self.qkv = None
        self._q_view = None
        self._k_view = None
        self._v_view = None
        self.output = None
        self._verify_input_buffers = {}
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_custom_metrics(self) -> Optional[dict]:
        metrics = compute_paged_attention_metrics(
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            bytes_per_element=2 if self.qkv is not None and self.qkv.element_size() == 2 else 4,
            elapsed_ms=getattr(self, "_last_elapsed_ms", None),
            block_size=self.block_size,
            chunk_size=self.chunk_size,
            uses_paged_kv=False,
        )
        metrics["paged_attn.backend_math"] = 1.0 if self.backend == "math" else 0.0
        metrics["paged_attn.backend_flash"] = 1.0 if self.backend == "flash" else 0.0
        return metrics


class LayoutPagedAttnBase(VerificationPayloadMixin, BaseBenchmark):
    """Decode benchmark that compares dense masked attention against a real block-sparse kernel."""

    uses_paged_kv = False
    nvtx_label = "paged_attn_layout"

    def __init__(self) -> None:
        super().__init__()
        self.q: Optional[torch.Tensor] = None
        self.k_dense: Optional[torch.Tensor] = None
        self.v_dense: Optional[torch.Tensor] = None
        self.block_table: Optional[torch.Tensor] = None
        self.dense_mask: Optional[torch.Tensor] = None
        self.block_mask: Optional[object] = None
        self._flex_attention_fn: Optional[Callable[..., torch.Tensor]] = None
        self._flex_compiled = False
        self.output: Optional[torch.Tensor] = None
        self.batch_size = 4
        self.num_heads = 16
        self.seq_len = 4096
        self.decode_tokens = 128
        self.head_dim = 64
        self.block_size = 128
        self.local_blocks = 4
        self.chunk_size = self.block_size * self.local_blocks
        self.q_offset = self.seq_len - self.decode_tokens
        self._empty_iteration_result = {}
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.decode_tokens),
        )
        self._enable_nvtx = False
        self._verify_input_buffers: dict[str, torch.Tensor] = {}
        self._verify_output_buffer: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        config = getattr(self, "_config", None) or self.get_config()
        self._enable_nvtx = get_nvtx_enabled(config) if config else False
        if not torch.cuda.is_available():
            raise RuntimeError("FAIL FAST: paged attention benchmark requires CUDA")
        if self.seq_len % self.block_size != 0:
            raise RuntimeError("FAIL FAST: paged_attn_layout requires seq_len divisible by block_size")
        if self.decode_tokens <= 0 or self.decode_tokens > self.seq_len:
            raise RuntimeError("FAIL FAST: decode_tokens must be in the range (0, seq_len]")
        self.q_offset = self.seq_len - self.decode_tokens

        b, h, s, d = self.batch_size, self.num_heads, self.seq_len, self.head_dim
        self.q = torch.randn(b, h, self.decode_tokens, d, device=self.device, dtype=torch.bfloat16).contiguous()
        self.k_dense = torch.randn(b, h, s, d, device=self.device, dtype=torch.bfloat16).contiguous()
        self.v_dense = torch.randn(b, h, s, d, device=self.device, dtype=torch.bfloat16).contiguous()
        self.block_table = self._build_block_table()
        self.dense_mask = self._build_dense_mask_from_block_table()
        self._verify_input_buffers = {
            "q": _empty_cpu_staging(self.q.shape, self.q.dtype),
            "k_dense": _empty_cpu_staging(self.k_dense.shape, self.k_dense.dtype),
            "v_dense": _empty_cpu_staging(self.v_dense.shape, self.v_dense.dtype),
            "block_table": _empty_cpu_staging(self.block_table.shape, self.block_table.dtype),
        }
        self._verify_output_buffer = _empty_cpu_staging(self.q.shape, self.q.dtype)
        _set_math_backend()
        if self.uses_paged_kv:
            self.block_mask = self._build_block_mask_from_block_table()
            self._flex_attention_fn = self._compile_flex_attention()
        for _ in range(16):
            _ = self._run_attention()
            torch.cuda.synchronize(self.device)
        self.output = None

    def _build_block_table(self) -> torch.Tensor:
        num_blocks = self.seq_len // self.block_size
        block_ids = torch.arange(num_blocks, device=self.device, dtype=torch.int64)
        batch_offsets = torch.arange(self.batch_size, device=self.device, dtype=torch.int64).unsqueeze(1)
        return (block_ids.unsqueeze(0) - batch_offsets).remainder_(num_blocks)

    def _build_dense_mask_from_block_table(self) -> torch.Tensor:
        if self.block_table is None:
            raise RuntimeError("FAIL FAST: block table must be initialized before building masks")
        q_ids = torch.arange(self.decode_tokens, device=self.device, dtype=torch.int64) + self.q_offset
        kv_ids = torch.arange(self.seq_len, device=self.device, dtype=torch.int64)
        logical_q_blocks = (q_ids // self.block_size).unsqueeze(0).expand(self.batch_size, -1)
        logical_kv_blocks = (kv_ids // self.block_size).unsqueeze(0).expand(self.batch_size, -1)
        physical_q_blocks = self.block_table.gather(1, logical_q_blocks)
        physical_kv_blocks = self.block_table.gather(1, logical_kv_blocks)
        physical_window = (physical_kv_blocks.unsqueeze(1) <= physical_q_blocks.unsqueeze(2)) & (
            (physical_q_blocks.unsqueeze(2) - physical_kv_blocks.unsqueeze(1)) < self.local_blocks
        )
        logical_q_positions = q_ids.unsqueeze(0).expand(self.batch_size, -1)
        causal = kv_ids.unsqueeze(0).unsqueeze(1) <= logical_q_positions.unsqueeze(2)
        allowed = physical_window & causal
        dense_mask = torch.full(
            (self.batch_size, 1, self.decode_tokens, self.seq_len),
            float("-inf"),
            device=self.device,
            dtype=self.q.dtype if self.q is not None else torch.bfloat16,
        )
        dense_mask[:, 0][allowed] = 0.0
        return dense_mask.contiguous()

    def _build_block_mask_from_block_table(self) -> object:
        if self.block_table is None:
            raise RuntimeError("FAIL FAST: block table must be initialized before building block masks")
        create_block_mask, _ = _require_flex_attention()
        block_table = self.block_table
        q_offset = self.q_offset
        block_size = self.block_size
        local_blocks = self.local_blocks

        def mask_fn(batch: torch.Tensor, head: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
            del head
            logical_q_pos = q_idx + q_offset
            logical_q_block = (q_idx + q_offset) // block_size
            logical_kv_block = kv_idx // block_size
            physical_q_block = block_table[batch, logical_q_block]
            physical_kv_block = block_table[batch, logical_kv_block]
            return (kv_idx <= logical_q_pos) & (physical_kv_block <= physical_q_block) & (
                (physical_q_block - physical_kv_block) < local_blocks
            )

        return create_block_mask(
            mask_fn,
            B=self.batch_size,
            H=self.num_heads,
            Q_LEN=self.decode_tokens,
            KV_LEN=self.seq_len,
            device=self.device,
            BLOCK_SIZE=self.block_size,
        )

    def _compile_flex_attention(self) -> Callable[..., torch.Tensor]:
        _, flex_attention = _require_flex_attention()
        try:
            compiled = torch.compile(flex_attention, mode="max-autotune")
        except Exception as exc:
            raise RuntimeError(
                f"FAIL FAST: torch.compile(flex_attention) failed for paged attention sparse path "
                f"({type(exc).__name__}: {exc})"
            ) from exc
        self._flex_compiled = True
        return compiled

    def _run_attention(self) -> torch.Tensor:
        if self.q is None or self.k_dense is None or self.v_dense is None or self.dense_mask is None:
            raise RuntimeError("FAIL FAST: attention tensors not initialized")
        if not self.uses_paged_kv:
            return F.scaled_dot_product_attention(self.q, self.k_dense, self.v_dense, attn_mask=self.dense_mask)
        if self.block_mask is None or self._flex_attention_fn is None:
            raise RuntimeError("FAIL FAST: block-sparse attention path is not initialized")
        return self._flex_attention_fn(self.q, self.k_dense, self.v_dense, block_mask=self.block_mask)

    def benchmark_fn(self) -> Optional[dict]:
        with nvtx_range(self.nvtx_label, enable=self._enable_nvtx):
            self.output = self._run_attention()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output")
        return self._empty_iteration_result

    def capture_verification_payload(self) -> None:
        if self.q is None or self.k_dense is None or self.v_dense is None or self.block_table is None or self.output is None:
            raise RuntimeError("FAIL FAST: paged_attn_layout verification requires setup() and benchmark_fn()")
        if self._verify_output_buffer is None:
            raise RuntimeError("FAIL FAST: paged_attn_layout verification output buffer is not initialized")
        for name, tensor in (
            ("q", self.q),
            ("k_dense", self.k_dense),
            ("v_dense", self.v_dense),
            ("block_table", self.block_table),
        ):
            self._verify_input_buffers[name].copy_(tensor, non_blocking=False)
        self._verify_output_buffer.copy_(self.output, non_blocking=False)
        self._set_verification_payload(
            # Move verification tensors off GPU so the worker can release device
            # memory before the next phase starts.
            inputs=self._verify_input_buffers,
            output=self._verify_output_buffer,
            batch_size=self.q.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": self.q.dtype == torch.float16,
                "bf16": self.q.dtype == torch.bfloat16,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(1e-2, 1e-2),
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def teardown(self) -> None:
        self.q = None
        self.k_dense = None
        self.v_dense = None
        self.block_table = None
        self.dense_mask = None
        self.block_mask = None
        self._flex_attention_fn = None
        self.output = None
        self._verify_input_buffers = {}
        self._verify_output_buffer = None
        torch.cuda.empty_cache()

    def get_custom_metrics(self) -> Optional[dict]:
        metrics = compute_paged_attention_metrics(
            batch_size=self.batch_size,
            num_heads=self.num_heads,
            seq_len=self.seq_len,
            head_dim=self.head_dim,
            bytes_per_element=2 if self.q is not None and self.q.element_size() == 2 else 4,
            elapsed_ms=getattr(self, "_last_elapsed_ms", None),
            block_size=self.block_size,
            chunk_size=self.chunk_size,
            uses_paged_kv=self.uses_paged_kv,
            query_len=self.decode_tokens,
        )
        metrics["paged_attn.backend_math"] = 1.0 if not self.uses_paged_kv else 0.0
        metrics["paged_attn.backend_flash"] = 0.0
        metrics["paged_attn.decode_tokens"] = float(self.decode_tokens)
        metrics["paged_attn.local_blocks"] = float(self.local_blocks)
        metrics["paged_attn.block_sparse_kernel"] = 1.0 if self.uses_paged_kv else 0.0
        metrics["paged_attn.block_mask_compiled"] = 1.0 if self._flex_compiled else 0.0
        return metrics
