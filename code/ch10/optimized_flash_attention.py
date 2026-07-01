"""optimized_flash_attention.py - FlashAttention via external kernels or SDPA fallback.

This module demonstrates how FlashAttention achieves O(seq_len) memory complexity
through the same intra-kernel pipelining and tiling concepts taught in Chapter 10.

FlashAttention's key insight (aligning with Ch10 concepts):
- Standard attention materializes the full [seq_len × seq_len] attention matrix
- FlashAttention tiles the computation, processing attention in SRAM blocks
- This is exactly the producer-consumer pipelining pattern from this chapter
- Memory: O(seq_len) instead of O(seq_len²) by never materializing full matrix

The optimization here is selecting the SDPA backend that uses tiled attention,
demonstrating the practical benefit of the intra-kernel pipelining concepts.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import contextmanager
from typing import Callable, Optional

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    WorkloadMetadata,
)
from ch10.flash_attention_common import (
    FLASH_ATTENTION_OUTPUT_TOLERANCE,
    compute_attention_backend_metrics,
    compute_attention_workload_metrics,
)

# Use new SDPA API when available (PyTorch 2.2+)
try:
    from torch.nn.attention import sdpa_kernel, SDPBackend
    _NEW_SDPA_API = True
except ImportError:
    sdpa_kernel = None  # type: ignore[assignment]
    SDPBackend = None  # type: ignore[assignment]
    _NEW_SDPA_API = False


@contextmanager
def sdpa_backend_context(backends: Optional[list[SDPBackend]] = None):
    """Context manager to select an SDPA backend in SDPA.
    
    FlashAttention uses the same principles taught in Chapter 10:
    - Tiled computation: processes attention in blocks that fit in SRAM
    - Pipelining: overlaps memory loads with softmax/matmul computation  
    - Never materializes the full O(seq_len²) attention matrix
    
    This is analogous to the double-buffered pipeline pattern but applied
    to the attention computation itself.
    """
    if _NEW_SDPA_API and sdpa_kernel is not None and backends:
        with sdpa_kernel(backends):
            yield
    else:
        yield


class TiledAttentionModule(nn.Module):
    """Attention module using tiled computation via SDPA.
    
    This demonstrates the practical application of Chapter 10's concepts:
    - The attention computation is broken into tiles
    - Each tile fits in shared memory (like our GEMM examples)
    - Softmax is computed incrementally to avoid storing full matrix
    """
    
    def __init__(self, hidden_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.dropout = dropout
        
        # Separate Q, K, V projections (could fuse, but keeping simple for ch10)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self._q_buffer: Optional[torch.Tensor] = None
        self._k_buffer: Optional[torch.Tensor] = None
        self._v_buffer: Optional[torch.Tensor] = None
        self._output_buffer: Optional[torch.Tensor] = None
        self._q_forward_view: Optional[torch.Tensor] = None
        self._k_forward_view: Optional[torch.Tensor] = None
        self._v_forward_view: Optional[torch.Tensor] = None
        self._output_forward_view: Optional[torch.Tensor] = None
        self._q_proj_weight_t: Optional[torch.Tensor] = None
        self._k_proj_weight_t: Optional[torch.Tensor] = None
        self._v_proj_weight_t: Optional[torch.Tensor] = None
        self._out_proj_weight_t: Optional[torch.Tensor] = None

    def cache_weight_views(self) -> None:
        self._q_proj_weight_t = self.q_proj.weight.t()
        self._k_proj_weight_t = self.k_proj.weight.t()
        self._v_proj_weight_t = self.v_proj.weight.t()
        self._out_proj_weight_t = self.out_proj.weight.t()

    def _workspace(
        self,
        name: str,
        shape: tuple[int, int, int],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        batch_size, seq_len, width = (int(shape[0]), int(shape[1]), int(shape[2]))
        numel = batch_size * seq_len * width
        buffer = getattr(self, name)
        if (
            not isinstance(buffer, torch.Tensor)
            or buffer.device != device
            or buffer.dtype != dtype
            or buffer.numel() < numel
        ):
            buffer = torch.empty(numel, device=device, dtype=dtype)
            setattr(self, name, buffer)
        return buffer[:numel].view(batch_size, seq_len, width)

    def _ensure_projection_buffers(
        self,
        x: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        shape = (batch_size, seq_len, self.hidden_dim)
        q = self._workspace("_q_buffer", shape, device=x.device, dtype=x.dtype)
        k = self._workspace("_k_buffer", shape, device=x.device, dtype=x.dtype)
        v = self._workspace("_v_buffer", shape, device=x.device, dtype=x.dtype)
        output = self._workspace("_output_buffer", shape, device=x.device, dtype=x.dtype)
        return q, k, v, output

    def prepare_projection_buffers(self, x: torch.Tensor) -> None:
        batch_size, seq_len, _ = x.shape
        (
            self._q_forward_view,
            self._k_forward_view,
            self._v_forward_view,
            self._output_forward_view,
        ) = self._ensure_projection_buffers(x, batch_size, seq_len)

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Project input into [batch, seq, heads, head_dim] tensors."""
        batch_size, seq_len, _ = x.shape
        if torch.is_grad_enabled():
            q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
            return q, k, v

        q_buffer, k_buffer, v_buffer, _output_buffer = self._ensure_projection_buffers(
            x,
            batch_size,
            seq_len,
        )
        if (
            self._q_proj_weight_t is None
            or self._k_proj_weight_t is None
            or self._v_proj_weight_t is None
        ):
            self.cache_weight_views()
        return self._project_qkv_into_buffers(x, q_buffer, k_buffer, v_buffer, batch_size, seq_len)

    def _project_qkv_prepared(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = x.shape
        q_buffer = self._q_forward_view
        k_buffer = self._k_forward_view
        v_buffer = self._v_forward_view
        if q_buffer is None or k_buffer is None or v_buffer is None:
            raise RuntimeError("forward_prepared() requires prepare_projection_buffers()")
        if (
            self._q_proj_weight_t is None
            or self._k_proj_weight_t is None
            or self._v_proj_weight_t is None
        ):
            self.cache_weight_views()
        return self._project_qkv_into_buffers(x, q_buffer, k_buffer, v_buffer, batch_size, seq_len)

    def _project_qkv_into_buffers(
        self,
        x: torch.Tensor,
        q_buffer: torch.Tensor,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        batch_size: int,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q = torch.matmul(x, self._q_proj_weight_t, out=q_buffer)
        k = torch.matmul(x, self._k_proj_weight_t, out=k_buffer)
        v = torch.matmul(x, self._v_proj_weight_t, out=v_buffer)
        return (
            q.view(batch_size, seq_len, self.num_heads, self.head_dim),
            k.view(batch_size, seq_len, self.num_heads, self.head_dim),
            v.view(batch_size, seq_len, self.num_heads, self.head_dim),
        )

    def _project_output(self, attn_output: torch.Tensor) -> torch.Tensor:
        """Project [batch, seq, heads, head_dim] attention output back to hidden_dim."""
        batch_size, seq_len, _, _ = attn_output.shape
        merged = attn_output.reshape(batch_size, seq_len, self.hidden_dim)
        if torch.is_grad_enabled():
            return self.out_proj(merged)
        _q_buffer, _k_buffer, _v_buffer, output_buffer = self._ensure_projection_buffers(
            merged,
            batch_size,
            seq_len,
        )
        if self._out_proj_weight_t is None:
            self.cache_weight_views()
        return torch.matmul(merged, self._out_proj_weight_t, out=output_buffer)

    def _project_output_prepared(self, attn_output: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _, _ = attn_output.shape
        output_buffer = self._output_forward_view
        if output_buffer is None:
            raise RuntimeError("forward_prepared() requires prepare_projection_buffers()")
        if self._out_proj_weight_t is None:
            self.cache_weight_views()
        merged = attn_output.reshape(batch_size, seq_len, self.hidden_dim)
        return torch.matmul(merged, self._out_proj_weight_t, out=output_buffer)
        
    def forward(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        """Forward pass using tiled attention.
        
        Args:
            x: Input tensor [batch, seq_len, hidden_dim]
            is_causal: If True, apply causal mask (autoregressive)
            
        Returns:
            Output tensor [batch, seq_len, hidden_dim]
        """
        q, k, v = self._project_qkv(x)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Tiled attention via SDPA
        # Internally, this uses the same tiling strategy discussed in Ch10:
        # - Break Q, K, V into tiles that fit in SRAM
        # - Compute partial attention scores per tile
        # - Accumulate softmax incrementally (online softmax)
        # - Never store the full [seq_len × seq_len] attention matrix
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        
        # Reshape back: [batch, seq, hidden]
        return self._project_output(attn_output.transpose(1, 2).contiguous())

    def forward_prepared(self, x: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        if torch.is_grad_enabled():
            return self.forward(x, is_causal=is_causal)
        q, k, v = self._project_qkv_prepared(x)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        return self._project_output_prepared(attn_output.transpose(1, 2).contiguous())

    def forward_external_flash(
        self,
        x: torch.Tensor,
        flash_attn_func: Callable[..., torch.Tensor],
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        """Run an external FlashAttention kernel when one is available."""
        q, k, v = self._project_qkv(x)
        attn_output = flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            causal=is_causal,
        )
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        return self._project_output(attn_output.contiguous())

    def forward_external_flash_prepared(
        self,
        x: torch.Tensor,
        flash_attn_func: Callable[..., torch.Tensor],
        *,
        is_causal: bool,
    ) -> torch.Tensor:
        if torch.is_grad_enabled():
            return self.forward_external_flash(
                x,
                flash_attn_func,
                is_causal=is_causal,
            )
        q, k, v = self._project_qkv_prepared(x)
        attn_output = flash_attn_func(
            q,
            k,
            v,
            dropout_p=0.0,
            causal=is_causal,
        )
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        return self._project_output_prepared(attn_output.contiguous())


class OptimizedFlashAttentionBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Tiled attention via FlashAttention SDPA backend.
    
    This benchmark demonstrates the practical benefit of Chapter 10's
    intra-kernel pipelining concepts applied to attention:
    
    Baseline (standard attention):
    - Computes full [seq_len × seq_len] attention matrix
    - Memory: O(seq_len²) 
    - For seq_len=4096: 4096² × 4 bytes × batch × heads = HUGE
    
    Optimized (FlashAttention/tiled):
    - Tiles the computation, never stores full matrix
    - Memory: O(seq_len)
    - Uses same producer-consumer pattern as Ch10's pipeline examples
    
    Expected improvement: Memory usage scales linearly, not quadratically.
    """
    
    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        
        # Use larger sizes to show the memory benefit of tiling
        self.batch_size = 4
        self.seq_len = 1024  # At 1024, O(n²) vs O(n) matters significantly
        self.hidden_dim = 512
        self.num_heads = 8
        self.use_causal = True
        
        self.input: Optional[torch.Tensor] = None
        
        tokens = self.batch_size * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.output = None
        self._verify_output_buffer: Optional[torch.Tensor] = None
        self._sdpa_backends: Optional[list[SDPBackend]] = None
        self._attention_runner: Optional[Callable[[torch.Tensor], torch.Tensor]] = None
        self._external_flash_func: Optional[Callable[..., torch.Tensor]] = None
        self._selected_engine_name = "sdpa"
        self._selected_backend_name = "default"
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def _candidate_backends(self) -> list[list[SDPBackend]]:
        if not _NEW_SDPA_API or sdpa_kernel is None or SDPBackend is None:
            return []
        candidates: list[list[SDPBackend]] = []
        major = 0
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
        if major >= 8:
            candidates.append([SDPBackend.FLASH_ATTENTION])
        elif torch.cuda.is_available():
            candidates.append([SDPBackend.EFFICIENT_ATTENTION])
        if hasattr(SDPBackend, "MATH"):
            candidates.append([SDPBackend.MATH])
        return candidates

    def _try_sdpa_backend(self, candidate: list[SDPBackend]) -> bool:
        if self.model is None or self.input is None:
            raise RuntimeError("setup() must initialize model and input before selecting SDPA backends")
        with torch.inference_mode(), sdpa_backend_context(candidate):
            _ = self.model(self.input[:1], is_causal=self.use_causal)
        self._sdpa_backends = candidate
        self._selected_backend_name = candidate[0].name.lower()
        return True

    def _resolve_sdpa_backends(self) -> None:
        if self.model is None or self.input is None:
            raise RuntimeError("setup() must initialize model and input before selecting SDPA backends")
        last_error: Optional[Exception] = None
        for candidate in self._candidate_backends():
            try:
                self._try_sdpa_backend(candidate)
                return
            except Exception as exc:  # pragma: no cover - CUDA/PyTorch dependent
                last_error = exc
                continue
        if _NEW_SDPA_API and torch.cuda.is_available():
            raise RuntimeError("No supported SDPA backend found for the Chapter 10 FlashAttention benchmark") from last_error
        self._sdpa_backends = None
        self._selected_backend_name = "default"

    def _resolve_external_flash(self) -> bool:
        if self.model is None or self.input is None or not torch.cuda.is_available():
            return False
        if self.input.dtype not in (torch.float16, torch.bfloat16):
            return False

        candidates: list[tuple[str, str]] = [
            ("flash_attn_3", "flash_attn_3.flash_attn_interface"),
            ("flash_attn", "flash_attn.flash_attn_interface"),
        ]

        for engine_name, module_name in candidates:
            try:
                module = __import__(module_name, fromlist=["flash_attn_func"])
                flash_attn_func = getattr(module, "flash_attn_func")
                with torch.inference_mode():
                    _ = self.model.forward_external_flash(
                        self.input[:1],
                        flash_attn_func,
                        is_causal=self.use_causal,
                    )
                self._external_flash_func = flash_attn_func
                self._selected_engine_name = engine_name
                self._selected_backend_name = engine_name
                self._attention_runner = lambda x: self.model.forward_external_flash_prepared(
                    x,
                    flash_attn_func,
                    is_causal=self.use_causal,
                )
                return True
            except Exception:
                continue
        return False

    def _resolve_attention_runner(self) -> None:
        if self.model is None or self.input is None:
            raise RuntimeError("setup() must initialize model and input before selecting the attention engine")
        if self._resolve_external_flash():
            return
        if torch.cuda.is_available():
            major, _minor = torch.cuda.get_device_capability()
            if major >= 10:
                if not _NEW_SDPA_API or SDPBackend is None:
                    raise RuntimeError(
                        "FAIL FAST: FlashAttention required for ch10 but no FlashAttention backend available on SM100. Install flash-attn>=2.7 or flash-attn-3."
                    )
                try:
                    self._try_sdpa_backend([SDPBackend.FLASH_ATTENTION])
                except Exception as exc:
                    raise RuntimeError(
                        "FAIL FAST: FlashAttention required for ch10 but no FlashAttention backend available on SM100. Install flash-attn>=2.7 or flash-attn-3."
                    ) from exc
                self._selected_engine_name = "sdpa"
                self._attention_runner = lambda x: self.model.forward_prepared(x, is_causal=self.use_causal)
                return
        self._resolve_sdpa_backends()
        self._selected_engine_name = "sdpa"
        self._attention_runner = lambda x: self.model.forward_prepared(x, is_causal=self.use_causal)

    def _run_attention(self, x: torch.Tensor) -> torch.Tensor:
        if self._attention_runner is None:
            raise RuntimeError("setup() must resolve the attention engine before benchmarking")
        if self._selected_engine_name == "sdpa":
            with sdpa_backend_context(self._sdpa_backends):
                return self._attention_runner(x)
        return self._attention_runner(x)
    
    def setup(self) -> None:
        """Setup: Initialize tiled attention model."""
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        
        # Use FP16 for tensor core acceleration
        self.model = TiledAttentionModule(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            dropout=0.0,
        ).to(self.device).half().eval()
        self.model.cache_weight_views()
        
        # Input tensor in FP16
        self.input = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=torch.float16
        )
        self._verify_output_buffer = torch.empty_like(self.input, dtype=torch.float32)
        self.model.prepare_projection_buffers(self.input)
        self._resolve_attention_runner()
        
        # Warmup the selected tiled attention engine.
        with torch.inference_mode():
            for _ in range(3):
                _ = self._run_attention(self.input)
        
        torch.cuda.synchronize(self.device)
    
    def benchmark_fn(self) -> None:
        """Benchmark: Tiled attention computation."""
        with self._nvtx_range("optimized_tiled_attention"):
            with torch.inference_mode():
                self.output = self._run_attention(self.input)
        
        if self.output is None or self.input is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_output_buffer is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._verify_output_buffer.copy_(self.output)
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self._verify_output_buffer,
            batch_size=self.batch_size,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=FLASH_ATTENTION_OUTPUT_TOLERANCE,
        )
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self.output = None
        self._verify_output_buffer = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return workload metrics derived from the real attention shape."""
        metrics = compute_attention_workload_metrics(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            num_heads=self.num_heads,
            is_causal=self.use_causal,
        )
        metrics.update(
            compute_attention_backend_metrics(
                engine=self._selected_engine_name,
                selected_backend=self._selected_backend_name,
            )
        )
        metrics.update(
            {
                "attention.backend_flash": 1.0
                if self._selected_backend_name in {"flash_attention", "flash_attn", "flash_attn_3"}
                else 0.0,
                "attention.backend_efficient": 1.0
                if self._selected_backend_name == "efficient_attention"
                else 0.0,
                "attention.backend_math": 1.0 if self._selected_backend_name == "math" else 0.0,
            }
        )
        return metrics

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        
        # Verify tiled attention produces valid output
        with torch.inference_mode():
            output = self._run_attention(self.input)
            if torch.isnan(output).any():
                return "NaN values in attention output"
        
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedFlashAttentionBenchmark()
