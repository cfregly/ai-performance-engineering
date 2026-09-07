#!/usr/bin/env python3
"""Real-world case study: Llama 3.1 8B optimization for Blackwell.

Demonstrates end-to-end optimization of Llama 3.1 8B:
- torch.compile with Blackwell-friendly settings
- Preferred SDPA backends (TE/Flash) via sdpa_kernel
- Materialized attention retained as an explicit numerical diagnostic
"""

from __future__ import annotations

from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.harness.arch_config import prefer_sdpa_backends
from core.utils.compile_utils import compile_model
from core.utils.logger import get_logger

logger = get_logger(__name__)

# BF16 eager, SDPA, and compiled paths compare the same model and inputs. Keep
# the pairwise gate tight enough that a zero or unrelated output cannot pass.
# Target-GPU calibration may tighten this bound; it must never be relaxed without
# fresh full-output evidence.
LLAMA_BF16_OUTPUT_TOLERANCE = (0.02, 0.02)

# Llama 3.1 uses a fixed RMSNorm epsilon. Leaving ``eps`` unset makes
# ``nn.RMSNorm`` choose the input dtype's machine epsilon (0.0078125 for
# BF16), which is not the model's normalization contract and can take a
# different arithmetic path when the full stack is compiled.
LLAMA_RMS_NORM_EPS = 1e-5

# Inductor normally elides BF16 downcast/upcast pairs between fused pointwise
# operations. Preserve the eager rounding boundaries so compiling the full
# decoder stack does not silently change the SiLU/gate and residual numerics.
LLAMA_EMULATE_EAGER_PRECISION_CASTS = True

LlamaAttentionMode = Literal["preferred_sdpa", "sdpa", "manual"]
LLAMA_ATTENTION_MODES = ("preferred_sdpa", "sdpa", "manual")


try:
    import triton
    import triton.language as tl
    from torch.library import triton_op, wrap_triton
except ImportError as exc:  # CPU correctness runs use the explicit PyTorch implementation.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]
    triton_op = None  # type: ignore[assignment]
    wrap_triton = None  # type: ignore[assignment]
    _TRITON_IMPORT_ERROR: ImportError | None = exc
else:
    _TRITON_IMPORT_ERROR = None


if triton is not None and triton_op is not None and wrap_triton is not None:

    @triton.jit
    def _llama_rms_norm_kernel(
        x_ptr,
        weight_ptr,
        output_ptr,
        n_cols: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Normalize one contiguous row per Triton program in FP32."""
        row = tl.program_id(axis=0)
        offsets = tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        row_offsets = row * n_cols + offsets
        values = tl.load(x_ptr + row_offsets, mask=mask, other=0.0).to(tl.float32)
        weight = tl.load(weight_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        mean_square = tl.sum(values * values, axis=0) / n_cols
        normalized = values * tl.rsqrt(mean_square + eps)
        tl.store(output_ptr + row_offsets, normalized * weight, mask=mask)

    @triton_op("ai_perf::llama_stable_rms_norm", mutates_args={})
    def _triton_llama_rms_norm(
        x: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        contiguous_x = x.contiguous()
        contiguous_weight = weight.contiguous()
        n_cols = contiguous_x.shape[-1]
        output = torch.empty_like(contiguous_x, dtype=torch.float32)
        rows = contiguous_x.numel() // n_cols
        block_size = triton.next_power_of_2(n_cols)
        num_warps = 8 if block_size >= 2048 else 4
        wrap_triton(_llama_rms_norm_kernel)[(rows,)](
            contiguous_x,
            contiguous_weight,
            output,
            n_cols,
            LLAMA_RMS_NORM_EPS,
            BLOCK_SIZE=block_size,
            num_warps=num_warps,
        )
        return output


else:
    _triton_llama_rms_norm = None


def _stable_llama_rms_norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Apply the shared FP32 RMSNorm contract on the active device."""
    if x.ndim == 0 or weight.ndim != 1 or x.shape[-1] != weight.numel():
        raise ValueError("RMSNorm requires a rank-one weight matching the input's final dimension")
    values = x.float()
    if values.is_cuda:
        if _triton_llama_rms_norm is None:
            detail = f": {_TRITON_IMPORT_ERROR}" if _TRITON_IMPORT_ERROR else ""
            raise RuntimeError(f"SKIPPED: CUDA Llama RMSNorm requires Triton{detail}")
        return _triton_llama_rms_norm(values, weight)
    mean_square = values.square().mean(dim=-1, keepdim=True)
    return values * torch.rsqrt(mean_square + LLAMA_RMS_NORM_EPS) * weight.float()


class StableLlamaRMSNorm(nn.Module):
    """RMSNorm with FP32 accumulation shared by eager and compiled arms."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = LLAMA_RMS_NORM_EPS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _stable_llama_rms_norm(x, self.weight)


def _max_autotune_compile_options() -> dict[str, Any]:
    """Return max-autotune options with eager BF16 rounding scoped to one compile."""
    inductor = getattr(torch, "_inductor", None)
    list_mode_options = getattr(inductor, "list_mode_options", None)
    if list_mode_options is None:
        raise RuntimeError(
            "SKIPPED: this PyTorch build cannot expand the max-autotune torch.compile options."
        )

    try:
        mode_options = list_mode_options("max-autotune")
    except Exception as exc:  # pragma: no cover - depends on the PyTorch build
        raise RuntimeError(
            "SKIPPED: this PyTorch build cannot expand the max-autotune torch.compile options."
        ) from exc
    if not isinstance(mode_options, dict) or mode_options.get("max_autotune") is not True:
        raise RuntimeError(
            "SKIPPED: this PyTorch build does not expose a max-autotune "
            "torch.compile option profile."
        )

    options = dict(mode_options)
    options["emulate_precision_casts"] = LLAMA_EMULATE_EAGER_PRECISION_CASTS
    return options


class Llama31_8B_Optimization:
    """Llama 3.1 8B optimization benchmark."""

    # Model specifications
    HIDDEN_SIZE = 4096
    NUM_HEADS = 32
    NUM_LAYERS = 32
    VOCAB_SIZE = 128256
    INTERMEDIATE_SIZE = 14336

    def __init__(
        self,
        batch_size: int = 1,
        seq_length: int = 2048,
        use_compile: bool = True,
        attention_mode: LlamaAttentionMode = "preferred_sdpa",
    ):
        if attention_mode not in LLAMA_ATTENTION_MODES:
            raise ValueError(
                f"attention_mode must be one of {LLAMA_ATTENTION_MODES}, got {attention_mode!r}"
            )
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_compile = use_compile
        self.attention_mode = attention_mode

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output: torch.Tensor | None = None
        self.model: nn.Module | None = None
        self.compile_options: dict[str, Any] | None = None

        logger.info("Llama 3.1 8B Optimization")
        logger.info("  Compile: %s", use_compile)
        logger.info("  Attention: %s", attention_mode)
        logger.info("  Residual dtype: FP32")

    def _create_attention_layer(self):
        """Create attention layer (simplified for benchmark)."""
        attention_mode = self.attention_mode

        class SimplifiedAttention(nn.Module):
            def __init__(self, hidden_size: int, num_heads: int, seq_len: int):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_heads = num_heads
                self.head_dim = hidden_size // num_heads
                self._scale = 1.0 / (self.head_dim**0.5)

                self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
                self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)

                # Materialized attention is retained only as a numerical diagnostic.
                # The benchmark pair uses identical preferred-SDPA execution.
                if attention_mode == "manual":
                    pos = torch.arange(seq_len)
                    causal = pos.unsqueeze(0) > pos.unsqueeze(1)
                else:
                    causal = None
                self.register_buffer("_causal_mask", causal, persistent=False)

            def forward(self, x):
                B, T, C = x.shape
                q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
                v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

                if attention_mode == "preferred_sdpa":
                    with prefer_sdpa_backends():
                        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                elif attention_mode == "sdpa":
                    attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                else:
                    if self._causal_mask is None or self._causal_mask.shape[0] != T:
                        raise RuntimeError(
                            f"Unexpected sequence length for manual attention: T={T}"
                        )
                    scores = torch.matmul(q, k.transpose(-2, -1)) * self._scale
                    scores.masked_fill_(self._causal_mask, float("-inf"))
                    probs = torch.softmax(scores, dim=-1)
                    attn = torch.matmul(probs, v)

                attn = attn.transpose(1, 2).contiguous().view(B, T, C)
                return self.o_proj(attn)

        return SimplifiedAttention(self.HIDDEN_SIZE, self.NUM_HEADS, self.seq_length)

    def _create_mlp_layer(self):
        """Create MLP layer (SwiGLU)."""

        class SimplifiedMLP(nn.Module):
            def __init__(self, hidden_size, intermediate_size):
                super().__init__()
                self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

            def forward(self, x):
                gate = self.gate_proj(x)
                up = self.up_proj(x)
                F.silu(gate, inplace=True)
                gate.mul_(up)
                return self.down_proj(gate)

        return SimplifiedMLP(self.HIDDEN_SIZE, self.INTERMEDIATE_SIZE)

    def setup(self):
        """Initialize Llama 3.1 8B model (simplified)."""
        if self.device.type == "cuda" and _triton_llama_rms_norm is None:
            detail = f": {_TRITON_IMPORT_ERROR}" if _TRITON_IMPORT_ERROR else ""
            raise RuntimeError(f"SKIPPED: CUDA Llama RMSNorm requires Triton{detail}")

        class SimplifiedLlamaLayer(nn.Module):
            def __init__(self, attention, mlp):
                super().__init__()
                self.attention = attention
                self.mlp = mlp
                self.input_layernorm = StableLlamaRMSNorm(Llama31_8B_Optimization.HIDDEN_SIZE)
                self.post_attention_layernorm = StableLlamaRMSNorm(
                    Llama31_8B_Optimization.HIDDEN_SIZE
                )

            def forward(self, x):
                residual = x.float()
                attention_input = self.input_layernorm(residual).to(torch.bfloat16)
                hidden = residual + self.attention(attention_input).float()
                mlp_input = self.post_attention_layernorm(hidden).to(torch.bfloat16)
                return hidden + self.mlp(mlp_input).float()

        class LayerStack(nn.Module):
            def __init__(self, layers: nn.ModuleList):
                super().__init__()
                self.layers = layers

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                for layer in self.layers:
                    x = layer(x)
                return x

        # Build the declared 32-layer Llama 3.1 8B-class stack. A reduced-depth
        # proxy must use a different benchmark identity and cannot inherit the
        # 8B label or its performance expectations.
        self.layers = (
            nn.ModuleList(
                [
                    SimplifiedLlamaLayer(self._create_attention_layer(), self._create_mlp_layer())
                    for _ in range(self.NUM_LAYERS)
                ]
            )
            .to(self.device)
            .to(torch.bfloat16)
            .eval()
        )

        self.model = LayerStack(self.layers)
        self.model.eval()

        # Materialize the caller-seeded input before compilation so compile setup
        # cannot affect baseline/optimized input identity.
        self.input = torch.randn(
            self.batch_size,
            self.seq_length,
            self.HIDDEN_SIZE,
            device=self.device,
            dtype=torch.bfloat16,
        )
        self.output = None

        # Apply torch.compile if requested. (The old sm_103 fallback to "default"
        # is retired: the tcgen05.wait.st abort it dodged was caused by the
        # sm_103a de-suffix in core/benchmark/triton_compat.py, fixed 2026-06-11;
        # max-autotune re-verified clean on GB300 / Triton 3.7.)
        if self.use_compile:
            # torch.compile rejects mode and options together. Expand PyTorch's
            # max-autotune profile and add the numerical policy to this compile
            # only, avoiding process-global Inductor configuration changes.
            self.compile_options = _max_autotune_compile_options()
            self.model = compile_model(
                self.model,
                mode=None,
                options=dict(self.compile_options),
                fullgraph=False,
                dynamic=False,
            )

        logger.info(f"Model setup complete: {self.seq_length} tokens")

    def run(self) -> None:
        """Execute forward pass (timed by the harness)."""
        if self.model is None:
            raise RuntimeError("Model not initialized (call setup() first)")
        with torch.inference_mode():
            self.output = self.model(self.input)

    def cleanup(self):
        """Clean up resources."""
        self.model = None
        self.layers = None
        self.input = None
        self.output = None
        self.compile_options = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Harness adapter
    def teardown(self):
        """Alias required by harness callers."""
        self.cleanup()


def run_benchmark(
    batch_size: int = 1,
    seq_length: int = 2048,
    use_compile: bool = True,
    attention_mode: LlamaAttentionMode = "preferred_sdpa",
    profile: str = "none",
    seed: int = 42,
) -> dict[str, Any]:
    """Run Llama 3.1 8B optimization benchmark."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    benchmark = Llama31_8B_Optimization(
        batch_size=batch_size,
        seq_length=seq_length,
        use_compile=use_compile,
        attention_mode=attention_mode,
    )
    benchmark.setup()
    benchmark.run()
    benchmark.cleanup()
    return {
        "seq_length": seq_length,
        "optimizations": {
            "compile": use_compile,
            "attention_mode": attention_mode,
            "fp32_residual": True,
            "stable_rms_norm": True,
        },
    }
