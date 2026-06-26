#!/usr/bin/env python3

"""
Dynamic Precision Switching for LLM Inference (Chapter 19)

Implements adaptive precision switching based on model confidence and memory pressure.
Automatically switches between FP16/BF16, FP8, and FP4 at runtime to maximize
throughput while maintaining quality.

Key features:
- Entropy-based confidence measurement
- Hysteretic switching to avoid flapping
- EMA smoothing for stability
- Memory-pressure-aware quantization
- Per-token and per-layer precision control

Usage:
    from dynamic_precision_switching import decode_with_dynamic_precision
    
    output = decode_with_dynamic_precision(
        model=model,
        tokens=input_ids,
        max_steps=128,
        enable_fp8=True
    )
"""

import contextlib
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
from enum import Enum


class PrecisionMode(Enum):
    """Available precision modes"""
    FP16 = "fp16"
    BF16 = "bf16"
    FP8 = "fp8"
    FP4 = "fp4"


@dataclass
class PrecisionStats:
    """Statistics for precision switching"""
    total_tokens: int = 0
    fp16_tokens: int = 0
    fp8_tokens: int = 0
    fp4_tokens: int = 0
    precision_switches: int = 0
    avg_confidence: float = 0.0
    
    @property
    def fp8_ratio(self) -> float:
        """Percentage of tokens generated in FP8"""
        return (self.fp8_tokens / self.total_tokens * 100) if self.total_tokens > 0 else 0.0
    
    def record_tokens(self, mode: PrecisionMode, batch_size: int):
        """Record token counts by precision mode."""
        self.total_tokens += batch_size
        if mode == PrecisionMode.FP4:
            self.fp4_tokens += batch_size
        elif mode == PrecisionMode.FP8:
            self.fp8_tokens += batch_size
        else:
            self.fp16_tokens += batch_size
    
    def print_summary(self):
        """Print statistics summary"""
        print("\n" + "="*60)
        print("Dynamic Precision Statistics")
        print("="*60)
        safe_total = max(self.total_tokens, 1)
        print(f"Total tokens:       {self.total_tokens}")
        print(f"FP16/BF16 tokens:   {self.fp16_tokens} ({self.fp16_tokens/safe_total*100:.1f}%)")
        print(f"FP8 tokens:         {self.fp8_tokens} ({self.fp8_tokens/safe_total*100:.1f}%)")
        print(f"FP4 tokens:         {self.fp4_tokens} ({self.fp4_tokens/safe_total*100:.1f}%)")
        print(f"Precision switches: {self.precision_switches}")
        print(f"Avg confidence:     {self.avg_confidence:.3f}")
        print("="*60 + "\n")


# Safe Transformer Engine (TE) FP8 autocast import
try:
    from transformer_engine.pytorch import fp8_autocast as _te_fp8_autocast
    _TE_AVAILABLE = True
except Exception:
    _TE_AVAILABLE = False
    print("Info: Transformer Engine not available. FP8 will use standard autocast.")
    
    # No-op stand-in so the code runs without TE installed
    class _NullCtx(contextlib.ContextDecorator):
        def __init__(self, **_):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False
    
    def _te_fp8_autocast(**_):
        return _NullCtx()


def _precision_context(
    device: torch.device,
    mode: PrecisionMode,
    prefer_bfloat16: bool,
    enable_fp8: bool
):
    """
    Get precision context for the specified device.
    
    Args:
        device: Target device
        mode: Desired precision mode
        prefer_bfloat16: Prefer BF16 over FP16
        enable_fp8: Allow FP8 if TE present
        
    Returns:
        Context manager for precision
    """
    if device.type != "cuda":
        return contextlib.nullcontext()

    if mode == PrecisionMode.FP8 and enable_fp8 and _TE_AVAILABLE:
        # Note: fp8_autocast affects only TE-enabled modules. Non-TE modules run at native dtypes.
        return _te_fp8_autocast(enabled=True)

    amp_dtype = torch.bfloat16 if prefer_bfloat16 else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)


def _simulate_fp4_quantize(tensor: torch.Tensor) -> torch.Tensor:
    """
    Approximate FP4 quantization by clamping to 4-bit range and de-quantizing
    back to the original dtype. This preserves tensor shape while emulating
    precision loss.
    """
    if not tensor.is_floating_point():
        return tensor

    # Compute per-row scale along last dimension to preserve structure
    abs_max = tensor.detach().abs()
    if tensor.dim() > 1:
        abs_max = abs_max.amax(dim=-1, keepdim=True)
    max_val = abs_max.clamp(min=1e-6)
    scale = max_val / 7.0  # 4-bit signed => [-8, 7]
    quantized = torch.clamp((tensor / scale).round(), min=-8, max=7)
    return (quantized * scale).to(tensor.dtype)


def _memory_utilization_percent(device: torch.device) -> float:
    """Return GPU memory utilization percentage for the given device."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0.0
    try:
        index = device.index if device.index is not None else torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(index)
        used = total_bytes - free_bytes
        return (used / total_bytes) * 100.0 if total_bytes else 0.0
    except Exception:
        return 0.0


@torch.no_grad()
def decode_with_dynamic_precision(
    model,
    tokens: torch.Tensor,
    max_steps: int,
    *,
    device: torch.device = torch.device("cuda"),
    prefer_bfloat16: bool = True,  # B200: prefer BF16 over FP16 for AMP
    enable_fp8: bool = True,  # Allow FP8 when TE present
    enable_fp4: bool = True,  # Allow simulated FP4 mode under high confidence/pressure
    enter_fp8_threshold: float = 6.0,  # hysteresis upper bound (logit margin average)
    exit_fp8_threshold: float = 3.0,  # hysteresis lower bound (avoid flapping)
    enter_fp4_threshold: float = 8.0,  # FP4 requires even higher confidence
    exit_fp4_threshold: float = 5.5,
    fp4_memory_enter: float = 90.0,  # trigger FP4 when memory pressure exceeds this percent
    fp4_memory_exit: float = 85.0,
    reeval_interval: int = 8,  # compute/inspect confidence every N steps to avoid per-step sync
    topk_dim: int = -1,  # last dimension holds vocabulary logits
    eos_id: Optional[int] = None,
    collect_stats: bool = True
) -> Tuple[torch.Tensor, Optional[PrecisionStats]]:
    """
    Autoregressive decode loop that smoothly switches between AMP (BF16/FP16) and
    FP8 (TE) without per-step host sync. Works even when TE is not installed;
    in that case, runs AMP only.
    
    Implements the dynamic precision approach from Chapter 19:
    - Confidence signal: mean(top1 - top2) logits margin across the batch
    - Smoothing: EMA + interval re-evaluation to minimize CPU-GPU sync pressure
    - Hysteresis: separate enter/exit thresholds to avoid precision flapping
    - Additional FP4 tier triggered when confidence is very high and memory pressure is elevated
    
    Args:
        model: The model to use for generation
        tokens: Input token IDs [batch_size, seq_len]
        max_steps: Maximum number of tokens to generate
        device: Device to run on
        prefer_bfloat16: Use BF16 instead of FP16 for AMP
        enable_fp8: Allow FP8 if Transformer Engine available
        enable_fp4: Enable FP4 simulation under high confidence + memory pressure
        enter_fp8_threshold: Confidence threshold to enter FP8
        exit_fp8_threshold: Confidence threshold to exit FP8
        enter_fp4_threshold: Confidence threshold to enter FP4
        exit_fp4_threshold: Confidence threshold to exit FP4
        fp4_memory_enter: Memory utilization threshold (%) to enter FP4
        fp4_memory_exit: Memory utilization threshold (%) to exit FP4
        reeval_interval: How often to reevaluate precision (steps)
        topk_dim: Dimension for vocabulary logits
        eos_id: End-of-sequence token ID
        collect_stats: Whether to collect statistics
        
    Returns:
        Tuple of (generated_tokens, statistics)
    """
    assert exit_fp8_threshold <= enter_fp8_threshold, \
        "Hysteresis requires exit <= enter threshold"
    if enable_fp4:
        assert exit_fp4_threshold <= enter_fp4_threshold, \
            "FP4 hysteresis requires exit <= enter threshold"
    
    model.eval()
    prompt = tokens.to(device, non_blocking=True)
    batch_size, prompt_len = prompt.shape
    generated = torch.empty(
        (batch_size, prompt_len + max_steps),
        device=device,
        dtype=prompt.dtype,
    )
    generated[:, :prompt_len].copy_(prompt)
    current_len = prompt_len
    
    # Internal state
    default_mode = PrecisionMode.BF16 if prefer_bfloat16 else PrecisionMode.FP16
    precision_mode: PrecisionMode = default_mode
    ema_conf: Optional[torch.Tensor] = None  # stays on device; host consults only at intervals
    alpha = 0.2  # EMA smoothing factor for confidence
    confidence_samples = 0
    
    # Statistics
    stats = PrecisionStats() if collect_stats else None
    
    # A tiny helper to update on-device EMA without host sync
    def _update_confidence_ema(logits: torch.Tensor) -> torch.Tensor:
        nonlocal ema_conf
        
        # logits: [B, vocab] or [B, T, vocab]. Use the last time-step if 3D.
        last = logits if logits.dim() == 2 else logits[:, -1, :]
        
        # Compute top-2 margin on-device
        top2 = torch.topk(last, k=2, dim=topk_dim).values  # [B, 2]
        margin = (top2[:, 0] - top2[:, 1]).mean()  # scalar tensor on device
        
        ema_conf = (1 - alpha) * (ema_conf if ema_conf is not None else margin) + alpha * margin
        return ema_conf  # device scalar
    
    # Decode
    for step in range(max_steps):
        active_tokens = generated[:, :current_len]

        # 1) Precision context (exactly one).
        # No nested contexts, no leakage across iterations.
        with _precision_context(device, precision_mode, prefer_bfloat16, enable_fp8):
            # Forward pass (HF-style or plain)
            try:
                logits = model(input_ids=active_tokens)
                if hasattr(logits, "logits"):
                    logits = logits.logits
            except TypeError:
                logits = model(active_tokens)

        if precision_mode == PrecisionMode.FP4:
            logits = _simulate_fp4_quantize(logits)
        
        # 2) Pick next token from the *last* position
        last_step_logits = logits if logits.dim() == 2 else logits[:, -1, :]
        next_token = torch.argmax(last_step_logits, dim=-1, keepdim=True)  # [B, 1]
        generated[:, current_len : current_len + 1].copy_(next_token)
        current_len += 1
        
        # 3) Update on-device EMA signal every step (no host sync yet)
        conf_dev = _update_confidence_ema(logits)
        
        # 4) Update statistics
        if stats:
            stats.record_tokens(precision_mode, batch_size)
        
        # 5) Periodically re-evaluate precision choice on host to avoid per-step sync
        if (step + 1) % reeval_interval == 0:
            conf_value = float(conf_dev)  # exactly one tiny sync every N steps
            confidence_samples += 1
            mem_util = _memory_utilization_percent(device)
            
            desired_mode = precision_mode
            
            if precision_mode == PrecisionMode.FP4:
                should_exit_fp4 = (
                    conf_value < exit_fp4_threshold or
                    mem_util < fp4_memory_exit
                )
                if should_exit_fp4:
                    if enable_fp8 and conf_value >= enter_fp8_threshold:
                        desired_mode = PrecisionMode.FP8
                    else:
                        desired_mode = default_mode
            else:
                can_enter_fp4 = (
                    enable_fp4 and
                    conf_value >= enter_fp4_threshold and
                    mem_util >= fp4_memory_enter
                )
                if can_enter_fp4:
                    desired_mode = PrecisionMode.FP4
                elif precision_mode == PrecisionMode.FP8:
                    if conf_value < exit_fp8_threshold:
                        desired_mode = default_mode
                else:
                    if enable_fp8 and conf_value >= enter_fp8_threshold:
                        desired_mode = PrecisionMode.FP8
            
            if stats:
                stats.avg_confidence = (
                    (stats.avg_confidence * (confidence_samples - 1)) + conf_value
                ) / max(confidence_samples, 1)

            if desired_mode == PrecisionMode.FP8:
                if not enable_fp8 or device.type != "cuda" or not _TE_AVAILABLE:
                    desired_mode = default_mode
            
            if desired_mode != precision_mode:
                precision_mode = desired_mode
                if stats:
                    stats.precision_switches += 1
            
        # 6) EOS handling
        if eos_id is not None:
            if (generated[:, current_len - 1] == eos_id).all():
                break
    
    return generated[:, :current_len].contiguous(), stats


class DynamicPrecisionModel(nn.Module):
    """
    Wrapper that applies dynamic precision to a model with per-layer control.
    
    This allows different layers to use different precisions based on their
    sensitivity and role in the model.
    """
    
    def __init__(
        self,
        model: nn.Module,
        layer_precision_map: Optional[Dict[str, PrecisionMode]] = None
    ):
        """
        Initialize dynamic precision model wrapper.
        
        Args:
            model: Base model to wrap
            layer_precision_map: Optional mapping of layer names to precision modes
        """
        super().__init__()
        self.model = model
        self.layer_precision_map = layer_precision_map or {}
        
    def forward(self, *args, **kwargs):
        """Forward pass with dynamic precision per layer"""
        # For simplicity, this example doesn't implement per-layer precision
        # In production, you'd hook into each layer and apply precision contexts
        return self.model(*args, **kwargs)


def compute_entropy(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Compute Shannon entropy of softmax distribution.
    
    Lower entropy indicates higher confidence (sharper distribution).
    Higher entropy indicates uncertainty (flatter distribution).
    
    Args:
        logits: Logit tensor
        dim: Dimension to compute entropy over
        
    Returns:
        Entropy values
    """
    probs = torch.softmax(logits, dim=dim)
    log_probs = torch.log_softmax(logits, dim=dim)
    entropy = -(probs * log_probs).sum(dim=dim)
    return entropy


def should_use_low_precision(
    logits: torch.Tensor,
    entropy_threshold: float = 2.0,
    max_prob_threshold: float = 0.8
) -> bool:
    """
    Determine if low precision (FP8/FP4) is safe based on model confidence.
    
    Args:
        logits: Model output logits [batch, vocab]
        entropy_threshold: Entropy below this = confident
        max_prob_threshold: Max probability above this = confident
        
    Returns:
        True if low precision is safe to use
    """
    probs = torch.softmax(logits, dim=-1)
    entropy, max_prob = torch.stack(
        (
            compute_entropy(logits).mean(),
            probs.max(dim=-1).values.mean(),
        )
    ).tolist()
    
    # Use low precision if confident (low entropy, high max prob)
    return entropy < entropy_threshold and max_prob > max_prob_threshold


def quantize_kv_cache_on_memory_pressure(
    kv_cache: torch.Tensor,
    memory_util_percent: float,
    threshold: float = 80.0,
    target_precision: PrecisionMode = PrecisionMode.FP8
) -> torch.Tensor:
    """
    Dynamically quantize KV cache when memory pressure is high.
    
    As described in Chapter 19: "If GPU memory usage is approaching its limit,
    the system can dynamically compress activations to a lower precision."
    
    Args:
        kv_cache: Key-value cache tensor
        memory_util_percent: Current GPU memory utilization (0-100)
        threshold: Memory threshold to trigger quantization
        target_precision: Target quantization precision
        
    Returns:
        Quantized cache if pressure is high, original otherwise
    """
    if memory_util_percent <= threshold:
        return kv_cache

    if target_precision == PrecisionMode.FP4:
        return _simulate_fp4_quantize(kv_cache)

    if target_precision == PrecisionMode.FP8:
        float8_dtype = getattr(torch, "float8_e4m3fn", None)
        if float8_dtype is not None:
            try:
                return kv_cache.to(float8_dtype)
            except (TypeError, RuntimeError):
                return _simulate_fp4_quantize(kv_cache)
        return _simulate_fp4_quantize(kv_cache)

    if target_precision == PrecisionMode.FP16:
        return kv_cache.to(torch.float16)

    return kv_cache


# Example usage and testing
if __name__ == '__main__':
    print("=" * 70)
    print("Dynamic Precision Switching Demo (Chapter 19)")
    print("=" * 70)
    
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("\nWarning: CUDA not available. This demo requires a GPU.")
        print("Exiting...")
        exit(0)
    
    device = torch.device("cuda")
    
    # Create a simple mock model for testing
    class SimpleModel(nn.Module):
        def __init__(self, vocab_size=1000):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, 512)
            self.transformer = nn.TransformerEncoderLayer(
                d_model=512,
                nhead=8,
                dim_feedforward=2048,
                batch_first=True
            )
            self.lm_head = nn.Linear(512, vocab_size)
        
        def forward(self, input_ids):
            x = self.embedding(input_ids)
            x = self.transformer(x)
            logits = self.lm_head(x)
            return logits
    
    print("\nInitializing model...")
    model = SimpleModel().to(device).eval()
    
    # Test with different confidence scenarios
    print("\n" + "=" * 70)
    print("Test 1: High confidence (should use FP8 more)")
    print("=" * 70)
    
    input_ids = torch.randint(0, 1000, (2, 10), device=device)
    output, stats = decode_with_dynamic_precision(
        model=model,
        tokens=input_ids,
        max_steps=50,
        device=device,
        enable_fp8=True,
        enter_fp8_threshold=3.0,  # Lower threshold for testing
        exit_fp8_threshold=1.0,
        reeval_interval=5
    )
    
    print(f"\nGenerated sequence shape: {output.shape}")
    if stats:
        stats.print_summary()
    
    # Test entropy computation
    print("\n" + "=" * 70)
    print("Test 2: Entropy-based confidence measurement")
    print("=" * 70)
    
    # High confidence logits (peaked distribution)
    high_conf_logits = torch.zeros(1, 1000, device=device)
    high_conf_logits[0, 42] = 10.0  # Very confident about token 42
    
    # Low confidence logits (flat distribution)
    low_conf_logits = torch.randn(1, 1000, device=device) * 0.1  # Flat
    
    high_entropy, low_entropy = torch.stack(
        (
            compute_entropy(high_conf_logits).mean(),
            compute_entropy(low_conf_logits).mean(),
        )
    ).tolist()
    
    print(f"High confidence entropy:  {high_entropy:.3f} (should be low)")
    print(f"Low confidence entropy:   {low_entropy:.3f} (should be high)")
    
    should_use_fp8_high = should_use_low_precision(high_conf_logits)
    should_use_fp8_low = should_use_low_precision(low_conf_logits)
    
    print(f"\nShould use FP8 for high confidence: {should_use_fp8_high}")
    print(f"Should use FP8 for low confidence:  {should_use_fp8_low}")
    
    # Test memory-pressure-based quantization
    print("\n" + "=" * 70)
    print("Test 3: Memory-pressure-based KV cache quantization")
    print("=" * 70)
    
    kv_cache = torch.randn(2, 8, 1024, 64, device=device, dtype=torch.float16)
    print(f"Original KV cache: {kv_cache.shape}, dtype={kv_cache.dtype}")
    
    # Simulate low memory pressure
    quantized_low = quantize_kv_cache_on_memory_pressure(kv_cache, memory_util_percent=50.0)
    print(f"Low memory pressure:  dtype={quantized_low.dtype} (should stay FP16)")
    
    # Simulate high memory pressure
    if hasattr(torch, 'float8_e4m3fn'):
        quantized_high = quantize_kv_cache_on_memory_pressure(kv_cache, memory_util_percent=90.0)
        print(f"High memory pressure: dtype={quantized_high.dtype} (should be FP8)")
    else:
        print("High memory pressure: FP8 dtype not available in this PyTorch version")
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)
    
    print("\nKey Insights from Chapter 19:")
    print("- Use lowest precision that maintains accuracy")
    print("- Switch to higher precision when confidence drops")
    print("- Hysteresis prevents precision flapping")
    print("- EMA smoothing reduces sync overhead")
    print("- Memory pressure can trigger KV cache quantization")
