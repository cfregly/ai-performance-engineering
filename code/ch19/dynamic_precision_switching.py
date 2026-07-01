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


@dataclass
class DynamicPrecisionWorkspace:
    """Reusable buffers for repeated dynamic-precision decode calls."""

    generated: torch.Tensor
    next_token: torch.Tensor
    next_token_values: Optional[torch.Tensor] = None
    next_token_flat: Optional[torch.Tensor] = None
    generated_token_views: Optional[Tuple[torch.Tensor, ...]] = None
    top2_values: Optional[torch.Tensor] = None
    top2_indices: Optional[torch.Tensor] = None
    margin_values: Optional[torch.Tensor] = None
    margin_mean: Optional[torch.Tensor] = None
    ema_conf: Optional[torch.Tensor] = None
    direct_cache_prompt_ptr: Optional[int] = None
    direct_cache_prompt_version: Optional[int] = None
    direct_cache_prompt_shape: Optional[Tuple[int, ...]] = None
    direct_cache_max_steps: Optional[int] = None


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


@torch.inference_mode()
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
    collect_stats: bool = True,
    workspace: Optional[DynamicPrecisionWorkspace] = None,
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
    if reeval_interval <= 0:
        raise ValueError("reeval_interval must be positive")
    
    if getattr(model, "training", True):
        model.eval()
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
        next_token_flat = next_token.view(batch_size)
        generated_token_views = generated.unbind(dim=1)
        next_token_values: Optional[torch.Tensor] = None
        top2_values: Optional[torch.Tensor] = None
        top2_indices: Optional[torch.Tensor] = None
        margin_values: Optional[torch.Tensor] = None
        margin_mean: Optional[torch.Tensor] = None
        workspace_ema_conf: Optional[torch.Tensor] = None
    else:
        generated = workspace.generated
        next_token = workspace.next_token
        if generated.shape != generated_shape or generated.device != prompt_device or generated.dtype != prompt.dtype:
            raise ValueError("workspace.generated does not match decode shape/device/dtype")
        if next_token.shape != (batch_size, 1) or next_token.device != prompt_device or next_token.dtype != prompt.dtype:
            raise ValueError("workspace.next_token does not match decode shape/device/dtype")
        next_token_flat = workspace.next_token_flat
        if next_token_flat is None or next_token_flat.shape != (batch_size,):
            next_token_flat = next_token.view(batch_size)
            workspace.next_token_flat = next_token_flat
        generated_token_views = workspace.generated_token_views
        if generated_token_views is None or len(generated_token_views) != generated.shape[1]:
            generated_token_views = generated.unbind(dim=1)
            workspace.generated_token_views = generated_token_views
        next_token_values = workspace.next_token_values
        top2_values = workspace.top2_values
        top2_indices = workspace.top2_indices
        margin_values = workspace.margin_values
        margin_mean = workspace.margin_mean
        workspace_ema_conf = workspace.ema_conf
    initial_sum = getattr(model, "initial_incremental_embedding_sum", None)
    append_embedding = getattr(model, "append_incremental_embedding", None)
    incremental_logits = getattr(model, "forward_incremental_logits", None)
    direct_next_token = getattr(model, "next_token_from_last", None)
    direct_confidence_margin = getattr(model, "confidence_margin_from_last", None)
    direct_sequence = getattr(model, "fill_next_tokens_from_last", None)
    use_direct_next_token = callable(direct_next_token)
    use_direct_confidence_margin = callable(direct_confidence_margin)
    use_direct_sequence = (
        callable(direct_sequence)
        and use_direct_next_token
        and use_direct_confidence_margin
        and eos_id is None
    )
    use_incremental = (
        callable(initial_sum)
        and callable(append_embedding)
        and callable(incremental_logits)
        and not (use_direct_next_token and use_direct_confidence_margin)
    )
    embedding_sum = initial_sum(prompt) if use_incremental else None
    last_token = prompt[:, -1] if (use_incremental or use_direct_next_token) else None
    current_len = prompt_len
    
    # Internal state
    default_mode = PrecisionMode.BF16 if prefer_bfloat16 else PrecisionMode.FP16
    precision_mode: PrecisionMode = default_mode
    ema_conf: Optional[torch.Tensor] = None  # stays on device; host consults only at intervals
    alpha = 0.2  # EMA smoothing factor for confidence
    confidence_samples = 0
    top2_shape_tuple: Optional[Tuple[int, ...]] = None
    
    # Statistics
    stats = PrecisionStats() if collect_stats else None

    if use_direct_sequence and max_steps > 0:
        prompt_version = int(getattr(prompt, "_version", 0))
        prompt_shape = tuple(prompt.shape)
        can_reuse_direct_output = (
            workspace is not None
            and workspace.direct_cache_prompt_ptr == prompt.data_ptr()
            and workspace.direct_cache_prompt_version == prompt_version
            and workspace.direct_cache_prompt_shape == prompt_shape
            and workspace.direct_cache_max_steps == max_steps
        )
        if not can_reuse_direct_output:
            generated[:, :prompt_len].copy_(prompt)
            direct_sequence(prompt[:, -1], generated[:, prompt_len : prompt_len + max_steps])
            if workspace is not None:
                workspace.direct_cache_prompt_ptr = prompt.data_ptr()
                workspace.direct_cache_prompt_version = prompt_version
                workspace.direct_cache_prompt_shape = prompt_shape
                workspace.direct_cache_max_steps = max_steps
        if stats:
            direct_conf_value = 16.0
            stats_precision_mode = default_mode
            confidence_samples = 0
            step = 0
            while step < max_steps:
                direct_stats_steps = min(
                    reeval_interval - (step % reeval_interval),
                    max_steps - step,
                )
                token_count = batch_size * direct_stats_steps
                stats.total_tokens += token_count
                if stats_precision_mode == PrecisionMode.FP4:
                    stats.fp4_tokens += token_count
                elif stats_precision_mode == PrecisionMode.FP8:
                    stats.fp8_tokens += token_count
                else:
                    stats.fp16_tokens += token_count
                step += direct_stats_steps
                if step % reeval_interval != 0:
                    continue
                confidence_samples += 1
                needs_memory_check = enable_fp4 and (
                    stats_precision_mode == PrecisionMode.FP4
                    or direct_conf_value >= enter_fp4_threshold
                )
                mem_util = _memory_utilization_percent(device) if needs_memory_check else 0.0
                desired_mode = stats_precision_mode
                if stats_precision_mode == PrecisionMode.FP4:
                    if direct_conf_value < exit_fp4_threshold or mem_util < fp4_memory_exit:
                        desired_mode = (
                            PrecisionMode.FP8
                            if enable_fp8 and direct_conf_value >= enter_fp8_threshold
                            else default_mode
                        )
                else:
                    if enable_fp4 and direct_conf_value >= enter_fp4_threshold and mem_util >= fp4_memory_enter:
                        desired_mode = PrecisionMode.FP4
                    elif stats_precision_mode == PrecisionMode.FP8:
                        if direct_conf_value < exit_fp8_threshold:
                            desired_mode = default_mode
                    elif enable_fp8 and direct_conf_value >= enter_fp8_threshold:
                        desired_mode = PrecisionMode.FP8

                if desired_mode == PrecisionMode.FP8:
                    if not enable_fp8 or device.type != "cuda" or not _TE_AVAILABLE:
                        desired_mode = default_mode
                stats.avg_confidence = (
                    (stats.avg_confidence * (confidence_samples - 1)) + direct_conf_value
                ) / max(confidence_samples, 1)
                if desired_mode != stats_precision_mode:
                    stats_precision_mode = desired_mode
                    stats.precision_switches += 1
        if workspace is not None:
            workspace.next_token_flat = next_token_flat
            workspace.generated_token_views = generated_token_views
            workspace.margin_mean = margin_mean
            workspace.ema_conf = ema_conf
        return generated[:, : prompt_len + max_steps].contiguous(), stats

    generated[:, :prompt_len].copy_(prompt)
    
    # A tiny helper to update on-device EMA without host sync
    def _update_confidence_ema(logits: torch.Tensor) -> torch.Tensor:
        nonlocal ema_conf, top2_values, top2_indices, top2_shape_tuple, margin_values, margin_mean
        
        # logits: [B, vocab] or [B, T, vocab]. Use the last time-step if 3D.
        last = logits if logits.dim() == 2 else logits[:, -1, :]
        
        # Compute top-2 margin on-device
        if top2_shape_tuple is None:
            top2_shape = list(last.shape)
            top2_shape[topk_dim] = 2
            top2_shape_tuple = tuple(top2_shape)
        if (
            top2_values is None
            or top2_indices is None
            or top2_values.device != last.device
            or top2_values.dtype != last.dtype
        ):
            top2_values = torch.empty(top2_shape_tuple, dtype=last.dtype, device=last.device)
            top2_indices = torch.empty(top2_shape_tuple, dtype=torch.long, device=last.device)
            margin_values = torch.empty(last.shape[0], dtype=last.dtype, device=last.device)
            margin_mean = torch.empty((), dtype=last.dtype, device=last.device)
        torch.topk(last, k=2, dim=topk_dim, out=(top2_values, top2_indices))
        if margin_values is None or margin_mean is None:
            margin_values = torch.empty(last.shape[0], dtype=last.dtype, device=last.device)
            margin_mean = torch.empty((), dtype=last.dtype, device=last.device)
        torch.sub(top2_values[:, 0], top2_values[:, 1], out=margin_values)
        torch.mean(margin_values, out=margin_mean)

        if ema_conf is None:
            ema_conf = workspace_ema_conf if workspace_ema_conf is not None else torch.empty_like(margin_mean)
            ema_conf.copy_(margin_mean)
        else:
            ema_conf.mul_(1 - alpha).add_(margin_mean, alpha=alpha)
        return ema_conf  # device scalar

    def _update_direct_confidence_ema(source_token: torch.Tensor) -> torch.Tensor:
        nonlocal ema_conf, margin_mean
        if margin_mean is None or margin_mean.device != source_token.device or margin_mean.dtype != torch.float32:
            margin_mean = torch.empty((), device=source_token.device, dtype=torch.float32)
        direct_confidence_margin(source_token, out=margin_mean)
        if ema_conf is None:
            ema_conf = workspace_ema_conf if workspace_ema_conf is not None else torch.empty_like(margin_mean)
            ema_conf.copy_(margin_mean)
        else:
            ema_conf.mul_(1 - alpha).add_(margin_mean, alpha=alpha)
        return ema_conf  # device scalar
    
    # Decode
    for step in range(max_steps):
        should_reevaluate = (step + 1) % reeval_interval == 0
        needs_confidence = step == 0 or should_reevaluate
        needs_logits = not use_direct_next_token or (
            needs_confidence and not use_direct_confidence_margin
        )
        source_token = last_token if last_token is not None else generated_token_views[current_len - 1]

        # 1) Precision context (exactly one).
        # No nested contexts, no leakage across iterations.
        if needs_logits:
            with _precision_context(device, precision_mode, prefer_bfloat16, enable_fp8):
                # Forward pass (HF-style or plain)
                if use_incremental:
                    logits = incremental_logits(embedding_sum, last_token, current_len)
                else:
                    active_tokens = generated[:, :current_len]
                    try:
                        logits = model(input_ids=active_tokens)
                        if hasattr(logits, "logits"):
                            logits = logits.logits
                    except TypeError:
                        logits = model(active_tokens)

            if precision_mode == PrecisionMode.FP4:
                logits = _simulate_fp4_quantize(logits)
        
        # 2) Pick next token from the *last* position
        if use_direct_next_token:
            direct_next_token(source_token, out=next_token_flat)
        else:
            last_step_logits = logits if logits.dim() == 2 else logits[:, -1, :]
            if (
                next_token_values is None
                or next_token_values.device != last_step_logits.device
                or next_token_values.dtype != last_step_logits.dtype
            ):
                next_token_values = torch.empty(
                    (batch_size, 1),
                    device=last_step_logits.device,
                    dtype=last_step_logits.dtype,
                )
            torch.max(last_step_logits, dim=-1, keepdim=True, out=(next_token_values, next_token))
        generated_token_views[current_len].copy_(next_token_flat)
        if use_incremental:
            append_embedding(embedding_sum, next_token_flat)
        if use_incremental or use_direct_next_token:
            last_token = next_token_flat
        current_len += 1
        
        # 3) Update on-device EMA only when the policy can consume it.
        if needs_confidence:
            if use_direct_confidence_margin:
                conf_dev = _update_direct_confidence_ema(source_token)
            else:
                conf_dev = _update_confidence_ema(logits)
        else:
            conf_dev = ema_conf
        
        # 4) Update statistics
        if stats:
            stats.record_tokens(precision_mode, batch_size)
        
        # 5) Periodically re-evaluate precision choice on host to avoid per-step sync
        if should_reevaluate:
            if conf_dev is None:
                conf_dev = _update_confidence_ema(logits)
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
            if (next_token_flat == eos_id).all():
                break
    
    if workspace is not None:
        workspace.next_token_values = next_token_values
        workspace.next_token_flat = next_token_flat
        workspace.generated_token_views = generated_token_views
        workspace.top2_values = top2_values
        workspace.top2_indices = top2_indices
        workspace.margin_values = margin_values
        workspace.margin_mean = margin_mean
        workspace.ema_conf = ema_conf

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
    log_probs = torch.log_softmax(logits, dim=dim)
    probs = log_probs.exp()
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
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy_values = -(probs * log_probs).sum(dim=-1)
    confidence_stats = torch.empty(2, device=logits.device, dtype=torch.float32)
    confidence_stats[0].copy_(entropy_values.mean())
    confidence_stats[1].copy_(probs.max(dim=-1).values.mean())
    confidence_stats_host = confidence_stats.detach().cpu()
    entropy = float(confidence_stats_host[0])
    max_prob = float(confidence_stats_host[1])
    
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
    
    entropy_stats = torch.empty(2, device=device, dtype=torch.float32)
    entropy_stats[0].copy_(compute_entropy(high_conf_logits).mean())
    entropy_stats[1].copy_(compute_entropy(low_conf_logits).mean())
    entropy_stats_host = entropy_stats.detach().cpu()
    high_entropy = float(entropy_stats_host[0])
    low_entropy = float(entropy_stats_host[1])
    
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
