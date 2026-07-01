"""Chapter 19: Token precision switching and cache quantization helpers."""
from __future__ import annotations

import contextlib
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F

from core.utils.compile_utils import enable_tf32

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

try:
    from hqq.core.quantize import Quantizer as HQQQuantizer
    HQQ_AVAILABLE = True
except Exception:
    HQQ_AVAILABLE = False

class PrecisionLevel(Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT8 = "int8"
    INT4 = "int4"


@dataclass
class ConfidenceMetrics:
    max_probability: float
    entropy: float
    logit_max_diff: float

    @property
    def confidence_score(self) -> float:
        entropy_score = max(0.0, 1.0 - self.entropy / 4.0)
        diff_score = min(1.0, self.logit_max_diff / 10.0)
        return 0.6 * self.max_probability + 0.4 * entropy_score * diff_score


class TokenPrecisionController:
    def __init__(self, model: torch.nn.Module, precision: PrecisionLevel = PrecisionLevel.FP16) -> None:
        self.model = model
        self.current_precision = precision
        self.threshold_high = 0.9
        self.threshold_low = 0.6
        self.switch_count = 0
        self.history: List[PrecisionLevel] = []
        self._next_token_buffer = None
        self._next_token_host_buffer = None
        self._token_buffer = None
        self._confidence_top2_values = None
        self._confidence_top2_indices = None
        self._confidence_metrics_buffer = None
        self._confidence_metrics_host_buffer = None

    def _next_token_buffers(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._next_token_buffer is None or self._next_token_buffer.device != device:
            self._next_token_buffer = torch.empty(1, dtype=torch.long, device=device)
            self._next_token_host_buffer = torch.empty(
                1,
                dtype=torch.long,
                device="cpu",
                pin_memory=device.type == "cuda",
            )
        return self._next_token_buffer, self._next_token_host_buffer

    def _generation_token_buffer(self, input_ids: torch.Tensor, total_len: int) -> torch.Tensor:
        batch_size = input_ids.size(0)
        if (
            self._token_buffer is None
            or self._token_buffer.device != input_ids.device
            or self._token_buffer.dtype != input_ids.dtype
            or self._token_buffer.size(0) < batch_size
            or self._token_buffer.size(1) < total_len
        ):
            self._token_buffer = torch.empty(
                (batch_size, total_len),
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
        return self._token_buffer[:batch_size, :total_len]

    def _confidence(self, logits: torch.Tensor, temperature: float = 1.0) -> ConfidenceMetrics:
        scaled = logits / temperature
        log_probs = F.log_softmax(scaled, dim=-1)
        probs = log_probs.exp()
        if (
            self._confidence_top2_values is None
            or self._confidence_top2_values.device != scaled.device
            or self._confidence_top2_values.dtype != scaled.dtype
        ):
            self._confidence_top2_values = torch.empty(2, device=scaled.device, dtype=scaled.dtype)
            self._confidence_top2_indices = torch.empty(2, device=scaled.device, dtype=torch.long)
        assert self._confidence_top2_indices is not None
        torch.topk(scaled, k=2, out=(self._confidence_top2_values, self._confidence_top2_indices))

        if self._confidence_metrics_buffer is None or self._confidence_metrics_buffer.device != scaled.device:
            self._confidence_metrics_buffer = torch.empty(3, device=scaled.device, dtype=torch.float32)
        metrics = self._confidence_metrics_buffer
        metrics[0].copy_(probs.max())
        metrics[1].copy_((-(probs * log_probs).sum()))
        metrics[2].copy_(self._confidence_top2_values[0] - self._confidence_top2_values[1])

        if metrics.device.type == "cuda":
            if (
                self._confidence_metrics_host_buffer is None
                or not self._confidence_metrics_host_buffer.is_pinned()
            ):
                self._confidence_metrics_host_buffer = torch.empty(
                    3,
                    dtype=torch.float32,
                    device="cpu",
                    pin_memory=True,
                )
            metrics_host = self._confidence_metrics_host_buffer
            metrics_host.copy_(metrics, non_blocking=False)
        else:
            metrics_host = metrics
        return ConfidenceMetrics(float(metrics_host[0]), float(metrics_host[1]), float(metrics_host[2]))

    def _choose_precision(self, score: float) -> PrecisionLevel:
        if score > self.threshold_high and self.current_precision == PrecisionLevel.FP16:
            return PrecisionLevel.INT8
        if score < self.threshold_low and self.current_precision == PrecisionLevel.INT8:
            return PrecisionLevel.FP16
        return self.current_precision

    def _cast_logits(self, logits: torch.Tensor, precision: PrecisionLevel) -> torch.Tensor:
        if precision == PrecisionLevel.FP32:
            return logits.float()
        if precision == PrecisionLevel.FP16:
            return logits.half()
        if precision == PrecisionLevel.BF16:
            return logits.bfloat16()
        if precision == PrecisionLevel.INT8:
            scale = logits.abs().max() / 127.0
            quant = torch.clamp((logits / scale).round(), -128, 127)
            return quant * scale
        if precision == PrecisionLevel.INT4:
            scale = logits.abs().max() / 7.0
            quant = torch.clamp((logits / scale).round(), -8, 7)
            return quant * scale
        return logits

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_length: int = 20, temperature: float = 1.0) -> Tuple[torch.Tensor, List[Dict[str, float]]]:
        batch_size, prompt_len = input_ids.shape
        tokens = self._generation_token_buffer(input_ids, prompt_len + max_length)
        tokens[:, :prompt_len].copy_(input_ids)
        current_len = prompt_len
        stats: List[Dict[str, float]] = []
        for _step in range(max_length):
            active_tokens = tokens[:, :current_len]
            logits = self.model(active_tokens).logits[0, -1, :]
            logits = self._cast_logits(logits, self.current_precision)
            metrics = self._confidence(logits, temperature)
            confidence_score = metrics.confidence_score
            next_precision = self._choose_precision(confidence_score)
            if next_precision != self.current_precision:
                self.switch_count += 1
            self.current_precision = next_precision
            probs = F.softmax(logits / temperature, dim=-1)
            next_token, next_token_host = self._next_token_buffers(probs.device)
            torch.multinomial(probs, num_samples=1, out=next_token)
            tokens[:, current_len : current_len + 1].copy_(next_token.view(1, 1))
            current_len += 1
            stats.append({"confidence": confidence_score, "precision": self.current_precision.value})
            next_token_host.copy_(next_token)
            if int(next_token_host[0]) == 0:
                break
        return tokens[:, :current_len].clone(memory_format=torch.contiguous_format), stats


#
# ===== BEGIN dynamic_precision_inference =====
# ----------------------------
# Optional global toggles (B200 best practices)
# ----------------------------
# NEW PyTorch 2.10 API (no warnings!)
enable_tf32()
_ARCH_PREFER_BFLOAT16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
# If you compile models elsewhere, keep it outside this loop; don't pay compile cost per-step.

# ----------------------------
# Safe Transformer Engine (TE) FP8 autocast import
# ----------------------------
try:
    # TE is only effective if your model actually uses TE-enabled layers (e.g., Linear, LayerNorm wrappers).
    from transformer_engine.pytorch import fp8_autocast as _te_fp8_autocast  # type: ignore
    _TE_AVAILABLE = True
except Exception:
    _TE_AVAILABLE = False
    # No-op stand-in so the code runs without TE installed. It never changes numerical behavior.
    class _NullCtx(contextlib.ContextDecorator):
        def __init__(self, **_): pass
        def __enter__(self): return self
        def __exit__(self, *exc): return False
    def _te_fp8_autocast(**_):
        return _NullCtx()

# ----------------------------
# Helper: choose the precision context *for this step* safely
# ----------------------------
def _precision_context_cuda(use_fp8: bool, prefer_bfloat16: bool, enable_fp8: bool):
    """
    Enter exactly one precision context. If FP8 isn't enabled or TE is missing/unused, fall back to AMP (BF16/FP16).
    """
    if use_fp8 and enable_fp8 and _TE_AVAILABLE:
        # Note: fp8_autocast affects only TE-enabled modules. Non-TE modules run at their native dtypes.
        return _te_fp8_autocast(enabled=True)  # On Blackwell, TE defaults to MXFP8 kernels.
    amp_dtype = torch.bfloat16 if prefer_bfloat16 else torch.float16
    return torch.autocast(device_type="cuda", dtype=amp_dtype)

def _precision_context(device: torch.device, use_fp8: bool, prefer_bfloat16: bool, enable_fp8: bool):
    return _precision_context_cuda(use_fp8, prefer_bfloat16, enable_fp8) if device.type == "cuda" else contextlib.nullcontext()

# ----------------------------
# Main decode loop with smoothed, hysteretic precision switching
# ----------------------------
@torch.inference_mode()
def decode_with_dynamic_precision(
    model,
    tokens: torch.Tensor,
    max_steps: int,
    *,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    prefer_bfloat16: bool = _ARCH_PREFER_BFLOAT16,  # B200: prefer BF16 over FP16 for AMP
    enable_fp8: bool = True,             # Set True to allow FP8 when TE present & stable confidence
    enter_fp8_threshold: float = 6.0,    # hysteresis upper bound (logit margin average)
    exit_fp8_threshold: float = 3.0,     # hysteresis lower bound (avoid flapping)
    reeval_interval: int = 8,            # compute/inspect confidence every N steps to avoid per-step sync
    topk_dim: int = -1,                  # last dimension holds vocabulary logits
    eos_id: int | None = None,
):
    """
    Autoregressive decode loop that *smoothly* switches between AMP (BF16/FP16) and FP8 (TE) without
    per-step host sync. Works even when TE is not installed; in that case, runs AMP only.

    - Confidence signal: mean(top1 - top2) logits margin across the batch.
    - Smoothing: EMA + interval re-evaluation to minimize CPU-GPU sync pressure.
    - Hysteresis: separate enter/exit thresholds to avoid precision flapping.
    """
    assert exit_fp8_threshold <= enter_fp8_threshold, "Hysteresis requires exit <= enter threshold"
    if reeval_interval <= 0:
        raise ValueError("reeval_interval must be positive")

    model.eval()
    prompt = tokens.to(device, non_blocking=True)
    batch_size, prompt_len = prompt.shape
    tokens = torch.empty(
        (batch_size, prompt_len + max_steps),
        device=device,
        dtype=prompt.dtype,
    )
    tokens[:, :prompt_len].copy_(prompt)
    current_len = prompt_len
    next_token = torch.empty((batch_size, 1), device=device, dtype=prompt.dtype)
    next_token_values: torch.Tensor | None = None
    top2_values: torch.Tensor | None = None
    top2_indices: torch.Tensor | None = None
    margin_values: torch.Tensor | None = None
    margin_mean: torch.Tensor | None = None

    # Internal state
    use_fp8: bool = False  # start in AMP; upgrade to FP8 when sustained confidence permits
    ema_conf: torch.Tensor | None = None  # stays on device; host consults only at intervals
    alpha = 0.2  # EMA smoothing factor for confidence
    top2_shape_tuple: tuple[int, ...] | None = None

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
            ema_conf = torch.empty_like(margin_mean)
            ema_conf.copy_(margin_mean)
        else:
            ema_conf.mul_(1 - alpha).add_(margin_mean, alpha=alpha)
        return ema_conf  # device scalar

    # Decode
    for step in range(max_steps):
        active_tokens = tokens[:, :current_len]
        # 1) Precision context (exactly one). No nested contexts, no leakage across iterations.
        with _precision_context(device, use_fp8, prefer_bfloat16, enable_fp8):
            # Forward pass (HF-style or plain)
            try:
                logits = model(input_ids=active_tokens)  # HF models often return logits tensor or a ModelOutput
                if hasattr(logits, "logits"):
                    logits = logits.logits
            except TypeError:
                logits = model(active_tokens)

            # 2) Pick next token from the *last* position
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
            tokens[:, current_len : current_len + 1].copy_(next_token)
            current_len += 1

        # 3) Update on-device EMA only when the policy can consume it.
        should_reevaluate = (step + 1) % reeval_interval == 0
        conf_dev = _update_confidence_ema(logits) if (step == 0 or should_reevaluate) else ema_conf

        # 4) Periodically re-evaluate precision choice on host to avoid per-step sync
        if should_reevaluate:
            if conf_dev is None:
                conf_dev = _update_confidence_ema(logits)
            conf_value = float(conf_dev)  # exactly one tiny sync every N steps
            if not use_fp8 and enable_fp8 and _TE_AVAILABLE and (conf_value > enter_fp8_threshold):
                use_fp8 = True
            elif use_fp8 and (conf_value < exit_fp8_threshold):
                use_fp8 = False
            # torch.autocast (AMP) in PyTorch 2.10 does not manage FP8/FP4; TE handles MXFP8/NVFP4 paths.

        # 5) EOS handling
        if eos_id is not None:
            if (tokens[:, current_len - 1] == eos_id).all():
                break

    return tokens[:, :current_len].contiguous()
# ===== END dynamic_precision_inference =====


# ----------------------------
# Example (commented):
# ----------------------------
# model = ...  # your TE-enabled model (or any torch.nn.Module)
# input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
# out = decode_with_dynamic_precision(model, input_ids, max_steps=128, eos_id=tokenizer.eos_token_id)
# print(out.shape)


class DynamicQuantizedCache:
    def __init__(self, threshold: float = 0.8) -> None:
        self.threshold = threshold
        self.executor = ThreadPoolExecutor(max_workers=1)
        self._ema_ratio: float | None = None

    def _memory_ratio(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        device = torch.cuda.current_device()
        used = torch.cuda.memory_reserved(device)
        total = torch.cuda.get_device_properties(device).total_memory
        raw = used / total
        if self._ema_ratio is None:
            self._ema_ratio = raw
        else:
            self._ema_ratio = 0.8 * self._ema_ratio + 0.2 * raw
        return self._ema_ratio

    def maybe_quantize(self, layers: List[object], policy: str = "conservative") -> None:
        if self._memory_ratio() < self.threshold:
            return
        nbits = 8 if policy == "conservative" else 4
        for layer in layers:
            for attr in ("key_cache", "value_cache"):
                tensor = getattr(layer, attr, None)
                if tensor is None:
                    continue
                if not HQQ_AVAILABLE:
                    scale = tensor.abs().max() / (127 if nbits == 8 else 7)
                    quant = torch.clamp((tensor / scale).round(), -128, 127) * scale
                    setattr(layer, attr, quant)
                else:
                    future = self.executor.submit(HQQQuantizer.quantize, tensor, nbits=nbits, optimize=False)
                    try:
                        quant, meta = future.result(timeout=2.0)
                        setattr(layer, attr, quant)
                        setattr(layer, f"{attr}_meta", meta)
                    except Exception as exc:
                        logger.warning("Quantization failed: %s", exc)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    class ToyModel(torch.nn.Module):
        def __init__(self, vocab=1000, dim=512) -> None:
            super().__init__()
            self.embed = torch.nn.Embedding(vocab, dim)
            self.linear = torch.nn.Linear(dim, vocab)
        def forward(self, input_ids: torch.Tensor):
            x = self.embed(input_ids)
            logits = self.linear(x)
            return type("Output", (), {"logits": logits})()

    model = ToyModel()
    controller = TokenPrecisionController(model)
    generated, stats = controller.generate(torch.randint(1, 100, (1, 8)))
    print("Generated:", generated.tolist())
    print("Switches:", controller.switch_count)
    print("Stats:", stats[:5])

    if _TE_AVAILABLE and torch.cuda.is_available():
        seq = decode_with_dynamic_precision(
            model.cuda(),
            torch.randint(1, 100, (1, 8), device="cuda"),
            max_steps=16,
        )
        print("Dynamic precision sequence length:", seq.shape[-1])
    else:
        print("Transformer Engine FP8 demo skipped (TE or CUDA not available).")

    class MockLayer:
        def __init__(self):
            self.key_cache = torch.randn(4, 128)
            self.value_cache = torch.randn(4, 128)

    cache = DynamicQuantizedCache(threshold=0.0)
    cache.maybe_quantize([MockLayer(), MockLayer()], policy="conservative")


if __name__ == "__main__":
    main()
