"""FP8 encoding, complete MoE output assembly and independent accuracy checks."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F


def quantize_e4m3(tensor: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """Encode x / scale into E4M3 and return its FP32 dequantization scale.

    Reduction, division, clamp and conversion are real work included by callers
    in their timed activation path. Nonfinite inputs remain invalid; this is not
    a nan_to_num repair. Zero tensors use scale one.
    """
    if tensor.shape != out.shape or out.dtype != torch.float8_e4m3fn:
        raise ValueError("FP8 destination must have the same shape and E4M3FN dtype")
    if tensor.device != out.device or not tensor.numel():
        raise ValueError("FP8 encoding requires nonempty tensors on the same device")
    value = tensor.float()
    amax = value.abs().amax()
    scale = (amax / 448.0).clamp_min(torch.finfo(torch.float32).tiny)
    # Update the computed scale directly; do not allocate a unit tensor in the
    # timed activation path. The scalar fill preserves exact zero handling.
    scale.masked_fill_(amax == 0, 1.0)
    out.copy_((value / scale).clamp(-448.0, 448.0))
    return scale


def combine_sorted_routes(sorted_output, inverse_order, top_k, unsorted, output):
    """Undo expert sorting, then sum every weighted route in FP32."""
    if top_k <= 0 or sorted_output.shape[0] != output.shape[0] * top_k:
        raise ValueError("All token routes must be present before combination")
    if unsorted.shape != sorted_output.shape:
        raise ValueError("Unsorted workspace must cover every routed output")
    torch.index_select(sorted_output, 0, inverse_order, out=unsorted)
    torch.sum(unsorted.view(output.shape[0], top_k, output.shape[1]),
              dim=1, dtype=torch.float32, out=output)


@torch.inference_mode()
def reference_moe(x, weights_cpu, expert_ids_cpu, routing_cpu):
    """Full BF16 reference from original weights and unsorted logical routing.

    Never reuse FP8 values, candidate expert sorting or candidate output storage.
    Original BF16 weights stay on the CPU between verification calls. Transfer
    and reference computation are outside benchmark timing.
    """
    result = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for expert in range(weights_cpu[0].shape[0]):
        rows, routes = torch.where(expert_ids_cpu == expert)
        if not rows.numel():
            continue
        w1, w3, w2 = (weight[expert].to(x.device) for weight in weights_cpu)
        for start in range(0, rows.numel(), 128):
            selected = rows[start:start + 128].to(x.device)
            tokens = x.index_select(0, selected)
            gate = F.silu(tokens @ w1)
            hidden = gate * (tokens @ w3)
            weighted = (hidden @ w2) * routing_cpu[
                rows[start:start + 128], routes[start:start + 128]].to(x.device).unsqueeze(1)
            result.index_add_(0, selected, weighted.float())
    return result


def full_output_errors(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype or not actual.numel():
        raise AssertionError("Full FP8 output/reference shape or dtype differs or is empty")
    if actual.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr():
        raise AssertionError("FP8 reference must not alias candidate output")
    squared_error = squared_reference = max_error = max_reference = 0.0
    got, ref = actual.reshape(-1), expected.reshape(-1)
    for start in range(0, got.numel(), 1 << 20):
        g, r = got[start:start + (1 << 20)].double(), ref[start:start + (1 << 20)].double()
        if not torch.isfinite(g).all() or not torch.isfinite(r).all():
            raise AssertionError("Nonfinite full FP8 output or BF16 reference")
        delta = g - r
        squared_error += float(delta.square().sum())
        squared_reference += float(r.square().sum())
        max_error = max(max_error, float(delta.abs().max()))
        max_reference = max(max_reference, float(r.abs().max()))
    return {
        "relative_l2": math.sqrt(squared_error / squared_reference) if squared_reference
        else (0.0 if squared_error == 0 else math.inf),
        "normalized_max_abs": max_error / max_reference if max_reference
        else (0.0 if max_error == 0 else math.inf),
    }


@dataclass(frozen=True)
class AccuracyLimits:
    relative_l2: float
    normalized_max_abs: float
    pairwise_rtol: float
    pairwise_atol: float

    def __post_init__(self):
        for name in ("relative_l2", "normalized_max_abs", "pairwise_rtol"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if not math.isfinite(self.pairwise_atol) or self.pairwise_atol < 0:
            raise ValueError("pairwise_atol must be finite and nonnegative")

    def check(self, errors):
        if any(not math.isfinite(value) or value > getattr(self, name)
               for name, value in errors.items()):
            raise AssertionError(f"Full FP8 output exceeds configured accuracy policy: {errors}")


def load_accuracy_limits():
    path = os.environ.get("AISP_NATIVE_FP8_ACCURACY_POLICY")
    if not path:
        raise RuntimeError(
            "Native FP8 MoE accuracy is uncalibrated: AISP_NATIVE_FP8_ACCURACY_POLICY "
            "must name an externally reviewed policy. Use the lab calibration command "
            "to collect full-output errors; calibration is not acceptance."
        )
    policy = json.loads(Path(path).read_text())
    if policy.get("schema_version") != 1:
        raise ValueError("Native FP8 accuracy policy requires schema_version=1")
    return AccuracyLimits(**policy["native_fp8"])
