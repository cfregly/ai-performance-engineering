"""Shared verification helpers to reduce benchmark boilerplate.

The mixin provides a single `_set_verification_payload()` call that wires up:
- get_verify_inputs()
- get_verify_output()
- get_input_signature()
- get_output_tolerance()

CRITICAL: `_set_verification_payload()` must be called from
`BaseBenchmark.capture_verification_payload()` (post-timing) to keep the timed
hot path clean. The harness calls `capture_verification_payload()` once after
measurement, and VerifyRunner calls it after verify runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import inspect
from typing import Any, Dict, Optional, Tuple, Union

import torch

from core.benchmark.verification import (
    InputSignature,
    PrecisionFlags,
    ToleranceSpec,
    coerce_input_signature,
    get_tolerance_for_dtype,
    simple_signature,
)


@dataclass
class VerificationPayload:
    """Container for verification metadata."""
    inputs: Dict[str, torch.Tensor]
    output: torch.Tensor
    batch_size: int
    parameter_count: int
    precision_flags: PrecisionFlags
    output_tolerance: Optional[ToleranceSpec] = None
    signature_overrides: Dict[str, Any] = field(default_factory=dict)


class VerificationPayloadMixin:
    """Mixin that supplies strict verification methods."""

    _ALLOWED_SIGNATURE_OVERRIDE_KEYS = {
        "world_size",
        "ranks",
        "shards",
        "pipeline_stages",
        "pipeline_stage_boundaries",
        "per_rank_batch_size",
        "collective_type",
        "collective_algorithm",
        "gradient_bucket_bytes",
        "barrier_policy",
        "async_completion_policy",
        "num_streams",
        "graph_capture_enabled",
        "pruning_enabled",
        "sparsity_ratio",
        "quantization_mode",
    }

    def _called_from_capture_hook(self) -> bool:
        frame = inspect.currentframe()
        if frame is None:
            return False
        frame = frame.f_back
        while frame is not None:
            if frame.f_code.co_name == "capture_verification_payload" and frame.f_locals.get("self") is self:
                return True
            frame = frame.f_back
        return False

    def _normalize_precision_flags(self, precision_flags: Optional[Dict[str, bool] | PrecisionFlags]) -> PrecisionFlags:
        if isinstance(precision_flags, PrecisionFlags):
            return precision_flags
        if precision_flags is None:
            tf32_enabled = False
            try:
                tf32_enabled = torch.cuda.is_available() and bool(torch.backends.cuda.matmul.allow_tf32)
            except Exception:
                tf32_enabled = False
            return PrecisionFlags(tf32=tf32_enabled)
        return PrecisionFlags.from_dict(dict(precision_flags))

    def _coerce_tolerance(self, tolerance: Optional[Union[ToleranceSpec, Tuple[float, float]]], output: torch.Tensor) -> ToleranceSpec:
        if isinstance(tolerance, ToleranceSpec):
            return tolerance
        if isinstance(tolerance, (tuple, list)) and len(tolerance) == 2:
            return ToleranceSpec(rtol=float(tolerance[0]), atol=float(tolerance[1]))
        return get_tolerance_for_dtype(output.dtype)

    def _normalize_signature_overrides(self, overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        if overrides is None:
            return {}
        if not isinstance(overrides, dict):
            raise TypeError(f"signature_overrides must be a dict, got {type(overrides)}")
        unknown = set(overrides.keys()) - self._ALLOWED_SIGNATURE_OVERRIDE_KEYS
        if unknown:
            raise ValueError(
                f"signature_overrides contains unsupported keys: {sorted(unknown)}. "
                f"Allowed: {sorted(self._ALLOWED_SIGNATURE_OVERRIDE_KEYS)}"
            )
        normalized: Dict[str, Any] = dict(overrides)
        if "pipeline_stages" in normalized and normalized["pipeline_stages"] is not None:
            normalized["pipeline_stages"] = int(normalized["pipeline_stages"])
        if "world_size" in normalized and normalized["world_size"] is not None:
            normalized["world_size"] = int(normalized["world_size"])
        if "shards" in normalized and normalized["shards"] is not None:
            normalized["shards"] = int(normalized["shards"])
        if "per_rank_batch_size" in normalized and normalized["per_rank_batch_size"] is not None:
            normalized["per_rank_batch_size"] = int(normalized["per_rank_batch_size"])
        if "gradient_bucket_bytes" in normalized and normalized["gradient_bucket_bytes"] is not None:
            if isinstance(normalized["gradient_bucket_bytes"], bool):
                raise TypeError("signature_overrides['gradient_bucket_bytes'] must be an integer, not bool")
            normalized["gradient_bucket_bytes"] = int(normalized["gradient_bucket_bytes"])
        if "num_streams" in normalized and normalized["num_streams"] is not None:
            normalized["num_streams"] = int(normalized["num_streams"])
        if "sparsity_ratio" in normalized and normalized["sparsity_ratio"] is not None:
            normalized["sparsity_ratio"] = float(normalized["sparsity_ratio"])
        if "ranks" in normalized and normalized["ranks"] is not None:
            ranks = normalized["ranks"]
            if not isinstance(ranks, list) or not all(isinstance(r, int) for r in ranks):
                raise TypeError("signature_overrides['ranks'] must be a list[int]")
        if "pipeline_stage_boundaries" in normalized and normalized["pipeline_stage_boundaries"] is not None:
            boundaries = normalized["pipeline_stage_boundaries"]
            if not isinstance(boundaries, list):
                raise TypeError("signature_overrides['pipeline_stage_boundaries'] must be a list of (start, end) pairs")
            normalized_boundaries = []
            for entry in boundaries:
                if not isinstance(entry, (list, tuple)) or len(entry) != 2:
                    raise TypeError(
                        "signature_overrides['pipeline_stage_boundaries'] entries must be (start, end) pairs"
                    )
                normalized_boundaries.append((int(entry[0]), int(entry[1])))
            normalized["pipeline_stage_boundaries"] = normalized_boundaries
        return normalized

    def _set_verification_payload(
        self,
        *,
        inputs: Dict[str, torch.Tensor],
        output: torch.Tensor,
        batch_size: int,
        parameter_count: int = 0,
        precision_flags: Optional[Dict[str, bool] | PrecisionFlags] = None,
        output_tolerance: Optional[Union[ToleranceSpec, Tuple[float, float]]] = None,
        signature_overrides: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Populate verification payload in a single call."""
        if not self._called_from_capture_hook():
            raise RuntimeError(
                "_set_verification_payload() must be called from capture_verification_payload() "
                "(post-timing) to keep benchmark_fn() hot path clean."
            )
        if not inputs:
            raise ValueError("inputs must be a non-empty dict of tensors")
        for name, tensor in inputs.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"inputs['{name}'] must be a torch.Tensor, got {type(tensor)}")
        if output is None or not isinstance(output, torch.Tensor):
            raise ValueError("output tensor is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        flags = self._normalize_precision_flags(precision_flags)
        tolerance_spec = self._coerce_tolerance(output_tolerance, output)
        signature_overrides_normalized = self._normalize_signature_overrides(signature_overrides)

        self._verification_payload = VerificationPayload(
            inputs=inputs,
            output=output,
            batch_size=int(batch_size),
            parameter_count=int(parameter_count),
            precision_flags=flags,
            output_tolerance=tolerance_spec,
            signature_overrides=signature_overrides_normalized,
        )

    # Public API ----------------------------------------------------------------
    def _require_payload(self) -> VerificationPayload:
        payload = getattr(self, "_verification_payload", None)
        if payload is None:
            raise RuntimeError("_set_verification_payload() must be called before verification")
        return payload

    def get_verify_inputs(self) -> Dict[str, torch.Tensor]:
        payload = self._require_payload()
        return {k: v for k, v in payload.inputs.items()}

    def get_verify_output(self) -> torch.Tensor:
        payload = self._require_payload()
        return payload.output.detach().clone()

    def get_input_signature(self) -> InputSignature:
        payload = self._require_payload()
        shapes = {name: tuple(t.shape) for name, t in payload.inputs.items()}
        dtypes = {name: str(t.dtype) for name, t in payload.inputs.items()}
        shapes["output"] = tuple(payload.output.shape)
        dtypes["output"] = str(payload.output.dtype)
        sig = InputSignature(
            shapes=shapes,
            dtypes=dtypes,
            batch_size=payload.batch_size,
            parameter_count=payload.parameter_count,
            precision_flags=payload.precision_flags,
            **dict(payload.signature_overrides),
        )
        # Validate eagerly to surface incomplete signatures early
        coerce_input_signature(sig)
        return sig

    def get_output_tolerance(self) -> Tuple[float, float]:
        payload = self._require_payload()
        tol = payload.output_tolerance or get_tolerance_for_dtype(payload.output.dtype)
        return (tol.rtol, tol.atol)


__all__ = [
    "VerificationPayload",
    "VerificationPayloadMixin",
    "simple_signature",
]
