"""Shared helpers for the FlashAttention-4 lab.

The optimized path targets a Blackwell-friendly FlexAttention configuration and
optionally attempts the experimental FLASH backend exposed by PyTorch 2.9's
FlexAttention integration. The lab keeps a dense reference implementation for
tests and explanation, but the benchmark pair itself compares eager
FlexAttention to a compiled, TMA-oriented kernel path.
"""

from __future__ import annotations

import inspect
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from core.benchmark.utils import scalar_tensor_to_float
from core.utils.compile_utils import compile_callable

try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
except Exception as exc:  # pragma: no cover - depends on local torch build
    SDPBackend = Any  # type: ignore[assignment]
    sdpa_kernel = None  # type: ignore[assignment]
    _SDPA_IMPORT_ERROR = exc
else:
    _SDPA_IMPORT_ERROR = None

try:
    from torch.nn.attention.flex_attention import BlockMask, create_block_mask, flex_attention
except Exception as exc:  # pragma: no cover - depends on local torch build
    BlockMask = Any  # type: ignore[assignment]
    create_block_mask = None  # type: ignore[assignment]
    flex_attention = None  # type: ignore[assignment]
    _FLEX_IMPORT_ERROR = exc
else:
    _FLEX_IMPORT_ERROR = None

SUPPORTED_MODES = (
    "dense",
    "causal",
    "alibi",
    "softcap",
    "windowed",
    "alibi_windowed",
)
SUPPORTED_BACKENDS = ("auto", "flash", "tma", "flex")
DEFAULT_FLASH_BACKEND_LITERAL = '"FLASH"'
SUPPORTED_PROVIDERS = ("flash_backend", "flex_tma", "flex_compiled")
BEST_AVAILABLE_PROVIDERS = ("cudnn_sdpa",) + SUPPORTED_PROVIDERS
FLASHATTENTION4_PROVIDER_ORDER = BEST_AVAILABLE_PROVIDERS + ("eager_flex",)
FLASHATTENTION4_PROVIDER_IDS = {
    provider: float(idx + 1) for idx, provider in enumerate(FLASHATTENTION4_PROVIDER_ORDER)
}
FLASHATTENTION4_CLAIM_TYPE_IDS = {
    "educational": 1.0,
    "absolute": 2.0,
    "reproduction": 3.0,
}
FLASHATTENTION4_EDUCATIONAL_TARGET = "labs/flashattention4:flashattention4"
FLASHATTENTION4_ABSOLUTE_TARGET = "labs/flashattention4:best_available_attention"
FLASHATTENTION4_REPRODUCTION_ENTRYPOINT = "labs/flashattention4/tflops_microbench.py"
_ALIBI_DISTANCE_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}
_ALIBI_SLOPE_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}
_DENSE_MASK_POSITION_CACHE: dict[tuple[int, torch.device], tuple[torch.Tensor, torch.Tensor]] = {}


@dataclass(frozen=True)
class FlashAttention4Config:
    """Configuration shared across baseline and optimized benchmarks."""

    batch: int = 2
    heads: int = 8
    seq_len: int = 2048
    head_dim: int = 64
    block_size: int = 128
    window_size: int = 256
    dtype: torch.dtype = torch.bfloat16
    mode: str = "alibi"
    backend: str = "auto"
    compile_mode: str = "max-autotune"


@dataclass
class FlashAttention4Inputs:
    """Prepared tensors and modifiers for the lab workloads."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    dense_mask: Optional[torch.Tensor]
    block_mask: Optional[BlockMask]
    alibi_slopes: Optional[torch.Tensor]
    softcap_scale: Optional[float]
    mode: str
    window_size: int
    block_size: int


@dataclass
class FlashAttention4Kernel:
    """Resolved compiled kernel/provider description."""

    fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]
    provider: str
    kernel_options: dict[str, Any]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class FlashAttention4Timing:
    """Basic CUDA timing stats for provider selection."""

    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


@dataclass
class FlashAttention4Selection:
    """Selected backend plus candidate timings from an auto-selection pass."""

    kernel: FlashAttention4Kernel
    candidate_median_ms: dict[str, float]
    candidate_errors: dict[str, str]


@dataclass(frozen=True)
class FlashAttention4ModeDecision:
    """Executable recommendation for how to interpret a given workload mode."""

    mode: str
    recommended_backend: str
    recommended_target: str
    recommended_claim_type: str
    educational_target: str
    reproduction_entrypoint: str
    evidence: str
    notes: str


FLASHATTENTION4_MODE_DECISION_TABLE = {
    "dense": FlashAttention4ModeDecision(
        mode="dense",
        recommended_backend="cudnn_sdpa",
        recommended_target=FLASHATTENTION4_ABSOLUTE_TARGET,
        recommended_claim_type="absolute",
        educational_target=FLASHATTENTION4_EDUCATIONAL_TARGET,
        reproduction_entrypoint=FLASHATTENTION4_REPRODUCTION_ENTRYPOINT,
        evidence="measured_local",
        notes="Use the best-available target for local peak latency claims; use the educational target for eager-vs-fused instruction.",
    ),
    "causal": FlashAttention4ModeDecision(
        mode="causal",
        recommended_backend="cudnn_sdpa",
        recommended_target=FLASHATTENTION4_ABSOLUTE_TARGET,
        recommended_claim_type="absolute",
        educational_target=FLASHATTENTION4_EDUCATIONAL_TARGET,
        reproduction_entrypoint=FLASHATTENTION4_REPRODUCTION_ENTRYPOINT,
        evidence="measured_local",
        notes="cuDNN is the stable local winner for standard causal attention on this stack.",
    ),
    "alibi": FlashAttention4ModeDecision(
        mode="alibi",
        recommended_backend="flex_tma",
        recommended_target=FLASHATTENTION4_ABSOLUTE_TARGET,
        recommended_claim_type="absolute",
        educational_target=FLASHATTENTION4_EDUCATIONAL_TARGET,
        reproduction_entrypoint=FLASHATTENTION4_REPRODUCTION_ENTRYPOINT,
        evidence="measured_local",
        notes="ALiBi is a Flex-only feature path; use the educational target when you want the simpler FA4-style story.",
    ),
    "softcap": FlashAttention4ModeDecision(
        mode="softcap",
        recommended_backend="flex_tma",
        recommended_target=FLASHATTENTION4_ABSOLUTE_TARGET,
        recommended_claim_type="absolute",
        educational_target=FLASHATTENTION4_EDUCATIONAL_TARGET,
        reproduction_entrypoint=FLASHATTENTION4_REPRODUCTION_ENTRYPOINT,
        evidence="selector_policy_inferred",
        notes="cuDNN is not applicable; use the best-available target for local backend selection within the Flex family.",
    ),
    "windowed": FlashAttention4ModeDecision(
        mode="windowed",
        recommended_backend="flex_tma",
        recommended_target=FLASHATTENTION4_ABSOLUTE_TARGET,
        recommended_claim_type="absolute",
        educational_target=FLASHATTENTION4_EDUCATIONAL_TARGET,
        reproduction_entrypoint=FLASHATTENTION4_REPRODUCTION_ENTRYPOINT,
        evidence="experimental",
        notes="Sliding-window paths are experimental on this local stack; treat results as probes, not canonical claims.",
    ),
    "alibi_windowed": FlashAttention4ModeDecision(
        mode="alibi_windowed",
        recommended_backend="flex_tma",
        recommended_target=FLASHATTENTION4_ABSOLUTE_TARGET,
        recommended_claim_type="absolute",
        educational_target=FLASHATTENTION4_EDUCATIONAL_TARGET,
        reproduction_entrypoint=FLASHATTENTION4_REPRODUCTION_ENTRYPOINT,
        evidence="experimental",
        notes="Combined ALiBi+windowed paths remain experimental; rely on the reproduction microbench for backend debugging.",
    ),
}


def flashattention4_provider_id(provider: str) -> float:
    """Encode provider names as stable numeric IDs for benchmark artifacts."""
    return FLASHATTENTION4_PROVIDER_IDS.get(provider, 0.0)


def flashattention4_claim_type_id(claim_type: str) -> float:
    """Encode claim types as stable numeric IDs for benchmark artifacts."""
    return FLASHATTENTION4_CLAIM_TYPE_IDS.get(claim_type, 0.0)


def resolve_flashattention4_mode_decision(mode: str) -> FlashAttention4ModeDecision:
    """Return the recommendation row for a given workload mode."""
    if mode not in FLASHATTENTION4_MODE_DECISION_TABLE:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {SUPPORTED_MODES}")
    return FLASHATTENTION4_MODE_DECISION_TABLE[mode]


def list_flashattention4_mode_decisions() -> tuple[FlashAttention4ModeDecision, ...]:
    """Return the full executable mode-decision table."""
    return tuple(FLASHATTENTION4_MODE_DECISION_TABLE[mode] for mode in SUPPORTED_MODES)


def resolve_flashattention4_claim_type(target_label: Optional[str], *, default: str) -> str:
    """Infer the claim type from the target label, falling back when absent."""
    normalized = (target_label or "").strip().lower()
    if "best_available_attention" in normalized:
        return "absolute"
    if "flashattention4" in normalized:
        return "educational"
    return default


def _slugify_artifact_component(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_") or "unknown"


def build_flashattention4_mode_table_payload(
    *,
    current_mode: str,
    run_claim_type: str,
    target_label: Optional[str],
    selected_provider: Optional[str] = None,
) -> dict[str, Any]:
    """Build the structured mode-decision payload for artifacts and microbench JSON."""
    decision = resolve_flashattention4_mode_decision(current_mode)
    return {
        "schema_version": "1.0",
        "current_run": {
            "mode": current_mode,
            "target_label": target_label,
            "run_claim_type": run_claim_type,
            "run_claim_type_id": flashattention4_claim_type_id(run_claim_type),
            "selected_provider": selected_provider,
            "selected_provider_id": flashattention4_provider_id(selected_provider or ""),
            "recommended_backend_for_mode": decision.recommended_backend,
            "recommended_backend_id": flashattention4_provider_id(decision.recommended_backend),
            "recommended_target_for_mode": decision.recommended_target,
            "recommended_claim_type_for_mode": decision.recommended_claim_type,
            "recommended_claim_type_id": flashattention4_claim_type_id(decision.recommended_claim_type),
            "educational_target": decision.educational_target,
            "reproduction_entrypoint": decision.reproduction_entrypoint,
            "evidence": decision.evidence,
            "notes": decision.notes,
        },
        "mode_table": [asdict(entry) for entry in list_flashattention4_mode_decisions()],
    }


def render_flashattention4_mode_table_markdown(payload: dict[str, Any]) -> str:
    """Render the structured mode-decision payload as a small markdown artifact."""
    current = payload["current_run"]
    lines = [
        "# FlashAttention-4 Mode Table",
        "",
        "## Current Run",
        "",
        "| Mode | Run Claim Type | Recommended Backend | Recommended Target | Selected Provider | Evidence |",
        "| --- | --- | --- | --- | --- | --- |",
        (
            f"| {current['mode']} | {current['run_claim_type']} | "
            f"{current['recommended_backend_for_mode']} | {current['recommended_target_for_mode']} | "
            f"{current.get('selected_provider') or 'n/a'} | {current['evidence']} |"
        ),
        "",
        current["notes"],
        "",
        "## Mode Decision Table",
        "",
        "| Mode | Recommended Backend | Recommended Target | Claim Type | Educational Target | Reproduction Entrypoint | Evidence |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in payload["mode_table"]:
        lines.append(
            f"| {row['mode']} | {row['recommended_backend']} | {row['recommended_target']} | "
            f"{row['recommended_claim_type']} | {row['educational_target']} | "
            f"{row['reproduction_entrypoint']} | {row['evidence']} |"
        )
    return "\n".join(lines) + "\n"


def emit_flashattention4_mode_table_artifacts(
    config: Any,
    *,
    current_mode: str,
    run_claim_type: str,
    selected_provider: Optional[str] = None,
) -> Optional[dict[str, str]]:
    """Write a JSON+markdown mode table into the current run's artifact directory."""
    output_root = getattr(config, "subprocess_stderr_dir", None) or getattr(config, "profiling_output_dir", None)
    if not output_root:
        return None
    output_dir = Path(output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_label = getattr(config, "target_label", None)
    target_slug = _slugify_artifact_component(target_label or run_claim_type)
    mode_slug = _slugify_artifact_component(current_mode)
    payload = build_flashattention4_mode_table_payload(
        current_mode=current_mode,
        run_claim_type=run_claim_type,
        target_label=target_label,
        selected_provider=selected_provider,
    )
    json_path = output_dir / f"flashattention4_mode_table__{target_slug}__{mode_slug}.json"
    md_path = output_dir / f"flashattention4_mode_table__{target_slug}__{mode_slug}.md"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="ascii")
    md_path.write_text(render_flashattention4_mode_table_markdown(payload), encoding="ascii")
    return {"json": str(json_path), "markdown": str(md_path)}


def load_lab_config() -> FlashAttention4Config:
    """Return the default lab configuration.

    FlashAttention-4 benchmark selection is target-driven. Mode/backend/shape
    changes should be expressed as explicit benchmark targets or code changes,
    not environment variables.
    """
    return FlashAttention4Config()


def resolve_cuda_device() -> torch.device:
    """Require CUDA for the runtime benchmark pair."""
    if not torch.cuda.is_available():
        raise RuntimeError("FlashAttention-4 lab requires CUDA.")
    return torch.device("cuda")


def _device_is_blackwell(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    return torch.cuda.get_device_capability(device) >= (10, 0)


def build_dense_attention_mask(
    mode: str,
    *,
    seq_len: int,
    window_size: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    """Create a dense boolean mask used by the reference implementation."""
    if mode == "dense":
        return None

    q_idx, kv_idx = _dense_mask_positions_for(seq_len, device)
    mask = q_idx >= kv_idx
    if mode in {"windowed", "alibi_windowed"}:
        mask = mask & ((q_idx - kv_idx) < window_size)
    return mask.unsqueeze(0).unsqueeze(0)


def _dense_mask_positions_for(
    seq_len: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = (int(seq_len), torch.device(device))
    cached = _DENSE_MASK_POSITION_CACHE.get(key)
    if cached is None:
        positions = torch.arange(seq_len, device=device)
        cached = (positions[:, None], positions[None, :])
        _DENSE_MASK_POSITION_CACHE[key] = cached
    return cached


def _alibi_distance_for(seq_len: int, device: torch.device) -> torch.Tensor:
    key = (int(seq_len), torch.device(device))
    cached = _ALIBI_DISTANCE_CACHE.get(key)
    if cached is None:
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        cached = positions[:, None] - positions[None, :]
        cached.clamp_min_(0)
        _ALIBI_DISTANCE_CACHE[key] = cached
    return cached


def build_alibi_slopes(num_heads: int, *, device: torch.device) -> torch.Tensor:
    """Standard ALiBi slope generation."""
    key = (int(num_heads), torch.device(device))
    cached = _ALIBI_SLOPE_CACHE.get(key)
    if cached is not None:
        return cached

    def _slopes_power_of_two(head_count: int) -> list[float]:
        start = 2 ** (-(2 ** -(math.log2(head_count) - 3)))
        ratio = start
        return [start * (ratio**idx) for idx in range(head_count)]

    if math.log2(num_heads).is_integer():
        slopes = _slopes_power_of_two(num_heads)
    else:
        closest = 2 ** math.floor(math.log2(num_heads))
        slopes = _slopes_power_of_two(closest)
        extra = _slopes_power_of_two(2 * closest)[0::2]
        slopes.extend(extra[: num_heads - closest])
    cached = torch.tensor(slopes, device=device, dtype=torch.float32)
    _ALIBI_SLOPE_CACHE[key] = cached
    return cached


def build_reference_inputs(
    config: FlashAttention4Config,
    *,
    device: torch.device,
    include_block_mask: bool,
) -> FlashAttention4Inputs:
    """Construct Q/K/V tensors plus dense and block masks."""
    generator = torch.Generator(device=device).manual_seed(0)
    shape = (config.batch, config.heads, config.seq_len, config.head_dim)
    q = torch.randn(shape, device=device, dtype=config.dtype, generator=generator)
    k = torch.randn(shape, device=device, dtype=config.dtype, generator=generator)
    v = torch.randn(shape, device=device, dtype=config.dtype, generator=generator)

    dense_mask = build_dense_attention_mask(
        config.mode,
        seq_len=config.seq_len,
        window_size=config.window_size,
        device=device,
    )
    block_mask = build_flex_block_mask(config, device=device) if include_block_mask else None
    alibi_slopes = build_alibi_slopes(config.heads, device=device) if "alibi" in config.mode else None
    softcap_scale = 32.0 if config.mode == "softcap" else None

    return FlashAttention4Inputs(
        q=q,
        k=k,
        v=v,
        dense_mask=dense_mask,
        block_mask=block_mask,
        alibi_slopes=alibi_slopes,
        softcap_scale=softcap_scale,
        mode=config.mode,
        window_size=config.window_size,
        block_size=config.block_size,
    )


def build_flex_block_mask(config: FlashAttention4Config, *, device: torch.device) -> Optional[BlockMask]:
    """Create the sparse/block mask used by FlexAttention."""
    if config.mode == "dense":
        return None
    if create_block_mask is None:
        raise RuntimeError(f"flex_attention.create_block_mask is unavailable ({_FLEX_IMPORT_ERROR})")

    def mask_mod(_b: int, _h: int, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        causal = q_idx >= kv_idx
        if config.mode in {"windowed", "alibi_windowed"}:
            return causal & ((q_idx - kv_idx) < config.window_size)
        return causal

    kwargs: dict[str, Any] = {"device": device, "BLOCK_SIZE": config.block_size}
    if "use_vmap" in inspect.signature(create_block_mask).parameters:
        kwargs["use_vmap"] = True

    return create_block_mask(  # type: ignore[misc]
        mask_mod,
        config.batch,
        config.heads,
        config.seq_len,
        config.seq_len,
        **kwargs,
    )


def build_score_mod(inputs: FlashAttention4Inputs) -> Optional[Callable[..., torch.Tensor]]:
    """Build the score_mod callback for FlexAttention."""
    if inputs.alibi_slopes is None and inputs.softcap_scale is None:
        return None

    slopes = inputs.alibi_slopes
    softcap_scale = inputs.softcap_scale

    def score_mod(
        score: torch.Tensor,
        _batch_idx: torch.Tensor,
        head_idx: torch.Tensor,
        q_idx: torch.Tensor,
        kv_idx: torch.Tensor,
    ) -> torch.Tensor:
        result = score
        if slopes is not None:
            distance = (q_idx - kv_idx).clamp_min(0).to(torch.float32)
            result = result - slopes[head_idx] * distance
        if softcap_scale is not None:
            result = softcap_scale * torch.tanh(result / softcap_scale)
        return result

    return score_mod


def reference_attention(inputs: FlashAttention4Inputs) -> torch.Tensor:
    """Dense math reference for tests and explanations."""
    scale = 1.0 / math.sqrt(float(inputs.q.size(-1)))
    scores = torch.matmul(inputs.q.float(), inputs.k.float().transpose(-1, -2)) * scale

    if inputs.alibi_slopes is not None:
        seq_len = inputs.q.size(-2)
        distance = _alibi_distance_for(seq_len, inputs.q.device)
        scores.addcmul_(
            inputs.alibi_slopes.view(1, -1, 1, 1),
            distance.view(1, 1, seq_len, seq_len),
            value=-1.0,
        )

    if inputs.softcap_scale is not None:
        scores.div_(inputs.softcap_scale).tanh_().mul_(inputs.softcap_scale)

    if inputs.dense_mask is not None:
        scores.masked_fill_(~inputs.dense_mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, inputs.v.float())


def eager_flex_attention(inputs: FlashAttention4Inputs) -> torch.Tensor:
    """Run the eager FlexAttention path used as the baseline benchmark."""
    if flex_attention is None:
        raise RuntimeError(f"flex_attention is unavailable ({_FLEX_IMPORT_ERROR})")
    return flex_attention(
        inputs.q,
        inputs.k,
        inputs.v,
        score_mod=build_score_mod(inputs),
        block_mask=inputs.block_mask,
    )


def _compile_candidate(
    inputs: FlashAttention4Inputs,
    *,
    kernel_options: dict[str, Any],
    provider: str,
    compile_mode: str,
    notes: tuple[str, ...] = (),
) -> FlashAttention4Kernel:
    if flex_attention is None:
        raise RuntimeError(f"flex_attention is unavailable ({_FLEX_IMPORT_ERROR})")

    score_mod = build_score_mod(inputs)

    def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        return flex_attention(
            q,
            k,
            v,
            score_mod=score_mod,
            block_mask=inputs.block_mask,
            kernel_options=kernel_options,
        )

    compiled = compile_callable(forward, mode=compile_mode, dynamic=False, fullgraph=False)
    with torch.inference_mode():
        compiled(inputs.q, inputs.k, inputs.v)
    return FlashAttention4Kernel(
        fn=compiled,
        provider=provider,
        kernel_options=dict(kernel_options),
        notes=notes,
    )


def measure_flashattention4_latency(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    iterations: int,
) -> FlashAttention4Timing:
    """Time a CUDA callable using events."""
    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    times_ms: list[float] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    current_stream = torch.cuda.current_stream()
    for _ in range(iterations):
        start.record(current_stream)
        _ = fn()
        end.record(current_stream)
        end.synchronize()
        times_ms.append(float(start.elapsed_time(end)))

    count = len(times_ms)
    total_ms = 0.0
    total_sq_ms = 0.0
    min_ms = float("inf")
    max_ms = float("-inf")
    for value in times_ms:
        total_ms += value
        total_sq_ms += value * value
        min_ms = min(min_ms, value)
        max_ms = max(max_ms, value)

    times_ms.sort()
    mid = count // 2
    if count % 2 == 1:
        median_ms = times_ms[mid]
    else:
        median_ms = (times_ms[mid - 1] + times_ms[mid]) / 2.0

    mean_ms = total_ms / count
    if count > 1:
        variance = (total_sq_ms - (total_ms * total_ms / count)) / (count - 1)
        std_ms = variance**0.5 if variance > 0.0 else 0.0
    else:
        std_ms = 0.0

    return FlashAttention4Timing(
        mean_ms=mean_ms,
        median_ms=median_ms,
        min_ms=min_ms,
        max_ms=max_ms,
        std_ms=std_ms,
    )


def count_nonmasked_attention_elements(
    mode: str,
    *,
    q_seq_len: int,
    kv_seq_len: int,
    window_size: int = 0,
) -> int:
    """Count the attention matrix elements that participate in the forward pass."""
    if mode in {"dense", "alibi", "softcap"}:
        return q_seq_len * kv_seq_len

    if mode == "causal":
        diagonal_rows = min(q_seq_len, kv_seq_len)
        triangular = diagonal_rows * (diagonal_rows + 1) // 2
        full_kv_rows = max(q_seq_len - kv_seq_len, 0)
        return triangular + full_kv_rows * kv_seq_len

    if mode in {"windowed", "alibi_windowed"}:
        if window_size < 1:
            raise ValueError("window_size must be >= 1 for windowed modes")
        active_kv = min(kv_seq_len, q_seq_len)
        full_window_cols = min(active_kv, max(q_seq_len - window_size + 1, 0))
        tail_cols = active_kv - full_window_cols
        if tail_cols == 0:
            return full_window_cols * window_size
        tail_first = full_window_cols
        tail_last = active_kv - 1
        tail_total = tail_cols * (2 * q_seq_len - tail_first - tail_last) // 2
        return full_window_cols * window_size + tail_total

    raise ValueError(f"Unsupported mode {mode!r}; expected one of {SUPPORTED_MODES}")


def estimate_attention_forward_flops(
    *,
    batch: int,
    heads: int,
    q_seq_len: int,
    kv_seq_len: int,
    head_dim: int,
    mode: str,
    window_size: int = 0,
) -> int:
    """Estimate forward-pass attention FLOPs using the common SDPA benchmark convention."""
    nonmasked_elements = count_nonmasked_attention_elements(
        mode,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        window_size=window_size,
    )
    return 4 * batch * heads * head_dim * nonmasked_elements


def get_flashattention4_candidate_kernel_options(
    config: FlashAttention4Config,
    *,
    device: torch.device,
) -> dict[str, dict[str, Any]]:
    """Return provider -> kernel_options for the FA4 lab backends."""
    common_options = {
        "BLOCK_M": config.block_size,
        "BLOCK_N": config.block_size,
        "FLOAT32_PRECISION": "'ieee'",
        "ROWS_GUARANTEED_SAFE": True,
    }
    providers: dict[str, dict[str, Any]] = {
        "flex_tma": {
            **common_options,
            "FORCE_USE_FLEX_ATTENTION": True,
            "USE_TMA": True,
        },
        "flex_compiled": {
            **common_options,
            "FORCE_USE_FLEX_ATTENTION": True,
            "USE_TMA": False,
        },
    }
    if _device_is_blackwell(device):
        providers["flash_backend"] = {
            **common_options,
            "BACKEND": DEFAULT_FLASH_BACKEND_LITERAL,
            "USE_TMA": True,
        }
    return providers


def mode_supports_cudnn_sdpa(mode: str) -> bool:
    """Return True when cuDNN SDPA can express the workload directly."""
    return mode in {"dense", "causal"}


def best_available_candidate_providers(
    mode: str,
    *,
    include_flash_backend: bool,
) -> tuple[str, ...]:
    """Return stable candidate providers for the local-best benchmark target."""
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {SUPPORTED_MODES}")

    if mode_supports_cudnn_sdpa(mode):
        # Dense/causal absolute-performance runs should stay on the stable
        # local winner; cross-backend comparisons belong in the TFLOPs microbench.
        return ("cudnn_sdpa",)

    candidates: list[str] = []
    if include_flash_backend:
        candidates.append("flash_backend")
    candidates.extend(("flex_tma", "flex_compiled"))
    return tuple(candidates)


def _resolve_cudnn_sdpa_backend() -> Any:
    if sdpa_kernel is None:
        raise RuntimeError(f"sdpa_kernel is unavailable ({_SDPA_IMPORT_ERROR})")
    backend = getattr(SDPBackend, "CUDNN_ATTENTION", None)
    if backend is None:
        backend = getattr(SDPBackend, "CUDNN", None)
    if backend is None:
        raise RuntimeError("PyTorch does not expose a cuDNN SDPA backend on this build")
    return backend


def build_cudnn_sdpa_kernel(inputs: FlashAttention4Inputs) -> FlashAttention4Kernel:
    """Build a cuDNN SDPA candidate for dense/causal workloads."""
    if not mode_supports_cudnn_sdpa(inputs.mode):
        raise ValueError(f"cudnn_sdpa does not support mode {inputs.mode!r}")
    backend = _resolve_cudnn_sdpa_backend()

    def forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode(), sdpa_kernel([backend]):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=None, is_causal=(inputs.mode == "causal"))

    with torch.inference_mode():
        forward(inputs.q, inputs.k, inputs.v)

    return FlashAttention4Kernel(
        fn=forward,
        provider="cudnn_sdpa",
        kernel_options={"sdpa_backend": "cudnn"},
        notes=(),
    )


def compile_flashattention4_provider(
    inputs: FlashAttention4Inputs,
    config: FlashAttention4Config,
    *,
    provider: str,
) -> FlashAttention4Kernel:
    """Compile a specific FA4 lab provider without fallback resolution."""
    provider_options = get_flashattention4_candidate_kernel_options(config, device=inputs.q.device)
    if provider not in provider_options:
        raise ValueError(f"Unsupported provider {provider!r}; expected one of {tuple(provider_options)}")
    return _compile_candidate(
        inputs,
        kernel_options=provider_options[provider],
        provider=provider,
        compile_mode=config.compile_mode,
    )


def select_lowest_latency_provider(candidate_median_ms: dict[str, float]) -> str:
    """Pick the fastest provider from a provider->median_ms mapping."""
    if not candidate_median_ms:
        raise ValueError("candidate_median_ms must not be empty")
    return min(candidate_median_ms.items(), key=lambda item: (item[1], item[0]))[0]


def _run_candidate_kernel(
    kernel: FlashAttention4Kernel,
    inputs: FlashAttention4Inputs,
) -> torch.Tensor:
    with torch.inference_mode():
        return kernel.fn(inputs.q, inputs.k, inputs.v)


def _candidate_matches_reference(
    inputs: FlashAttention4Inputs,
    kernel: FlashAttention4Kernel,
    reference_output: Optional[torch.Tensor],
) -> tuple[bool, Optional[torch.Tensor], str]:
    """Run a one-shot correctness smoke test for an optimized candidate."""
    if reference_output is None:
        with torch.inference_mode():
            reference_output = eager_flex_attention(inputs).float()

    candidate_output = _run_candidate_kernel(kernel, inputs).float()

    if not torch.isfinite(candidate_output).all():
        return False, reference_output, "non-finite output"

    if not torch.allclose(candidate_output, reference_output, atol=0.5, rtol=0.05):
        max_diff = scalar_tensor_to_float((candidate_output - reference_output).abs().max())
        return False, reference_output, f"max_diff={max_diff:.6f}"

    return True, reference_output, "ok"


def _experimental_windowed_skip_reason(
    mode: str,
    candidate_errors: dict[str, str] | list[str],
) -> Optional[str]:
    """Return a skip reason when experimental sliding-window paths are unsupported.

    The local torch 2.9.1 + sm_100 stack is known to produce non-finite outputs
    for some cold-start windowed kernels. Those targets should remain visible as
    explicit probes, but they should skip cleanly when every candidate fails for
    that known stack limitation rather than reporting a misleading lab failure.
    """
    if mode not in {"windowed", "alibi_windowed"}:
        return None

    messages = (
        candidate_errors.values()
        if isinstance(candidate_errors, dict)
        else candidate_errors
    )
    normalized = [str(message).lower() for message in messages]
    if normalized and all("non-finite output" in message for message in normalized):
        return (
            "experimental sliding-window FlashAttention-4 kernels are unstable on this "
            "torch 2.9.1 + sm_100 stack (all candidate providers produced non-finite outputs)."
        )
    return None


def resolve_best_available_attention_kernel(
    inputs: FlashAttention4Inputs,
    config: FlashAttention4Config,
    *,
    selection_warmup: int = 3,
    selection_iterations: int = 6,
) -> FlashAttention4Selection:
    """Measure all valid local backends and return the fastest correct one."""
    provider_options = get_flashattention4_candidate_kernel_options(config, device=inputs.q.device)
    candidate_providers = best_available_candidate_providers(
        config.mode,
        include_flash_backend="flash_backend" in provider_options,
    )

    candidate_median_ms: dict[str, float] = {}
    candidate_errors: dict[str, str] = {}
    reference_output: Optional[torch.Tensor] = None
    built_kernels: dict[str, FlashAttention4Kernel] = {}

    for provider in candidate_providers:
        try:
            if provider == "cudnn_sdpa":
                kernel = build_cudnn_sdpa_kernel(inputs)
            else:
                kernel = compile_flashattention4_provider(inputs, config, provider=provider)

            is_valid, reference_output, message = _candidate_matches_reference(
                inputs,
                kernel,
                reference_output,
            )
            if not is_valid:
                candidate_errors[provider] = f"failed correctness smoke test: {message}"
                continue

            timing = measure_flashattention4_latency(
                lambda: _run_candidate_kernel(kernel, inputs),
                warmup=selection_warmup,
                iterations=selection_iterations,
            )
            candidate_median_ms[provider] = timing.median_ms
            built_kernels[provider] = kernel
        except Exception as exc:  # pragma: no cover - depends on local backend availability
            candidate_errors[provider] = f"{exc.__class__.__name__}: {str(exc).splitlines()[0]}"

    if not candidate_median_ms:
        skip_reason = _experimental_windowed_skip_reason(config.mode, candidate_errors)
        if skip_reason is not None:
            raise RuntimeError(f"SKIPPED: {skip_reason}")
        detail = "\n".join(f"{provider}: {message}" for provider, message in candidate_errors.items())
        raise RuntimeError(f"Failed to resolve any best-available attention backend:\n{detail}")

    winner = select_lowest_latency_provider(candidate_median_ms)
    winner_kernel = built_kernels[winner]
    if candidate_errors:
        winner_kernel.notes = winner_kernel.notes + tuple(
            f"{provider}: {message}" for provider, message in candidate_errors.items()
        )
    return FlashAttention4Selection(
        kernel=winner_kernel,
        candidate_median_ms=candidate_median_ms,
        candidate_errors=candidate_errors,
    )


def resolve_flashattention4_kernel(
    inputs: FlashAttention4Inputs,
    config: FlashAttention4Config,
) -> FlashAttention4Kernel:
    """Resolve the best compiled provider for the current runtime."""
    candidates: list[tuple[str, dict[str, Any]]] = []
    provider_options = get_flashattention4_candidate_kernel_options(config, device=inputs.q.device)

    if config.backend in {"auto", "flash"} and "flash_backend" in provider_options:
        candidates.append(("flash_backend", provider_options["flash_backend"]))
    if config.backend in {"auto", "tma", "flash"}:
        candidates.append(("flex_tma", provider_options["flex_tma"]))
    candidates.append(("flex_compiled", provider_options["flex_compiled"]))

    errors: list[str] = []
    reference_output: Optional[torch.Tensor] = None
    for provider, kernel_options in candidates:
        try:
            kernel = _compile_candidate(
                inputs,
                kernel_options=kernel_options,
                provider=provider,
                compile_mode=config.compile_mode,
                notes=tuple(errors),
            )
            is_valid, reference_output, message = _candidate_matches_reference(
                inputs,
                kernel,
                reference_output,
            )
            if is_valid:
                return kernel
            errors.append(f"{provider} failed correctness smoke test: {message}")
        except Exception as exc:  # pragma: no cover - exercised on toolchain mismatch
            errors.append(f"{provider} failed: {exc.__class__.__name__}: {str(exc).splitlines()[0]}")

    detail = "\n".join(errors) if errors else "no providers attempted"
    skip_reason = _experimental_windowed_skip_reason(config.mode, errors)
    if skip_reason is not None:
        raise RuntimeError(f"SKIPPED: {skip_reason}")
    if config.mode in {"windowed", "alibi_windowed"}:
        detail = (
            f"{detail}\n"
            "note: sliding-window variants are exposed for experimentation, but this "
            "torch 2.9.1 + sm_100 build can produce non-finite outputs on cold-start "
            "compilation for windowed score/mask combinations."
        )
    raise RuntimeError(f"Failed to resolve a FlashAttention-4 kernel:\n{detail}")
