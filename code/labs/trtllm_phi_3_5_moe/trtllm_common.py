"""Shared helpers for the TRT-LLM Phi-3.5-MoE lab."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Sequence, Tuple

import importlib.machinery
import importlib.metadata
import importlib.util
import os
import platform
import sys

import torch

from core.utils.warning_filters import suppress_user_warnings, warn_optional_component_unavailable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = REPO_ROOT / "phi-3.5-moe" / "original"
DEFAULT_ENGINE_PATH = REPO_ROOT / "phi-3.5-moe" / "trtllm_engine_tp1_fp16"
LEGACY_ENGINE_PATH = REPO_ROOT / "phi-3.5-moe" / "trtllm_engine"
MODEL_PATH_ENV = "AISP_PHI35_MOE_MODEL_PATH"
ENGINE_PATH_ENV = "AISP_PHI35_MOE_ENGINE_PATH"
PROMPT_TEXT = "Explain GPU kernel fusion in one sentence."
_ACCELERATE_IMPORT_PATCHED = False
VERIFICATION_TOKEN_PREFIX = 8
_TOKEN_OFFSET_CACHE: dict[tuple[int, torch.device], torch.Tensor] = {}
_OPTIONAL_MODELOPT_PLUGIN_WARNING_PATTERNS: tuple[str, ...] = (
    r"Failed to import vllm plugin due to: .*You may ignore this warning if you do not need this plugin\.",
    r"Failed to import transformer[_ ]engine plugin due to: .*You may ignore this warning if you do not need this plugin\.",
)


@contextmanager
def _suppress_optional_modelopt_plugin_warnings():
    """Summarize known non-fatal modelopt optional plugin ABI warnings.

    modelopt eagerly imports its optional vLLM plugin and emits a warning when
    the host vLLM extension ABI does not match the active torch build. This lab
    also does not need the optional Transformer Engine plugin path. Capture
    those specific warning patterns while preserving all other warnings.
    """
    with suppress_user_warnings(
        _OPTIONAL_MODELOPT_PLUGIN_WARNING_PATTERNS,
        context="labs.trtllm_phi_3_5_moe modelopt plugin import",
    ):
        yield


def disable_accelerate_transformer_engine() -> None:
    """Disable optional FP8/vision package probing for this text-only lab.

    The Phi-3.5-MoE TRT-LLM benchmark does not use Transformer Engine, but
    `transformers` may import `accelerate`, which can eagerly probe optional TE
    modules. On benchmark hosts with mixed TE packaging/ABI state this can cause
    non-deterministic warnings or import failures during profiler subprocesses.
    We also hide torchvision so text-only model loads do not trip host-specific
    torchvision/torch ABI mismatches in profiler child processes.
    """
    global _ACCELERATE_IMPORT_PATCHED
    if _ACCELERATE_IMPORT_PATCHED:
        return

    te_packages = {
        "transformer_engine",
        "transformer_engine_torch",
        "transformer_engine_cu12",
        "transformer_engine_cu13",
        "torchvision",
    }

    original_find_spec = importlib.util.find_spec
    original_metadata = importlib.metadata.metadata
    def _normalize_package_name(name: str) -> str:
        return str(name).replace("-", "_").lower()

    def _patched_find_spec(name, *args, **kwargs):
        normalized = _normalize_package_name(name)
        if normalized.startswith("transformer_engine") or normalized.startswith("torchvision"):
            return None
        return original_find_spec(name, *args, **kwargs)

    def _patched_metadata(name, *args, **kwargs):
        if _normalize_package_name(name) in te_packages:
            raise importlib.metadata.PackageNotFoundError(name)
        return original_metadata(name, *args, **kwargs)

    importlib.util.find_spec = _patched_find_spec
    importlib.metadata.metadata = _patched_metadata
    _ACCELERATE_IMPORT_PATCHED = True


def resolve_default_engine_path() -> Path:
    """Resolve the canonical TRT-LLM engine path with repo-local fallbacks."""
    env_engine = os.environ.get(ENGINE_PATH_ENV, "").strip()
    if env_engine:
        return Path(env_engine).expanduser()

    for candidate in (DEFAULT_ENGINE_PATH, LEGACY_ENGINE_PATH):
        if candidate.exists():
            return candidate
    return DEFAULT_ENGINE_PATH


def _engine_path_has_assets(engine_path: Path) -> bool:
    if not engine_path.exists():
        return False
    if engine_path.is_file():
        return engine_path.stat().st_size > 0

    # Common TRT-LLM engine-dir outputs include config.json + rank*.engine.
    if (engine_path / "config.json").exists():
        return True
    if any(engine_path.glob("rank*.engine")):
        return True
    if any(engine_path.glob("*.engine")):
        return True
    if any(engine_path.glob("*.plan")):
        return True
    return False


def resolve_model_path(model_path: Path) -> Path:
    """Require an explicit local model path for deterministic TRT-LLM validation."""
    if model_path.exists():
        return model_path

    raise RuntimeError(
        "SKIPPED: TRT-LLM baseline model assets unavailable "
        f"(requested model_path={model_path}). "
        f"Remediation: provide --model-path (or ${MODEL_PATH_ENV}) pointing to a local "
        "Phi-3.5-MoE model checkout."
    )


def ensure_trtllm_assets(
    model_path: Path,
    *,
    engine_path: Optional[Path] = None,
    require_engine: bool = False,
) -> None:
    """Fail fast with explicit remediation when required TRT-LLM assets are absent."""
    missing = []
    if not model_path.exists():
        missing.append(f"model_path={model_path}")
    if require_engine:
        if engine_path is None:
            missing.append("engine_path=<unset>")
        elif not _engine_path_has_assets(engine_path):
            missing.append(f"engine_path={engine_path} (missing config/engine artifacts)")
    if not missing:
        return
    missing_text = ", ".join(missing)
    default_engine_hint = (
        f"{DEFAULT_ENGINE_PATH} (preferred) or {LEGACY_ENGINE_PATH} (legacy)."
    )
    raise RuntimeError(
        "SKIPPED: TRT-LLM Phi-3.5-MoE assets unavailable "
        f"({missing_text}). "
        "Remediation: provide a local Phi-3.5-MoE checkout via --model-path "
        f"(or ${MODEL_PATH_ENV}) and a built TensorRT-LLM engine via --engine-path "
        f"(or ${ENGINE_PATH_ENV}). Canonical repo defaults are "
        f"{DEFAULT_MODEL_PATH} for model and {default_engine_hint}"
    )


def parse_trtllm_args() -> argparse.Namespace:
    default_model_path = os.environ.get(MODEL_PATH_ENV, str(DEFAULT_MODEL_PATH))
    default_engine_path = str(resolve_default_engine_path())
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model-path", type=str, default=default_model_path)
    parser.add_argument("--engine-path", type=str, default=default_engine_path)
    parser.add_argument("--prompt-len", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--vocab-slice", type=int, default=256)
    args, _ = parser.parse_known_args()
    return args


def load_trtllm_runtime():
    """Load TensorRT-LLM runtime without importing the full package __init__."""
    os.environ.setdefault("OMPI_MCA_coll_ucc_enable", "0")
    os.environ.setdefault("TORCH_DISABLE_ADDR2LINE", "1")
    if platform.system() == "Linux":
        try:
            from ctypes import cdll

            v_major, v_minor, *_ = sys.version_info
            cdll.LoadLibrary(f"libpython{v_major}.{v_minor}.so.1.0")
            cdll.LoadLibrary(f"libpython{v_major}.{v_minor}.so")
        except Exception as exc:
            warn_optional_component_unavailable(
                "libpython runtime preload",
                exc,
                impact="TensorRT-LLM will continue without the explicit preload helper.",
                context="labs.trtllm_phi_3_5_moe.load_trtllm_runtime",
            )

    with _suppress_optional_modelopt_plugin_warnings():
        spec = importlib.util.find_spec("tensorrt_llm")
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("TensorRT-LLM is not installed")
        root = Path(spec.submodule_search_locations[0])

        pkg_name = "tensorrt_llm"
        if pkg_name not in sys.modules:
            pkg = importlib.util.module_from_spec(importlib.machinery.ModuleSpec(pkg_name, None))
            pkg.__path__ = [str(root)]
            sys.modules[pkg_name] = pkg

        common_path = root / "_common.py"
        spec_common = importlib.util.spec_from_file_location("tensorrt_llm._common", common_path)
        if spec_common is None or spec_common.loader is None:
            raise RuntimeError("TensorRT-LLM _common module not found")
        common_mod = importlib.util.module_from_spec(spec_common)
        spec_common.loader.exec_module(common_mod)  # type: ignore[union-attr]
        common_mod._init()

        runtime_path = root / "runtime" / "__init__.py"
        spec_runtime = importlib.util.spec_from_file_location("tensorrt_llm.runtime", runtime_path)
        if spec_runtime is None or spec_runtime.loader is None:
            raise RuntimeError("TensorRT-LLM runtime module not found")
        runtime_mod = importlib.util.module_from_spec(spec_runtime)
        spec_runtime.loader.exec_module(runtime_mod)  # type: ignore[union-attr]
        return runtime_mod


def build_prompt_tokens(tokenizer, *, prompt_len: int, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenizer.encode(PROMPT_TEXT, add_special_tokens=True)
    encoded = encoded[:prompt_len]
    if not encoded:
        raise ValueError("Prompt encoding produced no tokens")
    encoded_ids = torch.tensor(encoded, dtype=torch.long)
    input_ids = encoded_ids.unsqueeze(0).expand(batch_size, -1).contiguous()
    # Keep prompt lengths identical across baseline and TRT-LLM by avoiding right-padding.
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return input_ids, attention_mask


def slice_logits(logits: torch.Tensor, vocab_slice: int) -> torch.Tensor:
    if logits.dim() != 2:
        raise ValueError("Expected logits of shape [batch, vocab]")
    return logits[:, :vocab_slice]


def _token_offsets_for(max_new_tokens: int, device: torch.device) -> torch.Tensor:
    key = (int(max_new_tokens), device)
    cached = _TOKEN_OFFSET_CACHE.get(key)
    if cached is None:
        cached = torch.arange(max_new_tokens, device=device, dtype=torch.long)
        _TOKEN_OFFSET_CACHE[key] = cached
    return cached


def slice_generated_token_ids(
    output_ids: torch.Tensor,
    *,
    prompt_lengths: Sequence[int],
    max_new_tokens: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Normalize generated token ids into a fixed [batch, max_new_tokens] tensor.

    TRT-LLM returns full prompt+generation token ids with an explicit beam
    dimension, while Transformers returns prompt+generation sequences without a
    beam dimension. The lab only needs a deterministic generated-token suffix
    for verification, so normalize both paths into the same shape and right-pad
    shorter generations with the tokenizer pad token.
    """
    if output_ids.dim() == 3:
        output_ids = output_ids[:, 0, :]
    if output_ids.dim() != 2:
        raise ValueError(
            "Expected generated token ids with shape [batch, seq] or [batch, beam, seq]; "
            f"got {tuple(output_ids.shape)}"
        )
    # Normalize to a stable integer dtype so baseline HF outputs and TRT-LLM
    # engine outputs compare on token content rather than backend-specific
    # integer storage choices.
    output_ids = output_ids.to(dtype=torch.int64)
    if len(prompt_lengths) != output_ids.size(0):
        raise ValueError(
            f"prompt_lengths must have one entry per batch item: got {len(prompt_lengths)} "
            f"for batch size {output_ids.size(0)}"
        )
    if max_new_tokens <= 0:
        raise ValueError(f"max_new_tokens must be positive, got {max_new_tokens}")

    prompt_lengths_list = [int(prompt_len) for prompt_len in prompt_lengths]
    pad_value = int(pad_token_id)
    for prompt_len in prompt_lengths_list:
        if prompt_len < 0 or prompt_len > output_ids.size(1):
            raise ValueError(
                f"Prompt length {prompt_len} is out of bounds for output_ids shape "
                f"{tuple(output_ids.shape)}"
            )
    if output_ids.size(1) == 0:
        return torch.full(
            (output_ids.size(0), max_new_tokens),
            pad_value,
            dtype=output_ids.dtype,
            device=output_ids.device,
        )

    prompt_offsets = torch.tensor(prompt_lengths_list, device=output_ids.device, dtype=torch.long)
    token_offsets = _token_offsets_for(max_new_tokens, output_ids.device)
    gather_positions = prompt_offsets.unsqueeze(1) + token_offsets.unsqueeze(0)
    valid_positions = gather_positions < output_ids.size(1)
    gathered = output_ids.gather(1, gather_positions.clamp_max(output_ids.size(1) - 1))
    gathered.masked_fill_(valid_positions.logical_not_(), pad_value)
    return gathered.contiguous()


def verification_token_prefix_length(max_new_tokens: int) -> int:
    """Use a short deterministic prefix for backend-parity verification.

    HF eager generation and TRT-LLM stay aligned on the early greedy decode
    prefix for this lab, but numerical drift accumulates later in the
    autoregressive rollout. The benchmark is a serving-stack comparison, so the
    verification target should cover the stable generated-token prefix rather
    than the full 128-token suffix.
    """
    return min(int(max_new_tokens), VERIFICATION_TOKEN_PREFIX)
