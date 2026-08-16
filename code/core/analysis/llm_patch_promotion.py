from __future__ import annotations

import hashlib
import math
import re
import shutil
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from core.utils.logger import get_logger

logger = get_logger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[2]


def _sanitize_llm_variant_name(variant_name: str) -> str:
    """Convert LLM variant names into safe filename fragments."""
    safe = re.sub(r"[^a-zA-Z0-9_]+", "_", variant_name or "")
    safe = re.sub(r"_+", "_", safe).strip("_").lower()
    return safe


def is_verified_llm_patch(patch: dict[str, Any]) -> bool:
    """Return true only when a patch has an explicit passing verification record."""
    verification = patch.get("verification")
    return isinstance(verification, dict) and verification.get("verified") is True


def _positive_finite_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed) or parsed <= 0:
        return None
    return parsed


def is_improving_llm_patch(patch: dict[str, Any]) -> bool:
    """Return true only for an explicitly measured improvement over the baseline."""
    speedup = _positive_finite_float(patch.get("actual_speedup"))
    return speedup is not None and speedup > 1.0


def has_promotable_execution_evidence(patch: dict[str, Any]) -> bool:
    """Return true only for evidence produced by a hardened OS sandbox backend."""
    verification = patch.get("verification")
    verification_details = verification.get("details") if isinstance(verification, dict) else None
    verification_policy = (
        verification_details.get("execution_policy")
        if isinstance(verification_details, dict)
        else None
    )
    worker_attestation = (
        verification_details.get("worker_attestation")
        if isinstance(verification_details, dict)
        else None
    )
    timing = patch.get("rebenchmark_result")
    timing_result = timing if isinstance(timing, dict) else {}
    timing_policy = timing_result.get("execution_policy")
    verification_digest = (
        worker_attestation.get("candidate_sha256") if isinstance(worker_attestation, dict) else None
    )
    timing_digest = timing_result.get("candidate_sha256")
    patched_file = patch.get("patched_file")
    if (
        not isinstance(verification_digest, str)
        or not verification_digest
        or timing_digest != verification_digest
        or not isinstance(patched_file, str)
        or not patched_file
    ):
        return False
    try:
        current_digest_matches = _sha256(Path(patched_file)) == verification_digest
    except OSError:
        return False
    return bool(
        patch.get("promotable") is True
        and isinstance(verification, dict)
        and verification.get("promotable") is True
        and isinstance(verification_policy, dict)
        and verification_policy.get("hardened_os_sandbox") is True
        and verification_policy.get("promotable") is True
        and isinstance(verification_policy.get("sandbox_backend"), str)
        and verification_policy.get("sandbox_backend")
        and isinstance(timing_policy, dict)
        and timing_policy == verification_policy
        and timing_result.get("promotable") is True
        and current_digest_matches
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _has_promotion_approval(
    patched_path: Path,
    *,
    manual_approval: bool,
    campaign_gate_attestation: Mapping[str, Any] | None,
) -> bool:
    if manual_approval is True:
        return True
    if not isinstance(campaign_gate_attestation, Mapping):
        return False
    if (
        campaign_gate_attestation.get("attestation_type") != "campaign_gate"
        or campaign_gate_attestation.get("approved") is not True
    ):
        return False
    expected_digest = campaign_gate_attestation.get("candidate_sha256")
    if not isinstance(expected_digest, str) or not expected_digest:
        return False
    try:
        return _sha256(patched_path) == expected_digest
    except OSError:
        return False


def select_best_verified_llm_patch(
    patches: list[dict[str, Any]],
    baseline_time_ms: float,
    incumbent_time_ms: float,
) -> dict[str, Any] | None:
    """Select the fastest verified patch that improves on the current incumbent."""
    baseline_time = _positive_finite_float(baseline_time_ms)
    incumbent_time = _positive_finite_float(incumbent_time_ms)
    if baseline_time is None or incumbent_time is None:
        return None
    eligible: list[dict[str, Any]] = []
    for patch in patches:
        rebenchmark_result = patch.get("rebenchmark_result")
        if (
            not isinstance(rebenchmark_result, dict)
            or rebenchmark_result.get("success") is not True
        ):
            continue
        if not is_verified_llm_patch(patch):
            continue
        patch_time = _positive_finite_float(rebenchmark_result.get("median_ms"))
        if patch_time is None:
            continue
        patch["actual_speedup"] = baseline_time / patch_time
        patch["incumbent_speedup"] = incumbent_time / patch_time
        if is_improving_llm_patch(patch) and patch["incumbent_speedup"] > 1.0:
            eligible.append(patch)
    if not eligible:
        return None
    return max(eligible, key=lambda patch: patch.get("actual_speedup", 0))


def promote_best_llm_patch(
    best_patch: dict[str, Any],
    benchmark_result: dict[str, Any],
    chapter_dir: Path,
    *,
    manual_approval: bool = False,
    campaign_gate_attestation: Mapping[str, Any] | None = None,
) -> str | None:
    """Copy a reviewed candidate only after explicit manual or campaign approval."""
    patched_file = best_patch.get("patched_file")
    if not patched_file:
        return None

    patched_path = Path(patched_file)
    if patched_path.is_symlink() or not patched_path.is_file():
        logger.warning("    WARNING: Best patch file missing: %s", patched_file)
        return None

    if not is_verified_llm_patch(best_patch):
        logger.warning(
            "    WARNING: Best patch lacks explicit passing verification; skipping promotion."
        )
        return None
    if not is_improving_llm_patch(best_patch):
        logger.warning("    WARNING: Best patch is not a measured improvement; skipping promotion.")
        return None
    incumbent_speedup = _positive_finite_float(best_patch.get("incumbent_speedup"))
    if incumbent_speedup is None or incumbent_speedup <= 1.0:
        logger.warning(
            "    WARNING: Best patch did not improve on the current incumbent; skipping promotion."
        )
        return None
    if not has_promotable_execution_evidence(best_patch):
        logger.warning(
            "    WARNING: Patch evidence is not promotable OS-sandbox evidence; skipping promotion."
        )
        return None
    if not _has_promotion_approval(
        patched_path,
        manual_approval=manual_approval,
        campaign_gate_attestation=campaign_gate_attestation,
    ):
        logger.warning(
            "    WARNING: Explicit manual approval or campaign-gate attestation is required."
        )
        return None

    example_name = benchmark_result.get("example", "unknown")
    variant_name = _sanitize_llm_variant_name(best_patch.get("variant_name", ""))
    if variant_name:
        stem = f"optimized_{example_name}_llm_{variant_name}"
    else:
        stem = f"optimized_{example_name}_llm_best"

    target_path = chapter_dir / f"{stem}{patched_path.suffix}"

    if target_path.exists():
        try:
            if target_path.read_bytes() == patched_path.read_bytes():
                return _relative_path(target_path)
        except OSError as exc:
            logger.warning("    WARNING: Unable to read existing promoted patch: %s", exc)

        for idx in range(2, 100):
            candidate = chapter_dir / f"{stem}_v{idx}{patched_path.suffix}"
            if not candidate.exists():
                target_path = candidate
                break
        else:
            logger.warning(
                "    WARNING: No available filename to promote best patch for %s",
                example_name,
            )
            return None

    shutil.copy2(patched_path, target_path)
    logger.info("    Promoted best patch to %s", target_path.name)

    return _relative_path(target_path)


def _relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)
