from __future__ import annotations

import hashlib
from pathlib import Path

from core.analysis import llm_patch_promotion


def _resolve_promoted_path(promoted: str) -> Path:
    promoted_path = Path(promoted)
    if promoted_path.is_absolute():
        return promoted_path
    return llm_patch_promotion.REPO_ROOT / promoted_path


def _promotable_patch(patched_file: Path, *, variant_name: str = "fast-path") -> dict:
    candidate_sha256 = hashlib.sha256(patched_file.read_bytes()).hexdigest()
    execution_policy = {
        "sandbox_backend": "test-hardened-backend",
        "hardened_os_sandbox": True,
        "promotable": True,
    }
    return {
        "patched_file": str(patched_file),
        "variant_name": variant_name,
        "verification": {
            "verified": True,
            "errors": [],
            "promotable": True,
            "details": {
                "execution_policy": execution_policy,
                "worker_attestation": {"candidate_sha256": candidate_sha256},
            },
        },
        "rebenchmark_result": {
            "success": True,
            "median_ms": 8.0,
            "promotable": True,
            "execution_policy": execution_policy,
            "candidate_sha256": candidate_sha256,
        },
        "promotable": True,
        "actual_speedup": 1.25,
        "incumbent_speedup": 1.10,
    }


def test_promote_best_llm_patch_creates_file(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "ch10"
    patch_dir = chapter_dir / "llm_patches"
    patch_dir.mkdir(parents=True)
    patched_file = patch_dir / "optimized_toy_patch.py"
    patched_file.write_text("print('patch')\n")

    best_patch = _promotable_patch(patched_file)
    benchmark_result = {"example": "toy"}

    promoted = llm_patch_promotion.promote_best_llm_patch(
        best_patch,
        benchmark_result,
        chapter_dir,
        manual_approval=True,
    )
    assert promoted is not None

    promoted_path = _resolve_promoted_path(promoted)
    assert promoted_path.exists()
    assert promoted_path.name.startswith("optimized_toy_llm_fast_path")


def test_promote_best_llm_patch_reuses_existing(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "ch10"
    patch_dir = chapter_dir / "llm_patches"
    patch_dir.mkdir(parents=True)
    patched_file = patch_dir / "optimized_toy_patch.py"
    patched_file.write_text("print('patch')\n")

    best_patch = _promotable_patch(patched_file)
    benchmark_result = {"example": "toy"}

    promoted_first = llm_patch_promotion.promote_best_llm_patch(
        best_patch,
        benchmark_result,
        chapter_dir,
        manual_approval=True,
    )
    promoted_second = llm_patch_promotion.promote_best_llm_patch(
        best_patch,
        benchmark_result,
        chapter_dir,
        manual_approval=True,
    )

    assert promoted_first is not None
    assert promoted_second is not None
    assert _resolve_promoted_path(promoted_first) == _resolve_promoted_path(promoted_second)
    assert len(list(chapter_dir.glob("optimized_toy_llm_fast_path*.py"))) == 1


def test_promote_best_llm_patch_skips_failed_verification(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "ch10"
    patch_dir = chapter_dir / "llm_patches"
    patch_dir.mkdir(parents=True)
    patched_file = patch_dir / "optimized_toy_patch.py"
    patched_file.write_text("print('patch')\n")

    best_patch = {
        "patched_file": str(patched_file),
        "variant_name": "fast-path",
        "verification": {"verified": False, "errors": ["mismatch"]},
    }
    benchmark_result = {"example": "toy"}

    promoted = llm_patch_promotion.promote_best_llm_patch(best_patch, benchmark_result, chapter_dir)
    assert promoted is None
    assert not list(chapter_dir.glob("optimized_toy_llm_fast_path*.py"))


def test_promote_best_llm_patch_skips_missing_verification(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "ch10"
    patch_dir = chapter_dir / "llm_patches"
    patch_dir.mkdir(parents=True)
    patched_file = patch_dir / "optimized_toy_patch.py"
    patched_file.write_text("print('patch')\n")

    best_patch = {"patched_file": str(patched_file), "variant_name": "fast-path"}
    benchmark_result = {"example": "toy"}

    promoted = llm_patch_promotion.promote_best_llm_patch(best_patch, benchmark_result, chapter_dir)
    assert promoted is None
    assert not list(chapter_dir.glob("optimized_toy_llm_fast_path*.py"))


def test_promote_best_llm_patch_requires_explicit_approval(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "ch10"
    patch_dir = chapter_dir / "llm_patches"
    patch_dir.mkdir(parents=True)
    patched_file = patch_dir / "optimized_toy_patch.py"
    patched_file.write_text("print('patch')\n")
    best_patch = _promotable_patch(patched_file)

    promoted = llm_patch_promotion.promote_best_llm_patch(
        best_patch,
        {"example": "toy"},
        chapter_dir,
    )

    assert promoted is None
    assert patched_file.exists()
    assert not list(chapter_dir.glob("optimized_toy_llm_fast_path*.py"))


def test_promote_best_llm_patch_accepts_matching_campaign_gate(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "ch10"
    patch_dir = chapter_dir / "llm_patches"
    patch_dir.mkdir(parents=True)
    patched_file = patch_dir / "optimized_toy_patch.py"
    patched_file.write_text("print('patch')\n")
    best_patch = _promotable_patch(patched_file)
    candidate_sha256 = hashlib.sha256(patched_file.read_bytes()).hexdigest()

    promoted = llm_patch_promotion.promote_best_llm_patch(
        best_patch,
        {"example": "toy"},
        chapter_dir,
        campaign_gate_attestation={
            "attestation_type": "campaign_gate",
            "approved": True,
            "candidate_sha256": candidate_sha256,
        },
    )

    assert promoted is not None
    assert _resolve_promoted_path(promoted).exists()


def test_promote_best_llm_patch_rejects_non_promotable_evidence(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "ch10"
    patch_dir = chapter_dir / "llm_patches"
    patch_dir.mkdir(parents=True)
    patched_file = patch_dir / "optimized_toy_patch.py"
    patched_file.write_text("print('patch')\n")
    best_patch = _promotable_patch(patched_file)
    best_patch["rebenchmark_result"]["execution_policy"] = {
        "sandbox_backend": None,
        "hardened_os_sandbox": False,
        "promotable": False,
    }

    promoted = llm_patch_promotion.promote_best_llm_patch(
        best_patch,
        {"example": "toy"},
        chapter_dir,
        manual_approval=True,
    )

    assert promoted is None
    assert not list(chapter_dir.glob("optimized_toy_llm_fast_path*.py"))


def test_benchmark_orchestrator_does_not_auto_promote_generated_code() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "core" / "harness" / "run_benchmarks.py"
    ).read_text(encoding="utf-8")

    assert "promote_best_llm_patch(" not in source
    assert "manual_promotion_required" in source


def test_select_best_patch_excludes_a_faster_unverified_candidate() -> None:
    patches = [
        {
            "variant_name": "fast-unverified",
            "rebenchmark_result": {"success": True, "median_ms": 5.0},
        },
        {
            "variant_name": "verified",
            "rebenchmark_result": {"success": True, "median_ms": 8.0},
            "verification": {"verified": True},
        },
    ]

    best = llm_patch_promotion.select_best_verified_llm_patch(
        patches,
        baseline_time_ms=10.0,
        incumbent_time_ms=10.0,
    )

    assert best is not None
    assert best["variant_name"] == "verified"
    assert best["actual_speedup"] == 1.25


def test_select_best_patch_returns_none_without_passing_verification() -> None:
    patches = [
        {
            "variant_name": "unverified",
            "rebenchmark_result": {"success": True, "median_ms": 5.0},
        }
    ]

    best = llm_patch_promotion.select_best_verified_llm_patch(
        patches,
        baseline_time_ms=10.0,
        incumbent_time_ms=10.0,
    )

    assert best is None


def test_select_best_patch_excludes_a_verified_regression() -> None:
    patches = [
        {
            "variant_name": "slower",
            "rebenchmark_result": {"success": True, "median_ms": 20.0},
            "verification": {"verified": True},
        }
    ]

    best = llm_patch_promotion.select_best_verified_llm_patch(
        patches,
        baseline_time_ms=10.0,
        incumbent_time_ms=10.0,
    )

    assert best is None


def test_promote_best_llm_patch_skips_a_verified_regression(tmp_path: Path) -> None:
    chapter_dir = tmp_path / "ch10"
    patch_dir = chapter_dir / "llm_patches"
    patch_dir.mkdir(parents=True)
    patched_file = patch_dir / "optimized_toy_patch.py"
    patched_file.write_text("print('patch')\n")
    best_patch = {
        "patched_file": str(patched_file),
        "variant_name": "slow-path",
        "verification": {"verified": True},
        "actual_speedup": 0.5,
        "incumbent_speedup": 0.5,
    }

    promoted = llm_patch_promotion.promote_best_llm_patch(
        best_patch, {"example": "toy"}, chapter_dir
    )

    assert promoted is None
    assert not list(chapter_dir.glob("optimized_toy_llm_slow_path*.py"))


def test_select_best_patch_rejects_a_candidate_slower_than_the_incumbent() -> None:
    patches = [
        {
            "variant_name": "baseline-win-incumbent-loss",
            "rebenchmark_result": {"success": True, "median_ms": 8.0},
            "verification": {"verified": True},
        }
    ]

    best = llm_patch_promotion.select_best_verified_llm_patch(
        patches,
        baseline_time_ms=10.0,
        incumbent_time_ms=5.0,
    )

    assert best is None
    assert patches[0]["actual_speedup"] == 1.25
    assert patches[0]["incumbent_speedup"] == 0.625


def test_select_best_patch_rejects_a_candidate_slower_than_the_baseline() -> None:
    patches = [
        {
            "variant_name": "incumbent-win-baseline-loss",
            "rebenchmark_result": {"success": True, "median_ms": 15.0},
            "verification": {"verified": True},
        }
    ]

    best = llm_patch_promotion.select_best_verified_llm_patch(
        patches,
        baseline_time_ms=10.0,
        incumbent_time_ms=20.0,
    )

    assert best is None
    assert patches[0]["actual_speedup"] == 10.0 / 15.0
    assert patches[0]["incumbent_speedup"] == 20.0 / 15.0
