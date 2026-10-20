"""W1-031 regression: the FP4 decode lab must run Transformer Engine NVFP4.

Transformer Engine recipes are pydantic dataclasses that drop unknown keyword
arguments. The decode lab used to build
``DelayedScaling(float8_block_scaling=NVFP4BlockScaling())``, which silently
became plain FP8 delayed scaling while the benchmark still reported
``use_fp4=1``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch

from labs.decode_optimization import decode_common
from labs.decode_optimization.baseline_decode_fp4 import get_benchmark as get_baseline_decode_fp4
from labs.decode_optimization.optimized_decode_fp4 import get_benchmark as get_optimized_decode_fp4

CODE_ROOT = Path(__file__).resolve().parents[1]
VENDORED_TE = CODE_ROOT / "third_party" / "TransformerEngine" / "transformer_engine"


def _load_vendored_module(relative_path: str, module_name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, VENDORED_TE / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Pydantic resolves postponed annotations through sys.modules.
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


@pytest.fixture
def vendored_te_recipe() -> Iterator[types.ModuleType]:
    module = _load_vendored_module("common/recipe/__init__.py", "audit_w1031_te_recipe")
    yield module
    sys.modules.pop("audit_w1031_te_recipe", None)


def test_vendored_delayed_scaling_silently_drops_nvfp4_keyword(vendored_te_recipe) -> None:
    """Pin the mechanism: the old construct produced an FP8 recipe, not NVFP4."""
    wrapped = vendored_te_recipe.DelayedScaling(
        float8_block_scaling=vendored_te_recipe.NVFP4BlockScaling()
    )
    assert wrapped.delayed()
    assert not wrapped.nvfp4()


def test_optimized_decode_fp4_entrypoint_selects_nvfp4_recipe(
    monkeypatch: pytest.MonkeyPatch,
    vendored_te_recipe: types.ModuleType,
) -> None:
    """The real FP4 entrypoint must hand TE an NVFP4BlockScaling recipe.

    Only the import and hardware gates are replaced so the constructor's recipe
    selection runs on a host without Transformer Engine. The CUDA test below
    covers the real runtime path.
    """
    monkeypatch.setitem(sys.modules, "transformer_engine.common.recipe", vendored_te_recipe)
    monkeypatch.setattr(decode_common, "TE_AVAILABLE", True)
    monkeypatch.setattr(
        decode_common, "te_constants", types.SimpleNamespace(NVFP4_BLOCK_SCALING_SIZE=16)
    )
    monkeypatch.setattr(decode_common, "_is_blackwell_family", lambda: True)

    bench = get_optimized_decode_fp4()

    recipe = bench.fp8_recipe
    assert type(recipe) is vendored_te_recipe.NVFP4BlockScaling
    assert recipe.nvfp4()
    assert not recipe.delayed()
    assert recipe.fp4_format is vendored_te_recipe.Format.E2M1
    assert bench._fp4_enabled is True
    assert bench._fp8_enabled is False


def _nvfp4_runtime_ready() -> bool:
    if not torch.cuda.is_available() or not decode_common.TE_AVAILABLE:
        return False
    from transformer_engine.pytorch import is_nvfp4_available

    return bool(is_nvfp4_available())


requires_nvfp4_runtime = pytest.mark.skipif(
    not _nvfp4_runtime_ready(),
    reason="SKIPPED: CUDA, Transformer Engine, and NVFP4-capable hardware are required",
)


@requires_nvfp4_runtime
def test_real_gpu_optimized_decode_fp4_quantizes_weights_to_nvfp4() -> None:
    from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Tensor

    torch.manual_seed(0)
    bench = get_optimized_decode_fp4()
    try:
        bench.setup()
        assert bench.fp8_recipe.nvfp4()
        for weight in (
            bench.prefill_mlp.fc1_weight,
            bench.prefill_mlp.fc2_weight,
            bench.decode_mlp.fc1_weight,
            bench.decode_mlp.fc2_weight,
            bench.lm_head.weight,
        ):
            assert isinstance(weight, NVFP4Tensor), type(weight).__name__

        bench.benchmark_fn()
        torch.cuda.synchronize()
        bench.capture_verification_payload()
        assert bench.validate_result() is None
        metrics = bench.get_custom_metrics()
        assert metrics["use_fp4"] == 1.0
        assert metrics["use_fp8"] == 0.0
    finally:
        bench.teardown()


def _summary_output(factory) -> torch.Tensor:
    torch.manual_seed(0)
    bench = factory()
    try:
        bench.setup()
        bench.benchmark_fn()
        torch.cuda.synchronize()
        bench.capture_verification_payload()
        assert bench.output is not None
        return bench.output.detach().float().cpu().clone()
    finally:
        bench.teardown()


@requires_nvfp4_runtime
def test_real_gpu_decode_fp4_output_matches_bf16_baseline_within_declared_tolerance() -> None:
    baseline = _summary_output(get_baseline_decode_fp4)
    optimized = _summary_output(get_optimized_decode_fp4)
    assert torch.isfinite(optimized).all()
    # Same (rtol, atol) the decode benchmark declares in its verification payload.
    torch.testing.assert_close(optimized, baseline, rtol=0.1, atol=1.0)
