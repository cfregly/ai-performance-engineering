from __future__ import annotations

import ast
import hashlib
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
COMMON_PATH = CODE_ROOT / "labs" / "block_scaling" / "block_scaling_common.py"
VENDOR_PATH = (
    CODE_ROOT
    / "labs"
    / "block_scaling"
    / "vendor"
    / "sm100_dense_blockscaled_gemm_persistent.py"
)
EXPECTED_CUTLASS_4_5_2_SHA256 = (
    "98e9974d42e888a27f02f0c52582c218d922b6cd748d78e7e62af19923efd7d8"
)


def _function(tree: ast.Module | ast.ClassDef, name: str) -> ast.FunctionDef:
    return next(
        child
        for child in tree.body
        if isinstance(child, ast.FunctionDef) and child.name == name
    )


def _class(tree: ast.Module, name: str) -> ast.ClassDef:
    return next(
        child
        for child in tree.body
        if isinstance(child, ast.ClassDef) and child.name == name
    )


def _arg_names(function: ast.FunctionDef) -> list[str]:
    return [arg.arg for arg in function.args.args]


def test_sm100_vendor_is_exact_cutlass_4_5_2_source() -> None:
    source = VENDOR_PATH.read_bytes()

    assert hashlib.sha256(source).hexdigest() == EXPECTED_CUTLASS_4_5_2_SHA256
    text = source.decode("utf-8")
    assert "SPDX-License-Identifier: BSD-3-Clause" in text
    assert "cute.arch.ProxyKind" not in text
    assert "cute.make_rmem_tensor" in text


def test_sm100_vendor_exports_production_pointer_contract() -> None:
    tree = ast.parse(VENDOR_PATH.read_text(encoding="utf-8"))
    kernel = _class(tree, "Sm100BlockScaledPersistentDenseGemmKernel")

    assert _arg_names(_function(kernel, "__init__")) == [
        "self",
        "sf_vec_size",
        "mma_tiler_mn",
        "cluster_shape_mn",
    ]
    assert _arg_names(_function(kernel, "__call__")) == [
        "self",
        "a_ptr",
        "b_ptr",
        "sfa_ptr",
        "sfb_ptr",
        "c_ptr",
        "layouts",
        "problem_mnkl",
        "max_active_clusters",
        "stream",
        "epilogue_op",
    ]
    assert _arg_names(_function(kernel, "can_implement")) == [
        "mnkl",
        "ab_dtype",
        "sf_dtype",
        "c_dtype",
        "a_major",
        "b_major",
        "c_major",
        "sf_vec_size",
        "mma_tiler_mn",
        "cluster_shape_mn",
    ]
    assert _arg_names(_function(tree, "cvt_sf_MKL_to_M32x4xrm_K4xrk_L")) == [
        "sf_ref_ptr",
        "sf_mma_ptr",
        "mn",
        "sf_k",
        "l",
        "mma_shape",
    ]
    assert _arg_names(_function(tree, "create_and_reorder_scale_factor_tensor")) == [
        "l",
        "mn",
        "k",
        "sf_vec_size",
        "sf_dtype",
        "torch_tensor",
    ]


def test_common_adapts_release_pointer_abi_without_global_cute_patch() -> None:
    source = COMMON_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    compile_adapter = _function(tree, "_compile_sm100_pointer_kernel")
    invoke = _function(compile_adapter, "invoke")
    calls = [node for node in ast.walk(compile_adapter) if isinstance(node, ast.Call)]

    assert "vendor\" / \"sm100_dense_blockscaled_gemm_persistent.py" in source
    assert "_ensure_cute_fragment_compatibility" not in source
    assert "cute_module.make_fragment" not in source
    assert _arg_names(invoke) == [
        "a_tensor",
        "b_tensor",
        "sfa_tensor",
        "sfb_tensor",
        "c_tensor",
        "stream",
    ]
    scaled_mm_call = next(
        call
        for call in calls
        if isinstance(call.func, ast.Attribute) and call.func.attr == "scaled_mm"
    )
    pointer_call = next(
        call for call in calls if isinstance(call.func, ast.Name) and call.func.id == "pointer_gemm"
    )
    assert len(scaled_mm_call.args) == 9
    assert [(keyword.arg, ast.literal_eval(keyword.value)) for keyword in scaled_mm_call.keywords] == [
        ("options", "--opt-level 2")
    ]
    assert len(pointer_call.args) == 7
