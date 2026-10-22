"""PyTorch 2.14.1 compiler and JIT-cache contracts for native extensions."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from labs.fast_cu.build import _build_toolchain_identity, extension_name
from labs.persistent_decode import optimized_tma_prefill_decode, tma_extension

CODE_ROOT = Path(__file__).resolve().parents[1]

TORCH_EXTENSION_LOADERS = (
    "core/benchmark/cutlass_binding.py",
    "core/benchmark/memory_kernels.py",
    "core/profiling/occupancy.py",
    "core/profiling/stream_ordered.py",
    "labs/moe_cuda/decode_kernels.py",
    "labs/nvfp4_group_gemm/custom_cuda_submission.py",
    "labs/software_pipelining/software_pipelining_extension.py",
    "labs/persistent_decode/optimized_tma_prefill_decode.py",
    "labs/persistent_decode/tma_extension.py",
    "labs/top_k_kernel/top_k_kernel_extension.py",
    "labs/fast_cu/b200_epilogue.py",
    "labs/fast_cu/h100_common.py",
    "labs/fast_cu/nvfp4.py",
    "labs/fast_cu/nvfp4_sm100.py",
)

STANDALONE_NATIVE_LOADERS = (
    "core/profiling/nvtx_stub.py",
    "core/verification/tma_cuda_probe.py",
)


def _cxx_standard_flags(relative_path: str) -> list[str]:
    tree = ast.parse((CODE_ROOT / relative_path).read_text())
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and re.fullmatch(r"--?std=c\+\+\d+", node.value)
    ]


@pytest.mark.parametrize("relative_path", TORCH_EXTENSION_LOADERS)
def test_torch_extensions_request_cxx20(relative_path: str) -> None:
    flags = _cxx_standard_flags(relative_path)

    assert flags, f"{relative_path} must explicitly select the PyTorch C++ standard"
    assert set(flags) <= {"-std=c++20", "--std=c++20"}


@pytest.mark.parametrize("relative_path", STANDALONE_NATIVE_LOADERS)
def test_standalone_native_probes_keep_their_cxx17_contract(relative_path: str) -> None:
    flags = _cxx_standard_flags(relative_path)

    assert flags
    assert set(flags) <= {"-std=c++17", "--std=c++17"}


def test_fast_cu_cache_fingerprint_changes_with_torch_or_cuda(tmp_path: Path) -> None:
    source = tmp_path / "extension.cu"
    source.write_text("__global__ void kernel() {}\n")
    manifest = {"revision": "fixed-revision", "files": {"header.cuh": "sha256"}}
    nvcc_path = tmp_path / "cuda" / "bin" / "nvcc"

    def name_for(torch_version: str, cuda_version: str) -> str:
        identity = _build_toolchain_identity(
            torch_version=torch_version,
            torch_cuda_version=cuda_version,
            nvcc_path=nvcc_path,
            nvcc_version="Cuda compilation tools, release 13.0, V13.0.88",
        )
        return extension_name(
            "fast_cu_contract",
            [source],
            ["-std=c++20"],
            [],
            manifest,
            toolchain_identity=identity,
        )

    torch_214 = name_for("2.14.1+cu130", "13.0")

    assert torch_214 == name_for("2.14.1+cu130", "13.0")
    assert torch_214 != name_for("2.15.0+cu130", "13.0")
    assert torch_214 != name_for("2.14.1+cu131", "13.1")


def test_persistent_decode_cache_name_changes_with_torch_or_cuda() -> None:
    name = tma_extension._torch_extension_cache_name(
        "persistent_decode_ext",
        torch_version="2.14.1+cu130",
        cuda_version="13.0",
    )

    assert name.isidentifier()
    assert name == tma_extension._torch_extension_cache_name(
        "persistent_decode_ext",
        torch_version="2.14.1+cu130",
        cuda_version="13.0",
    )
    assert name != tma_extension._torch_extension_cache_name(
        "persistent_decode_ext",
        torch_version="2.15.0+cu130",
        cuda_version="13.0",
    )
    assert name != tma_extension._torch_extension_cache_name(
        "persistent_decode_ext",
        torch_version="2.14.1+cu131",
        cuda_version="13.1",
    )


def test_direct_load_inline_paths_use_fingerprinted_names_and_cxx20(monkeypatch) -> None:
    captured: list[dict] = []

    def fake_load_inline(*args, **kwargs):
        captured.append({"args": args, **kwargs})
        return object()

    monkeypatch.setattr(tma_extension, "load_inline", fake_load_inline)
    monkeypatch.setattr(optimized_tma_prefill_decode, "_TMA_CP_ASYNC_EXT", None)
    monkeypatch.setattr("torch.utils.cpp_extension.load_inline", fake_load_inline)

    tma_extension._try_build_extension()
    optimized_tma_prefill_decode._load_cp_async_tma_ext()

    expected_names = {
        tma_extension._torch_extension_cache_name("persistent_decode_thread_async_copy_ext"),
        tma_extension._torch_extension_cache_name("persistent_decode_tma_cp_async_ext"),
    }
    actual_names = {
        call.get("name", call["args"][0] if call["args"] else None) for call in captured
    }

    assert actual_names == expected_names
    assert all(
        "--std=c++20" in call["extra_cuda_cflags"]
        and "--std=c++17" not in call["extra_cuda_cflags"]
        for call in captured
    )
