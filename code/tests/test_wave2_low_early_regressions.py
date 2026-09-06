"""Focused host-side regressions for Wave 2 low findings W2-097 through W2-110."""

from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from ch02 import cpu_gpu_topology_aware
from ch04 import dist_allreduce

CODE = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (CODE / path).read_text(encoding="utf-8")


class _FakeTensor:
    def __getitem__(self, _index: object) -> _FakeTensor:
        return self

    def item(self) -> float:
        return 2.0


def test_w2_097_fp32_fusion_uses_strict_output_tolerance() -> None:
    text = source("ch01/optimized_performance_fusion.py")
    assert "output_tolerance=(1e-4, 1e-5)" in text
    assert "output_tolerance=(0.5, 0.5)" not in text


def _patch_arm_cpu_probe(monkeypatch: pytest.MonkeyPatch, cpuinfo: str) -> None:
    monkeypatch.setattr(cpu_gpu_topology_aware.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(cpu_gpu_topology_aware.platform, "processor", lambda: "")
    monkeypatch.setattr(
        cpu_gpu_topology_aware.psutil,
        "cpu_count",
        lambda logical=True: 16 if logical else 8,
    )
    monkeypatch.setattr(
        cpu_gpu_topology_aware.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=64 * 1024**3),
    )
    monkeypatch.setattr(
        cpu_gpu_topology_aware.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError()),
    )
    monkeypatch.setattr("builtins.open", lambda *_args, **_kwargs: io.StringIO(cpuinfo))


def test_w2_098_generic_neoverse_cpu_is_not_classified_as_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_arm_cpu_probe(monkeypatch, "model name : ARM Neoverse V2\n")

    info = cpu_gpu_topology_aware.detect_cpu_info()

    assert info["cpu_type"] == "ARM Neoverse"
    assert info["is_grace"] is False


def test_w2_098_explicit_nvidia_grace_cpu_is_classified_as_grace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_arm_cpu_probe(monkeypatch, "model name : NVIDIA Grace CPU\n")

    info = cpu_gpu_topology_aware.detect_cpu_info()

    assert info["cpu_type"] == "NVIDIA Grace"
    assert info["is_grace"] is True
    interconnect = cpu_gpu_topology_aware.detect_interconnect_type(
        info,
        {"family": "Blackwell", "nvlink_capable": False},
    )
    assert interconnect == "NVLink-C2C (detected; bandwidth unmeasured)"
    assert "900 GB/s" not in interconnect


def test_w2_099_nccl_allows_one_local_gpu_in_a_multi_node_job(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {"set_device": [], "init": None, "tensor_device": None}
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setattr(
        dist_allreduce.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(data_size=16, backend="nccl"),
    )
    monkeypatch.setattr(dist_allreduce.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        dist_allreduce.torch.cuda,
        "set_device",
        lambda device_id: captured["set_device"].append(device_id),
    )
    monkeypatch.setattr(
        dist_allreduce.dist,
        "init_process_group",
        lambda **kwargs: captured.__setitem__("init", kwargs),
    )
    monkeypatch.setattr(dist_allreduce.dist, "get_rank", lambda: 1)
    monkeypatch.setattr(dist_allreduce.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist_allreduce.dist, "barrier", lambda: None)
    monkeypatch.setattr(dist_allreduce.dist, "all_reduce", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dist_allreduce.dist, "destroy_process_group", lambda: None)
    monkeypatch.setattr(
        dist_allreduce.torch,
        "ones",
        lambda *_args, **kwargs: captured.__setitem__("tensor_device", kwargs["device"])
        or _FakeTensor(),
    )

    dist_allreduce.main()

    assert captured["set_device"] == [0]
    assert str(captured["tensor_device"]) == "cuda:0"
    assert captured["init"] == {
        "backend": "nccl",
        "init_method": "env://",
        "device_id": 0,
    }


def test_w2_100_through_w2_102_tma_copies_compare_nonuniform_full_outputs() -> None:
    async_copy = source("ch07/async_prefetch_2d_demo.cu")
    bulk_copy = source("ch07/optimized_tma_bulk_tensor_2d.cu")
    optimized_copy = source("ch07/optimized_tma_copy.cu")

    assert "row * 131 + col * 17" in async_copy
    assert "h_out[idx] != h_in[idx]" in async_copy
    assert "h_out[idx] != 1.0f" not in async_copy

    assert "h_dst[idx] != h_src[idx]" in bulk_copy
    assert "return copy_ok ? 0 : 2;" in bulk_copy

    assert "idx < h_matrix_out.size()" in optimized_copy
    assert "tiled_neighbor_reference_at(h_matrix, M, N, idx)" in optimized_copy
    assert "h_matrix_out[idx] != expected" in optimized_copy
    assert "h_matrix_out[idx] != h_matrix[idx]" not in optimized_copy
    assert "2D TMA tiled-neighbor mismatch at %zu" in optimized_copy
    assert "2D TMA validation failed" in optimized_copy
    assert "std::exit(EXIT_FAILURE);" in optimized_copy


def test_w2_103_through_w2_106_cuda_descriptions_match_the_code() -> None:
    cublaslt = source("ch09/optimized_cublaslt_gemm.cu")
    cublaslt_fp4_header = source("ch09/optimized_cublaslt_gemm_fp4.cu").split(
        "#include",
        1,
    )[0]
    cutlass_fp16_header = source("ch09/optimized_cutlass_gemm_fp16.cu").split(
        "#include",
        1,
    )[0]
    tiled = source("ch09/optimized_micro_tiling_matmul.cu")
    readme = source("ch09/README.md")
    readme_generator = source("core/scripts/refresh_readmes.py")

    assert "CUBLASLT_ALGO_CAP_PROGRAMMATIC_DEPENDENT_LAUNCH" not in cublaslt
    assert "One strided-batched TN" in cublaslt_fp4_header
    assert "single-matrix" not in cublaslt_fp4_header
    assert "LayoutB=RowMajor" in cutlass_fp16_header
    assert "LayoutB=ColumnMajor" not in cutlass_fp16_header
    assert "16x16 shared-memory tiled matmul" in tiled
    assert "Register-tiled" not in tiled
    honest_description = (
        "Naive and 16x16 shared-memory tiled matmuls with per-thread scalar accumulation."
    )
    assert honest_description in readme
    assert honest_description in readme_generator


def test_w2_107_flash_attention_uses_its_explicit_col_row_descriptor_builder() -> None:
    text = source("ch10/optimized_flash_attn_tma_micro_pipeline.cu")
    assert "inline bool make_2d_tensor_map_col_row(" in text
    assert "std::uint64_t dims[rank]" in text
    assert "static_cast<std::uint64_t>(width)" in text
    assert "static_cast<std::uint64_t>(height)" in text
    assert text.count("make_2d_tensor_map_col_row(") >= 3


def test_w2_108_tma_pipeline_accounts_for_barriers_and_partial_stores() -> None:
    text = source("ch10/tma_2d_pipeline_blackwell.cu")
    compact = "".join(text.split())
    assert "barrier_arrive_tx(bar,1,BYTES_PER_CHUNK)" in compact
    assert "constboolcan_use_tma_store=full_columns&&full_rows;" in compact
    assert "if(can_use_tma_store)" in compact
    assert "h_verify" in text


def test_w2_109_multicast_gemm_uses_row_major_b_descriptor_and_indexing() -> None:
    text = source("ch10/tma_multicast_cluster.cu")
    compact = "".join(text.split())
    assert "B[global_k*N+global_n]" in compact
    assert "/*width=*/N" in text
    assert "/*height=*/K" in text
    assert "/*ld=*/N" in text
    assert "/*box_width=*/TILE_N" in text
    assert "/*box_height=*/TILE_K" in text


def test_w2_110_cuda_graph_autofree_flag_is_not_macro_shadowed() -> None:
    text = source("ch12/cuda_extensions/cuda_graphs_kernels.cu")
    compact = "".join(text.split())
    assert "#ifndefcudaGraphInstantiateFlagAutoFreeOnLaunch" not in compact
    assert "#definecudaGraphInstantiateFlagAutoFreeOnLaunch" not in compact
    assert "#ifCUDART_VERSION>=12000" in compact
    assert "cudaGraphInstantiateFlagAutoFreeOnLaunch" in text
