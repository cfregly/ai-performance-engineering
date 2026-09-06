"""Regression contracts for NVSHMEM examples that require vendor headers to build."""

from __future__ import annotations

import re
from pathlib import Path


CH04 = Path(__file__).resolve().parents[1] / "ch04"


def _source(name: str) -> str:
    return (CH04 / name).read_text(encoding="utf-8")


def _body_at(source: str, start: int) -> str:
    opening = source.index("{", start)
    depth = 0
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1 : index]
    raise AssertionError("unterminated function body")


def _function_body(source: str, name: str) -> str:
    match = re.search(rf"\b{name}\s*\(", source)
    assert match is not None, f"missing function {name}"
    return _body_at(source, match.start())


def _global_kernel_bodies(source: str) -> dict[str, str]:
    kernels: dict[str, str] = {}
    for match in re.finditer(r"__global__\s+void\s+(\w+)\s*\(", source):
        kernels[match.group(1)] = _body_at(source, match.start())
    return kernels


def test_host_stream_collectives_never_run_inside_cuda_kernels() -> None:
    for filename in (
        "nvshmem_advanced_multigpu.cu",
        "nvshmem_tensor_parallel.cu",
    ):
        kernels = _global_kernel_bodies(_source(filename))
        assert kernels, f"no CUDA kernels found in {filename}"
        offenders = [
            name
            for name, body in kernels.items()
            if "nvshmemx_barrier_all_on_stream" in body
        ]
        assert offenders == [], f"host collective used by kernels: {offenders}"


def test_multi_block_communication_has_host_stream_barriers() -> None:
    advanced = _source("nvshmem_advanced_multigpu.cu")
    for name in (
        "run_ring_reduce_scatter",
        "run_ring_allgather",
        "run_double_buffered_reduce_scatter",
        "benchmark_recursive_halving_doubling",
    ):
        assert "nvshmemx_barrier_all_on_stream" in _function_body(advanced, name)

    tensor = _source("nvshmem_tensor_parallel.cu")
    for name in (
        "nvshmem_allgather",
        "nvshmem_reduce_scatter_ring",
        "nvshmem_allreduce_ring",
    ):
        assert "nvshmemx_barrier_all_on_stream" in _function_body(tensor, name)


def test_pipeline_uses_ordered_ready_and_reuse_handshakes() -> None:
    source = _source("nvshmem_pipeline_patterns.cu")
    pipeline = _function_body(source, "double_buffer_pipeline_demo")

    assert "static __global__ void fill_chunk" in source
    assert "nvshmem_int_wait_until(" not in pipeline
    assert pipeline.count("nvshmemx_int_wait_until_on_stream") == 2
    assert "nvshmemx_int_p_on_stream(flags + buf, 1, stage0, stream)" in pipeline
    assert "nvshmemx_int_p_on_stream(flags + buf, 1, stage1, stream)" in pipeline
    assert "nvshmemx_int_p_on_stream(flags + buf, 0, stage0, stream)" in pipeline

    data_put = pipeline.index("nvshmemx_float_put_on_stream")
    data_quiet = pipeline.index("nvshmemx_quiet_on_stream", data_put)
    ready = pipeline.index(
        "nvshmemx_int_p_on_stream(flags + buf, 1, stage1, stream)", data_quiet
    )
    assert data_put < data_quiet < ready


def test_multinode_reduction_has_collective_team_setup_and_no_racy_atomic() -> None:
    source = _source("nvshmem_multinode_example.cu")
    setup = _function_body(source, "build_node_context")
    reduction = _function_body(source, "hierarchical_reduce")

    assert "for (int node = 0; node < ctx.num_nodes; ++node)" in setup
    assert re.search(r"&config,\s*0L,\s*&candidate", setup)
    assert "nvshmem_float_atomic_add" not in source
    assert "scratch[" not in reduction
    assert "for (int i = 0; i < node_members; ++i)" in reduction
    assert "nvshmem_team_destroy(ctx.node_team)" in source


def test_tensor_parallel_uses_device_linkable_binary16_transport() -> None:
    source = _source("nvshmem_tensor_parallel.cu")
    assert "nvshmem_half_p" not in source
    assert source.count("nvshmem_ushort_p") == 3
    assert source.count("__half_as_ushort") == 3
