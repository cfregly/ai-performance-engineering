"""Host-only source contracts for Chapter 12 CUDA graph lifetime rules.

Actual CUDA compilation and runtime acceptance remain in tests/cuda.
"""

from pathlib import Path

CODE = Path(__file__).resolve().parents[1]


def source(path: str) -> str:
    return (CODE / path).read_text()


def test_conditional_graph_populates_cuda_owned_if_else_bodies_after_node_creation() -> None:
    text = source("ch12/optimized_graph_conditional_runtime.cu")
    compact = "".join(text.split())

    assert "#if CUDA_VERSION >= 12040" in text
    assert "#if CUDART_VERSION >= 13000" in text
    assert "cudaGraphCondTypeWhile" not in text
    assert "cond_node_params.conditional.type=cudaGraphCondTypeIf;" in compact
    assert "cond_node_params.conditional.size=2;" in compact
    assert ".conditional.phGraph_out=" not in compact
    assert "cudaGraphDestroy(if_body)" not in text

    add_conditional = compact.index("cudaGraphAddNode(&cond_node,graph,")
    get_body = compact.index(
        "cudaGraph_tif_body=cond_node_params.conditional.phGraph_out[0];"
    )
    get_else_body = compact.index(
        "cudaGraph_telse_body=cond_node_params.conditional.phGraph_out[1];"
    )
    populate_body = compact.index(
        "cudaGraphAddKernelNode(&expensive_node,if_body,nullptr,0,&expensive_params)"
    )
    populate_else_body = compact.index(
        "cudaGraphAddKernelNode(&cheap_node,else_body,nullptr,0,&cheap_params)"
    )
    assert add_conditional < get_body < get_else_body < populate_body < populate_else_body


def test_cuda_graph_cache_keys_capture_on_the_tensor_data_pointer() -> None:
    text = source("ch12/cuda_extensions/cuda_graphs_kernels.cu")
    graph_replay = text.split("void graph_replay", 1)[1].split("PYBIND11_MODULE", 1)[0]
    compact = "".join(graph_replay.split())

    assert "float*data_ptr=nullptr;" in "".join(
        text.split("struct GraphCache", 1)[1].split("};", 1)[0].split()
    )
    assert "float*constdata_ptr=data.data_ptr<float>();" in compact
    assert "(g_graph_cache.data_ptr!=data_ptr)" in compact
    assert compact.count(">(data_ptr,n);") == 3
    assert "g_graph_cache.data_ptr=data_ptr;" in compact
