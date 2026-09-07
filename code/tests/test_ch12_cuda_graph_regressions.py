"""Host-only source contracts for Chapter 12 CUDA graph lifetime rules.

Actual CUDA compilation and runtime acceptance remain in tests/cuda.
"""

from array import array
from pathlib import Path

import pytest

from ch12.baseline_cuda_graphs_conditional import (
    CONDITIONAL_ELEMENTS,
    BaselineCudaGraphsConditionalBenchmark,
)
from ch12.optimized_cuda_graphs_conditional import OptimizedCudaGraphsConditionalBenchmark
from core.benchmark.cuda_binary_benchmark import BinaryRunResult

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
    get_body = compact.index("cudaGraph_tif_body=cond_node_params.conditional.phGraph_out[0];")
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


@pytest.mark.parametrize(
    "benchmark_type",
    (BaselineCudaGraphsConditionalBenchmark, OptimizedCudaGraphsConditionalBenchmark),
)
def test_conditional_binary_retains_complete_timed_output(benchmark_type, monkeypatch) -> None:
    benchmark = benchmark_type()
    monkeypatch.setattr(benchmark, "_build_binary", lambda verify_mode=False: Path("unused"))
    benchmark.setup()

    def fake_run() -> BinaryRunResult:
        with benchmark._output_path.open("wb") as output_file:
            array("f", [1.25] * CONDITIONAL_ELEMENTS).tofile(output_file)
        return BinaryRunResult(1.0, f"OUTPUT_DUMPED: {CONDITIONAL_ELEMENTS}\n", "")

    monkeypatch.setattr(benchmark, "_run_once", fake_run)
    benchmark.benchmark_fn()
    verified = benchmark.get_verify_output()
    assert verified.shape == (CONDITIONAL_ELEMENTS,)
    assert verified[0].item() == 1.25 and verified[-1].item() == 1.25
    assert verified.data_ptr() != benchmark.output.data_ptr()
    metadata = benchmark.get_workload_metadata()
    assert metadata.custom_units_per_iteration == float(CONDITIONAL_ELEMENTS * 5000)
    assert benchmark.get_input_signature().shapes["workload"] == (CONDITIONAL_ELEMENTS, 1024, 5000)
    benchmark.teardown()


def test_conditional_native_binaries_dump_after_timing() -> None:
    for name in ("baseline_cuda_graphs_conditional.cu", "optimized_cuda_graphs_conditional.cu"):
        text = source(f"ch12/{name}")
        assert "--dump-output" in text and "OUTPUT_DUMPED: %zu" in text
        assert "std::fwrite(output.data(), sizeof(float), output.size(), file)" in text
        assert text.index("cudaEventElapsedTime") < text.rindex(
            "write_output_dump(dump_path, h_data)"
        )
